//! A small render graph: passes declare the resources they read and write,
//! the graph validates the wiring and records them in order.
//!
//! This is deliberately *not* a full automatic-scheduling graph. It gives
//! the three things that actually hurt as pass count grows:
//!
//! 1. **A declared frame structure** — the pass list is data, printable and
//!    testable, instead of an implicit sequence buried in one long function.
//! 2. **Read/write validation** — a pass that reads a resource nothing has
//!    written, or two passes writing the same resource in one frame, is a
//!    hard error at build time rather than a silently black screen.
//! 3. **One place to add cross-cutting behavior** — timing scopes, debug
//!    markers, or a "disable this pass" toggle apply to every pass at once.
//!
//! Execution order stays explicit (the order passes are added), because at
//! this scale a topological sort would hide more than it automates.

#[cfg(test)]
use super::gpu_timing::TimedPass;
use std::collections::HashSet;

/// Named GPU resources the passes exchange within a frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    /// Cascaded shadow depth array.
    ShadowMap,
    /// MSAA depth buffer written by the forward pass.
    DepthMsaa,
    /// Single-sample depth resolved from `DepthMsaa`; SSAO and the cull
    /// pass's input.
    Depth,
    /// HDR scene color.
    HdrColor,
    /// Occlusion/visibility result of the cull pass. Consumed by the
    /// *following* frame's forward pass (a one-frame-latent feedback loop),
    /// so nothing reads it within this frame's graph.
    Visibility,
    /// Blurred bloom contribution.
    Bloom,
    /// Blurred ambient occlusion factor.
    Ambient,
    /// Luminance histogram bins.
    Histogram,
    /// Adapted exposure value reduced from `Histogram`.
    Exposure,
    /// Final display target (swapchain frame or readback texture).
    Output,
}

/// A recorded pass: what it touches, and how to encode it.
pub struct PassDesc<'a> {
    pub name: &'static str,
    pub reads: &'a [Resource],
    pub writes: &'a [Resource],
}

#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A pass reads a resource that no earlier pass wrote.
    UndefinedRead {
        pass: &'static str,
        resource: Resource,
    },
    /// Two passes write the same resource in one frame.
    DuplicateWrite {
        pass: &'static str,
        resource: Resource,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::UndefinedRead { pass, resource } => write!(
                f,
                "pass '{pass}' reads {resource:?} before anything writes it"
            ),
            GraphError::DuplicateWrite { pass, resource } => {
                write!(f, "pass '{pass}' writes {resource:?} twice in one frame")
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Validates a frame's pass list: every read must be satisfied by an
/// earlier write, and no resource may be written twice.
///
/// Resources listed in `external` are considered already available (e.g.
/// the swapchain frame the caller hands in).
pub fn validate(passes: &[PassDesc<'_>], external: &[Resource]) -> Result<(), GraphError> {
    let mut written: HashSet<Resource> = external.iter().copied().collect();
    for pass in passes {
        for resource in pass.reads {
            if !written.contains(resource) {
                return Err(GraphError::UndefinedRead {
                    pass: pass.name,
                    resource: *resource,
                });
            }
        }
        for resource in pass.writes {
            if !written.insert(*resource) {
                return Err(GraphError::DuplicateWrite {
                    pass: pass.name,
                    resource: *resource,
                });
            }
        }
    }
    Ok(())
}

/// The frame graph the forward renderer records, as data. Kept next to the
/// recording code so the two stay in step; `validate` proves the wiring.
///
/// This is the *maximal* frame: passes the renderer may skip at runtime
/// (bloom and SSAO at zero strength, occlusion cull when disabled, and the
/// histogram build when auto-exposure is off) are still declared here, since
/// the graph documents wiring, not what a given frame actually recorded. Do
/// not add conditional rows for that - `validate` only needs to prove a
/// legal read/write order exists, not reproduce runtime toggles.
pub fn forward_frame_graph() -> Vec<PassDesc<'static>> {
    vec![
        PassDesc {
            name: "shadow",
            reads: &[],
            writes: &[Resource::ShadowMap],
        },
        PassDesc {
            name: "forward+sky",
            reads: &[Resource::ShadowMap],
            writes: &[Resource::HdrColor, Resource::DepthMsaa],
        },
        PassDesc {
            name: "depth_resolve",
            reads: &[Resource::DepthMsaa],
            writes: &[Resource::Depth],
        },
        PassDesc {
            name: "occlusion_cull",
            reads: &[Resource::Depth],
            writes: &[Resource::Visibility],
        },
        PassDesc {
            name: "bloom",
            reads: &[Resource::HdrColor],
            writes: &[Resource::Bloom],
        },
        PassDesc {
            name: "ssao",
            reads: &[Resource::Depth],
            writes: &[Resource::Ambient],
        },
        PassDesc {
            name: "histogram",
            reads: &[Resource::HdrColor],
            writes: &[Resource::Histogram],
        },
        PassDesc {
            name: "exposure_reduce",
            reads: &[Resource::Histogram],
            writes: &[Resource::Exposure],
        },
        PassDesc {
            name: "tonemap",
            reads: &[
                Resource::HdrColor,
                Resource::Bloom,
                Resource::Ambient,
                Resource::Exposure,
            ],
            writes: &[Resource::Output],
        },
    ]
}

/// Maps a timed pass to the name of the graph row that records it. A
/// `match` (rather than a string lookup) makes a new `TimedPass` variant a
/// compile error here, which is what keeps this mapping - and the test built
/// on it - honest.
#[cfg(test)]
fn graph_pass_name(pass: TimedPass) -> &'static str {
    match pass {
        TimedPass::ShadowCascades => "shadow",
        TimedPass::Forward => "forward+sky",
        TimedPass::OcclusionCull => "occlusion_cull",
        TimedPass::Bloom => "bloom",
        TimedPass::Ssao => "ssao",
        TimedPass::Histogram => "histogram",
        TimedPass::ExposureReduce => "exposure_reduce",
        TimedPass::Tonemap => "tonemap",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_graph_is_valid() {
        validate(&forward_frame_graph(), &[]).expect("the shipped frame graph must validate");
    }

    #[test]
    fn every_timed_pass_has_a_graph_row() {
        let graph = forward_frame_graph();
        for pass in TimedPass::ALL {
            let expected = graph_pass_name(pass);
            assert!(
                graph.iter().any(|p| p.name == expected),
                "TimedPass::{pass:?} maps to graph row '{expected}', but no such row exists"
            );
        }
    }

    #[test]
    fn detects_reading_undefined_resource() {
        let passes = vec![PassDesc {
            name: "tonemap",
            reads: &[Resource::Bloom],
            writes: &[Resource::Output],
        }];
        assert_eq!(
            validate(&passes, &[]),
            Err(GraphError::UndefinedRead {
                pass: "tonemap",
                resource: Resource::Bloom
            })
        );
    }

    #[test]
    fn detects_double_write() {
        let passes = vec![
            PassDesc {
                name: "a",
                reads: &[],
                writes: &[Resource::HdrColor],
            },
            PassDesc {
                name: "b",
                reads: &[],
                writes: &[Resource::HdrColor],
            },
        ];
        assert_eq!(
            validate(&passes, &[]),
            Err(GraphError::DuplicateWrite {
                pass: "b",
                resource: Resource::HdrColor
            })
        );
    }

    #[test]
    fn external_resources_satisfy_reads() {
        let passes = vec![PassDesc {
            name: "overlay",
            reads: &[Resource::Output],
            writes: &[],
        }];
        assert!(validate(&passes, &[Resource::Output]).is_ok());
    }
}
