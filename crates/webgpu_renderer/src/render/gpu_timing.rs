//! Per-pass GPU timestamp queries.
//!
//! The point of this module is comparability: the C++ Vulkan engine in the
//! sibling tree reports per-pass GPU times (`GUIRendererSharedVars::gpuTimings`,
//! keyed by a `GpuTimedPass` enum), and a side-by-side harness can only compare
//! numbers that are shaped the same way. So the output here is deliberately the
//! same shape - a fixed set of named passes, each a duration in milliseconds.
//!
//! Three constraints drove the design:
//!
//! 1. **`TIMESTAMP_QUERY` is optional, and this renderer targets the web.**
//!    Browsers gate the feature behind a flag and most do not expose it at all.
//!    So the subsystem has an explicit "unavailable" state carrying no wgpu
//!    resources ([`GpuTiming::unavailable`]), every pass asks for its timestamp
//!    writes and gets `None`, and the frame records exactly as it did before.
//!    Nothing here may panic or unwrap on a missing feature.
//!
//! 2. **A pass is not a render pass.** Shadows are `CASCADE_COUNT` render
//!    passes, bloom is three, SSAO two, the histogram two compute passes. WebGPU
//!    can only stamp a timestamp at the start or end of a pass, so a scope
//!    spans sub-passes: the first sub-pass writes the begin query, the last
//!    writes the end query, and the middle ones write nothing. That is what
//!    [`PassScope::render_writes`] encodes.
//!
//! 3. **Reading results must not stall the frame.** Timestamps land in a ring
//!    of `SLOT_COUNT` slots; a slot's readback is mapped asynchronously after
//!    submit and consumed whenever it happens to be ready, which in practice is
//!    two to three frames later. A slot still in flight is simply skipped for
//!    that frame rather than waited on, so the frame path never blocks.
//!
//! Rejected alternative: `CommandEncoder::write_timestamp`, which reads much
//! more naturally (stamp anywhere, no sub-pass bookkeeping). It needs
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS`, a second optional feature that is rarer
//! than the first and absent on WebGPU entirely - it would have made the web
//! target permanently untimed.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// The passes we time, in the order they are recorded in a frame.
///
/// Opaque and blended geometry are *not* split, though they are conceptually
/// two passes: they share one `wgpu::RenderPass` (the blended draws need the
/// opaque depth buffer without a store/load round trip), and a timestamp can
/// only be written at a pass boundary. Splitting the render pass purely to
/// time it would change what is being measured.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TimedPass {
    /// All shadow cascades, one render pass each.
    ShadowCascades,
    /// Opaque geometry, sky, and blended geometry in one render pass.
    Forward,
    /// The occlusion-query bounding-box pass (one render pass, only recorded
    /// when occlusion culling is on). Its cost is the overhead the cull must
    /// beat: measuring it is how you tell whether culling actually pays.
    OcclusionCull,
    /// Brightpass plus separable blur (three render passes).
    Bloom,
    /// Occlusion plus blur (two render passes).
    Ssao,
    /// Histogram clear plus build (two compute passes).
    Histogram,
    /// Histogram -> adapted exposure reduction (one compute pass).
    ExposureReduce,
    /// ACES tonemap and composite to the output target.
    Tonemap,
}

impl TimedPass {
    /// Every pass, in record order. Iteration order is the reporting order.
    pub const ALL: [TimedPass; 8] = [
        TimedPass::ShadowCascades,
        TimedPass::Forward,
        TimedPass::OcclusionCull,
        TimedPass::Bloom,
        TimedPass::Ssao,
        TimedPass::Histogram,
        TimedPass::ExposureReduce,
        TimedPass::Tonemap,
    ];

    /// Stable identifier for reports and for matching against the C++ engine.
    pub fn name(self) -> &'static str {
        match self {
            TimedPass::ShadowCascades => "ShadowCascades",
            TimedPass::Forward => "Forward",
            TimedPass::OcclusionCull => "OcclusionCull",
            TimedPass::Bloom => "Bloom",
            TimedPass::Ssao => "Ssao",
            TimedPass::Histogram => "Histogram",
            TimedPass::ExposureReduce => "ExposureReduce",
            TimedPass::Tonemap => "Tonemap",
        }
    }

    fn index(self) -> usize {
        match self {
            TimedPass::ShadowCascades => 0,
            TimedPass::Forward => 1,
            TimedPass::OcclusionCull => 2,
            TimedPass::Bloom => 3,
            TimedPass::Ssao => 4,
            TimedPass::Histogram => 5,
            TimedPass::ExposureReduce => 6,
            TimedPass::Tonemap => 7,
        }
    }
}

/// Number of timed passes.
pub const PASS_COUNT: usize = TimedPass::ALL.len();
/// Two queries (begin, end) per pass.
const QUERIES_PER_SLOT: u32 = PASS_COUNT as u32 * 2;
/// Frames of timestamp storage in flight.
///
/// Two would cover wgpu's `desired_maximum_frame_latency: 2`, but a slot is
/// only recycled once its map callback has fired, and that fires on a poll -
/// one spare slot means a frame where the poll came up empty still finds a
/// free slot instead of dropping the sample.
const SLOT_COUNT: usize = 3;
/// Samples averaged for the reported value.
///
/// Raw per-frame GPU times jitter by tens of percent (clock/residency noise),
/// which makes an on-screen number unreadable. 32 frames is roughly half a
/// second at 60 Hz: steady enough to read, short enough to react to a change.
const AVERAGE_WINDOW: usize = 32;

/// Fixed-window mean over the most recent samples.
///
/// A ring rather than an exponential decay because the window is an exact,
/// statable number of frames - "the mean of the last 32 frames" is something a
/// comparison harness can reproduce, where a decay constant is not.
#[derive(Clone, Debug)]
pub struct RollingAverage {
    samples: [f32; AVERAGE_WINDOW],
    /// Where the next sample lands.
    next: usize,
    /// Samples recorded so far, saturating at the window size.
    filled: usize,
    sum: f32,
}

impl Default for RollingAverage {
    fn default() -> Self {
        Self {
            samples: [0.0; AVERAGE_WINDOW],
            next: 0,
            filled: 0,
            sum: 0.0,
        }
    }
}

impl RollingAverage {
    /// Records one sample, evicting the oldest once the window is full.
    pub fn push(&mut self, value: f32) {
        if self.filled == AVERAGE_WINDOW {
            self.sum -= self.samples[self.next];
        } else {
            self.filled += 1;
        }
        self.samples[self.next] = value;
        self.sum += value;
        self.next = (self.next + 1) % AVERAGE_WINDOW;
    }

    /// Mean of the recorded samples, or `None` before the first sample.
    ///
    /// `None` rather than 0.0 on purpose: "this pass has never reported" and
    /// "this pass took no measurable time" are different facts, and a caller
    /// printing 0.00 ms for the former is exactly the fake number this
    /// subsystem exists to avoid.
    pub fn average(&self) -> Option<f32> {
        if self.filled == 0 {
            None
        } else {
            Some(self.sum / self.filled as f32)
        }
    }

    /// Number of samples currently in the window.
    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }
}

/// Converts a GPU timestamp delta to milliseconds.
///
/// `period_ns` is whatever `Queue::get_timestamp_period()` reports - it varies
/// by an order of magnitude across vendors (~1 ns on current NVIDIA, ~40 ns on
/// some AMD parts), so it is never assumed.
///
/// The arithmetic is done in `f64`: a tick count is a `u64`, and on a 1 ns
/// timer a GPU that has been up for an hour is already past `f32`'s 24-bit
/// mantissa, so an `f32` multiply would quantise the delta.
pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f32 {
    (ticks as f64 * period_ns as f64 / 1_000_000.0) as f32
}

/// The timestamp writes one pass should hand to its render/compute passes.
///
/// Cheap to copy and safe to ask for when timing is off, in which case every
/// accessor yields `None` and call sites need no branch of their own.
#[derive(Copy, Clone)]
pub struct PassScope<'a> {
    query_set: Option<&'a wgpu::QuerySet>,
    begin: u32,
    end: u32,
}

// The write descriptors borrow the query set (`'a`), NOT the scope: a call site
// that builds a scope inline and immediately asks it for writes would otherwise
// be handing wgpu a borrow of a temporary.
impl<'a> PassScope<'a> {
    /// A scope that writes nothing - timing off, or feature unavailable.
    pub fn disabled() -> Self {
        Self {
            query_set: None,
            begin: 0,
            end: 0,
        }
    }

    /// True when this scope will actually write timestamps.
    pub fn is_active(&self) -> bool {
        self.query_set.is_some()
    }

    /// Timestamp writes for sub-pass `sub` of `count` in this scope.
    ///
    /// The begin query rides the first sub-pass and the end query the last, so
    /// the measured span covers the whole scope including the gaps between its
    /// sub-passes. With `count == 1` both land on the same pass.
    pub fn render_writes(
        &self,
        sub: usize,
        count: usize,
    ) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
        let (query_set, beginning, end) = self.indices(sub, count)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: beginning,
            end_of_pass_write_index: end,
        })
    }

    /// As [`Self::render_writes`], for compute passes.
    pub fn compute_writes(
        &self,
        sub: usize,
        count: usize,
    ) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        let (query_set, beginning, end) = self.indices(sub, count)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: beginning,
            end_of_pass_write_index: end,
        })
    }

    fn indices(
        &self,
        sub: usize,
        count: usize,
    ) -> Option<(&'a wgpu::QuerySet, Option<u32>, Option<u32>)> {
        let query_set = self.query_set?;
        let (beginning, end) = sub_pass_write_indices(sub, count, self.begin, self.end)?;
        Some((query_set, beginning, end))
    }
}

/// Which of a scope's two queries sub-pass `sub` of `count` should write.
///
/// Split out from [`PassScope`] because it is the whole of the sub-pass
/// bookkeeping and it is pure - it can be tested without a GPU. `None` means
/// this sub-pass stamps nothing, which must become "no descriptor at all":
/// wgpu rejects a timestamp-writes descriptor with both indices unset.
fn sub_pass_write_indices(
    sub: usize,
    count: usize,
    begin: u32,
    end: u32,
) -> Option<(Option<u32>, Option<u32>)> {
    let beginning = (sub == 0).then_some(begin);
    let last = (count > 0 && sub + 1 == count).then_some(end);
    if beginning.is_none() && last.is_none() {
        return None;
    }
    Some((beginning, last))
}

/// Map state of one readback slot, shared with the `map_async` callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

struct Slot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// `Some` while a map is outstanding; the callback flips it to
    /// READY/FAILED and the next `end_frame` drains it.
    map_state: Option<Arc<AtomicU8>>,
}

struct Resources {
    query_set: wgpu::QuerySet,
    slots: Vec<Slot>,
    /// Slot the current frame writes into.
    current: usize,
    period_ns: f32,
}

/// Per-pass GPU timings, or an inert stand-in when timestamps are unavailable.
pub struct GpuTiming {
    /// `None` when `TIMESTAMP_QUERY` is absent or timing was never enabled.
    resources: Option<Resources>,
    /// Set by `begin_frame` when this frame found a free slot to record into.
    recording: bool,
    averages: Vec<RollingAverage>,
}

impl GpuTiming {
    /// The state every caller can always construct: no queries, no timings.
    ///
    /// Deliberately takes no device, so the unsupported path is reachable in a
    /// unit test on a machine with no GPU at all.
    pub fn unavailable() -> Self {
        Self {
            resources: None,
            recording: false,
            averages: vec![RollingAverage::default(); PASS_COUNT],
        }
    }

    /// Allocates query and readback resources, or returns the inert instance
    /// when the device lacks `TIMESTAMP_QUERY`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            log::info!("TIMESTAMP_QUERY unavailable; per-pass GPU timings disabled");
            return Self::unavailable();
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timing_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERIES_PER_SLOT * SLOT_COUNT as u32,
        });
        let bytes = u64::from(QUERIES_PER_SLOT) * 8;
        let slots = (0..SLOT_COUNT)
            .map(|i| Slot {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("gpu_timing_resolve_{i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("gpu_timing_readback_{i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                map_state: None,
            })
            .collect();

        Self {
            resources: Some(Resources {
                query_set,
                slots,
                current: 0,
                period_ns: queue.get_timestamp_period(),
            }),
            recording: false,
            averages: vec![RollingAverage::default(); PASS_COUNT],
        }
    }

    /// True when timestamps are being collected.
    pub fn is_available(&self) -> bool {
        self.resources.is_some()
    }

    /// Nanoseconds per timestamp tick, as reported by the queue.
    pub fn timestamp_period_ns(&self) -> Option<f32> {
        self.resources.as_ref().map(|r| r.period_ns)
    }

    /// Averaged duration of each pass that has reported at least once.
    ///
    /// Empty while unavailable, and short of `PASS_COUNT` entries until every
    /// pass has produced a sample - a caller must not read absence as zero.
    pub fn timings_ms(&self) -> Vec<(&'static str, f32)> {
        TimedPass::ALL
            .iter()
            .filter_map(|pass| {
                self.averages[pass.index()]
                    .average()
                    .map(|ms| (pass.name(), ms))
            })
            .collect()
    }

    /// Opens the frame: claims a slot if one is free.
    ///
    /// A slot whose readback is still mapping cannot be overwritten, so if
    /// every slot is busy this frame goes untimed. Dropping a sample is the
    /// right trade against either stalling or corrupting an in-flight one.
    pub fn begin_frame(&mut self) {
        self.recording = false;
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        for offset in 0..SLOT_COUNT {
            let candidate = (resources.current + offset) % SLOT_COUNT;
            if resources.slots[candidate].map_state.is_none() {
                resources.current = candidate;
                self.recording = true;
                return;
            }
        }
    }

    /// The scope `pass` should stamp into this frame.
    pub fn scope(&self, pass: TimedPass) -> PassScope<'_> {
        if !self.recording {
            return PassScope::disabled();
        }
        let Some(resources) = self.resources.as_ref() else {
            return PassScope::disabled();
        };
        let base = resources.current as u32 * QUERIES_PER_SLOT + pass.index() as u32 * 2;
        PassScope {
            query_set: Some(&resources.query_set),
            begin: base,
            end: base + 1,
        }
    }

    /// Resolves this frame's queries into the slot's readback buffer. Must be
    /// recorded into the same encoder as the passes, before submit.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.recording {
            return;
        }
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        let slot = &resources.slots[resources.current];
        let first = resources.current as u32 * QUERIES_PER_SLOT;
        encoder.resolve_query_set(
            &resources.query_set,
            first..first + QUERIES_PER_SLOT,
            &slot.resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, slot.resolve.size());
    }

    /// Closes the frame: starts this slot's readback and consumes any slot
    /// whose readback has since completed. Never blocks.
    pub fn end_frame(&mut self, device: &wgpu::Device) {
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        if self.recording {
            let state = Arc::new(AtomicU8::new(MAP_PENDING));
            let callback_state = Arc::clone(&state);
            resources.slots[resources.current]
                .readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let code = if result.is_ok() {
                        MAP_READY
                    } else {
                        MAP_FAILED
                    };
                    callback_state.store(code, Ordering::Release);
                });
            resources.slots[resources.current].map_state = Some(state);
            resources.current = (resources.current + 1) % SLOT_COUNT;
            self.recording = false;
        }

        // Non-blocking: this only lets already-finished map callbacks run. A
        // waiting poll here would reintroduce exactly the stall the ring
        // exists to avoid.
        let _ = device.poll(wgpu::PollType::Poll);

        for index in 0..resources.slots.len() {
            let Some(state) = resources.slots[index].map_state.as_ref() else {
                continue;
            };
            match state.load(Ordering::Acquire) {
                MAP_READY => {
                    let ticks = read_ticks(&resources.slots[index].readback);
                    resources.slots[index].readback.unmap();
                    resources.slots[index].map_state = None;
                    for pass in TimedPass::ALL {
                        let i = pass.index();
                        let (begin, end) = (ticks[i * 2], ticks[i * 2 + 1]);
                        // Queries a skipped pass never wrote resolve to
                        // undefined contents; a real GPU timestamp is never 0
                        // and never runs backwards, so those two checks reject
                        // the garbage without inventing a number for it.
                        if begin == 0 || end == 0 || end < begin {
                            continue;
                        }
                        self.averages[i].push(ticks_to_ms(end - begin, resources.period_ns));
                    }
                }
                MAP_FAILED => {
                    resources.slots[index].map_state = None;
                }
                _ => {}
            }
        }
    }
}

/// Reads `QUERIES_PER_SLOT` u64 ticks out of a mapped readback buffer.
fn read_ticks(buffer: &wgpu::Buffer) -> Vec<u64> {
    let view = buffer.slice(..).get_mapped_range();
    let ticks = view
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("chunks_exact(8) yields 8 bytes")))
        .collect();
    drop(view);
    ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_reports_nothing_and_hands_out_inert_scopes() {
        let mut timing = GpuTiming::unavailable();
        assert!(!timing.is_available());
        assert_eq!(timing.timestamp_period_ns(), None);
        timing.begin_frame();
        for pass in TimedPass::ALL {
            let scope = timing.scope(pass);
            assert!(!scope.is_active());
            assert!(scope.render_writes(0, 1).is_none());
            assert!(scope.compute_writes(0, 1).is_none());
        }
        assert!(timing.timings_ms().is_empty());
    }

    #[test]
    fn ticks_convert_with_the_reported_period() {
        // 1 000 000 ticks at 1 ns each is exactly one millisecond.
        assert_eq!(ticks_to_ms(1_000_000, 1.0), 1.0);
        // A coarse timer (some AMD parts report ~40 ns) scales accordingly.
        assert_eq!(ticks_to_ms(25_000, 40.0), 1.0);
        assert_eq!(ticks_to_ms(0, 1.0), 0.0);
    }

    #[test]
    fn tick_conversion_keeps_precision_on_a_large_timestamp_delta() {
        // A delta past f32's 24-bit mantissa: computed in f32 this rounds to a
        // multiple of 2 ticks and the sub-microsecond digits vanish.
        let ticks = 16_777_217u64;
        assert_eq!(ticks_to_ms(ticks, 1.0), 16.777217);
    }

    #[test]
    fn rolling_average_starts_empty_then_means_its_samples() {
        let mut average = RollingAverage::default();
        assert_eq!(average.average(), None);
        assert!(average.is_empty());
        average.push(2.0);
        average.push(4.0);
        assert_eq!(average.len(), 2);
        assert_eq!(average.average(), Some(3.0));
    }

    #[test]
    fn rolling_average_evicts_beyond_its_window() {
        let mut average = RollingAverage::default();
        for _ in 0..AVERAGE_WINDOW {
            average.push(10.0);
        }
        assert_eq!(average.average(), Some(10.0));
        // One full window of a new value must displace the old one entirely.
        for _ in 0..AVERAGE_WINDOW {
            average.push(20.0);
        }
        assert_eq!(average.len(), AVERAGE_WINDOW);
        assert_eq!(average.average(), Some(20.0));
    }

    #[test]
    fn pass_names_are_unique_and_indices_are_dense() {
        let mut indices: Vec<usize> = TimedPass::ALL.iter().map(|p| p.index()).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..PASS_COUNT).collect::<Vec<_>>());
        let mut names: Vec<&str> = TimedPass::ALL.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PASS_COUNT);
    }

    #[test]
    fn a_single_sub_pass_scope_stamps_both_ends() {
        assert_eq!(sub_pass_write_indices(0, 1, 4, 5), Some((Some(4), Some(5))));
    }

    #[test]
    fn a_multi_sub_pass_scope_stamps_only_its_outer_passes() {
        // Bloom: brightpass, blur H, blur V. The middle one must get no
        // descriptor at all rather than one with both indices unset.
        assert_eq!(sub_pass_write_indices(0, 3, 4, 5), Some((Some(4), None)));
        assert_eq!(sub_pass_write_indices(1, 3, 4, 5), None);
        assert_eq!(sub_pass_write_indices(2, 3, 4, 5), Some((None, Some(5))));
    }
}
