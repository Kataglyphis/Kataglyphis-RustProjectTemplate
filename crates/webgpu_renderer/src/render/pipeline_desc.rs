//! `wgpu::RenderPipeline`/`wgpu::PipelineLayout` constructors for the shapes
//! shared by `bloom.rs` (3 pipelines), `ssao.rs` (2), `tonemap.rs` (1) and
//! `ibl.rs` (5) - eleven fullscreen-triangle passes with no vertex buffers,
//! one color target and no depth test. Each site repeated the same ~20
//! fields (`vertex.buffers: &[]`, `primitive: PrimitiveState::default()`,
//! `depth_stencil: None`, `multisample: MultisampleState::default()`,
//! `multiview_mask: None`, `cache: None`, ...) around the 4 that actually
//! differ (label, entry point, format, blend), and had already grown two
//! spellings of "no blending" (`None` vs `Some(BlendState::REPLACE)`) between
//! copies. One helper per shape keeps the differing fields visible at the
//! call site instead of buried in a twenty-line literal.
//!
//! `single_layout` covers the sibling repetition: nine call sites across the
//! crate build a `PipelineLayout` from exactly one `BindGroupLayout` with no
//! immediate data. Sites that pass more than one bind group layout (the
//! forward, shadow and shadow-masked pipelines in `forward.rs`) are not this
//! shape and stay as they are.

/// One fullscreen-triangle render pipeline: a single color target, no vertex
/// buffers, no depth test. `label`, `vs_entry`, `fs_entry`, `format` and
/// `blend` are the fields that differ per pipeline; everything else is baked
/// into [`create_fullscreen_pipeline`].
pub struct FullscreenPipeline<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vs_entry: &'a str,
    pub fs_entry: &'a str,
    pub format: wgpu::TextureFormat,
    pub blend: Option<wgpu::BlendState>,
}

/// Builds a fullscreen-triangle render pipeline from a [`FullscreenPipeline`].
///
/// Any pass that needs a vertex buffer, MSAA or a depth test (e.g.
/// `create_forward_pipeline_set` in `forward.rs`) is a different shape and
/// should build its `RenderPipelineDescriptor` inline rather than force a
/// baked-in field here to become a parameter.
pub fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    desc: FullscreenPipeline<'_>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(desc.layout),
        vertex: wgpu::VertexState {
            module: desc.module,
            entry_point: Some(desc.vs_entry),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: desc.module,
            entry_point: Some(desc.fs_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: desc.format,
                blend: desc.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Builds a `PipelineLayout` from a single bind group layout with no
/// immediate data - the shape shared by nine pipeline-layout call sites
/// across the crate.
pub fn single_layout(
    device: &wgpu::Device,
    label: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    })
}

#[cfg(test)]
mod tests {
    /// Regression guard: the four fullscreen modules migrated in this task
    /// must go through `create_fullscreen_pipeline` rather than hand-rolling
    /// a `RenderPipelineDescriptor` again. If this fails, route the new
    /// pipeline through `create_fullscreen_pipeline`.
    #[test]
    fn the_fullscreen_passes_do_not_hand_roll_a_pipeline_descriptor() {
        for (name, src) in [
            ("bloom.rs", include_str!("bloom.rs")),
            ("ssao.rs", include_str!("ssao.rs")),
            ("tonemap.rs", include_str!("tonemap.rs")),
            ("ibl.rs", include_str!("ibl.rs")),
        ] {
            assert!(
                !src.contains("RenderPipelineDescriptor {"),
                "{name} hand-rolls a RenderPipelineDescriptor - use create_fullscreen_pipeline instead"
            );
        }
    }
}
