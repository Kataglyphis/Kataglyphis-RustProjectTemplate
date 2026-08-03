//! Fullscreen ACES tonemap pass: HDR (Rgba16Float) input -> display target.

use crate::context::GpuContext;
use crate::render::bind_layout;
use crate::render::gpu_timing::PassScope;

pub struct TonemapPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    /// Whether the shader must gamma-encode its output itself.
    ///
    /// An sRGB target encodes in hardware, so the shader emits linear and the
    /// flag is false. WebGPU canvases expose no sRGB surface format - the
    /// browser hands back something like `Bgra8Unorm` - and writing linear
    /// values to a non-sRGB target displays them uncorrected, which is why the
    /// web demo rendered noticeably dark. There the shader has to apply the
    /// transfer function itself.
    encode_srgb: bool,
}

impl TonemapPass {
    /// `output_format` is the format of the view passed to [`Self::render`], e.g.
    /// the surface format or `Rgba8UnormSrgb` for readback targets.
    pub fn new(gpu: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tonemap.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_bind_group_layout"),
            entries: &[
                bind_layout::texture_2d(0, wgpu::ShaderStages::FRAGMENT, true),
                bind_layout::sampler(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::SamplerBindingType::Filtering,
                ),
                bind_layout::texture_2d(2, wgpu::ShaderStages::FRAGMENT, true),
                bind_layout::uniform(3, wgpu::ShaderStages::FRAGMENT),
                bind_layout::texture_2d(4, wgpu::ShaderStages::FRAGMENT, true),
                // Exposure comes from the auto-exposure reduction rather than
                // a CPU uniform, so no frame has to wait on a readback.
                bind_layout::storage_buffer(5, wgpu::ShaderStages::FRAGMENT, true),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemap_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap_uniforms"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            bind_group: None,
            encode_srgb: !output_format.is_srgb(),
        }
    }

    /// Per-frame parameters.
    pub fn set_params(&self, queue: &wgpu::Queue, bloom_strength: f32, ssao_strength: f32) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&[
                bloom_strength,
                ssao_strength,
                // params.z: unused. Exposure comes from exposureState (see
                // TonemapUniforms in tonemap.slang) so this used to carry a
                // dead exposure_ev.exp2() write; kept as a field because the
                // uniform layout is pinned by the shader struct.
                0.0,
                // params.w: see `encode_srgb`. Decided by the output format at
                // pipeline creation, not per frame, but it rides along here
                // because the uniform already exists and its w was unused.
                if self.encode_srgb { 1.0 } else { 0.0 },
            ]),
        );
    }

    /// (Re)binds the HDR + bloom inputs; call whenever they are recreated.
    pub fn set_input(
        &mut self,
        gpu: &GpuContext,
        hdr_view: &wgpu::TextureView,
        bloom_view: &wgpu::TextureView,
        ao_view: &wgpu::TextureView,
        exposure_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: exposure_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        scope: PassScope<'_>,
    ) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            log::error!("TonemapPass::render called before set_input");
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: scope.render_writes(0, 1),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
