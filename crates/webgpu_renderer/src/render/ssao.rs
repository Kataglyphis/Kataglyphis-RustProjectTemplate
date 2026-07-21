//! Half-resolution depth-only SSAO with a 3x3 blur. The blurred AO factor
//! is multiplied into the HDR image by the tonemap pass.

use crate::context::GpuContext;
use crate::render::gpu_timing::PassScope;

const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

pub struct SsaoPass {
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    raw: Option<wgpu::TextureView>,
    blurred: Option<wgpu::TextureView>,
    bg_ssao: Option<wgpu::BindGroup>,
    bg_blur: Option<wgpu::BindGroup>,
}

impl SsaoPass {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make = |entry: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: AO_FORMAT,
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
            })
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ssao_pipeline: make("fs_ssao", "ssao_pipeline"),
            blur_pipeline: make("fs_blur", "ssao_blur_pipeline"),
            bind_group_layout,
            uniform_buffer,
            raw: None,
            blurred: None,
            bg_ssao: None,
            bg_blur: None,
        }
    }

    /// (Re)creates the half-res AO chain for a new depth buffer.
    pub fn rebuild(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        depth: &wgpu::TextureView,
    ) {
        let (w, h) = ((width / 2).max(1), (height / 2).max(1));
        let make_tex = |label: &str| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: AO_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let raw = make_tex("ssao_raw");
        let blurred = make_tex("ssao_blurred");

        let bind = |ao_input: &wgpu::TextureView, label: &str| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ao_input),
                    },
                ],
            })
        };
        // The ssao pass ignores binding 2; feed it the blurred texture (any
        // non-target texture) to satisfy the layout.
        self.bg_ssao = Some(bind(&blurred, "ssao_bg"));
        self.bg_blur = Some(bind(&raw, "ssao_blur_bg"));
        self.raw = Some(raw);
        self.blurred = Some(blurred);
    }

    /// Per-frame camera matrices + tuning.
    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        proj: glam::Mat4,
        radius: f32,
        bias: f32,
        intensity: f32,
    ) {
        let uniforms = SsaoUniforms {
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            params: [radius, bias, intensity, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// The blurred AO output (valid after `rebuild` + `encode`).
    pub fn output(&self) -> Option<&wgpu::TextureView> {
        self.blurred.as_ref()
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, scope: PassScope<'_>) {
        let (Some(raw), Some(blurred), Some(bg_ssao), Some(bg_blur)) = (
            self.raw.as_ref(),
            self.blurred.as_ref(),
            self.bg_ssao.as_ref(),
            self.bg_blur.as_ref(),
        ) else {
            return;
        };
        let steps: [(&wgpu::RenderPipeline, &wgpu::BindGroup, &wgpu::TextureView); 2] = [
            (&self.ssao_pipeline, bg_ssao, raw),
            (&self.blur_pipeline, bg_blur, blurred),
        ];
        let step_count = steps.len();
        for (step, (pipeline, bind_group, target)) in steps.into_iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: scope.render_writes(step, step_count),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
