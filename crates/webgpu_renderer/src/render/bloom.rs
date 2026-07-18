//! Half-resolution bloom chain: bright-pass -> horizontal blur -> vertical
//! blur, ping-ponging two Rgba16Float textures. The blurred result is
//! composited by the tonemap pass.

use crate::context::GpuContext;
use crate::render::forward::HDR_FORMAT;

pub struct BloomPass {
    brightpass: wgpu::RenderPipeline,
    blur_h: wgpu::RenderPipeline,
    blur_v: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture_a: Option<wgpu::TextureView>,
    texture_b: Option<wgpu::TextureView>,
    bg_bright: Option<wgpu::BindGroup>,
    bg_h: Option<wgpu::BindGroup>,
    bg_v: Option<wgpu::BindGroup>,
    size: (u32, u32),
}

impl BloomPass {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            brightpass: make("fs_brightpass", "bloom_brightpass"),
            blur_h: make("fs_blur_h", "bloom_blur_h"),
            blur_v: make("fs_blur_v", "bloom_blur_v"),
            bind_group_layout,
            sampler,
            texture_a: None,
            texture_b: None,
            bg_bright: None,
            bg_h: None,
            bg_v: None,
            size: (0, 0),
        }
    }

    /// (Re)creates the half-res chain for a new HDR target.
    pub fn rebuild(&mut self, gpu: &GpuContext, width: u32, height: u32, hdr: &wgpu::TextureView) {
        let (w, h) = ((width / 2).max(1), (height / 2).max(1));
        self.size = (w, h);
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
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let a = make_tex("bloom_a");
        let b = make_tex("bloom_b");

        let bind = |src: &wgpu::TextureView, label: &str| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        self.bg_bright = Some(bind(hdr, "bloom_bg_bright"));
        self.bg_h = Some(bind(&a, "bloom_bg_h"));
        self.bg_v = Some(bind(&b, "bloom_bg_v"));
        self.texture_a = Some(a);
        self.texture_b = Some(b);
    }

    /// The blurred bloom output (valid after `rebuild` + `encode`).
    pub fn output(&self) -> Option<&wgpu::TextureView> {
        self.texture_a.as_ref()
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(a), Some(b), Some(bg_bright), Some(bg_h), Some(bg_v)) = (
            self.texture_a.as_ref(),
            self.texture_b.as_ref(),
            self.bg_bright.as_ref(),
            self.bg_h.as_ref(),
            self.bg_v.as_ref(),
        ) else {
            return;
        };
        let steps: [(&wgpu::RenderPipeline, &wgpu::BindGroup, &wgpu::TextureView); 3] = [
            (&self.brightpass, bg_bright, a),
            (&self.blur_h, bg_h, b),
            (&self.blur_v, bg_v, a),
        ];
        for (pipeline, bind_group, target) in steps {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
