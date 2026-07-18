//! Forward render pass over a `CpuScene` into an internal HDR target,
//! composited to the output through the ACES tonemap pass. Also provides
//! headless render-to-pixels for golden tests and CI.

use anyhow::Context as _;
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt as _;

use crate::context::GpuContext;
use crate::render::tonemap::TonemapPass;
use crate::scene::camera::OrbitCamera;
use crate::scene::{CpuScene, CpuTexture, Vertex};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_space: [[f32; 4]; 4],
    base_color: [f32; 4],
    light_dir_ambient: [f32; 4],
    light_color_intensity: [f32; 4],
}

struct GpuPrimitive {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Uniforms-only group for the shadow pass: the full group samples the
    /// shadow map, which the shadow pass writes — an exclusive-usage conflict.
    shadow_bind_group: wgpu::BindGroup,
    model: Mat4,
    base_color: [f32; 4],
}

pub struct ForwardRenderer {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    shadow_bind_group_layout: wgpu::BindGroupLayout,
    default_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    white_texture_view: wgpu::TextureView,
    shadow_view: wgpu::TextureView,
    primitives: Vec<GpuPrimitive>,
    scene_bounds: Option<(Vec3, Vec3)>,
    depth: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    target_size: (u32, u32),
    hdr_rebound_needed: bool,
    /// Direction towards the light (world space) + ambient strength.
    pub light_dir_ambient: Vec4,
    /// Light color (rgb) + intensity multiplier (w). Values > 1 are the
    /// point of the HDR pipeline.
    pub light_color_intensity: Vec4,
}

impl ForwardRenderer {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/forward.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[&shadow_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Depth-only pipeline rendering the scene from the light's POV.
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("base_color_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let white_texture_view = create_base_color_texture(
            gpu,
            &CpuTexture {
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
            },
            Some("white_fallback"),
        );

        let shadow_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow_map"),
                size: wgpu::Extent3d {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth = create_depth_texture(device, width, height);
        let hdr_view = create_hdr_texture(device, width, height);

        Self {
            pipeline,
            shadow_pipeline,
            bind_group_layout,
            shadow_bind_group_layout,
            default_sampler,
            shadow_sampler,
            white_texture_view,
            shadow_view,
            primitives: Vec::new(),
            scene_bounds: None,
            depth,
            hdr_view,
            target_size: (width.max(1), height.max(1)),
            hdr_rebound_needed: true,
            light_dir_ambient: Vec4::new(0.5, 0.8, 0.3, 0.15),
            light_color_intensity: Vec4::new(1.0, 0.97, 0.92, 1.6),
        }
    }

    /// Uploads a CPU scene, replacing any previously uploaded one.
    pub fn upload_scene(&mut self, gpu: &GpuContext, scene: &CpuScene) {
        let device = &gpu.device;
        self.primitives.clear();
        self.scene_bounds = compute_world_bounds(scene);

        for (i, prim) in scene.primitives.iter().enumerate() {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vertices_{i}")),
                contents: bytemuck::cast_slice(&prim.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("indices_{i}")),
                contents: bytemuck::cast_slice(&prim.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("uniforms_{i}")),
                size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let texture_view = match prim.material.base_color_texture.as_deref() {
                Some(texture) => {
                    create_base_color_texture(gpu, texture, Some(&format!("base_color_{i}")))
                }
                None => self.white_texture_view.clone(),
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bind_group_{i}")),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
            });

            let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_bind_group_{i}")),
                layout: &self.shadow_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            self.primitives.push(GpuPrimitive {
                vertex_buffer,
                index_buffer,
                index_count: prim.indices.len() as u32,
                uniform_buffer,
                bind_group,
                shadow_bind_group,
                model: prim.transform,
                base_color: prim.material.base_color,
            });
        }
    }

    /// Renders the scene HDR->tonemap into `output_view` (surface frame or
    /// offscreen texture). `width`/`height` must match `output_view`.
    pub fn render_tonemapped(
        &mut self,
        gpu: &GpuContext,
        tonemap: &mut TonemapPass,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
    ) {
        let (width, height) = (width.max(1), height.max(1));
        if self.target_size != (width, height) {
            self.depth = create_depth_texture(&gpu.device, width, height);
            self.hdr_view = create_hdr_texture(&gpu.device, width, height);
            self.target_size = (width, height);
            self.hdr_rebound_needed = true;
        }
        if self.hdr_rebound_needed {
            tonemap.set_input(gpu, &self.hdr_view);
            self.hdr_rebound_needed = false;
        }

        let aspect = width as f32 / height as f32;
        let view_proj = camera.view_projection(aspect);
        let light_space = self.light_space_matrix();

        for prim in &self.primitives {
            let uniforms = Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                model: prim.model.to_cols_array_2d(),
                light_space: light_space.to_cols_array_2d(),
                base_color: prim.base_color,
                light_dir_ambient: self.light_dir_ambient.to_array(),
                light_color_intensity: self.light_color_intensity.to_array(),
            };
            gpu.queue
                .write_buffer(&prim.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            for prim in &self.primitives {
                pass.set_bind_group(0, &prim.shadow_bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("forward_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            for prim in &self.primitives {
                pass.set_bind_group(0, &prim.bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }
        }

        tonemap.render(&mut encoder, output_view);
        gpu.queue.submit(Some(encoder.finish()));
    }

    /// Headless helper: renders one tonemapped frame into a fresh RGBA8
    /// texture and returns the pixel bytes (RGBA, row-major, tightly packed).
    pub fn render_to_pixels(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
    ) -> anyhow::Result<Vec<u8>> {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut tonemap = TonemapPass::new(gpu, format);

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Force a rebind: this pass instance has never seen the HDR view.
        self.hdr_rebound_needed = true;
        self.render_tonemapped(gpu, &mut tonemap, &view, width, height, camera);

        let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer_size = (bytes_per_row * height) as wgpu::BufferAddress;
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("Device poll failed while mapping readback buffer")?;
        rx.recv()
            .context("Readback mapping callback dropped")?
            .context("Failed to map readback buffer")?;

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
        }
        drop(data);
        readback.unmap();
        Ok(pixels)
    }
}

impl ForwardRenderer {
    /// Orthographic world->light-clip matrix fitted to the scene bounds.
    fn light_space_matrix(&self) -> Mat4 {
        let (min, max) = self
            .scene_bounds
            .unwrap_or((Vec3::splat(-1.0), Vec3::splat(1.0)));
        let center = (min + max) * 0.5;
        let radius = ((max - min).length() * 0.5).max(1e-3);

        let light_dir = self.light_dir_ambient.truncate().normalize_or_zero();
        let light_dir = if light_dir == Vec3::ZERO {
            Vec3::Y
        } else {
            light_dir
        };
        let up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };

        let eye = center + light_dir * (radius * 2.0);
        let view = Mat4::look_at_rh(eye, center, up);
        // glam's orthographic_rh uses 0..1 depth, matching WebGPU clip space.
        let projection =
            Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 4.0);
        projection * view
    }
}

fn compute_world_bounds(scene: &CpuScene) -> Option<(Vec3, Vec3)> {
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for prim in &scene.primitives {
        for vertex in &prim.vertices {
            let world = prim
                .transform
                .transform_point3(Vec3::from_array(vertex.position));
            bounds = Some(match bounds {
                None => (world, world),
                Some((min, max)) => (min.min(world), max.max(world)),
            });
        }
    }
    bounds
}

fn create_base_color_texture(
    gpu: &GpuContext,
    texture: &CpuTexture,
    label: Option<&str>,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: texture.width.max(1),
        height: texture.height.max(1),
        depth_or_array_layers: 1,
    };
    let gpu_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Base color is authored in sRGB; sampling converts to linear.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texture.rgba8,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );
    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_color"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
