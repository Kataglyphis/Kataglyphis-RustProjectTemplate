//! Forward PBR pass over a `CpuScene` into an internal HDR target,
//! composited to the output through the ACES tonemap pass. Also provides
//! headless render-to-pixels for golden tests and CI.

use anyhow::Context as _;
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt as _;

use crate::context::GpuContext;
use crate::render::bloom::BloomPass;
use crate::render::tonemap::TonemapPass;
use crate::scene::camera::OrbitCamera;
use crate::scene::{
    AlphaMode, CpuLight, CpuLightKind, CpuSampler, CpuScene, CpuTexture, CpuWrap, Vertex,
};

pub const MAX_PUNCTUAL_LIGHTS: usize = 4;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    inv_view_proj: [[f32; 4]; 4],
    light_dir_intensity: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    light_space: [[f32; 4]; 4],
    base_color: [f32; 4],
    light_dir_ambient: [f32; 4],
    light_color_intensity: [f32; 4],
    material_factors: [f32; 4],
    emissive_factor: [f32; 4],
    // xyz: world-space camera position, w: active punctual light count
    camera_position: [f32; 4],
    // Per light: [pos.xyz, kind], [color*intensity.rgb, range],
    // [dir.xyz, cos_inner], [cos_outer, 0, 0, 0]
    punctual_lights: [[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4],
    // KHR_texture_transform rows for the base color UV.
    base_uv_row0: [f32; 4],
    base_uv_row1: [f32; 4],
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
    material_factors: [f32; 4],
    emissive_factor: [f32; 4],
    base_uv_transform: [[f32; 3]; 2],
    double_sided: bool,
    alpha_blend: bool,
    casts_shadow: bool,
    world_center: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

fn pack_punctual_lights(lights: &[CpuLight]) -> ([[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4], u32) {
    let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
    let count = lights.len().min(MAX_PUNCTUAL_LIGHTS);
    for (i, light) in lights.iter().take(count).enumerate() {
        let (kind, cos_inner, cos_outer) = match light.kind {
            CpuLightKind::Point => (1.0, 0.0, 0.0),
            CpuLightKind::Spot {
                cos_inner,
                cos_outer,
            } => (2.0, cos_inner, cos_outer),
            CpuLightKind::Directional => (3.0, 0.0, 0.0),
        };
        let base = i * 4;
        packed[base] = [
            light.position[0],
            light.position[1],
            light.position[2],
            kind,
        ];
        packed[base + 1] = [
            light.color[0] * light.intensity,
            light.color[1] * light.intensity,
            light.color[2] * light.intensity,
            light.range,
        ];
        packed[base + 2] = [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            cos_inner,
        ];
        packed[base + 3] = [cos_outer, 0.0, 0.0, 0.0];
    }
    (packed, count as u32)
}

pub struct ForwardRenderer {
    pipeline: wgpu::RenderPipeline,
    pipeline_double_sided: wgpu::RenderPipeline,
    pipeline_blend: wgpu::RenderPipeline,
    pipeline_blend_double_sided: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shadow_pipeline_layout: wgpu::PipelineLayout,
    sky_pipeline_layout: wgpu::PipelineLayout,
    sky_uniform_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    shadow_bind_group_layout: wgpu::BindGroupLayout,
    shadow_sampler: wgpu::Sampler,
    white_texture_view: wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    shadow_view: wgpu::TextureView,
    primitives: Vec<GpuPrimitive>,
    scene_bounds: Option<(Vec3, Vec3)>,
    depth: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    target_size: (u32, u32),
    hdr_rebound_needed: bool,
    bloom: BloomPass,
    /// Bloom contribution mixed in by the tonemap pass.
    pub bloom_strength: f32,
    punctual_lights: [[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4],
    punctual_light_count: u32,
    /// Direction towards the light (world space) + ambient strength.
    pub light_dir_ambient: Vec4,
    /// Light color (rgb) + intensity multiplier (w). Values > 1 are the
    /// point of the HDR pipeline; the BRDF divides diffuse by PI.
    pub light_color_intensity: Vec4,
}

impl ForwardRenderer {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/forward.wgsl").into()),
        });

        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
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
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ];
        // Five material texture/sampler pairs: base color, metallic-roughness,
        // normal, emissive, occlusion (bindings 3..=12).
        for slot in 0..5u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 3 + slot * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 4 + slot * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward_bind_group_layout"),
            entries: &entries,
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

        let (pipeline, pipeline_double_sided, pipeline_blend, pipeline_blend_double_sided) =
            create_forward_pipeline_set(device, &shader, &pipeline_layout);

        // Procedural sky: fullscreen triangle at far depth, only where no
        // geometry was drawn (LessEqual vs the cleared 1.0, no depth writes).
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });
        let sky_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let sky_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky_pipeline_layout"),
                bind_group_layouts: &[&sky_bind_group_layout],
                push_constant_ranges: &[],
            });
        let sky_pipeline = create_sky_pipeline(device, &sky_shader, &sky_pipeline_layout);
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_uniforms"),
            size: std::mem::size_of::<SkyUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bind_group"),
            layout: &sky_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_uniform_buffer.as_entire_binding(),
            }],
        });

        let shadow_pipeline = create_shadow_pipeline(device, &shader, &shadow_pipeline_layout);

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let white_texture_view = create_material_texture(
            gpu,
            &CpuTexture {
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
            },
            false,
            Some("white_fallback"),
        );
        // Flat tangent-space normal (0, 0, 1) encoded as RGBA8.
        let flat_normal_view = create_material_texture(
            gpu,
            &CpuTexture {
                width: 1,
                height: 1,
                rgba8: vec![128, 128, 255, 255],
            },
            false,
            Some("flat_normal_fallback"),
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
            pipeline_double_sided,
            pipeline_blend,
            pipeline_blend_double_sided,
            shadow_pipeline,
            sky_pipeline,
            pipeline_layout,
            shadow_pipeline_layout,
            sky_pipeline_layout,
            sky_uniform_buffer,
            sky_bind_group,
            bind_group_layout,
            shadow_bind_group_layout,
            shadow_sampler,
            white_texture_view,
            flat_normal_view,
            shadow_view,
            primitives: Vec::new(),
            scene_bounds: None,
            depth,
            hdr_view,
            target_size: (width.max(1), height.max(1)),
            hdr_rebound_needed: true,
            bloom: BloomPass::new(gpu),
            bloom_strength: 0.6,
            punctual_lights: [[0.0; 4]; MAX_PUNCTUAL_LIGHTS * 4],
            punctual_light_count: 0,
            light_dir_ambient: Vec4::new(0.5, 0.8, 0.3, 0.35),
            // The BRDF divides diffuse by PI: intensity ~5 restores the
            // pre-PBR brightness ballpark.
            light_color_intensity: Vec4::new(1.0, 0.97, 0.92, 5.0),
        }
    }

    /// Uploads a CPU scene, replacing any previously uploaded one.
    pub fn upload_scene(&mut self, gpu: &GpuContext, scene: &CpuScene) {
        let device = &gpu.device;
        self.primitives.clear();
        self.scene_bounds = compute_world_bounds(scene);
        let (packed, count) = pack_punctual_lights(&scene.lights);
        self.punctual_lights = packed;
        self.punctual_light_count = count;
        if scene.lights.len() > MAX_PUNCTUAL_LIGHTS {
            log::warn!(
                "Scene has {} punctual lights; only the first {} are used.",
                scene.lights.len(),
                MAX_PUNCTUAL_LIGHTS
            );
        }

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

            let prim_bounds = primitive_world_aabb(prim);
            let material = &prim.material;
            let slots = [
                (&material.base_color_texture, &self.white_texture_view),
                (&material.metallic_roughness_texture, &self.white_texture_view),
                (&material.normal_texture, &self.flat_normal_view),
                (&material.emissive_texture, &self.white_texture_view),
                (&material.occlusion_texture, &self.white_texture_view),
            ];

            let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(5);
            let mut samplers: Vec<wgpu::Sampler> = Vec::with_capacity(5);
            for (slot_index, (texture_ref, fallback)) in slots.iter().enumerate() {
                match texture_ref {
                    Some(tex_ref) => {
                        views.push(create_material_texture(
                            gpu,
                            &tex_ref.texture,
                            tex_ref.srgb,
                            Some(&format!("material_{i}_slot_{slot_index}")),
                        ));
                        samplers.push(create_sampler(device, &tex_ref.sampler));
                    }
                    None => {
                        views.push((*fallback).clone());
                        samplers.push(create_sampler(device, &CpuSampler::default()));
                    }
                }
            }

            let mut bind_entries: Vec<wgpu::BindGroupEntry> = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
            ];
            for slot in 0..5usize {
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: 3 + slot as u32 * 2,
                    resource: wgpu::BindingResource::TextureView(&views[slot]),
                });
                bind_entries.push(wgpu::BindGroupEntry {
                    binding: 4 + slot as u32 * 2,
                    resource: wgpu::BindingResource::Sampler(&samplers[slot]),
                });
            }

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bind_group_{i}")),
                layout: &self.bind_group_layout,
                entries: &bind_entries,
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
                base_color: material.base_color,
                material_factors: [
                    material.metallic_factor,
                    material.roughness_factor,
                    material.occlusion_strength,
                    material.normal_scale,
                ],
                emissive_factor: [
                    material.emissive_factor[0],
                    material.emissive_factor[1],
                    material.emissive_factor[2],
                    // w carries the MASK alpha cutoff (0 = never discard).
                    match material.alpha_mode {
                        AlphaMode::Mask(cutoff) => cutoff,
                        _ => 0.0,
                    },
                ],
                base_uv_transform: material.base_uv_transform,
                double_sided: material.double_sided,
                alpha_blend: material.alpha_mode == AlphaMode::Blend,
                // Transparents cast no shadow (v1); a MASK primitive whose
                // base alpha is fully below the cutoff is invisible and must
                // not shadow either. Per-pixel alpha-tested shadows for
                // textured masks are a later refinement.
                casts_shadow: match material.alpha_mode {
                    AlphaMode::Blend => false,
                    AlphaMode::Mask(cutoff) => material.base_color[3] >= cutoff,
                    AlphaMode::Opaque => true,
                },
                world_center: primitive_world_center(prim),
                aabb_min: prim_bounds.0,
                aabb_max: prim_bounds.1,
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
            self.bloom.rebuild(gpu, width, height, &self.hdr_view);
            let bloom_out = self
                .bloom
                .output()
                .expect("bloom output exists after rebuild")
                .clone();
            tonemap.set_input(gpu, &self.hdr_view, &bloom_out);
            self.hdr_rebound_needed = false;
        }
        tonemap.set_params(&gpu.queue, self.bloom_strength);

        let aspect = width as f32 / height as f32;
        let view_proj = camera.view_projection(aspect);
        let frustum = Frustum::from_view_proj(&view_proj);
        let light_space = self.light_space_matrix();
        let eye = camera.eye();

        let sky_uniforms = SkyUniforms {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            light_dir_intensity: [
                self.light_dir_ambient.x,
                self.light_dir_ambient.y,
                self.light_dir_ambient.z,
                self.light_color_intensity.w,
            ],
        };
        gpu.queue.write_buffer(
            &self.sky_uniform_buffer,
            0,
            bytemuck::bytes_of(&sky_uniforms),
        );

        for prim in &self.primitives {
            let uniforms = Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                model: prim.model.to_cols_array_2d(),
                normal_matrix: prim.model.inverse().transpose().to_cols_array_2d(),
                light_space: light_space.to_cols_array_2d(),
                base_color: prim.base_color,
                light_dir_ambient: self.light_dir_ambient.to_array(),
                light_color_intensity: self.light_color_intensity.to_array(),
                material_factors: prim.material_factors,
                emissive_factor: prim.emissive_factor,
                camera_position: [eye.x, eye.y, eye.z, self.punctual_light_count as f32],
                punctual_lights: self.punctual_lights,
                base_uv_row0: [
                    prim.base_uv_transform[0][0],
                    prim.base_uv_transform[0][1],
                    prim.base_uv_transform[0][2],
                    0.0,
                ],
                base_uv_row1: [
                    prim.base_uv_transform[1][0],
                    prim.base_uv_transform[1][1],
                    prim.base_uv_transform[1][2],
                    0.0,
                ],
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
            for prim in self.primitives.iter().filter(|p| p.casts_shadow) {
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

            for prim in self
                .primitives
                .iter()
                .filter(|p| !p.alpha_blend && frustum.intersects_aabb(p.aabb_min, p.aabb_max))
            {
                let pipeline = if prim.double_sided {
                    &self.pipeline_double_sided
                } else {
                    &self.pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &prim.bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }

            // Sky fills every pixel geometry left untouched.
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.sky_bind_group, &[]);
            pass.draw(0..3, 0..1);

            // Transparents last, farthest first, no depth writes.
            let mut blended: Vec<&GpuPrimitive> = self
                .primitives
                .iter()
                .filter(|p| p.alpha_blend && frustum.intersects_aabb(p.aabb_min, p.aabb_max))
                .collect();
            blended.sort_by(|a, b| {
                let da = a.world_center.distance_squared(eye);
                let db = b.world_center.distance_squared(eye);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            for prim in blended {
                let pipeline = if prim.double_sided {
                    &self.pipeline_blend_double_sided
                } else {
                    &self.pipeline_blend
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &prim.bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }
        }

        self.bloom.encode(&mut encoder);
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

    /// Rebuilds the scene/shadow/sky pipelines from new WGSL sources.
    /// Invalid shaders are rejected (wgpu validation error scope) and the
    /// previous pipelines stay active.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reload_shaders(
        &mut self,
        gpu: &GpuContext,
        forward_wgsl: &str,
        sky_wgsl: &str,
    ) -> anyhow::Result<()> {
        let device = &gpu.device;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_shader_reloaded"),
            source: wgpu::ShaderSource::Wgsl(forward_wgsl.into()),
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader_reloaded"),
            source: wgpu::ShaderSource::Wgsl(sky_wgsl.into()),
        });
        let set = create_forward_pipeline_set(device, &shader, &self.pipeline_layout);
        let shadow = create_shadow_pipeline(device, &shader, &self.shadow_pipeline_layout);
        let sky = create_sky_pipeline(device, &sky_shader, &self.sky_pipeline_layout);
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            anyhow::bail!("shader reload rejected: {err}");
        }
        (
            self.pipeline,
            self.pipeline_double_sided,
            self.pipeline_blend,
            self.pipeline_blend_double_sided,
        ) = set;
        self.shadow_pipeline = shadow;
        self.sky_pipeline = sky;
        Ok(())
    }

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

type ForwardPipelineSet = (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
);

fn create_forward_pipeline_set(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
) -> ForwardPipelineSet {
    let make = |cull_mode: Option<wgpu::Face>, blend: bool, label: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::LAYOUT],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(if blend {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: !blend,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    };
    (
        make(Some(wgpu::Face::Back), false, "forward_pipeline"),
        make(None, false, "forward_pipeline_double_sided"),
        make(Some(wgpu::Face::Back), true, "forward_pipeline_blend"),
        make(None, true, "forward_pipeline_blend_double_sided"),
    )
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
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
        })
}

fn create_sky_pipeline(
    device: &wgpu::Device,
    sky_shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: sky_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: sky_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
}

/// View-frustum from a wgpu-convention (0..1 depth) view-projection matrix
/// (Gribb-Hartmann plane extraction; plane normals point inward).
pub(crate) struct Frustum {
    planes: [glam::Vec4; 6],
}

impl Frustum {
    pub(crate) fn from_view_proj(m: &Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);
        Self {
            planes: [
                r3 + r0, // left
                r3 - r0, // right
                r3 + r1, // bottom
                r3 - r1, // top
                r2,      // near (z >= 0 in 0..1 depth)
                r3 - r2, // far
            ],
        }
    }

    /// Positive-vertex test: the AABB is outside when its most favorable
    /// corner is behind any plane.
    pub(crate) fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );
            if plane.truncate().dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

fn primitive_world_aabb(prim: &crate::scene::CpuPrimitive) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &prim.vertices {
        let world = prim
            .transform
            .transform_point3(Vec3::from_array(vertex.position));
        min = min.min(world);
        max = max.max(world);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

fn primitive_world_center(prim: &crate::scene::CpuPrimitive) -> Vec3 {
    if prim.vertices.is_empty() {
        return prim.transform.transform_point3(Vec3::ZERO);
    }
    let mut sum = Vec3::ZERO;
    for vertex in &prim.vertices {
        sum += Vec3::from_array(vertex.position);
    }
    prim.transform
        .transform_point3(sum / prim.vertices.len() as f32)
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

fn create_sampler(device: &wgpu::Device, desc: &CpuSampler) -> wgpu::Sampler {
    let wrap = |mode: CpuWrap| match mode {
        CpuWrap::Repeat => wgpu::AddressMode::Repeat,
        CpuWrap::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        CpuWrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
    };
    let filter = |nearest: bool| {
        if nearest {
            wgpu::FilterMode::Nearest
        } else {
            wgpu::FilterMode::Linear
        }
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material_sampler"),
        address_mode_u: wrap(desc.wrap_u),
        address_mode_v: wrap(desc.wrap_v),
        mag_filter: filter(desc.mag_nearest),
        min_filter: filter(desc.min_nearest),
        mipmap_filter: filter(desc.mip_nearest),
        ..Default::default()
    })
}

fn srgb_to_linear(byte: u8) -> f32 {
    let c = byte as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let c = if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Full mip chain via 2x2 box filtering. sRGB data is averaged in linear
/// space; data maps (normals, metallic-roughness) are averaged raw.
pub(crate) fn generate_mips(base: &CpuTexture, srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(base.width, base.height, base.rgba8.clone())];
    let (mut w, mut h) = (base.width, base.height);

    while w > 1 || h > 1 {
        let (pw, ph, prev) = levels.last().unwrap();
        let (pw, ph) = (*pw, *ph);
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);

        for y in 0..nh {
            for x in 0..nw {
                let x0 = (x * 2).min(pw - 1);
                let x1 = (x * 2 + 1).min(pw - 1);
                let y0 = (y * 2).min(ph - 1);
                let y1 = (y * 2 + 1).min(ph - 1);
                for channel in 0..4usize {
                    let fetch = |px: u32, py: u32| prev[((py * pw + px) * 4) as usize + channel];
                    let samples = [fetch(x0, y0), fetch(x1, y0), fetch(x0, y1), fetch(x1, y1)];
                    // Alpha is linear even for sRGB textures.
                    let value = if srgb && channel < 3 {
                        let sum: f32 = samples.iter().map(|&b| srgb_to_linear(b)).sum();
                        linear_to_srgb(sum / 4.0)
                    } else {
                        (samples.iter().map(|&b| b as u32).sum::<u32>() / 4) as u8
                    };
                    next.push(value);
                }
            }
        }
        levels.push((nw, nh, next));
        w = nw;
        h = nh;
    }

    levels
}

fn create_material_texture(
    gpu: &GpuContext,
    texture: &CpuTexture,
    srgb: bool,
    label: Option<&str>,
) -> wgpu::TextureView {
    let mips = generate_mips(texture, srgb);
    let size = wgpu::Extent3d {
        width: texture.width.max(1),
        height: texture.height.max(1),
        depth_or_array_layers: 1,
    };
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let gpu_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (w, h, data)) in mips.iter().enumerate() {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(*h),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::camera::OrbitCamera;

    #[test]
    fn frustum_culls_out_of_view_aabbs() {
        let camera = OrbitCamera::default();
        let frustum = Frustum::from_view_proj(&camera.view_projection(1.0));

        // Cube at the orbit target is visible.
        assert!(frustum.intersects_aabb(Vec3::splat(-0.5), Vec3::splat(0.5)));
        // A cube far off to the side is culled.
        assert!(!frustum.intersects_aabb(
            Vec3::new(1000.0, -0.5, -0.5),
            Vec3::new(1001.0, 0.5, 0.5)
        ));
        // Behind the camera is culled.
        let eye = camera.eye();
        let behind = eye + (eye - Vec3::ZERO).normalize() * 10.0;
        assert!(!frustum.intersects_aabb(behind - Vec3::splat(0.4), behind + Vec3::splat(0.4)));
    }
}
