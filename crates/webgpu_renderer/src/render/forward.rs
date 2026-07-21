//! Forward PBR pass over a `CpuScene` into an internal HDR target,
//! composited to the output through the ACES tonemap pass. Also provides
//! headless render-to-pixels for golden tests and CI.

use anyhow::Context as _;
use glam::{Mat4, Quat, Vec3, Vec4};
use wgpu::util::DeviceExt as _;

use crate::context::GpuContext;
use crate::render::bloom::BloomPass;
use crate::render::gpu_timing::{GpuTiming, TimedPass};
use crate::render::ibl::{BrdfLut, EquirectImage, IblEnvironment, IblFallback};
use crate::render::occlusion::OcclusionQueries;
use crate::render::ssao::SsaoPass;
use crate::render::tonemap::TonemapPass;
use crate::scene::camera::OrbitCamera;
use crate::scene::{
    AlphaMode, ChannelValues, CpuAnimation, CpuLight, CpuLightKind, CpuNode, CpuSampler, CpuScene,
    CpuSkin, CpuTexture, CpuWrap, InstanceRaw, Interpolation, MorphTarget, Vertex,
};

/// Upper bound on joints per skin (storage buffer is sized to the skin).
pub const MAX_JOINTS: usize = 256;

pub const MAX_PUNCTUAL_LIGHTS: usize = 4;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SHADOW_MAP_SIZE: u32 = 2048;
/// Cascaded shadow map layers (split by view distance).
pub const CASCADE_COUNT: usize = 3;

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
    light_space_1: [[f32; 4]; 4],
    light_space_2: [[f32; 4]; 4],
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
    cascade_splits: [f32; 4],
    /// x: 1.0 when KHR_materials_unlit, else 0.0. Remaining lanes reserved for
    /// further material flags so the next one does not have to squat in an
    /// unrelated vector.
    material_flags: [f32; 4],
}

/// One pre-uploaded LOD level: its own vertex and index buffer, built once.
///
/// Every level is simplified and uploaded at `upload_scene` time. Simplifying
/// on demand inside a frame would make frame cost depend on how the camera
/// moved, which is exactly the hitch LOD exists to avoid; this is a demo
/// renderer, so the VRAM for all levels at once is the cheaper trade and
/// nothing is streamed or evicted.
struct LodLevel {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
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
    /// Per-instance transforms. Always at least one (identity), so every draw
    /// binds slot 1 and there is no un-instanced code path to diverge.
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    /// Instance transforms as last set, empty for the default identity. Kept on
    /// the CPU so culling bounds can be recomposed whenever EITHER side moves:
    /// the instances (`set_instances`) or the posed geometry they replicate
    /// (`set_animation_time`).
    instance_transforms: Vec<Mat4>,
    /// World bounds BEFORE instance transforms are applied - the box each
    /// instance replicates. Stored so recomposition never has to re-derive the
    /// skinned/morphed pose.
    pre_instance_aabb: (Vec3, Vec3),
    model: Mat4,
    base_color: [f32; 4],
    material_factors: [f32; 4],
    emissive_factor: [f32; 4],
    base_uv_transform: [[f32; 3]; 2],
    double_sided: bool,
    /// KHR_materials_unlit: skip lighting entirely and emit the base color.
    unlit: bool,
    alpha_blend: bool,
    casts_shadow: bool,
    world_center: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
    node_index: Option<usize>,
    skin_index: Option<usize>,
    joint_buffer: wgpu::Buffer,
    local_aabb_min: Vec3,
    local_aabb_max: Vec3,
    /// Simplified levels, coarsest last. Empty when LOD is off, which is what
    /// makes the disabled path identical to the pre-LOD renderer rather than
    /// merely equivalent to it.
    lod_levels: Vec<LodLevel>,
    /// Switch distance per entry in `lod_levels`. Held separately from the
    /// levels so per-frame selection scans a contiguous f32 slice and never
    /// allocates.
    lod_min_distances: Vec<f32>,
    /// Un-morphed vertices, kept only when this primitive has morph targets so
    /// each frame re-blends from the neutral pose rather than accumulating.
    /// Empty (and cheap) for the overwhelming majority of primitives.
    base_vertices: Vec<Vertex>,
    /// POSITION/NORMAL deltas, one entry per morph target.
    morph_targets: Vec<MorphTarget>,
    /// Current per-target weights (animation-driven or the mesh defaults).
    morph_weights: Vec<f32>,
    /// Set when `morph_weights` changed and the vertex buffer needs re-blending
    /// on the next render. Starts true so non-zero default weights apply once.
    morph_dirty: bool,
}

impl GpuPrimitive {
    /// Vertex buffer, index buffer and index count the draw path should use
    /// from a camera at `eye`.
    ///
    /// Distance is measured to `world_center`, matching the metric the sorted
    /// blend pass already uses, so a primitive cannot be "far" for one pass
    /// and "near" for another.
    fn geometry_for(&self, eye: Vec3) -> (&wgpu::Buffer, &wgpu::Buffer, u32) {
        let distance = self.world_center.distance(eye);
        match crate::scene::lod::select_lod_by_distance(&self.lod_min_distances, distance) {
            Some(level) => {
                let lod = &self.lod_levels[level];
                (&lod.vertex_buffer, &lod.index_buffer, lod.index_count)
            }
            None => (&self.vertex_buffer, &self.index_buffer, self.index_count),
        }
    }
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
    /// Group 1 of the forward pipeline: the IBL maps, bound once per pass.
    ibl_bind_group_layout: wgpu::BindGroupLayout,
    ibl_uniform_buffer: wgpu::Buffer,
    ibl_bind_group: wgpu::BindGroup,
    /// 1x1 stand-ins so group 1 is always bindable, environment or not.
    ibl_fallback: IblFallback,
    /// `None` until [`Self::set_environment`]; the fallback analytic path runs
    /// until then, unchanged.
    ibl_environment: Option<IblEnvironment>,
    /// Environment-independent, so it is baked at most once and survives every
    /// later `set_environment`. Lazily built rather than built in `new`: a
    /// renderer that never sets an environment must not pay for a 256x256 x
    /// 1024-sample pass it will never sample.
    brdf_lut: Option<BrdfLut>,
    shadow_sampler: wgpu::Sampler,
    white_texture_view: wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    shadow_view: wgpu::TextureView,
    shadow_layer_views: Vec<wgpu::TextureView>,
    cascade_index_bind_groups: Vec<wgpu::BindGroup>,
    cascade_matrices: [Mat4; CASCADE_COUNT],
    cascade_splits: [f32; 4],
    /// Caster draws submitted / considered across all cascades in the last
    /// frame. Summed over cascades, so one mesh under three cascades
    /// considers 3 - same convention as the C++ engine's counters.
    /// Opaque primitives drawn / considered by the camera pass last frame,
    /// after both frustum and (when enabled) occlusion culling. `drawn <
    /// considered` is the observable proof occlusion culling engaged.
    occlusion_drawn: u32,
    occlusion_considered: u32,
    shadow_casters_drawn: u32,
    shadow_casters_considered: u32,
    primitives: Vec<GpuPrimitive>,
    scene_bounds: Option<(Vec3, Vec3)>,
    /// GPU textures actually created by the last `upload_scene` (shared images
    /// are uploaded once). Exposed so tests can prove the dedup, not infer it.
    uploaded_texture_count: usize,
    depth: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
    target_size: (u32, u32),
    hdr_rebound_needed: bool,
    bloom: BloomPass,
    ssao: SsaoPass,
    /// Bloom contribution mixed in by the tonemap pass.
    pub bloom_strength: f32,
    /// SSAO strength applied by the tonemap pass (0 = off).
    pub ssao_strength: f32,
    /// Exposure in EV stops (0 = neutral).
    pub exposure_ev: f32,
    /// Drive exposure from the scene histogram instead of `exposure_ev`.
    pub auto_exposure: bool,
    /// Adaptation rate; higher settles faster.
    pub auto_exposure_speed: f32,
    /// Seconds since the previous frame, for exposure adaptation. A field
    /// rather than a render() parameter so existing call sites keep working;
    /// leaving it at the default just means adaptation runs at a nominal
    /// 60 Hz rate.
    pub frame_delta_seconds: f32,
    histogram: crate::render::histogram::HistogramPass,
    punctual_lights: [[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4],
    punctual_light_count: u32,
    nodes: Vec<CpuNode>,
    animations: Vec<CpuAnimation>,
    skins: Vec<CpuSkin>,
    pending_joint_world: Option<Vec<Mat4>>,
    /// Direction towards the light (world space) + ambient strength.
    pub light_dir_ambient: Vec4,
    /// Light color (rgb) + intensity multiplier (w). Values > 1 are the
    /// point of the HDR pipeline; the BRDF divides diffuse by PI.
    pub light_color_intensity: Vec4,
    /// Build and draw per-primitive LOD chains.
    ///
    /// Off by default, and read only by `upload_scene`: with it off no chain
    /// is built and every primitive keeps an empty level list, so the draw
    /// path is the pre-LOD one instruction for instruction and every golden
    /// test keeps the meaning it was written with. Set it (and
    /// `lod_switch_distances`) BEFORE `upload_scene`.
    pub lod_enabled: bool,
    /// Multiplier on the environment's contribution. 1.0 means "use the
    /// panorama's radiance as authored"; it is not the ambient slider, which
    /// is `light_dir_ambient.w` and scales both IBL paths alike.
    pub ibl_intensity: f32,
    /// Per-pass GPU timestamps. Inert until [`Self::enable_gpu_timing`].
    gpu_timing: GpuTiming,
    /// Per-primitive hardware occlusion detection over the forward depth.
    /// Resources always exist; recording only happens when the flag below is
    /// set. See [`crate::render::occlusion`].
    occlusion: OcclusionQueries,
    /// Record the occlusion detection pass after the forward pass.
    ///
    /// Off by default, like `lod_enabled` and `enable_gpu_timing`: it adds a
    /// depth-only pass plus a query resolve, and this increment only DETECTS
    /// occlusion (reads back per-primitive visibility) - it does not yet skip
    /// any draw, so the frame renders exactly as before whether it is on or
    /// off. The visibility is exposed via [`Self::occlusion_visibility`].
    pub occlusion_queries_enabled: bool,
    /// Camera distances at which successive levels take over, ascending.
    ///
    /// Two levels by default: the geometry roughly halves at each, so the
    /// coarsest is a quarter of the original triangle budget. Tuned for the
    /// unit-scale test assets; a scene in metres wants larger numbers.
    pub lod_switch_distances: Vec<f32>,
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
                    view_dimension: wgpu::TextureViewDimension::D2Array,
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

        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward_bind_group_layout"),
            entries: &entries,
        });

        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let ibl_bind_group_layout = create_ibl_bind_group_layout(device);

        // vs_shadow touches nothing in group 1, so the shadow pipeline layout
        // stays single-group: a pipeline layout only has to cover the bindings
        // its entry points actually use.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&ibl_bind_group_layout)],
            immediate_size: 0,
        });

        // One tiny static buffer per cascade, bound at group(1) of the shadow
        // pipeline, telling vs_shadow which cascade matrix to project with.
        //
        // This is the fix for a real ordering bug, not a convenience. The
        // cascade index used to be written into every primitive's SHARED
        // uniform buffer once per cascade inside a single encoder, and
        // Queue::write_buffer applies all its writes before the command buffer
        // executes - so every one of the three shadow passes saw the LAST
        // cascade's index, and all three depth layers were rendered with
        // cascade 2's matrix while the fragment stage still sampled them as
        // 0/1/2. The structural shadow tests could not see it (a shadow still
        // appeared); it showed up as coarse, slightly-wrong cascades. Static
        // per-cascade buffers written once cannot express the bug, and they
        // also delete 3 x primitive-count queue writes per frame.
        let cascade_index_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_cascade_index_layout"),
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
        let cascade_index_bind_groups: Vec<wgpu::BindGroup> = (0..CASCADE_COUNT as u32)
            .map(|cascade| {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("shadow_cascade_index_{cascade}")),
                    contents: bytemuck::bytes_of(&[cascade, 0u32, 0u32, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("shadow_cascade_index_bg_{cascade}")),
                    layout: &cascade_index_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            })
            .collect();

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[Some(&shadow_bind_group_layout), Some(&cascade_index_layout)],
                immediate_size: 0,
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
        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pipeline_layout"),
            bind_group_layouts: &[Some(&sky_bind_group_layout)],
            immediate_size: 0,
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
                compressed: None,
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
                compressed: None,
            },
            false,
            Some("flat_normal_fallback"),
        );

        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map_array"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_map_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_layer_views: Vec<wgpu::TextureView> = (0..CASCADE_COUNT as u32)
            .map(|layer| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow_map_layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let depth = create_depth_texture(device, width, height);
        let hdr_view = create_hdr_texture(device, width, height);
        let mut histogram = crate::render::histogram::HistogramPass::new(gpu);
        histogram.set_input(gpu, &hdr_view);

        let ibl_fallback = IblFallback::new(device);
        let ibl_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibl_params"),
            contents: bytemuck::bytes_of(&IblUniforms::disabled()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let ibl_bind_group = create_ibl_bind_group(
            device,
            &ibl_bind_group_layout,
            &ibl_uniform_buffer,
            &ibl_fallback.irradiance,
            &ibl_fallback.prefiltered,
            &ibl_fallback.brdf_lut,
            &ibl_fallback.sampler,
        );

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
            ibl_bind_group_layout,
            ibl_uniform_buffer,
            ibl_bind_group,
            ibl_fallback,
            ibl_environment: None,
            brdf_lut: None,
            ibl_intensity: 1.0,
            shadow_sampler,
            white_texture_view,
            flat_normal_view,
            shadow_view,
            shadow_layer_views,
            cascade_index_bind_groups,
            cascade_matrices: [Mat4::IDENTITY; CASCADE_COUNT],
            occlusion_drawn: 0,
            occlusion_considered: 0,
            shadow_casters_drawn: 0,
            shadow_casters_considered: 0,
            cascade_splits: [10.0, 30.0, CASCADE_COUNT as f32, 0.0],
            primitives: Vec::new(),
            gpu_timing: GpuTiming::unavailable(),
            occlusion: OcclusionQueries::new(device),
            occlusion_queries_enabled: false,
            lod_enabled: false,
            lod_switch_distances: vec![8.0, 24.0],
            scene_bounds: None,
            uploaded_texture_count: 0,
            depth,
            hdr_view,
            target_size: (width.max(1), height.max(1)),
            hdr_rebound_needed: true,
            bloom: BloomPass::new(gpu),
            ssao: SsaoPass::new(gpu),
            bloom_strength: 0.6,
            ssao_strength: 0.7,
            exposure_ev: 0.0,
            auto_exposure: false,
            auto_exposure_speed: 3.0,
            frame_delta_seconds: 1.0 / 60.0,
            histogram,
            punctual_lights: [[0.0; 4]; MAX_PUNCTUAL_LIGHTS * 4],
            punctual_light_count: 0,
            nodes: Vec::new(),
            animations: Vec::new(),
            skins: Vec::new(),
            pending_joint_world: None,
            light_dir_ambient: Vec4::new(0.5, 0.8, 0.3, 0.35),
            // The BRDF divides diffuse by PI: intensity ~5 restores the
            // pre-PBR brightness ballpark.
            light_color_intensity: Vec4::new(1.0, 0.97, 0.92, 5.0),
        }
    }

    /// Uploads a CPU scene, replacing any previously uploaded one.
    /// Replaces the instance transforms of one primitive.
    ///
    /// Passing an empty slice restores the single identity instance rather
    /// than drawing nothing: a primitive with zero instances silently
    /// disappears, which is indistinguishable from a culling or upload bug
    /// when you are looking at the frame.
    ///
    /// Reallocates when the count grows; a same-size update just writes.
    pub fn set_instances(&mut self, gpu: &GpuContext, primitive_index: usize, transforms: &[Mat4]) {
        let Some(primitive) = self.primitives.get_mut(primitive_index) else {
            return;
        };

        if transforms.is_empty() {
            gpu.queue.write_buffer(
                &primitive.instance_buffer,
                0,
                bytemuck::bytes_of(&InstanceRaw::IDENTITY),
            );
            primitive.instance_count = 1;
            // Back to the single identity instance: bounds collapse to the
            // posed box.
            primitive.instance_transforms.clear();
            let (min, max) = primitive.pre_instance_aabb;
            primitive.aabb_min = min;
            primitive.aabb_max = max;
            primitive.world_center = (min + max) * 0.5;
            self.recompute_scene_bounds();
            return;
        }

        let raw: Vec<InstanceRaw> = transforms
            .iter()
            .map(|m| InstanceRaw {
                model: m.to_cols_array_2d(),
            })
            .collect();
        let bytes = bytemuck::cast_slice(&raw);

        if (bytes.len() as u64) <= primitive.instance_buffer.size() {
            gpu.queue.write_buffer(&primitive.instance_buffer, 0, bytes);
        } else {
            primitive.instance_buffer =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("instances"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
        }
        primitive.instance_count = raw.len() as u32;
        // Culling bounds must span every instance, not just the base position.
        primitive.instance_transforms = transforms.to_vec();
        let (min, max) = instanced_bounds(primitive.pre_instance_aabb, transforms);
        primitive.aabb_min = min;
        primitive.aabb_max = max;
        primitive.world_center = (min + max) * 0.5;
        self.recompute_scene_bounds();
    }

    /// Instance count of a primitive, for tests and diagnostics.
    pub fn instance_count(&self, primitive_index: usize) -> u32 {
        self.primitives
            .get(primitive_index)
            .map(|p| p.instance_count)
            .unwrap_or(0)
    }

    /// GPU textures created by the most recent [`Self::upload_scene`]. Shared
    /// images upload once, so this is < the number of material slots whenever a
    /// scene reuses a texture.
    pub fn uploaded_texture_count(&self) -> usize {
        self.uploaded_texture_count
    }

    /// Whole-scene world bounds, exactly as cascade fitting reads them. `None`
    /// before anything is uploaded.
    pub fn scene_bounds(&self) -> Option<(Vec3, Vec3)> {
        self.scene_bounds
    }

    /// World-space culling bounds of a primitive, exactly as the frustum test
    /// reads them. Exposed so tests can assert on the bounds that actually gate
    /// drawing rather than on a recomputed copy.
    pub fn primitive_world_aabb(&self, primitive_index: usize) -> Option<(Vec3, Vec3)> {
        self.primitives
            .get(primitive_index)
            .map(|p| (p.aabb_min, p.aabb_max))
    }

    /// Number of pre-uploaded LOD levels for a primitive (0 when LOD is off).
    pub fn lod_level_count(&self, primitive_index: usize) -> usize {
        self.primitives
            .get(primitive_index)
            .map(|p| p.lod_levels.len())
            .unwrap_or(0)
    }

    /// Index count of one pre-uploaded level, `None` if it does not exist.
    pub fn lod_level_index_count(&self, primitive_index: usize, level: usize) -> Option<u32> {
        self.primitives
            .get(primitive_index)?
            .lod_levels
            .get(level)
            .map(|l| l.index_count)
    }

    /// Index count the draw path would actually issue for a primitive with the
    /// camera at `eye`.
    ///
    /// This calls the same `geometry_for` the render pass calls, so a test
    /// against it fails if selection is computed but not bound. A test that
    /// only asked "is LOD enabled?" would pass against wiring that draws the
    /// full-detail buffer every frame, which is the bug this whole feature was
    /// added to fix.
    pub fn selected_index_count(&self, primitive_index: usize, eye: Vec3) -> Option<u32> {
        let prim = self.primitives.get(primitive_index)?;
        Some(prim.geometry_for(eye).2)
    }

    /// Starts collecting per-pass GPU timings; returns whether it took effect.
    ///
    /// Off by default, and a method rather than a `pub` flag because switching
    /// it on allocates a query set and readback buffers, which needs the
    /// device. `false` means the adapter has no `TIMESTAMP_QUERY` (every
    /// browser today, and some native drivers) - the caller gets no timings and
    /// the frame renders exactly as before.
    ///
    /// Off by default for the same reason as `lod_enabled`: it is not free.
    /// Every timed pass gains two timestamp writes, and the frame gains a query
    /// resolve plus a buffer copy. Small, but it is a diagnostic, and a golden
    /// or perf test must measure the shipping path unless it asked not to.
    pub fn enable_gpu_timing(&mut self, gpu: &GpuContext) -> bool {
        self.gpu_timing = GpuTiming::new(&gpu.device, &gpu.queue);
        self.gpu_timing.is_available()
    }

    /// Stops collecting timings and releases the query resources.
    pub fn disable_gpu_timing(&mut self) {
        self.gpu_timing = GpuTiming::unavailable();
    }

    /// Averaged per-pass GPU durations in milliseconds, in record order.
    ///
    /// Empty until timing is enabled AND the first readback has landed (a few
    /// frames in, by design - see [`crate::render::gpu_timing`]). A pass absent
    /// from the list has not reported yet; that is not the same as zero.
    /// Shadow caster draws submitted / considered in the last frame, summed
    /// over cascades. `drawn < considered` is the observable proof that
    /// per-cascade culling actually engaged.
    pub fn shadow_caster_stats(&self) -> (u32, u32) {
        (self.shadow_casters_drawn, self.shadow_casters_considered)
    }

    /// Opaque primitives drawn / considered by the camera pass last frame.
    /// With occlusion queries enabled, `drawn < considered` once a primitive
    /// has been occluded for a frame - the observable proof the skip engaged.
    pub fn occlusion_cull_stats(&self) -> (u32, u32) {
        (self.occlusion_drawn, self.occlusion_considered)
    }

    pub fn gpu_timings_ms(&self) -> Vec<(&'static str, f32)> {
        self.gpu_timing.timings_ms()
    }

    /// Per-primitive occlusion visibility from the most recent completed
    /// readback, index-aligned to the uploaded primitives: `true` when the
    /// primitive's world AABB had > 0 fragments pass the depth test.
    ///
    /// Empty until occlusion queries are enabled AND the first readback has
    /// landed (a frame or two after the first recorded frame - the readback is
    /// asynchronous, exactly like the GPU timings). A primitive absent from the
    /// slice has not been measured yet; that is not the same as "occluded".
    pub fn occlusion_visibility(&self) -> &[bool] {
        self.occlusion.visibility()
    }

    /// Raw occlusion sample counts behind [`Self::occlusion_visibility`], for
    /// tests and diagnostics that want the fragment counts, not just the bool.
    pub fn occlusion_samples(&self) -> &[u64] {
        self.occlusion.samples()
    }

    /// True while per-pass GPU timings are being collected.
    pub fn gpu_timing_available(&self) -> bool {
        self.gpu_timing.is_available()
    }

    /// Bakes `equirect` into irradiance/prefiltered maps and lights the scene
    /// with them from the next frame on.
    ///
    /// Off by default, like `lod_enabled` and `enable_gpu_timing`: with no
    /// environment the forward shader takes the analytic hemisphere path it
    /// always took, so every golden test keeps the meaning it was written
    /// with. Setting one is the only thing that changes shading.
    ///
    /// Blocks for the length of the bake (one submit, tens of milliseconds) -
    /// it is a load-time operation, not a per-frame one. Nothing is recomputed
    /// afterwards; the frame path only samples the results.
    pub fn set_environment(&mut self, gpu: &GpuContext, equirect: &EquirectImage) {
        // The BRDF table integrates the GGX BRDF against a white furnace and
        // never touches the environment, so it is baked once and reused for
        // every environment this renderer is ever given.
        let brdf_lut = self.brdf_lut.get_or_insert_with(|| BrdfLut::new(gpu));
        let environment = IblEnvironment::bake(gpu, equirect);

        gpu.queue.write_buffer(
            &self.ibl_uniform_buffer,
            0,
            bytemuck::bytes_of(&IblUniforms::enabled(
                environment.max_prefiltered_mip(),
                self.ibl_intensity,
            )),
        );
        self.ibl_bind_group = create_ibl_bind_group(
            &gpu.device,
            &self.ibl_bind_group_layout,
            &self.ibl_uniform_buffer,
            environment.irradiance_view(),
            environment.prefiltered_view(),
            brdf_lut.view(),
            &self.ibl_fallback.sampler,
        );
        self.ibl_environment = Some(environment);
    }

    /// Drops the environment and returns to the analytic hemisphere path.
    pub fn clear_environment(&mut self, gpu: &GpuContext) {
        self.ibl_environment = None;
        gpu.queue.write_buffer(
            &self.ibl_uniform_buffer,
            0,
            bytemuck::bytes_of(&IblUniforms::disabled()),
        );
        self.ibl_bind_group = create_ibl_bind_group(
            &gpu.device,
            &self.ibl_bind_group_layout,
            &self.ibl_uniform_buffer,
            &self.ibl_fallback.irradiance,
            &self.ibl_fallback.prefiltered,
            &self.ibl_fallback.brdf_lut,
            &self.ibl_fallback.sampler,
        );
    }

    /// True when a baked environment is lighting the scene.
    pub fn environment_enabled(&self) -> bool {
        self.ibl_environment.is_some()
    }

    /// The baked environment, for tests and diagnostics that want to inspect
    /// the maps the frame is actually sampling.
    pub fn environment(&self) -> Option<&IblEnvironment> {
        self.ibl_environment.as_ref()
    }

    /// The shared split-sum BRDF table, once an environment has been set.
    pub fn brdf_lut(&self) -> Option<&BrdfLut> {
        self.brdf_lut.as_ref()
    }

    pub fn upload_scene(&mut self, gpu: &GpuContext, scene: &CpuScene) {
        let device = &gpu.device;
        self.primitives.clear();
        // Per-index occlusion visibility is only valid for the primitive list it
        // was measured against; a new scene must not inherit it (a smaller scene
        // would see stale culls at shared indices).
        self.occlusion.reset();
        self.scene_bounds = compute_world_bounds(scene);
        let (packed, count) = pack_punctual_lights(&scene.lights);
        self.punctual_lights = packed;
        self.punctual_light_count = count;
        self.nodes = scene.nodes.clone();
        self.animations = scene.animations.clone();
        self.skins = scene.skins.clone();
        if scene.lights.len() > MAX_PUNCTUAL_LIGHTS {
            log::warn!(
                "Scene has {} punctual lights; only the first {} are used.",
                scene.lights.len(),
                MAX_PUNCTUAL_LIGHTS
            );
        }

        // Resolved once for the whole upload: skinned primitives need the joint
        // nodes' world matrices to size their bounds (below).
        let upload_node_world = CpuScene::compute_world_transforms(&scene.nodes);

        // One GPU texture per (image, color space), not one per primitive that
        // references it. Without this a 200-primitive glTF sharing a single 4K
        // atlas ran 200 CPU mip-chain builds - `generate_mips` does a per-texel
        // powf for the sRGB-correct average - and uploaded the same pixels 200
        // times. That is the load hitch AND the VRAM ceiling. The CPU side is
        // already `Arc<CpuTexture>`, so the pointer is a free identity key;
        // `srgb` joins it because the same image can legitimately be uploaded
        // both ways (base color vs a data map).
        let mut texture_cache: std::collections::HashMap<(usize, bool), wgpu::TextureView> =
            std::collections::HashMap::new();
        let mut sampler_cache: std::collections::HashMap<CpuSampler, wgpu::Sampler> =
            std::collections::HashMap::new();
        let mut uploaded_textures = 0usize;

        for (i, prim) in scene.primitives.iter().enumerate() {
            // Morphed primitives are re-blended and re-uploaded per frame, so
            // their vertex buffer needs COPY_DST. Everyone else stays upload-once.
            let has_morph = !prim.morph_targets.is_empty();
            let vertex_usage = if has_morph {
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
            } else {
                wgpu::BufferUsages::VERTEX
            };
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vertices_{i}")),
                contents: bytemuck::cast_slice(&prim.vertices),
                usage: vertex_usage,
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

            let local_bounds = primitive_local_aabb(prim);
            // Skinned bounds matter even before any animation runs: a scene can
            // be authored with its joints already posed away from the bind pose,
            // and set_animation_time (which also widens) bails out early when the
            // scene carries no animations.
            let mut prim_bounds = primitive_world_aabb(prim);
            if let Some(skin) = prim.skin_index.and_then(|s| scene.skins.get(s)) {
                prim_bounds = widen_bounds_for_skin(
                    prim_bounds,
                    local_bounds.0,
                    local_bounds.1,
                    skin,
                    &upload_node_world,
                );
            }
            let material = &prim.material;
            let slots = [
                (&material.base_color_texture, &self.white_texture_view),
                (
                    &material.metallic_roughness_texture,
                    &self.white_texture_view,
                ),
                (&material.normal_texture, &self.flat_normal_view),
                (&material.emissive_texture, &self.white_texture_view),
                (&material.occlusion_texture, &self.white_texture_view),
            ];

            let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(5);
            let mut samplers: Vec<wgpu::Sampler> = Vec::with_capacity(5);
            for (slot_index, (texture_ref, fallback)) in slots.iter().enumerate() {
                match texture_ref {
                    Some(tex_ref) => {
                        let key = (std::sync::Arc::as_ptr(&tex_ref.texture) as usize, tex_ref.srgb);
                        let view = match texture_cache.get(&key) {
                            Some(v) => v.clone(),
                            None => {
                                let v = create_material_texture(
                                    gpu,
                                    &tex_ref.texture,
                                    tex_ref.srgb,
                                    Some(&format!("material_{i}_slot_{slot_index}")),
                                );
                                uploaded_textures += 1;
                                texture_cache.insert(key, v.clone());
                                v
                            }
                        };
                        views.push(view);
                        samplers.push(
                            sampler_cache
                                .entry(tex_ref.sampler)
                                .or_insert_with(|| create_sampler(device, &tex_ref.sampler))
                                .clone(),
                        );
                    }
                    None => {
                        views.push((*fallback).clone());
                        let d = CpuSampler::default();
                        samplers.push(
                            sampler_cache
                                .entry(d)
                                .or_insert_with(|| create_sampler(device, &d))
                                .clone(),
                        );
                    }
                }
            }

            // Joint matrices: sized to the skin (identity when unskinned).
            let joint_count = prim
                .skin_index
                .and_then(|s| scene.skins.get(s))
                .map(|s| s.joints.len().clamp(1, MAX_JOINTS))
                .unwrap_or(1);
            let identity = vec![Mat4::IDENTITY.to_cols_array_2d(); joint_count];
            let joint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("joints_{i}")),
                contents: bytemuck::cast_slice(&identity),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

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

            bind_entries.push(wgpu::BindGroupEntry {
                binding: 13,
                resource: joint_buffer.as_entire_binding(),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bind_group_{i}")),
                layout: &self.bind_group_layout,
                entries: &bind_entries,
            });

            let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_bind_group_{i}")),
                layout: &self.shadow_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: joint_buffer.as_entire_binding(),
                    },
                ],
            });

            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("instances_{i}")),
                contents: bytemuck::bytes_of(&InstanceRaw::IDENTITY),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            // Quadric decimation, not vertex clustering: clustering's first
            // level at cell ratio 0.02 is a no-op on anything but a dense
            // photogrammetry mesh (every vertex lands in its own cell), so a
            // chain built with it would draw the same triangle count at every
            // distance and the whole system would look wired but do nothing.
            // QEM's ratio is a triangle budget, so level 0 halves regardless
            // of how dense the input is.
            // Morphed primitives are excluded from LOD: `apply_morph_targets`
            // re-blends only the full-res `vertex_buffer`, so a simplified LOD
            // level would draw the un-morphed neutral pose and the object would
            // visibly pop to its rest shape at distance. Keep them full-res.
            let (lod_levels, lod_min_distances) = if self.lod_enabled && !has_morph {
                let chain = crate::scene::lod::build_lod_chain_with(
                    prim,
                    &self.lod_switch_distances,
                    crate::scene::lod::Simplifier::Quadric,
                );
                let mut levels = Vec::with_capacity(chain.len());
                let mut distances = Vec::with_capacity(chain.len());
                for (level, lod) in chain.iter().enumerate() {
                    // A level that decimated to nothing ends the chain: a
                    // zero-length buffer is invalid in wgpu, and there is no
                    // sensible thing to draw past the point where the mesh has
                    // no triangles left. The primitive then simply stops
                    // getting coarser beyond that distance.
                    if lod.primitive.indices.is_empty() || lod.primitive.vertices.is_empty() {
                        break;
                    }
                    levels.push(LodLevel {
                        vertex_buffer: device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some(&format!("vertices_{i}_lod_{level}")),
                                contents: bytemuck::cast_slice(&lod.primitive.vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        ),
                        index_buffer: device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some(&format!("indices_{i}_lod_{level}")),
                                contents: bytemuck::cast_slice(&lod.primitive.indices),
                                usage: wgpu::BufferUsages::INDEX,
                            },
                        ),
                        index_count: lod.primitive.indices.len() as u32,
                    });
                    distances.push(lod.min_distance);
                }
                (levels, distances)
            } else {
                (Vec::new(), Vec::new())
            };

            self.primitives.push(GpuPrimitive {
                vertex_buffer,
                index_buffer,
                index_count: prim.indices.len() as u32,
                uniform_buffer,
                bind_group,
                shadow_bind_group,
                instance_buffer,
                instance_count: 1,
                instance_transforms: Vec::new(),
                pre_instance_aabb: prim_bounds,
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
                node_index: prim.node_index,
                skin_index: prim.skin_index,
                joint_buffer,
                local_aabb_min: local_bounds.0,
                local_aabb_max: local_bounds.1,
                double_sided: material.double_sided,
                unlit: material.unlit,
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
                // AABB centre, NOT the vertex centroid: every other site that
                // maintains this field uses the box centre, so seeding it with a
                // centroid meant the value silently changed definition the first
                // time an animation or instance update ran. For an unevenly
                // tessellated mesh the two differ, so `set_animation_time(0.0)` -
                // no movement at all - could flip the LOD level across a switch
                // distance and reorder the transparent draw list.
                world_center: (prim_bounds.0 + prim_bounds.1) * 0.5,
                aabb_min: prim_bounds.0,
                aabb_max: prim_bounds.1,
                lod_levels,
                lod_min_distances,
                // Keep the neutral pose only for morphed primitives; the render
                // path re-blends from it each time the weights move.
                base_vertices: if has_morph {
                    prim.vertices.clone()
                } else {
                    Vec::new()
                },
                morph_targets: prim.morph_targets.clone(),
                morph_weights: prim.morph_weights.clone(),
                // Apply once up front so non-zero mesh default weights show
                // even before any animation channel drives them.
                morph_dirty: has_morph && prim.morph_weights.iter().any(|w| *w != 0.0),
            });
        }
        self.uploaded_texture_count = uploaded_textures;
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
            // A stale view here reads a destroyed texture.
            self.histogram.set_input(gpu, &self.hdr_view);
            self.target_size = (width, height);
            self.hdr_rebound_needed = true;
        }
        if self.hdr_rebound_needed {
            self.bloom.rebuild(gpu, width, height, &self.hdr_view);
            self.ssao.rebuild(gpu, width, height, &self.depth);
            let bloom_out = self
                .bloom
                .output()
                .expect("bloom output exists after rebuild")
                .clone();
            let ao_out = self
                .ssao
                .output()
                .expect("ssao output exists after rebuild")
                .clone();
            tonemap.set_input(
                gpu,
                &self.hdr_view,
                &bloom_out,
                &ao_out,
                self.histogram.exposure_buffer(),
            );
            self.hdr_rebound_needed = false;
        }
        tonemap.set_params(
            &gpu.queue,
            self.bloom_strength,
            self.ssao_strength,
            self.exposure_ev,
        );
        // Manual mode still routes through the reduction, which copies the
        // slider value into the same buffer the tonemap reads - one source of
        // truth, and switching modes cannot strand a stale value.
        self.histogram.set_exposure_settings(
            &gpu.queue,
            crate::render::histogram::ExposureSettings {
                delta_time_seconds: self.frame_delta_seconds.max(0.0),
                speed: self.auto_exposure_speed,
                auto_enabled: self.auto_exposure,
                manual_ev: self.exposure_ev,
            },
        );

        self.update_joint_matrices(gpu);
        self.apply_morph_targets(gpu);

        let aspect = width as f32 / height as f32;
        let view_proj = camera.view_projection(aspect);
        let frustum = Frustum::from_view_proj(&view_proj);
        self.ssao
            .write_uniforms(&gpu.queue, camera.projection(aspect), 0.6, 0.02, 1.0);
        self.update_cascades(camera);
        let light_space = self.cascade_matrices[0];
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
                normal_matrix: normal_matrix_of(prim.model).to_cols_array_2d(),
                light_space: light_space.to_cols_array_2d(),
                light_space_1: self.cascade_matrices[1].to_cols_array_2d(),
                light_space_2: self.cascade_matrices[2].to_cols_array_2d(),
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
                cascade_splits: self.cascade_splits,
                material_flags: [if prim.unlit { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            };
            gpu.queue
                .write_buffer(&prim.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        // The frame's pass wiring is declared in render::graph and validated
        // here in debug builds: a read of an unwritten resource (or a double
        // write) fails loudly instead of rendering black.
        debug_assert!(
            crate::render::graph::validate(&crate::render::graph::forward_frame_graph(), &[])
                .is_ok(),
            "forward frame graph is invalid"
        );

        // Claimed before any pass asks for a scope: begin_frame decides which
        // ring slot this frame writes into, and every scope handed out below
        // must agree on that answer.
        self.gpu_timing.begin_frame();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_encoder"),
            });
        let shadow_scope = self.gpu_timing.scope(TimedPass::ShadowCascades);
        // Locals rather than the fields: the draw loop immutably borrows
        // self.primitives, so the fields are written once, after the loop.
        let mut casters_drawn = 0u32;
        let mut casters_considered = 0u32;
        for cascade in 0..CASCADE_COUNT {
            // Cull casters against THIS cascade's light frustum - never the
            // camera's, which would delete shadows cast by geometry beside or
            // behind the viewer. Measured motivation: the comparison harness
            // put this pass at 0.119 ms against the C++ engine's 0.067 ms on
            // identical geometry, and per-cascade culling is the difference
            // between the two shadow paths.
            let light_frustum = Frustum::from_view_proj(&self.cascade_matrices[cascade]);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_layer_views[cascade],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: shadow_scope.render_writes(cascade, CASCADE_COUNT),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(1, &self.cascade_index_bind_groups[cascade], &[]);
            // Shadow casters draw at FULL detail regardless of LOD.
            //
            // Two reasons. First, camera distance is the wrong metric here at
            // all: the cascade renders from the light, and a primitive far
            // from the camera can be the one occluder filling a near cascade.
            // Second, a popping shadow silhouette is far more visible than a
            // popping mesh silhouette - the mesh pops at a size where it is a
            // handful of pixels, while its shadow can land right next to the
            // viewer at full size. If the shadow pass ever needs its own LOD,
            // it needs its own distance metric (per cascade, from the light),
            // not a share of this one.
            for prim in self.primitives.iter().filter(|p| p.casts_shadow) {
                casters_considered += 1;
                if !light_frustum.intersects_aabb_as_caster(prim.aabb_min, prim.aabb_max) {
                    continue;
                }
                casters_drawn += 1;
                pass.set_bind_group(0, &prim.shadow_bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, prim.instance_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..prim.instance_count);
            }
        }
        self.shadow_casters_drawn = casters_drawn;
        self.shadow_casters_considered = casters_considered;
        // Occlusion-cull stats, filled by the opaque loop below and read back
        // out to the fields once the pass (which borrows self) has ended.
        let mut occ_drawn = 0u32;
        let mut occ_considered = 0u32;
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
                        // Stored: SSAO reconstructs positions from it.
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self
                    .gpu_timing
                    .scope(TimedPass::Forward)
                    .render_writes(0, 1),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Group 1 is per-frame, not per-draw, so it is set once here and
            // again after the sky. The sky pipeline has its own layout, which
            // invalidates every group binding when it is bound.
            pass.set_bind_group(1, &self.ibl_bind_group, &[]);
            for (i, prim) in self.primitives.iter().enumerate() {
                // Frustum first (cheap, no dependency on last frame).
                if prim.alpha_blend || !frustum.intersects_aabb(prim.aabb_min, prim.aabb_max) {
                    continue;
                }
                occ_considered += 1;
                // Occlusion skip uses LAST frame's query result (this frame's
                // is still being recorded below). One-frame latency is the
                // accepted cost; `visible` defaults to true for primitives the
                // readback has not covered, so nothing pops on the first frames
                // or right after a scene change. The occlusion pass still
                // queries EVERY primitive, so a skipped one is re-evaluated and
                // reappears the frame its occluder moves away.
                // Never skip a primitive the camera is INSIDE. Its proxy box
                // surrounds the eye, so the box's front faces are near-plane
                // clipped and only back faces rasterise; those sit behind the
                // primitive's own surface and fail LessEqual, so the query reports
                // zero, the skip empties that surface from the depth buffer, the
                // box passes next frame and the object strobes at frame rate.
                //
                // Cheap insurance rather than a fix for an observed failure: with
                // the current closed, back-face-culled fixtures nothing renders
                // from inside at all (so depth stays empty and the box passes
                // regardless), and I could not build a failing case from them.
                // Interior/double-sided geometry does reach the bad path.
                if self.occlusion_queries_enabled
                    && !self.occlusion.visible(i)
                    && !aabb_contains_point(prim.aabb_min, prim.aabb_max, eye)
                {
                    continue;
                }
                occ_drawn += 1;
                let pipeline = if prim.double_sided {
                    &self.pipeline_double_sided
                } else {
                    &self.pipeline
                };
                // Selection sits beside the frustum test on purpose: both are
                // per-primitive, per-frame decisions about what this draw
                // costs, and splitting them apart is how one of them ends up
                // computed and then ignored.
                let (vertex_buffer, index_buffer, index_count) = prim.geometry_for(eye);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &prim.bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, prim.instance_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..index_count, 0, 0..prim.instance_count);
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
            if !blended.is_empty() {
                pass.set_bind_group(1, &self.ibl_bind_group, &[]);
            }
            for prim in blended {
                let pipeline = if prim.double_sided {
                    &self.pipeline_blend_double_sided
                } else {
                    &self.pipeline_blend
                };
                let (vertex_buffer, index_buffer, index_count) = prim.geometry_for(eye);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &prim.bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, prim.instance_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..index_count, 0, 0..prim.instance_count);
            }
        }

        self.occlusion_drawn = occ_drawn;
        self.occlusion_considered = occ_considered;

        // Occlusion DETECTION rides the forward depth: a separate depth-only
        // pass draws each primitive's world AABB with depth-write off, so a box
        // fully behind other geometry counts zero fragments. It records after
        // the forward pass (which filled the depth) and before SSAO reads that
        // depth back - depth-write off means the two see identical depth. Same
        // view_proj as the forward draws, so the boxes line up with the
        // geometry. This increment only measures; nothing is skipped yet.
        if self.occlusion_queries_enabled && !self.primitives.is_empty() {
            let aabbs: Vec<(Vec3, Vec3)> = self
                .primitives
                .iter()
                .map(|p| (p.aabb_min, p.aabb_max))
                .collect();
            self.occlusion.record(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &self.depth,
                view_proj,
                &aabbs,
                self.gpu_timing.scope(TimedPass::OcclusionCull),
            );
        }

        // Skip these outright at zero strength. The tonemap composites bloom as
        // `bloom * params.x` and AO as `mix(1.0, ao_raw, params.y)`, so at zero
        // the (now stale) texture provably contributes nothing - while the passes
        // themselves still cost a brightpass plus a separable Gaussian, and a
        // half-res depth pass plus a 3x3 blur, every single frame. Turning the
        // overlay slider to 0 used to cost exactly as much as leaving it on.
        if self.bloom_strength > 0.0 {
            self.bloom
                .encode(&mut encoder, self.gpu_timing.scope(TimedPass::Bloom));
        }
        if self.ssao_strength > 0.0 {
            self.ssao
                .encode(&mut encoder, self.gpu_timing.scope(TimedPass::Ssao));
        }
        // Histogram and reduction run against THIS frame's HDR target, before
        // the tonemap reads the exposure they produce. Measuring the frame it
        // is about to expose costs one extra pass over the HDR image and
        // avoids the frame-of-lag a previous-frame measurement would add.
        self.histogram.encode(
            &mut encoder,
            width,
            height,
            self.gpu_timing.scope(TimedPass::Histogram),
        );
        self.histogram.encode_reduce(
            &mut encoder,
            self.gpu_timing.scope(TimedPass::ExposureReduce),
        );
        tonemap.render(
            &mut encoder,
            output_view,
            self.gpu_timing.scope(TimedPass::Tonemap),
        );
        // Resolve rides the frame's own encoder: a second submit purely to
        // copy 112 bytes would serialise against the frame it is measuring.
        self.gpu_timing.resolve(&mut encoder);
        gpu.queue.submit(Some(encoder.finish()));
        self.gpu_timing.end_frame(&gpu.device);
        // Consume any occlusion readback that has landed. Non-blocking: results
        // lag the frame they measured by one or more frames, which is fine -
        // detection is advisory and a later increment reads last-known
        // visibility. When disabled this is a cheap poll over empty slots.
        self.occlusion.end_frame(&gpu.device);
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
        self.render_to_pixels_with_format(
            gpu,
            width,
            height,
            camera,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
    }

    /// As [`Self::render_to_pixels`], but with an explicit target format.
    ///
    /// Exists so tests can exercise a NON-sRGB target, which is what browsers
    /// hand out: WebGPU canvases expose no sRGB surface format. The tonemap
    /// pass has to gamma-encode itself there, and that path is otherwise
    /// unreachable from a headless test - every readback target is sRGB.
    pub fn render_to_pixels_with_format(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<Vec<u8>> {
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
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
        if let Some(err) = pollster::block_on(error_scope.pop()) {
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

    /// Uploads joint matrices for skinned primitives (skin joint world
    /// transform * inverse bind matrix), called once per frame.
    fn update_joint_matrices(&mut self, gpu: &GpuContext) {
        let Some(world) = self.pending_joint_world.take() else {
            return;
        };
        if self.skins.is_empty() {
            return;
        }
        for prim in &self.primitives {
            let Some(skin_index) = prim.skin_index else {
                continue;
            };
            let Some(skin) = self.skins.get(skin_index) else {
                continue;
            };
            let matrices: Vec<[[f32; 4]; 4]> = skin
                .joints
                .iter()
                .take(MAX_JOINTS)
                .enumerate()
                .map(|(i, &node)| {
                    let joint_world = world.get(node).copied().unwrap_or(Mat4::IDENTITY);
                    let inverse_bind = skin
                        .inverse_bind_matrices
                        .get(i)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    (joint_world * inverse_bind).to_cols_array_2d()
                })
                .collect();
            if !matrices.is_empty() {
                gpu.queue
                    .write_buffer(&prim.joint_buffer, 0, bytemuck::cast_slice(&matrices));
            }
        }
    }

    /// Re-blend and re-upload the vertex buffer of every morphed primitive whose
    /// weights moved since the last frame. The dirty flag gates the work, so a
    /// paused animation (or a scene with no morph targets) costs nothing here,
    /// and only morphed primitives carry the `base_vertices` copy this reads.
    /// LOD levels are left un-morphed (simplified meshes drop morphing, v1).
    fn apply_morph_targets(&mut self, gpu: &GpuContext) {
        for prim in &mut self.primitives {
            if !prim.morph_dirty || prim.base_vertices.is_empty() {
                continue;
            }
            let blended = crate::scene::blend_morph_targets(
                &prim.base_vertices,
                &prim.morph_targets,
                &prim.morph_weights,
            );
            gpu.queue
                .write_buffer(&prim.vertex_buffer, 0, bytemuck::cast_slice(&blended));
            prim.morph_dirty = false;
        }
    }

    /// True when the uploaded scene carries animations.
    pub fn has_animations(&self) -> bool {
        !self.animations.is_empty()
    }

    /// Longest animation duration in seconds (0 when none).
    pub fn animation_duration(&self) -> f32 {
        self.animations
            .iter()
            .map(|a| a.duration)
            .fold(0.0, f32::max)
    }

    /// Samples every animation at `time` (seconds), recomputes node world
    /// transforms and retargets the affected primitives (transforms, AABBs,
    /// blend-sort centers).
    pub fn set_animation_time(&mut self, time: f32) {
        if self.animations.is_empty() || self.nodes.is_empty() {
            return;
        }
        for animation in &self.animations {
            for channel in &animation.channels {
                let Some(node) = self.nodes.get_mut(channel.node) else {
                    continue;
                };
                let t = if animation.duration > 0.0 {
                    time % animation.duration
                } else {
                    0.0
                };
                let (i0, i1, frac) = keyframe_lerp_indices(&channel.times, t);
                let dt = match (channel.times.get(i0), channel.times.get(i1)) {
                    (Some(a), Some(b)) => b - a,
                    _ => 0.0,
                };
                match &channel.values {
                    ChannelValues::Translation(values) => {
                        if let Some(v) =
                            sample_vec3(values, channel.interpolation, i0, i1, frac, dt)
                        {
                            node.translation = v;
                        }
                    }
                    ChannelValues::Rotation(values) => {
                        if let Some(q) =
                            sample_quat(values, channel.interpolation, i0, i1, frac, dt)
                        {
                            node.rotation = q;
                        }
                    }
                    ChannelValues::Scale(values) => {
                        if let Some(v) =
                            sample_vec3(values, channel.interpolation, i0, i1, frac, dt)
                        {
                            node.scale = v;
                        }
                    }
                    // Morph weights don't drive a node transform; they re-pose
                    // the primitives on this node, handled in a separate pass
                    // below so it doesn't fight the `nodes` mutable borrow.
                    ChannelValues::MorphWeights(_) => {}
                }
            }
        }

        // Morph-weight pass: sample per-target weights and mark affected
        // primitives dirty. `num_targets` comes from each primitive, so the
        // sampler strides the flattened channel correctly regardless of mesh.
        for animation in &self.animations {
            let t = if animation.duration > 0.0 {
                time % animation.duration
            } else {
                0.0
            };
            for channel in &animation.channels {
                let ChannelValues::MorphWeights(values) = &channel.values else {
                    continue;
                };
                let (i0, i1, frac) = keyframe_lerp_indices(&channel.times, t);
                let dt = match (channel.times.get(i0), channel.times.get(i1)) {
                    (Some(a), Some(b)) => b - a,
                    _ => 0.0,
                };
                for prim in &mut self.primitives {
                    if prim.node_index != Some(channel.node) || prim.morph_targets.is_empty() {
                        continue;
                    }
                    let n = prim.morph_targets.len();
                    let w = sample_morph_weights(values, n, channel.interpolation, i0, i1, frac, dt);
                    if w != prim.morph_weights {
                        prim.morph_weights = w;
                        prim.morph_dirty = true;
                    }
                }
            }
        }

        let world = CpuScene::compute_world_transforms(&self.nodes);
        self.pending_joint_world = Some(world.clone());
        for prim in &mut self.primitives {
            if let Some(node) = prim.node_index {
                if let Some(m) = world.get(node) {
                    prim.model = *m;
                    let mut bounds =
                        transform_aabb(*m, prim.local_aabb_min, prim.local_aabb_max);
                    if let Some(skin) = prim.skin_index.and_then(|s| self.skins.get(s)) {
                        bounds = widen_bounds_for_skin(
                            bounds,
                            prim.local_aabb_min,
                            prim.local_aabb_max,
                            skin,
                            &world,
                        );
                    }
                    // The posed box just moved, so the instances that replicate
                    // it have to be re-applied on top.
                    prim.pre_instance_aabb = bounds;
                    let (min, max) = instanced_bounds(bounds, &prim.instance_transforms);
                    prim.aabb_min = min;
                    prim.aabb_max = max;
                    prim.world_center = (min + max) * 0.5;
                }
            }
        }
        self.recompute_scene_bounds();
    }

    /// Re-derive `scene_bounds` from the current per-primitive world AABBs.
    ///
    /// These bounds are the ONLY input to cascade fitting (`update_cascades`), so
    /// anything that moves a primitive's box has to call this or the shadow
    /// cascades stay fitted to a scene that no longer exists - geometry then falls
    /// outside every cascade and neither receives nor casts shadows. Kept as one
    /// function precisely because the bug it fixes was two call sites disagreeing
    /// about whose job this was.
    fn recompute_scene_bounds(&mut self) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for prim in &self.primitives {
            min = min.min(prim.aabb_min);
            max = max.max(prim.aabb_max);
        }
        if min.x <= max.x {
            self.scene_bounds = Some((min, max));
        }
    }

    /// Fits one orthographic light matrix per cascade: cascade 0 hugs the
    /// camera focus (crisp near shadows), the last covers the whole scene.
    fn update_cascades(&mut self, camera: &OrbitCamera) {
        let (min, max) = self
            .scene_bounds
            .unwrap_or((Vec3::splat(-1.0), Vec3::splat(1.0)));
        let scene_center = (min + max) * 0.5;
        let scene_radius = ((max - min).length() * 0.5).max(1e-3);

        let near_radius = (scene_radius * 0.35).max(0.5);
        let mid_radius = (scene_radius * 0.7).max(1.0);
        self.cascade_splits = [
            near_radius * 2.0,
            mid_radius * 2.0,
            CASCADE_COUNT as f32,
            0.0,
        ];

        let focus_near = camera.target.lerp(camera.eye(), 0.15);
        let cascades = [
            (focus_near, near_radius),
            (camera.target, mid_radius),
            (scene_center, scene_radius),
        ];
        for (i, (center, radius)) in cascades.into_iter().enumerate() {
            self.cascade_matrices[i] = self.light_matrix_for(center, radius);
        }
    }

    fn light_matrix_for(&self, center: Vec3, radius: f32) -> Mat4 {
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
        // Pull the eye far back so casters outside the cascade box still fit
        // in the depth range.
        let eye = center + light_dir * (radius * 4.0);
        let view = Mat4::look_at_rh(eye, center, up);
        let projection = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 8.0);
        projection * view
    }

    /// Orthographic world->light-clip matrix fitted to the scene bounds.
    #[allow(dead_code)]
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
        let projection = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 4.0);
        projection * view
    }
}

/// Group 1 uniforms, mirroring `IblParams` in forward.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct IblUniforms {
    // x: enabled flag, y: highest prefiltered mip, z: intensity, w: unused
    enabled_maxmip_intensity: [f32; 4],
}

impl IblUniforms {
    fn disabled() -> Self {
        Self {
            enabled_maxmip_intensity: [0.0, 0.0, 1.0, 0.0],
        }
    }

    fn enabled(max_mip: f32, intensity: f32) -> Self {
        Self {
            enabled_maxmip_intensity: [1.0, max_mip, intensity, 0.0],
        }
    }
}

fn create_ibl_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture =
        |binding: u32, dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dimension,
                multisampled: false,
            },
            count: None,
        };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ibl_forward_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            texture(1, wgpu::TextureViewDimension::Cube),
            texture(2, wgpu::TextureViewDimension::Cube),
            texture(3, wgpu::TextureViewDimension::D2),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_ibl_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    irradiance: &wgpu::TextureView,
    prefiltered: &wgpu::TextureView,
    brdf_lut: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl_forward_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(irradiance),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(prefiltered),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(brdf_lut),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
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
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
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
                depth_write_enabled: Some(!blend),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
            buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
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
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
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
        self.test_planes(min, max, &self.planes)
    }

    /// Visibility test for SHADOW CASTERS: identical, except the near plane
    /// is ignored - a correctness requirement, not an optimisation. A
    /// cascade's ortho box is fitted to the camera slice it covers; geometry
    /// between the light and that box lies outside the near plane but still
    /// casts into it along the box's own depth axis. Culling it produces the
    /// missing-shadow-from-tall-geometry bug the C++ engine documents in
    /// scene/Frustum.ixx. Side and far planes are safe: a caster outside them
    /// in the light's XY casts its shadow outside the map too.
    pub(crate) fn intersects_aabb_as_caster(&self, min: Vec3, max: Vec3) -> bool {
        let [left, right, bottom, top, _near, far] = &self.planes;
        self.test_planes(min, max, &[*left, *right, *bottom, *top, *far])
    }

    fn test_planes(&self, min: Vec3, max: Vec3, planes: &[glam::Vec4]) -> bool {
        for plane in planes {
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

/// Inverse-transpose of a model matrix, guarded against singular input.
///
/// A zero-scale node is how Blender (and plenty of exporters) hide an object, so
/// singular model matrices arrive in ordinary files. `Mat4::inverse` on one
/// yields inf/NaN, which poisons the normal matrix and shades that primitive as
/// garbage. Fall back to identity: the geometry is degenerate anyway, and a
/// finite wrong normal is strictly better than a NaN one.
fn normal_matrix_of(model: Mat4) -> Mat4 {
    let inv = model.inverse();
    if inv.is_finite() {
        inv.transpose()
    } else {
        Mat4::IDENTITY
    }
}

/// True when `p` lies inside the AABB expanded by the SAME margin the occlusion
/// proxy box uses (`occlusion_bbox.wgsl`: 2% of the half-extent plus 1 cm), so
/// this CPU test agrees with the box actually rasterised rather than a slightly
/// different one.
fn aabb_contains_point(min: Vec3, max: Vec3, p: Vec3) -> bool {
    let margin = (max - min) * 0.5 * 0.02 + Vec3::splat(0.01);
    p.cmpge(min - margin).all() && p.cmple(max + margin).all()
}

/// Bounds covering every instance of `pre`.
///
/// See `docs/renderer-bounds-invariant.md` for the rule these helpers exist to
/// uphold, the full list of consumers that read bounds, and why each over-cover
/// argument is a proof rather than a fudge factor. Eight bugs in this renderer
/// were the same bug; that document is the checklist for not writing a ninth.
///
/// The shader builds its world position as `instance_matrix * skin_matrix * v`,
/// so instance transforms apply ON TOP of the posed box. With bounds left at the
/// un-instanced position the frustum test culls the whole primitive - every
/// instance with it - as soon as the base position leaves the view, even while
/// the instances are on screen. Empty means the default single identity
/// instance, for which the bounds are unchanged.
fn instanced_bounds(pre: (Vec3, Vec3), instances: &[Mat4]) -> (Vec3, Vec3) {
    if instances.is_empty() {
        return pre;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for m in instances {
        let (lo, hi) = transform_aabb(*m, pre.0, pre.1);
        min = min.min(lo);
        max = max.max(hi);
    }
    (min, max)
}

/// Widen world bounds so they cover every pose the skin can put this geometry
/// in.
///
/// A skinned vertex ignores the node/model matrix entirely (`skin_matrix` in
/// forward.wgsl returns the joint blend and only falls back to `uniforms.model`
/// when the weights are zero), so node-derived bounds describe the bind pose,
/// not what is drawn. Skin weights are normalized, which makes the skinned
/// position a CONVEX COMBINATION of `J_i * v`; it therefore lies inside the
/// union of the per-joint boxes, so unioning them is conservative and never
/// culls visible geometry. The caller's node-derived box is kept in the union
/// because a skinned primitive may still contain zero-weight vertices.
fn widen_bounds_for_skin(
    bounds: (Vec3, Vec3),
    local_min: Vec3,
    local_max: Vec3,
    skin: &CpuSkin,
    world: &[Mat4],
) -> (Vec3, Vec3) {
    let (mut min, mut max) = bounds;
    for (i, &joint_node) in skin.joints.iter().take(MAX_JOINTS).enumerate() {
        let jw = world.get(joint_node).copied().unwrap_or(Mat4::IDENTITY);
        let ib = skin
            .inverse_bind_matrices
            .get(i)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        let (lo, hi) = transform_aabb(jw * ib, local_min, local_max);
        min = min.min(lo);
        max = max.max(hi);
    }
    (min, max)
}

fn primitive_world_aabb(prim: &crate::scene::CpuPrimitive) -> (Vec3, Vec3) {
    // Morphed primitives go through the pose-covering local bounds and are then
    // transformed as a box, which stays conservative under rotation. Everything
    // else keeps the exact per-vertex transform - a tighter box, and the path
    // every non-morph primitive used before morph targets existed.
    if !prim.morph_targets.is_empty() {
        let (lo, hi) = primitive_local_aabb(prim);
        return transform_aabb(prim.transform, lo, hi);
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &prim.vertices {
        let world = prim
            .transform
            .transform_point3(Vec3::from_array(vertex.position));
        // Same reasoning as primitive_local_aabb: never let a non-finite vertex
        // (or a non-finite transform) reach the cascade fitting.
        if !world.is_finite() {
            continue;
        }
        min = min.min(world);
        max = max.max(world);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

fn keyframe_lerp_indices(times: &[f32], t: f32) -> (usize, usize, f32) {
    if times.is_empty() {
        return (0, 0, 0.0);
    }
    if t <= times[0] {
        return (0, 0, 0.0);
    }
    if t >= *times.last().unwrap() {
        let last = times.len() - 1;
        return (last, last, 0.0);
    }
    let mut i = 0;
    while i + 1 < times.len() && times[i + 1] < t {
        i += 1;
    }
    let span = (times[i + 1] - times[i]).max(1e-6);
    (i, i + 1, (t - times[i]) / span)
}

/// glTF CUBICSPLINE Hermite basis weights for (value0, out_tangent0, value1,
/// in_tangent1) at local time `t` in [0,1] over a segment of duration `td`.
/// The tangent weights are scaled by `td` per the glTF spec.
fn cubic_spline_weights(t: f32, td: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        td * (t3 - 2.0 * t2 + t),
        -2.0 * t3 + 3.0 * t2,
        td * (t3 - t2),
    )
}

/// Sample a Vec3 channel (translation/scale) between keyframes `i0` and `i1` at
/// fraction `frac`, honoring the interpolation mode. `dt` is the segment's time
/// span (used by CubicSpline). For CubicSpline the array is 3x `times`
/// (in-tangent, value, out-tangent per keyframe).
fn sample_vec3(
    values: &[Vec3],
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Option<Vec3> {
    match interp {
        Interpolation::Step => values.get(i0).copied(),
        Interpolation::Linear => Some(values.get(i0)?.lerp(*values.get(i1)?, frac)),
        Interpolation::CubicSpline => {
            let p0 = *values.get(3 * i0 + 1)?;
            let m0 = *values.get(3 * i0 + 2)?;
            let m1 = *values.get(3 * i1)?;
            let p1 = *values.get(3 * i1 + 1)?;
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            Some(p0 * w0 + m0 * w1 + p1 * w2 + m1 * w3)
        }
    }
}

/// Sample a rotation channel. CubicSpline interpolates the quaternion components
/// with the Hermite basis and renormalizes (the glTF-spec approximation);
/// Linear uses slerp; Step holds the keyframe.
fn sample_quat(
    values: &[Quat],
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Option<Quat> {
    match interp {
        Interpolation::Step => values.get(i0).copied(),
        Interpolation::Linear => Some(values.get(i0)?.slerp(*values.get(i1)?, frac)),
        Interpolation::CubicSpline => {
            let p0 = *values.get(3 * i0 + 1)?;
            let m0 = *values.get(3 * i0 + 2)?;
            let m1 = *values.get(3 * i1)?;
            let p1 = *values.get(3 * i1 + 1)?;
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            let v4 = |q: Quat| Vec4::new(q.x, q.y, q.z, q.w);
            let v = v4(p0) * w0 + v4(m0) * w1 + v4(p1) * w2 + v4(m1) * w3;
            Some(Quat::from_vec4(v.normalize()))
        }
    }
}

/// Sample a morph-weights channel: `n` weights per keyframe, returned as a Vec
/// of length `n`. Under CubicSpline the channel stores `3 * n` per keyframe as
/// three contiguous `n`-blocks (in-tangents, values, out-tangents), each target
/// Hermite-interpolated with the same basis the vector/quaternion paths use. On
/// any out-of-range index the weight falls back to 0 rather than panicking.
fn sample_morph_weights(
    values: &[f32],
    n: usize,
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let at = |idx: usize| values.get(idx).copied().unwrap_or(0.0);
    match interp {
        Interpolation::Step => (0..n).map(|k| at(i0 * n + k)).collect(),
        Interpolation::Linear => (0..n)
            .map(|k| {
                let a = at(i0 * n + k);
                let b = at(i1 * n + k);
                a + (b - a) * frac
            })
            .collect(),
        Interpolation::CubicSpline => {
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            (0..n)
                .map(|k| {
                    let p0 = at(i0 * 3 * n + n + k); // value at start
                    let m0 = at(i0 * 3 * n + 2 * n + k); // out-tangent at start
                    let m1 = at(i1 * 3 * n + k); // in-tangent at end
                    let p1 = at(i1 * 3 * n + n + k); // value at end
                    p0 * w0 + m0 * w1 + p1 * w2 + m1 * w3
                })
                .collect()
        }
    }
}

/// Local-space bounds that cover every pose the primitive can reach.
///
/// Morph targets move vertices away from the neutral pose, so bounds taken from
/// `vertices` alone would be too small and frustum culling would drop a morphed
/// primitive while it is still on screen. For weights in [0,1] the reachable
/// extremes of a vertex are its position plus the sum of the negative deltas
/// (lower corner) and plus the sum of the positive deltas (upper corner), so
/// this is exact over that weight range and conservative outside it. Primitives
/// without morph targets are unaffected.
fn primitive_local_aabb(prim: &crate::scene::CpuPrimitive) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (i, vertex) in prim.vertices.iter().enumerate() {
        let p = Vec3::from_array(vertex.position);
        // Skip non-finite positions instead of letting them propagate. A single
        // NaN/inf vertex otherwise NaNs these bounds, then the scene bounds, then
        // the cascade radius - and ALL THREE cascade matrices become NaN, so
        // shadows break for every object in the scene, not just the bad mesh.
        // (`Frustum::test_planes` also treats NaN as visible, so the bad
        // primitive would additionally never cull.)
        if !p.is_finite() {
            continue;
        }
        let (mut lo, mut hi) = (p, p);
        for target in &prim.morph_targets {
            if let Some(d) = target.position_deltas.get(i) {
                lo += d.min(Vec3::ZERO);
                hi += d.max(Vec3::ZERO);
            }
        }
        min = min.min(lo);
        max = max.max(hi);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

fn transform_aabb(m: Mat4, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let mut out_min = Vec3::splat(f32::INFINITY);
    let mut out_max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let p = m.transform_point3(corner);
        out_min = out_min.min(p);
        out_max = out_max.max(p);
    }
    (out_min, out_max)
}

fn compute_world_bounds(scene: &CpuScene) -> Option<(Vec3, Vec3)> {
    // Union the per-primitive world AABBs rather than raw vertices: that helper
    // covers the morphed pose, so a morphing scene cannot report bounds smaller
    // than the geometry it actually draws (these bounds fit the shadow
    // cascades). For primitives without morph targets the helper still does the
    // exact per-vertex transform, so this is identical to the previous result.
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for prim in &scene.primitives {
        if prim.vertices.is_empty() {
            continue;
        }
        let (lo, hi) = primitive_world_aabb(prim);
        bounds = Some(match bounds {
            None => (lo, hi),
            Some((min, max)) => (min.min(lo), max.max(hi)),
        });
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
        mipmap_filter: if desc.mip_nearest { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
        anisotropy_clamp: anisotropy_for(desc),
        ..Default::default()
    })
}

/// Anisotropy for a glTF sampler.
///
/// The mip chain is already generated correctly, so this is the cheapest visible
/// quality win available: without it a floor or wall seen at a grazing angle -
/// i.e. most of any architectural or photogrammetry scene - is over-blurred by
/// several mip levels.
///
/// wgpu REQUIRES min/mag/mipmap to all be Linear before anisotropy above 1, and
/// validates it, so a sampler that asked for Nearest anywhere must stay at 1.
/// That is not a nicety: returning 16 there is a device-lost-grade validation
/// error, and the nearest-filtered assets are exactly the pixel-art ones whose
/// look the author chose deliberately.
fn anisotropy_for(desc: &CpuSampler) -> u16 {
    if desc.mag_nearest || desc.min_nearest || desc.mip_nearest {
        1
    } else {
        16
    }
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

fn compressed_wgpu_format(
    format: crate::scene::CompressedFormat,
    srgb: bool,
) -> wgpu::TextureFormat {
    use crate::scene::CompressedFormat as F;
    match (format, srgb) {
        (F::Bc1RgbaUnorm, false) => wgpu::TextureFormat::Bc1RgbaUnorm,
        (F::Bc1RgbaUnorm, true) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        (F::Bc3RgbaUnorm, false) => wgpu::TextureFormat::Bc3RgbaUnorm,
        (F::Bc3RgbaUnorm, true) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
        (F::Bc5RgUnorm, _) => wgpu::TextureFormat::Bc5RgUnorm,
        (F::Bc7RgbaUnorm, false) => wgpu::TextureFormat::Bc7RgbaUnorm,
        (F::Bc7RgbaUnorm, true) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    }
}

/// Uploads a pre-compressed (BCn) mip chain without touching the pixels.
fn create_compressed_texture(
    gpu: &GpuContext,
    texture: &CpuTexture,
    compressed: &crate::scene::CompressedTexture,
    srgb: bool,
    label: Option<&str>,
) -> wgpu::TextureView {
    let format = compressed_wgpu_format(compressed.format, srgb);
    let block_bytes = compressed.format.block_bytes();
    let gpu_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: wgpu::Extent3d {
            width: texture.width.max(1),
            height: texture.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: compressed.mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, data) in compressed.mips.iter().enumerate() {
        let w = (texture.width >> level).max(1);
        let h = (texture.height >> level).max(1);
        let blocks_wide = w.div_ceil(4);
        let blocks_high = h.div_ceil(4);
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
                bytes_per_row: Some(blocks_wide * block_bytes),
                rows_per_image: Some(blocks_high),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_material_texture(
    gpu: &GpuContext,
    texture: &CpuTexture,
    srgb: bool,
    label: Option<&str>,
) -> wgpu::TextureView {
    if let Some(compressed) = texture.compressed.as_ref() {
        if gpu.supports_bc {
            return create_compressed_texture(gpu, texture, compressed, srgb, label);
        }
        log::warn!("block-compressed texture but no BC support; falling back to white");
        return create_material_texture(
            gpu,
            &CpuTexture {
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
                compressed: None,
            },
            srgb,
            label,
        );
    }
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

    #[test]
    fn caster_test_ignores_the_near_plane_and_nothing_else() {
        // Ortho box [-1,1]^3: something BEHIND the near plane (z just above
        // 1 in view space, i.e. between the light and the box) must survive
        // the caster test - its shadow still falls into the box - while the
        // full test rejects it. Anything outside a SIDE plane must fail both.
        let light = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, Vec3::Y);

        let behind_near = (Vec3::new(-0.1, -0.1, 1.4), Vec3::new(0.1, 0.1, 1.6));
        let frustum = Frustum::from_view_proj(&light);
        assert!(
            !frustum.intersects_aabb(behind_near.0, behind_near.1),
            "the full test must reject a box behind the near plane"
        );
        assert!(
            frustum.intersects_aabb_as_caster(behind_near.0, behind_near.1),
            "a caster between the light and the box still shadows into it"
        );

        let beside = (Vec3::new(5.0, -0.1, 0.0), Vec3::new(5.2, 0.1, 0.2));
        assert!(
            !frustum.intersects_aabb_as_caster(beside.0, beside.1),
            "outside a side plane, the shadow lands outside the map too"
        );

        let inside = (Vec3::splat(-0.2), Vec3::splat(0.2));
        assert!(frustum.intersects_aabb_as_caster(inside.0, inside.1));
    }
    use super::*;
    use crate::scene::camera::OrbitCamera;

    #[test]
    fn cubic_spline_weights_collapse_to_the_keyframe_at_the_ends() {
        // At t=0 only the value0 weight is 1; at t=1 only the value1 weight is 1.
        let (a, b, c, d) = cubic_spline_weights(0.0, 3.0);
        assert!((a - 1.0).abs() < 1e-6 && b.abs() < 1e-6 && c.abs() < 1e-6 && d.abs() < 1e-6);
        let (a, b, c, d) = cubic_spline_weights(1.0, 3.0);
        assert!(a.abs() < 1e-6 && b.abs() < 1e-6 && (c - 1.0).abs() < 1e-6 && d.abs() < 1e-6);
    }

    #[test]
    fn cubic_spline_vec3_hits_keyframe_values_at_segment_ends() {
        // 2 keyframes, cubic layout: [in0, v0, out0, in1, v1, out1]. The tangents
        // are non-zero to prove the ends ignore them.
        let v0 = Vec3::new(1.0, 2.0, 3.0);
        let v1 = Vec3::new(4.0, 5.0, 6.0);
        let vals = vec![Vec3::ONE, v0, Vec3::splat(9.0), Vec3::splat(-7.0), v1, Vec3::ONE];
        let start = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 0.0, 2.0).unwrap();
        let end = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 1.0, 2.0).unwrap();
        assert!((start - v0).length() < 1e-5, "t=0 should be v0, got {start:?}");
        assert!((end - v1).length() < 1e-5, "t=1 should be v1, got {end:?}");
    }

    #[test]
    fn cubic_spline_vec3_zero_tangents_reduce_to_smoothstep_midpoint() {
        // With zero tangents the Hermite basis is h00·v0 + h01·v1; at t=0.5 both
        // are 0.5, so the midpoint is the linear midpoint.
        let vals = vec![Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ONE, Vec3::ZERO];
        let mid = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 0.5, 1.0).unwrap();
        assert!(
            (mid - Vec3::splat(0.5)).length() < 1e-5,
            "zero-tangent cubic midpoint should be 0.5, got {mid:?}"
        );
    }

    #[test]
    fn step_holds_and_linear_interpolates_vec3() {
        let vals = vec![Vec3::ZERO, Vec3::splat(2.0)];
        assert_eq!(
            sample_vec3(&vals, Interpolation::Step, 0, 1, 0.9, 1.0).unwrap(),
            Vec3::ZERO,
            "Step must hold the left keyframe"
        );
        let lin = sample_vec3(&vals, Interpolation::Linear, 0, 1, 0.5, 1.0).unwrap();
        assert!((lin - Vec3::splat(1.0)).length() < 1e-6);
    }

    #[test]
    fn cubic_spline_quat_ends_are_the_normalized_keyframes() {
        let q0 = Quat::from_rotation_y(0.3);
        let q1 = Quat::from_rotation_y(1.1);
        let vals = vec![Quat::IDENTITY, q0, Quat::IDENTITY, Quat::IDENTITY, q1, Quat::IDENTITY];
        let start = sample_quat(&vals, Interpolation::CubicSpline, 0, 1, 0.0, 1.0).unwrap();
        let end = sample_quat(&vals, Interpolation::CubicSpline, 0, 1, 1.0, 1.0).unwrap();
        // dot ~= 1 means (near-)identical orientation (sign-agnostic).
        assert!(start.dot(q0).abs() > 0.999, "t=0 should match q0");
        assert!(end.dot(q1).abs() > 0.999, "t=1 should match q1");
        assert!((start.length() - 1.0).abs() < 1e-5, "result must be a unit quaternion");
    }

    #[test]
    fn a_singular_model_matrix_yields_a_finite_normal_matrix() {
        // Zero scale on an axis is how Blender hides an object, so singular
        // matrices arrive in ordinary files. inverse() is inf/NaN there, and a
        // NaN normal matrix shades the primitive as garbage.
        let squashed = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(!squashed.inverse().is_finite(), "precondition: inverse is not finite");
        assert!(normal_matrix_of(squashed).is_finite(), "guard must return something finite");
        // A well-formed matrix must still get the real inverse-transpose.
        let ok = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        let expected = ok.inverse().transpose();
        assert!((normal_matrix_of(ok) - expected).abs_diff_eq(Mat4::ZERO, 1e-6));
    }

    #[test]
    fn a_non_finite_vertex_cannot_poison_the_bounds() {
        // One bad vertex used to NaN these bounds, then the scene bounds, then
        // the cascade radius - breaking shadows for EVERY object in the scene.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets/cube.gltf");
        let scene = crate::load_gltf(path).expect("cube.gltf must load");
        let mut prim = scene.primitives[0].clone();
        prim.vertices[0].position = [f32::NAN, 1.0, f32::INFINITY];

        let (min, max) = primitive_local_aabb(&prim);
        assert!(
            min.is_finite() && max.is_finite(),
            "local bounds must stay finite, got {min:?}..{max:?}"
        );
        let (wmin, wmax) = primitive_world_aabb(&prim);
        assert!(
            wmin.is_finite() && wmax.is_finite(),
            "world bounds must stay finite, got {wmin:?}..{wmax:?}"
        );
        // And the surviving vertices must still define a real box.
        assert!(max.x > min.x, "the good vertices must still bound something");
    }

    #[test]
    fn anisotropy_is_requested_only_when_every_filter_is_linear() {
        // wgpu validates this: anisotropy > 1 with any Nearest filter is an
        // error, not a hint. The all-linear default must still get the win.
        let linear = CpuSampler::default();
        assert_eq!(anisotropy_for(&linear), 16, "all-linear sampler should get 16x");

        for (label, s) in [
            ("mag", CpuSampler { mag_nearest: true, ..Default::default() }),
            ("min", CpuSampler { min_nearest: true, ..Default::default() }),
            ("mip", CpuSampler { mip_nearest: true, ..Default::default() }),
        ] {
            assert_eq!(
                anisotropy_for(&s),
                1,
                "{label}-nearest sampler must stay at 1x or wgpu rejects it"
            );
        }
    }

    #[test]
    fn aabb_contains_point_matches_the_occlusion_proxy_margin() {
        // The guard that keeps a camera-containing primitive drawn must agree
        // with the box occlusion_bbox.wgsl actually rasterises, or it protects a
        // slightly different volume than the one that misbehaves. That shader
        // expands by `half * 0.02 + 0.01`.
        let (min, max) = (Vec3::splat(-0.5), Vec3::splat(0.5));
        let margin = 0.5 * 0.02 + 0.01; // half-extent 0.5

        assert!(aabb_contains_point(min, max, Vec3::ZERO), "centre is inside");
        // Just inside the expanded face.
        let inside = 0.5 + margin - 1e-4;
        assert!(aabb_contains_point(min, max, Vec3::new(inside, 0.0, 0.0)));
        // Just outside it.
        let outside = 0.5 + margin + 1e-3;
        assert!(!aabb_contains_point(min, max, Vec3::new(outside, 0.0, 0.0)));
        // Must hold on every axis, not just x.
        assert!(!aabb_contains_point(min, max, Vec3::new(0.0, 0.0, outside)));
    }

    #[test]
    fn morph_targets_expand_the_culling_bounds() {
        // cube_morph.gltf is the unit cube (positions in [-0.5, 0.5]) plus one
        // target that lifts every vertex +1.0 in Y. Bounds taken from the
        // neutral pose alone would stop at y = 0.5, so a fully-weighted morph
        // would sit outside its own AABB and the frustum test could cull it
        // while it is plainly on screen.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets/cube_morph.gltf");
        let scene = crate::load_gltf(path).expect("cube_morph.gltf must load");
        let prim = &scene.primitives[0];
        assert_eq!(prim.morph_targets.len(), 1, "fixture must carry a target");

        let (min, max) = primitive_local_aabb(prim);
        assert!(
            (max.y - 1.5).abs() < 1e-5,
            "bounds must reach the fully-morphed pose (0.5 + 1.0), got max.y={}",
            max.y
        );
        // The neutral pose stays inside: the +Y target only has positive deltas,
        // so the lower bound must NOT be dragged upward.
        assert!(
            (min.y + 0.5).abs() < 1e-5,
            "unmorphed extent must be preserved, got min.y={}",
            min.y
        );
        // X/Z are untouched by the target and must stay exactly the cube's.
        assert!((max.x - 0.5).abs() < 1e-5 && (min.x + 0.5).abs() < 1e-5);
    }

    #[test]
    fn morph_weights_linear_and_step_stride_two_targets() {
        // 2 targets, 2 keyframes, flattened [k0t0, k0t1, k1t0, k1t1].
        let vals = vec![0.0, 1.0, 1.0, 3.0];
        let lin = sample_morph_weights(&vals, 2, Interpolation::Linear, 0, 1, 0.5, 1.0);
        assert_eq!(lin, vec![0.5, 2.0], "each target lerps independently");
        let step = sample_morph_weights(&vals, 2, Interpolation::Step, 0, 1, 0.9, 1.0);
        assert_eq!(step, vec![0.0, 1.0], "Step holds the left keyframe per target");
    }

    #[test]
    fn morph_weights_cubic_hits_keyframe_values_at_ends() {
        // 1 target, 2 keyframes, cubic layout is [in, value, out] per keyframe:
        // [in0, v0, out0, in1, v1, out1] with non-zero tangents to prove the
        // ends ignore them.
        let vals = vec![9.0, 0.2, -4.0, 7.0, 0.8, -3.0];
        let start = sample_morph_weights(&vals, 1, Interpolation::CubicSpline, 0, 1, 0.0, 1.0);
        let end = sample_morph_weights(&vals, 1, Interpolation::CubicSpline, 0, 1, 1.0, 1.0);
        assert!((start[0] - 0.2).abs() < 1e-5, "t=0 is value0, got {}", start[0]);
        assert!((end[0] - 0.8).abs() < 1e-5, "t=1 is value1, got {}", end[0]);
    }

    #[test]
    fn morph_weights_out_of_range_falls_back_to_zero() {
        // A malformed/short channel must not panic; missing samples read 0.
        let vals = vec![0.5];
        let w = sample_morph_weights(&vals, 3, Interpolation::Linear, 0, 1, 0.5, 1.0);
        assert_eq!(w.len(), 3);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert!(sample_morph_weights(&vals, 0, Interpolation::Linear, 0, 1, 0.5, 1.0).is_empty());
    }

    #[test]
    fn frustum_culls_out_of_view_aabbs() {
        let camera = OrbitCamera::default();
        let frustum = Frustum::from_view_proj(&camera.view_projection(1.0));

        // Cube at the orbit target is visible.
        assert!(frustum.intersects_aabb(Vec3::splat(-0.5), Vec3::splat(0.5)));
        // A cube far off to the side is culled.
        assert!(
            !frustum.intersects_aabb(Vec3::new(1000.0, -0.5, -0.5), Vec3::new(1001.0, 0.5, 0.5))
        );
        // Behind the camera is culled.
        let eye = camera.eye();
        let behind = eye + (eye - Vec3::ZERO).normalize() * 10.0;
        assert!(!frustum.intersects_aabb(behind - Vec3::splat(0.4), behind + Vec3::splat(0.4)));
    }
}
