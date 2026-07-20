//! Split-sum image-based lighting: the precompute that turns one
//! equirectangular HDR panorama into the three maps `forward.wgsl` samples.
//!
//! Karis' split-sum approximation factors the specular environment integral
//! into two independent pieces, each precomputable:
//!
//! ```text
//!   L_o ~= prefiltered(R, roughness) * (F0 * lut.r + lut.g)
//! ```
//!
//! so this module bakes, in order:
//!
//! 1. **Environment cube** ([`ENV_SIZE`], with a mip chain) - the panorama
//!    reprojected onto six faces. Everything downstream samples the cube, not
//!    the panorama, so the equirect distortion is paid for exactly once.
//! 2. **Irradiance cube** ([`IRRADIANCE_SIZE`]) - cosine convolution, for
//!    diffuse. Stores E / PI, the quantity that multiplies albedo directly.
//! 3. **Prefiltered cube** ([`PREFILTER_SIZE`], [`PREFILTER_MIPS`] mips) - GGX
//!    importance sampling, roughness = mip / (mips - 1).
//! 4. **BRDF LUT** ([`BRDF_LUT_SIZE`], `Rg16Float`) - the second split-sum
//!    factor.
//!
//! **The BRDF LUT is environment-independent by construction**: it integrates
//! the GGX BRDF against a *white furnace*, with no environment term anywhere in
//! [`crate::render::ibl`]'s `fs_brdf_lut`. It is therefore baked once by
//! [`BrdfLut`] and shared by every [`IblEnvironment`] the renderer ever sets.
//! Baking it per environment would be pure waste; hard-coding the Karis
//! analytic fit instead (which `forward.wgsl` still uses on the fallback path)
//! would have saved the 256x256 pass at the cost of a visible error at high
//! roughness, and the whole point of this feature is to stop approximating.
//!
//! **Precompute is one-shot.** [`IblEnvironment::bake`] records every pass into
//! one encoder and submits once; the frame path only ever samples the results.
//! Nothing here runs per frame, which is why none of it is wired into
//! [`crate::render::gpu_timing`].
//!
//! Decoding image files is still not this module's job: the entry point takes
//! decoded linear float pixels ([`EquirectImage`]) and the caller chooses how
//! they were produced - [`crate::asset::hdr::decode_hdr`] for Radiance `.hdr`
//! files, or [`EquirectImage::sky`] for the procedural fallback.
//! [`IblEnvironment::bake_hdr`] composes decode and bake for the common case.

use crate::context::GpuContext;

/// Environment cube face resolution.
///
/// 128 rather than the 512-1024 an offline baker would use: the prefiltered
/// cube is what actually carries sharp reflections and it is baked at the same
/// resolution, so this only bounds mirror-like detail. 128 keeps a full bake
/// (including the 8192-sample-per-texel irradiance convolution) inside a few
/// milliseconds, which matters because a bake blocks the frame that requests it.
pub const ENV_SIZE: u32 = 128;
/// Mips of the environment cube. Only the prefilter reads them, to pick a
/// footprint matching each GGX sample's solid angle.
pub const ENV_MIPS: u32 = 5;
/// Irradiance cube face resolution. Cosine convolution is a very low-pass
/// filter - 32 is the standard choice and holds all the signal there is.
pub const IRRADIANCE_SIZE: u32 = 32;
/// Prefiltered specular cube face resolution (mip 0).
pub const PREFILTER_SIZE: u32 = 128;
/// Roughness levels in the prefiltered cube: roughness = mip / (MIPS - 1).
pub const PREFILTER_MIPS: u32 = 5;
/// GGX samples per prefiltered texel.
pub const PREFILTER_SAMPLES: u32 = 256;
/// Split-sum BRDF lookup table edge length.
pub const BRDF_LUT_SIZE: u32 = 256;

// A mip chain deeper than its base resolution would ask wgpu to render into a
// zero-texel level. Checked at compile time because the sizes are constants and
// a runtime test would only notice on a machine that ran the bake.
const _: () = assert!(PREFILTER_SIZE >> (PREFILTER_MIPS - 1) >= 1);
const _: () = assert!(ENV_SIZE >> (ENV_MIPS - 1) >= 1);

/// Roughness the prefilter bakes into mip `mip`.
///
/// `forward.wgsl` inverts this as `roughness * max_prefiltered_mip`, so the two
/// have to be exact inverses or a material samples the wrong sharpness. Written
/// once here and used by the bake so there is only one place to get it wrong.
pub fn prefilter_roughness(mip: u32) -> f32 {
    mip as f32 / (PREFILTER_MIPS - 1) as f32
}

const CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// The panorama is uploaded as-is, in f32. `Float32Filterable` is an optional
/// WebGPU feature the context does not request, so `ibl.wgsl` filters it by
/// hand with textureLoad; that keeps the CPU side free of an f32 -> f16
/// conversion and its rounding decisions.
const EQUIRECT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// A decoded equirectangular panorama: linear radiance, RGBA, row-major.
#[derive(Clone, Debug)]
pub struct EquirectImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` floats. Alpha is ignored.
    pub rgba32f: Vec<f32>,
}

impl EquirectImage {
    /// Wraps decoded pixels, checking the buffer actually matches the extent.
    pub fn new(width: u32, height: u32, rgba32f: Vec<f32>) -> anyhow::Result<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        anyhow::ensure!(width > 0 && height > 0, "equirect image must be non-empty");
        anyhow::ensure!(
            rgba32f.len() == expected,
            "equirect image is {}x{} ({expected} floats) but got {}",
            width,
            height,
            rgba32f.len()
        );
        Ok(Self {
            width,
            height,
            rgba32f,
        })
    }

    /// A constant-radiance panorama.
    ///
    /// The only environment whose convolution has a closed form, which makes it
    /// the reference the irradiance pass is checked against.
    pub fn constant(width: u32, height: u32, radiance: [f32; 3]) -> Self {
        let mut rgba32f = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            rgba32f.extend_from_slice(&[radiance[0], radiance[1], radiance[2], 1.0]);
        }
        Self {
            width,
            height,
            rgba32f,
        }
    }

    /// The analytic sky of `sky.wgsl`, panoramised.
    ///
    /// Exists so the renderer has a usable environment with no asset pipeline
    /// and no decoder: setting it swaps the fallback's analytic hemisphere for
    /// a real convolution of the same sky, which is the cleanest way to see
    /// that the IBL path is doing something and doing it plausibly.
    pub fn sky(width: u32, height: u32) -> Self {
        const ZENITH: [f32; 3] = [0.09, 0.16, 0.35];
        const HORIZON: [f32; 3] = [0.55, 0.62, 0.72];
        const GROUND: [f32; 3] = [0.18, 0.16, 0.15];

        let mut rgba32f = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            // v = 0 is +Y, matching `equirect_uv` in ibl.wgsl.
            let v = (y as f32 + 0.5) / height as f32;
            let dir_y = (0.5 - v) * std::f32::consts::PI;
            let dir_y = dir_y.sin();
            let color = if dir_y >= 0.0 {
                lerp3(HORIZON, ZENITH, dir_y.clamp(0.0, 1.0).powf(0.7))
            } else {
                lerp3(HORIZON, GROUND, (-dir_y * 3.0).clamp(0.0, 1.0))
            };
            for _ in 0..width {
                rgba32f.extend_from_slice(&[color[0], color[1], color[2], 1.0]);
            }
        }
        Self {
            width,
            height,
            rgba32f,
        }
    }

    /// Largest radiance in any colour channel. The energy bound the irradiance
    /// map must respect is derived from this - see `tests/ibl.rs`.
    pub fn max_radiance(&self) -> f32 {
        self.rgba32f
            .chunks_exact(4)
            .flat_map(|texel| texel[..3].iter().copied())
            .fold(0.0f32, f32::max)
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// IEEE-754 binary16 to f32.
///
/// Hand-rolled rather than pulling in `half`: the crate ships to wasm and this
/// is needed only to read `Rgba16Float`/`Rg16Float` targets back for tests and
/// diagnostics - the frame path never sees a half on the CPU.
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let out = match exponent {
        0 if mantissa == 0 => sign << 31,
        0 => {
            // Subnormal: shift the implicit leading 1 into place and pay for
            // it in the exponent. Value is (m >> k)/1024 * 2^-14, i.e.
            // 1.xxx * 2^(-14-k), so the biased f32 exponent is 113 - k.
            let mut m = mantissa;
            let mut k = 0u32;
            while m & 0x400 == 0 {
                m <<= 1;
                k += 1;
            }
            (sign << 31) | ((113 - k) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (mantissa << 13),
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(out)
}

/// Shared pipeline set for every precompute pass.
///
/// One bind group layout serves all four entry points, with the texture a pass
/// does not read bound to a 1x1 dummy. Four near-identical layouts would be the
/// tidier abstraction and buy nothing: this is one-shot code that runs at most
/// once per environment change.
struct Precompute {
    bind_group_layout: wgpu::BindGroupLayout,
    equirect_to_cube: wgpu::RenderPipeline,
    downsample_cube: wgpu::RenderPipeline,
    irradiance: wgpu::RenderPipeline,
    prefilter: wgpu::RenderPipeline,
    brdf_lut: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    dummy_equirect: wgpu::TextureView,
    dummy_cube: wgpu::TextureView,
}

impl Precompute {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl_bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Not filterable, and not sampled: `sample_equirect`
                        // uses textureLoad. Declaring it filterable would need
                        // the FLOAT32_FILTERABLE feature, which browsers gate.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make = |entry_point: &str, format: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
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

        Self {
            equirect_to_cube: make("fs_equirect_to_cube", CUBE_FORMAT, "ibl_equirect_to_cube"),
            downsample_cube: make("fs_downsample_cube", CUBE_FORMAT, "ibl_downsample_cube"),
            irradiance: make("fs_irradiance", CUBE_FORMAT, "ibl_irradiance"),
            prefilter: make("fs_prefilter", CUBE_FORMAT, "ibl_prefilter"),
            brdf_lut: make("fs_brdf_lut", LUT_FORMAT, "ibl_brdf_lut"),
            bind_group_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("ibl_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            dummy_equirect: dummy_2d(device),
            dummy_cube: dummy_cube(device),
        }
    }
}

/// Per-draw uniforms, mirroring `Params` in ibl.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IblParams {
    face_roughness_samples_mip: [f32; 4],
    source_resolution: [f32; 4],
}

fn dummy_2d(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_dummy_2d"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: EQUIRECT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn dummy_cube(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_dummy_cube"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CUBE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl_dummy_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        })
}

/// The 1x1 stand-ins `forward.wgsl` samples when no environment is set.
///
/// The forward shader samples all three IBL maps unconditionally and `select`s
/// between the environment and the analytic result, so the bindings must be
/// valid even on the fallback path. Sampling a 1x1 texture and discarding the
/// result is the price; the alternative - two forward pipelines, one per
/// branch - doubles the pipeline set and the shader-reload path to save three
/// texture fetches.
pub struct IblFallback {
    pub irradiance: wgpu::TextureView,
    pub prefiltered: wgpu::TextureView,
    pub brdf_lut: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl IblFallback {
    pub fn new(device: &wgpu::Device) -> Self {
        let lut = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ibl_fallback_lut"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            irradiance: dummy_cube(device),
            prefiltered: dummy_cube(device),
            brdf_lut: lut,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("ibl_fallback_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }
}

/// The environment-independent split-sum BRDF table, baked once.
pub struct BrdfLut {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl BrdfLut {
    /// Bakes the table. Roughly `BRDF_LUT_SIZE^2 * 1024` GGX samples; measured
    /// at a few milliseconds on a discrete GPU, and it happens once per
    /// renderer, the first time an environment is set.
    pub fn new(gpu: &GpuContext) -> Self {
        let precompute = Precompute::new(&gpu.device);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_brdf_lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ibl_brdf_lut_encoder"),
            });
        draw_fullscreen(
            &gpu.device,
            &mut encoder,
            &precompute,
            &precompute.brdf_lut,
            &view,
            IblParams::default(),
            &precompute.dummy_equirect,
            &precompute.dummy_cube,
            "ibl_brdf_lut_pass",
        );
        gpu.queue.submit(Some(encoder.finish()));

        Self { texture, view }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// (scale, bias) per texel, row-major, `BRDF_LUT_SIZE^2` entries.
    /// `u` (column) is N.V, `v` (row) is roughness, both at texel centres.
    pub fn read_back(&self, gpu: &GpuContext) -> Vec<[f32; 2]> {
        let halves = read_texture_halves(gpu, &self.texture, 0, 0, BRDF_LUT_SIZE, BRDF_LUT_SIZE, 2);
        halves.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }
}

/// A baked environment: the three maps `forward.wgsl` binds.
///
/// The BRDF LUT is not here on purpose - it does not depend on the
/// environment, so it lives beside the renderer and outlives every
/// environment set on it.
pub struct IblEnvironment {
    environment: wgpu::Texture,
    irradiance: wgpu::Texture,
    prefiltered: wgpu::Texture,
    irradiance_view: wgpu::TextureView,
    prefiltered_view: wgpu::TextureView,
}

impl IblEnvironment {
    /// Bakes every environment-dependent map from `equirect`, in one submit.
    pub fn bake(gpu: &GpuContext, equirect: &EquirectImage) -> Self {
        let device = &gpu.device;
        let precompute = Precompute::new(device);

        let source = upload_equirect(gpu, equirect);
        let environment = create_cube(device, "ibl_environment", ENV_SIZE, ENV_MIPS);
        let irradiance = create_cube(device, "ibl_irradiance", IRRADIANCE_SIZE, 1);
        let prefiltered = create_cube(device, "ibl_prefiltered", PREFILTER_SIZE, PREFILTER_MIPS);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl_bake_encoder"),
        });

        // 1. Panorama -> cube, mip 0.
        for face in 0..6u32 {
            draw_fullscreen(
                device,
                &mut encoder,
                &precompute,
                &precompute.equirect_to_cube,
                &face_view(&environment, face, 0),
                IblParams {
                    face_roughness_samples_mip: [face as f32, 0.0, 0.0, 0.0],
                    source_resolution: [equirect.width as f32, 0.0, 0.0, 0.0],
                },
                &source,
                &precompute.dummy_cube,
                "ibl_equirect_to_cube_pass",
            );
        }

        // 2. Environment mip chain. Each level reads the one above it, so the
        // source view is restricted to that single mip - binding the whole
        // chain while rendering into part of it is a usage conflict.
        for mip in 1..ENV_MIPS {
            let source_mip = cube_view_of_mip(&environment, mip - 1);
            for face in 0..6u32 {
                draw_fullscreen(
                    device,
                    &mut encoder,
                    &precompute,
                    &precompute.downsample_cube,
                    &face_view(&environment, face, mip),
                    IblParams {
                        face_roughness_samples_mip: [face as f32, 0.0, 0.0, 0.0],
                        source_resolution: [(ENV_SIZE >> (mip - 1)) as f32, 0.0, 0.0, 0.0],
                    },
                    &precompute.dummy_equirect,
                    &source_mip,
                    "ibl_downsample_pass",
                );
            }
        }

        // 3. Diffuse irradiance, from mip 0 of the environment.
        let environment_all_mips = cube_view_all_mips(&environment);
        for face in 0..6u32 {
            draw_fullscreen(
                device,
                &mut encoder,
                &precompute,
                &precompute.irradiance,
                &face_view(&irradiance, face, 0),
                IblParams {
                    face_roughness_samples_mip: [face as f32, 0.0, 0.0, 0.0],
                    source_resolution: [ENV_SIZE as f32, 0.0, 0.0, 0.0],
                },
                &precompute.dummy_equirect,
                &environment_all_mips,
                "ibl_irradiance_pass",
            );
        }

        // 4. Prefiltered specular, one mip per roughness level.
        for mip in 0..PREFILTER_MIPS {
            let roughness = prefilter_roughness(mip);
            for face in 0..6u32 {
                draw_fullscreen(
                    device,
                    &mut encoder,
                    &precompute,
                    &precompute.prefilter,
                    &face_view(&prefiltered, face, mip),
                    IblParams {
                        face_roughness_samples_mip: [
                            face as f32,
                            roughness,
                            PREFILTER_SAMPLES as f32,
                            0.0,
                        ],
                        source_resolution: [ENV_SIZE as f32, 0.0, 0.0, 0.0],
                    },
                    &precompute.dummy_equirect,
                    &environment_all_mips,
                    "ibl_prefilter_pass",
                );
            }
        }

        gpu.queue.submit(Some(encoder.finish()));

        let irradiance_view = cube_view_all_mips(&irradiance);
        let prefiltered_view = cube_view_all_mips(&prefiltered);
        Self {
            environment,
            irradiance,
            prefiltered,
            irradiance_view,
            prefiltered_view,
        }
    }

    /// Radiance `.hdr` bytes straight to a baked environment.
    ///
    /// Just [`crate::asset::hdr::decode_hdr`] into [`Self::bake`], so a caller
    /// holding a downloaded panorama does not have to learn the intermediate
    /// [`EquirectImage`] step. The only error is the decode - the bake itself
    /// cannot fail.
    pub fn bake_hdr(
        gpu: &GpuContext,
        hdr_bytes: &[u8],
    ) -> Result<Self, crate::asset::hdr::HdrError> {
        Ok(Self::bake(gpu, &crate::asset::hdr::decode_hdr(hdr_bytes)?))
    }

    pub fn irradiance_view(&self) -> &wgpu::TextureView {
        &self.irradiance_view
    }

    pub fn prefiltered_view(&self) -> &wgpu::TextureView {
        &self.prefiltered_view
    }

    /// Highest prefiltered mip index, i.e. the roughness-1.0 level. The
    /// forward shader multiplies material roughness by this to pick a mip.
    pub fn max_prefiltered_mip(&self) -> f32 {
        (PREFILTER_MIPS - 1) as f32
    }

    /// RGB of one irradiance cube face, row-major.
    pub fn read_irradiance_face(&self, gpu: &GpuContext, face: u32) -> Vec<[f32; 3]> {
        read_cube_face_rgb(gpu, &self.irradiance, face, 0, IRRADIANCE_SIZE)
    }

    /// RGB of one prefiltered cube face at `mip`, row-major.
    pub fn read_prefiltered_face(&self, gpu: &GpuContext, face: u32, mip: u32) -> Vec<[f32; 3]> {
        read_cube_face_rgb(gpu, &self.prefiltered, face, mip, PREFILTER_SIZE >> mip)
    }

    /// RGB of one environment cube face at `mip`, row-major.
    pub fn read_environment_face(&self, gpu: &GpuContext, face: u32, mip: u32) -> Vec<[f32; 3]> {
        read_cube_face_rgb(gpu, &self.environment, face, mip, ENV_SIZE >> mip)
    }
}

fn create_cube(device: &wgpu::Device, label: &str, size: u32, mips: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CUBE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Render-target view of one face at one mip (a 2D view of one array layer).
fn face_view(texture: &wgpu::Texture, face: u32, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ibl_face_target"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

fn cube_view_all_mips(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ibl_cube_view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

fn cube_view_of_mip(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ibl_cube_view_single_mip"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn draw_fullscreen(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    precompute: &Precompute,
    pipeline: &wgpu::RenderPipeline,
    target: &wgpu::TextureView,
    params: IblParams,
    equirect: &wgpu::TextureView,
    cube: &wgpu::TextureView,
    label: &str,
) {
    // A fresh uniform buffer and bind group per draw. Dynamic offsets into one
    // buffer would be the frame-path answer; here there are at most ~60 draws
    // total, once, and per-draw buffers remove the 256-byte alignment
    // arithmetic that is the usual source of an off-by-one-face bug.
    let uniforms = wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("ibl_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl_bind_group"),
        layout: &precompute.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(equirect),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(cube),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&precompute.sampler),
            },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
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
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn upload_equirect(gpu: &GpuContext, image: &EquirectImage) -> wgpu::TextureView {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ibl_equirect_source"),
        size: wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: EQUIRECT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&image.rgba32f),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 16),
            rows_per_image: Some(image.height),
        },
        wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn read_cube_face_rgb(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    face: u32,
    mip: u32,
    size: u32,
) -> Vec<[f32; 3]> {
    let halves = read_texture_halves(gpu, texture, face, mip, size, size, 4);
    halves.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Blocking readback of a half-float texture subresource, decoded to f32.
///
/// Diagnostics and tests only: it stalls the queue. Nothing on the frame path
/// reads any of these maps back.
fn read_texture_halves(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    layer: u32,
    mip: u32,
    width: u32,
    height: u32,
    channels: u32,
) -> Vec<f32> {
    let unpadded = width * channels * 2;
    let bytes_per_row = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ibl_readback"),
        size: (bytes_per_row * height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl_readback_encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
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

    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let _ = receiver.recv();

    let values = {
        let data = slice.get_mapped_range();
        let mut values = Vec::with_capacity((width * height * channels) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            let end = start + unpadded as usize;
            for pair in data[start..end].chunks_exact(2) {
                values.push(half_to_f32(u16::from_le_bytes([pair[0], pair[1]])));
            }
        }
        values
    };
    buffer.unmap();
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_decoding_covers_normals_subnormals_and_specials() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        // 0x3555 is the nearest half to 1/3.
        assert!((half_to_f32(0x3555) - 1.0 / 3.0).abs() < 1e-3);
        // Largest finite half.
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        // Smallest positive subnormal: 2^-24. The subnormal branch is the one
        // a naive implementation gets wrong, and it is what a near-black
        // irradiance texel decodes through.
        assert_eq!(half_to_f32(0x0001), 2.0f32.powi(-24));
        // Largest subnormal: 1023 * 2^-24.
        assert_eq!(half_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        // Smallest normal: 2^-14.
        assert_eq!(half_to_f32(0x0400), 2.0f32.powi(-14));
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn a_constant_panorama_reports_its_own_radiance_as_the_maximum() {
        let image = EquirectImage::constant(8, 4, [0.25, 0.5, 0.75]);
        assert_eq!(image.rgba32f.len(), 8 * 4 * 4);
        assert_eq!(image.max_radiance(), 0.75);
    }

    #[test]
    fn equirect_construction_rejects_a_mismatched_buffer() {
        assert!(EquirectImage::new(4, 2, vec![0.0; 4 * 2 * 4]).is_ok());
        assert!(EquirectImage::new(4, 2, vec![0.0; 4 * 2 * 3]).is_err());
        assert!(EquirectImage::new(0, 2, Vec::new()).is_err());
    }

    #[test]
    fn the_procedural_sky_is_brightest_at_the_horizon_and_darkest_below() {
        // Rows run +Y (v = 0) to -Y. The analytic sky in sky.wgsl peaks at the
        // horizon, so the panorama must too - if the latitude mapping were
        // flipped, ground and zenith would swap and nothing else would notice.
        let height = 64;
        let sky = EquirectImage::sky(4, height);
        let luminance = |row: usize| {
            let base = row * 4 * 4;
            sky.rgba32f[base] + sky.rgba32f[base + 1] + sky.rgba32f[base + 2]
        };
        let zenith = luminance(0);
        let horizon = luminance(height as usize / 2);
        let ground = luminance(height as usize - 1);
        assert!(horizon > zenith, "horizon {horizon} vs zenith {zenith}");
        assert!(horizon > ground, "horizon {horizon} vs ground {ground}");
        assert!(ground < zenith, "ground {ground} vs zenith {zenith}");
    }

    #[test]
    fn prefilter_roughness_round_trips_through_the_shader_mip_selection() {
        // forward.wgsl picks a mip as `roughness * max_prefiltered_mip`. That
        // is the inverse of `prefilter_roughness` only if the endpoints line up
        // and nothing rounds, so walk the chain and check both directions.
        let max_mip = (PREFILTER_MIPS - 1) as f32;
        assert_eq!(prefilter_roughness(0), 0.0);
        assert_eq!(prefilter_roughness(PREFILTER_MIPS - 1), 1.0);
        for mip in 0..PREFILTER_MIPS {
            let selected = prefilter_roughness(mip) * max_mip;
            assert_eq!(
                selected,
                mip as f32,
                "roughness {} selects mip {selected}, not {mip}",
                prefilter_roughness(mip)
            );
        }
    }
}
