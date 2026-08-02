//! Texture, sampler and mip-chain creation, split out of `forward.rs`.

use crate::context::GpuContext;
use crate::render::forward::{DEPTH_FORMAT, HDR_FORMAT};
use crate::scene::{CompressedFormat, CompressedTexture, CpuSampler, CpuTexture, CpuWrap};

pub(crate) fn create_sampler(device: &wgpu::Device, desc: &CpuSampler) -> wgpu::Sampler {
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
pub(crate) fn anisotropy_for(desc: &CpuSampler) -> u16 {
    if desc.mag_nearest || desc.min_nearest || desc.mip_nearest {
        1
    } else {
        16
    }
}

pub(crate) fn srgb_to_linear(byte: u8) -> f32 {
    let c = byte as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn linear_to_srgb(value: f32) -> u8 {
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

pub(crate) fn compressed_wgpu_format(
    format: CompressedFormat,
    srgb: bool,
) -> wgpu::TextureFormat {
    use CompressedFormat as F;
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
pub(crate) fn create_compressed_texture(
    gpu: &GpuContext,
    texture: &CpuTexture,
    compressed: &CompressedTexture,
    srgb: bool,
    label: Option<&str>,
) -> wgpu::TextureView {
    if let Some(declared) = compressed.declared_srgb {
        if declared != srgb {
            log::warn!(
                "{}: KTX2 container declares {} but the material uses it as {}; \
                 glTF usage decides the GPU format, so this may render with the wrong gamma",
                label.unwrap_or("<unlabeled texture>"),
                if declared { "sRGB" } else { "linear" },
                if srgb { "sRGB" } else { "linear" },
            );
        }
    }
    let format = compressed_wgpu_format(compressed.format, srgb);
    // `compressed.mips` is trusted here (level count and per-level byte size both
    // match `texture.width`/`texture.height`) because `ktx2_loader::validate_mip_chain`
    // already rejected anything that wouldn't.
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

pub(crate) fn create_material_texture(
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

pub(crate) fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

pub(crate) fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_color"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
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
    use crate::scene::CpuTexture;

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
    fn compressed_wgpu_format_is_decided_by_usage_not_by_the_container() {
        use CompressedFormat as F;
        // The KTX2 container's declared colour space must never change which
        // wgpu format is picked; only the material's `srgb` usage flag does.
        let cases = [
            (F::Bc1RgbaUnorm, false, wgpu::TextureFormat::Bc1RgbaUnorm),
            (F::Bc1RgbaUnorm, true, wgpu::TextureFormat::Bc1RgbaUnormSrgb),
            (F::Bc3RgbaUnorm, false, wgpu::TextureFormat::Bc3RgbaUnorm),
            (F::Bc3RgbaUnorm, true, wgpu::TextureFormat::Bc3RgbaUnormSrgb),
            (F::Bc5RgUnorm, false, wgpu::TextureFormat::Bc5RgUnorm),
            (F::Bc7RgbaUnorm, false, wgpu::TextureFormat::Bc7RgbaUnorm),
            (F::Bc7RgbaUnorm, true, wgpu::TextureFormat::Bc7RgbaUnormSrgb),
        ];
        for (format, srgb, expected) in cases {
            assert_eq!(compressed_wgpu_format(format, srgb), expected);
        }
    }

    #[test]
    fn srgb_round_trip_is_stable_for_every_byte() {
        for b in 0..=255u8 {
            assert_eq!(
                linear_to_srgb(srgb_to_linear(b)),
                b,
                "sRGB byte {b} must round-trip through linear space exactly"
            );
        }
    }

    #[test]
    fn generate_mips_halves_down_to_one_by_one() {
        let base = CpuTexture {
            width: 4,
            height: 4,
            rgba8: vec![128u8; 4 * 4 * 4],
            compressed: None,
        };
        let levels = generate_mips(&base, false);
        let dims: Vec<(u32, u32)> = levels.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(dims, vec![(4, 4), (2, 2), (1, 1)]);
        for (w, h, data) in &levels {
            assert_eq!(data.len(), (*w as usize) * (*h as usize) * 4);
        }
    }

    #[test]
    fn generate_mips_averages_srgb_in_linear_space() {
        // One black and one white texel: a raw byte average gives 127/128,
        // but averaging in linear space (as sRGB data must be) gives a
        // darker result because sRGB is a nonlinear encoding.
        let base = CpuTexture {
            width: 2,
            height: 1,
            rgba8: vec![0, 0, 0, 0, 255, 255, 255, 255],
            compressed: None,
        };
        let levels = generate_mips(&base, true);
        let (w, h, data) = &levels[1];
        assert_eq!((*w, *h), (1, 1));

        let expected_rgb = linear_to_srgb((srgb_to_linear(0) + srgb_to_linear(255)) / 2.0);
        for &channel in &data[0..3] {
            assert_eq!(
                channel, expected_rgb,
                "sRGB channels must be averaged in linear space, not as raw bytes"
            );
            assert_ne!(
                channel, 127,
                "a raw byte average would incorrectly give ~127/128"
            );
        }
        // Alpha is linear even for sRGB textures: raw byte average of two 0s and two 255s.
        assert_eq!(data[3], 127);
    }
}
