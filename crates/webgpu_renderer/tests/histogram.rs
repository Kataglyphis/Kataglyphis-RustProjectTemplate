//! The GPU luminance histogram, checked against the CPU binning it must match.
//!
//! render::auto_exposure::histogram_bin and histogram.wgsl's histogram_bin are
//! the same function written twice, in two languages, and auto-exposure is
//! only correct if they agree. The CPU one is unit-tested; this pins the
//! shader to it.
//!
//! Runs over a texture with known contents rather than a rendered frame, so a
//! disagreement points at the binning rather than at whatever the renderer
//! happened to draw.

use kataglyphis_webgpu_renderer::context::GpuContext;
use kataglyphis_webgpu_renderer::render::auto_exposure::{histogram_bin, HISTOGRAM_BINS};
use kataglyphis_webgpu_renderer::render::histogram::HistogramPass;

/// Builds an Rgba32Float texture whose pixels have the given luminances.
///
/// Grey pixels (r = g = b) so luminance equals the channel value under any
/// sane weighting - the test is about binning, not about the luma constants.
fn texture_with_luminances(gpu: &GpuContext, luminances: &[f32], width: u32) -> wgpu::TextureView {
    let height = luminances.len() as u32 / width;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("histogram_test_source"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut pixels: Vec<f32> = Vec::with_capacity(luminances.len() * 4);
    for &l in luminances {
        pixels.extend_from_slice(&[l, l, l, 1.0]);
    }

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn build_histogram(gpu: &GpuContext, luminances: &[f32], width: u32) -> Vec<u32> {
    let view = texture_with_luminances(gpu, luminances, width);
    let mut pass = HistogramPass::new(gpu);
    pass.set_input(gpu, &view);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    pass.encode(&mut encoder, width, luminances.len() as u32 / width);
    pass.encode_readback(&mut encoder);
    gpu.queue.submit(Some(encoder.finish()));

    pass.read_back(gpu)
}

#[test]
fn gpu_binning_matches_the_cpu_binning() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // A spread covering black, sub-range, in-range across many decades, and
    // above-range - i.e. every branch of the binning function.
    let width = 16u32;
    let luminances: Vec<f32> = (0..256)
        .map(|i| match i % 8 {
            0 => 0.0,
            1 => 1e-9,
            2 => 1e-5,
            3 => 0.01,
            4 => 0.18,
            5 => 1.0,
            6 => 50.0,
            _ => 1e9,
        })
        .collect();

    let gpu_histogram = build_histogram(&gpu, &luminances, width);
    assert_eq!(gpu_histogram.len(), HISTOGRAM_BINS);

    let mut expected = vec![0u32; HISTOGRAM_BINS];
    for &l in &luminances {
        expected[histogram_bin(l)] += 1;
    }

    assert_eq!(
        gpu_histogram, expected,
        "shader binning disagrees with render::auto_exposure::histogram_bin"
    );
}

#[test]
fn every_pixel_is_counted_exactly_once() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // 40x24 is deliberately NOT a multiple of the 16x16 workgroup. Rounding
    // the dispatch down would silently drop the right and bottom edges;
    // rounding up without the bounds check would double-count them. Either
    // way the total stops matching the pixel count.
    let width = 40u32;
    let height = 24u32;
    let luminances = vec![0.5f32; (width * height) as usize];

    let histogram = build_histogram(&gpu, &luminances, width);
    let total: u32 = histogram.iter().sum();

    assert_eq!(
        total,
        width * height,
        "histogram counted {total} samples for a {width}x{height} image"
    );
}

#[test]
fn the_histogram_is_cleared_between_builds() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // Reusing one pass across two frames must not accumulate. Without the
    // clear dispatch the second build doubles every count, which reads as a
    // scene twice as populated and skews nothing visibly - the exposure just
    // drifts.
    let width = 16u32;
    let luminances = vec![0.25f32; 256];
    let view = texture_with_luminances(&gpu, &luminances, width);
    let mut pass = HistogramPass::new(&gpu);
    pass.set_input(&gpu, &view);

    let mut totals = Vec::new();
    for _ in 0..3 {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        pass.encode(&mut encoder, width, 256 / width);
        pass.encode_readback(&mut encoder);
        gpu.queue.submit(Some(encoder.finish()));
        totals.push(pass.read_back(&gpu).iter().sum::<u32>());
    }

    assert_eq!(totals[0], 256);
    assert_eq!(totals, vec![256, 256, 256], "counts accumulated across builds: {totals:?}");
}
