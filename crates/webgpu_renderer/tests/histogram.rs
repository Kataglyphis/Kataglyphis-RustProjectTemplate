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

/// Runs build + reduce over a known image and returns (adapted EV, target EV).
fn reduce_exposure(
    gpu: &GpuContext,
    luminances: &[f32],
    width: u32,
    settings: kataglyphis_webgpu_renderer::render::histogram::ExposureSettings,
    start_ev: f32,
) -> (f32, f32) {
    let view = texture_with_luminances(gpu, luminances, width);
    let mut pass = HistogramPass::new(gpu);
    pass.set_input(gpu, &view);
    pass.reset_exposure(&gpu.queue, start_ev);
    pass.set_exposure_settings(&gpu.queue, settings);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    pass.encode(&mut encoder, width, luminances.len() as u32 / width);
    pass.encode_reduce(&mut encoder);
    pass.encode_exposure_readback(&mut encoder);
    gpu.queue.submit(Some(encoder.finish()));

    pass.read_back_exposure(gpu)
}

#[test]
fn gpu_reduction_matches_the_cpu_exposure_maths() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    use kataglyphis_webgpu_renderer::render::auto_exposure::{
        average_luminance, exposure_ev_for_luminance,
    };
    use kataglyphis_webgpu_renderer::render::histogram::ExposureSettings;

    let width = 16u32;
    for &scene_luminance in &[0.01f32, 0.18, 1.0, 25.0] {
        let luminances = vec![scene_luminance; 256];

        // speed 0 disables smoothing, so the adapted value IS the target and
        // this compares the maths rather than the adaptation curve.
        let settings = ExposureSettings {
            delta_time_seconds: 1.0 / 60.0,
            speed: 0.0,
            auto_enabled: true,
            manual_ev: 0.0,
        };
        let (_adapted, gpu_target) = reduce_exposure(&gpu, &luminances, width, settings, 0.0);

        let mut expected_histogram = vec![0u32; HISTOGRAM_BINS];
        for &l in &luminances {
            expected_histogram[histogram_bin(l)] += 1;
        }
        let cpu_target =
            exposure_ev_for_luminance(average_luminance(&expected_histogram).expect("populated"));

        assert!(
            (gpu_target - cpu_target).abs() < 0.05,
            "scene luminance {scene_luminance}: GPU target EV {gpu_target}, CPU {cpu_target}"
        );
    }
}

#[test]
fn a_dark_scene_exposes_up_and_a_bright_scene_exposes_down() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    use kataglyphis_webgpu_renderer::render::histogram::ExposureSettings;

    let settings = ExposureSettings {
        speed: 0.0,
        ..ExposureSettings::default()
    };

    let (_, dark_target) = reduce_exposure(&gpu, &vec![0.005f32; 256], 16, settings, 0.0);
    let (_, bright_target) = reduce_exposure(&gpu, &vec![20.0f32; 256], 16, settings, 0.0);

    assert!(dark_target > 0.0, "a dark scene must expose up, got {dark_target}");
    assert!(bright_target < 0.0, "a bright scene must expose down, got {bright_target}");
}

#[test]
fn adaptation_moves_toward_the_target_without_jumping_to_it() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    use kataglyphis_webgpu_renderer::render::histogram::ExposureSettings;

    // One frame at 60 Hz with a moderate rate: exposure should move a
    // fraction of the way, not snap. Snapping is what makes auto-exposure
    // look like a flicker rather than an eye adjusting.
    let settings = ExposureSettings {
        delta_time_seconds: 1.0 / 60.0,
        speed: 3.0,
        auto_enabled: true,
        manual_ev: 0.0,
    };
    let (adapted, target) = reduce_exposure(&gpu, &vec![0.005f32; 256], 16, settings, 0.0);

    assert!(target > 1.0, "test needs a target well away from the start, got {target}");
    assert!(adapted > 0.0, "exposure moved the wrong way: {adapted}");
    assert!(
        adapted < target * 0.5,
        "one 16ms frame should cover a fraction of the distance, got {adapted} of {target}"
    );
}

#[test]
fn an_all_black_frame_holds_the_previous_exposure() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    use kataglyphis_webgpu_renderer::render::histogram::ExposureSettings;

    // A scene that has not loaded yet. Deriving an exposure from an empty
    // histogram would divide by zero and blow the frame out; holding is the
    // only safe answer.
    let start_ev = 1.75f32;
    let (adapted, _target) = reduce_exposure(
        &gpu,
        &vec![0.0f32; 256],
        16,
        ExposureSettings::default(),
        start_ev,
    );

    assert!(
        (adapted - start_ev).abs() < 1e-4,
        "an all-black frame changed exposure from {start_ev} to {adapted}"
    );
}

#[test]
fn manual_mode_writes_the_slider_value_through() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    use kataglyphis_webgpu_renderer::render::histogram::ExposureSettings;

    // Manual mode still writes the buffer so the tonemap has one source of
    // truth; switching modes must not leave a stale auto value behind.
    let settings = ExposureSettings {
        auto_enabled: false,
        manual_ev: -2.5,
        ..ExposureSettings::default()
    };
    let (adapted, target) = reduce_exposure(&gpu, &vec![0.005f32; 256], 16, settings, 4.0);

    assert!((adapted + 2.5).abs() < 1e-4, "manual EV did not reach the buffer: {adapted}");
    assert!((target + 2.5).abs() < 1e-4, "manual EV must overwrite the target too: {target}");
}
