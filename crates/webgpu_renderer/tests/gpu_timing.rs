//! Per-pass GPU timestamp queries against a real adapter.
//!
//! The pure arithmetic (tick conversion, rolling average) and the
//! feature-absent path are unit-tested inside `render::gpu_timing`; this file
//! covers the thing only a GPU can answer - that every named pass actually
//! stamps its queries during a real frame and reports a sane duration.

use kataglyphis_webgpu_renderer::render::gpu_timing::{GpuTiming, TimedPass};
use kataglyphis_webgpu_renderer::{
    load_gltf, ForwardRenderer, GpuContext, OrbitCamera, TonemapPass,
};

/// The same cube the golden tests use, so every pass has real work to do.
fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

#[test]
fn every_pass_reports_a_finite_non_negative_duration() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    if !gpu.supports_timestamps {
        eprintln!("adapter has no TIMESTAMP_QUERY; skipping");
        return;
    }

    let (width, height) = (256u32, 256u32);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    assert!(
        renderer.enable_gpu_timing(&gpu),
        "the adapter reports TIMESTAMP_QUERY, so enabling must succeed"
    );
    renderer.upload_scene(&gpu, &load_gltf(cube_path()).expect("cube.gltf must load"));
    let camera = OrbitCamera::default();
    let mut tonemap = TonemapPass::new(&gpu, wgpu::TextureFormat::Rgba8UnormSrgb);

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu_timing_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Results arrive some frames after the frame they measure - the readback is
    // mapped asynchronously and never waited on. 64 frames both clears that
    // latency and fills the 32-frame averaging window, so what gets asserted is
    // an averaged number rather than a single sample that happened to land.
    for _ in 0..64 {
        renderer.render_tonemapped(&gpu, &mut tonemap, &view, width, height, &camera);
        // The frame path only polls without waiting, so a headless test that
        // never presents has to give the map callbacks somewhere to run.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let timings = renderer.gpu_timings_ms();
    assert_eq!(
        timings.len(),
        TimedPass::ALL.len(),
        "not every pass reported: {timings:?}"
    );
    for (expected, (name, ms)) in TimedPass::ALL.iter().zip(&timings) {
        assert_eq!(*name, expected.name(), "passes must report in record order");
        assert!(ms.is_finite(), "{name} reported a non-finite duration");
        assert!(*ms >= 0.0, "{name} reported a negative duration {ms}");
        // One pass over a 256x256 cube taking longer than a second means the
        // tick scaling is wrong, not that the GPU is slow.
        assert!(*ms < 1000.0, "{name} reported an implausible {ms} ms");
        eprintln!("{name}: {ms:.4} ms");
    }
}

#[test]
fn timings_stay_empty_until_enabled() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let renderer = ForwardRenderer::new(&gpu, 64, 64);
    // Default-off is the contract: an untimed renderer records exactly the
    // passes it always did, and reports nothing rather than zeroes.
    assert!(!renderer.gpu_timing_available());
    assert!(renderer.gpu_timings_ms().is_empty());
}

#[test]
fn a_disabled_subsystem_needs_no_device() {
    let timing = GpuTiming::unavailable();
    assert!(!timing.is_available());
    assert!(timing.timings_ms().is_empty());
}
