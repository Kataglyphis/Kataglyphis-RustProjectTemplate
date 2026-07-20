//! Renders headlessly and writes per-pass GPU timings as JSON.
//!
//! The schema deliberately matches the C++ engine's KATAGLYPHIS_GPU_TIMING_JSON
//! export, so the side-by-side comparison script can read both files with one
//! parser:
//!
//! ```json
//! { "frames_measured": 96, "timestamps_supported": true,
//!   "passes": { "Forward": 0.02, ... } }
//! ```
//!
//! Usage: `cargo run --example dump_gpu_timings -- <out.json> [scene.gltf]`
//! Defaults to the bundled test cube when no scene is given.
//!
//! JSON is written by hand: the fixed schema is four lines, and this crate
//! deliberately carries no JSON dependency.

use kataglyphis_webgpu_renderer::{
    load_gltf, ForwardRenderer, GpuContext, OrbitCamera, TonemapPass,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = args
        .next()
        .unwrap_or_else(|| "gpu-timings-rust.json".to_string());
    let scene_path = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
        });

    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; writing an unsupported dump");
        write_json(&out_path, 0, false, &[]);
        return;
    };

    let (width, height) = (1280u32, 720u32);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    let supported = renderer.enable_gpu_timing(&gpu);
    renderer.upload_scene(&gpu, &load_gltf(scene_path).expect("scene must load"));
    let camera = OrbitCamera::default();
    let mut tonemap = TonemapPass::new(&gpu, wgpu::TextureFormat::Rgba8UnormSrgb);

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("timing_dump_target"),
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

    // Matches what the C++ export measures over a golden-test run, and covers
    // the readback latency plus the 32-frame averaging window (see
    // tests/gpu_timing.rs for why 64 is the minimum that yields averages).
    let frames = 96u32;
    for _ in 0..frames {
        renderer.render_tonemapped(&gpu, &mut tonemap, &view, width, height, &camera);
        // Headless: nothing presents, so the map callbacks need a poll to run.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let timings = renderer.gpu_timings_ms();
    write_json(&out_path, frames, supported, &timings);
    eprintln!("wrote {out_path} ({} passes)", timings.len());
}

fn write_json(path: &str, frames: u32, supported: bool, passes: &[(&'static str, f32)]) {
    let body: Vec<String> = passes
        .iter()
        .map(|(name, ms)| format!("    \"{name}\": {ms}"))
        .collect();
    let json = format!(
        "{{\n  \"frames_measured\": {frames},\n  \"timestamps_supported\": {supported},\n  \"passes\": {{\n{}\n  }}\n}}\n",
        body.join(",\n")
    );
    if let Err(err) = std::fs::write(path, json) {
        eprintln!("could not write {path}: {err}");
        std::process::exit(1);
    }
}
