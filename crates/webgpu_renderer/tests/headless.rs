//! Headless golden tests: load the bundled cube glTF, render a frame to an
//! offscreen texture, and assert structural pixel properties (robust across
//! GPUs/drivers, unlike exact image comparison).

use kataglyphis_webgpu_renderer::{load_gltf, ForwardRenderer, GpuContext, OrbitCamera};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

#[test]
fn gltf_loader_reads_cube() {
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    assert_eq!(scene.primitives.len(), 1);
    assert_eq!(scene.vertex_count(), 24);
    assert_eq!(scene.triangle_count(), 12);

    let material = scene.primitives[0].material;
    assert!((material.base_color[0] - 0.8).abs() < 1e-6);
    assert!((material.base_color[3] - 1.0).abs() < 1e-6);
}

#[test]
fn renders_cube_headless() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (256, 256);
    let mut renderer =
        ForwardRenderer::new(&gpu, wgpu::TextureFormat::Rgba8UnormSrgb, width, height);
    renderer.upload_scene(&gpu, &scene);

    let camera = OrbitCamera::default();
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");
    assert_eq!(pixels.len(), (width * height * 4) as usize);

    let pixel = |x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };

    // NOTE: the target is Rgba8UnormSrgb, so all read-back bytes are
    // sRGB-encoded (linear 0.05 clear -> byte ~63, not ~13).

    // Center: lit red-ish cube — red clearly dominant over green/blue.
    let center = pixel(width / 2, height / 2);
    assert!(
        center[0] > 110 && center[0] > center[1] + 40 && center[0] > center[2] + 40,
        "center pixel should be the red cube, got {center:?}"
    );

    // Corner: dark clear color (linear 0.05/0.05/0.08), untouched by the cube.
    let corner = pixel(2, 2);
    assert!(
        corner[0] < 80 && corner[1] < 80 && corner[2] < 100,
        "corner pixel should be the clear color, got {corner:?}"
    );

    // The cube must cover a plausible portion of the frame: count lit pixels.
    let lit = pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 110 && p[0] > p[1] + 40)
        .count();
    let total = (width * height) as usize;
    assert!(
        lit > total / 20 && lit < total / 2,
        "cube coverage out of range: {lit}/{total} lit pixels"
    );
}

#[test]
fn resize_handles_zero_dimensions() {
    let Ok(mut gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    // Headless context has no surface: resize must be a no-op, not a crash —
    // same contract the windowed path relies on when minimized.
    gpu.resize(0, 0);
    gpu.resize(800, 600);
    gpu.reconfigure();
}
