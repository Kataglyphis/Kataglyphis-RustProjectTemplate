//! Headless golden tests: load the bundled cube glTF, render a frame to an
//! offscreen texture, and assert structural pixel properties (robust across
//! GPUs/drivers, unlike exact image comparison).

use kataglyphis_webgpu_renderer::{load_gltf, ForwardRenderer, GpuContext, OrbitCamera};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

fn textured_cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_textured.gltf")
}

#[test]
fn gltf_loader_reads_cube() {
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    assert_eq!(scene.primitives.len(), 1);
    assert_eq!(scene.vertex_count(), 24);
    assert_eq!(scene.triangle_count(), 12);

    let material = &scene.primitives[0].material;
    assert!((material.base_color[0] - 0.8).abs() < 1e-6);
    assert!((material.base_color[3] - 1.0).abs() < 1e-6);
    assert!(material.base_color_texture.is_none());
}

#[test]
fn gltf_loader_reads_base_color_texture() {
    let scene = load_gltf(textured_cube_path()).expect("cube_textured.gltf must load");
    let material = &scene.primitives[0].material;

    let texture = material
        .base_color_texture
        .as_ref()
        .expect("textured cube must expose its base color texture");
    assert_eq!((texture.width, texture.height), (2, 2));
    assert_eq!(texture.rgba8.len(), 16);
    // 2x2 checker: green at (0,0), magenta at (1,0).
    assert_eq!(&texture.rgba8[0..4], &[40, 220, 60, 255]);
    assert_eq!(&texture.rgba8[4..8], &[220, 40, 200, 255]);
}

#[test]
fn renders_cube_headless() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
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
fn renders_textured_cube_headless() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(textured_cube_path()).expect("cube_textured.gltf must load");
    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);

    let camera = OrbitCamera::default();
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");

    // The checker must produce BOTH green-dominant and magenta-dominant
    // pixels — proving the base color texture is actually sampled.
    let mut green = 0usize;
    let mut magenta = 0usize;
    for p in pixels.chunks_exact(4) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        if g > 100 && g > r + 30 && g > b + 30 {
            green += 1;
        } else if r > 100 && b > 80 && r > g + 30 {
            magenta += 1;
        }
    }
    assert!(
        green > 200 && magenta > 200,
        "expected both checker colors, got {green} green / {magenta} magenta pixels"
    );
}

#[test]
fn shadow_darkens_plane_under_cube() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_on_plane.gltf");
    let scene = load_gltf(path).expect("cube_on_plane.gltf must load");
    assert_eq!(scene.primitives.len(), 2);

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    // Low light from -x/-z so the floating cube casts a long shadow onto the
    // +x/+z plane area the camera looks at (default light is too steep — the
    // shadow hides directly beneath the cube).
    renderer.light_dir_ambient = glam::Vec4::new(-1.0, 0.7, -0.3, 0.15);

    // Look down from above so the plane fills most of the frame.
    let camera = OrbitCamera {
        radius: 6.0,
        pitch_deg: 55.0,
        ..OrbitCamera::default()
    };
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");

    // Plane pixels are near-neutral (white albedo). The shadowed patch under
    // the cube only receives ambient light and is therefore much darker than
    // sunlit plane areas — both populations must exist.
    let mut lit_plane = 0usize;
    let mut shadowed_plane = 0usize;
    for p in pixels.chunks_exact(4) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        let neutral = (r - g).abs() < 25 && (g - b).abs() < 25 && (r - b).abs() < 25;
        if !neutral {
            continue; // red cube or blue-ish clear color
        }
        if r > 180 {
            lit_plane += 1;
        } else if (80..150).contains(&r) {
            shadowed_plane += 1;
        }
    }
    assert!(
        lit_plane > 1000,
        "expected a large sunlit plane area, got {lit_plane} pixels"
    );
    assert!(
        shadowed_plane > 150,
        "expected a shadowed patch under the cube, got {shadowed_plane} pixels"
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
