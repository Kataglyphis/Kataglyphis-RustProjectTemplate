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

    let texture_ref = material
        .base_color_texture
        .as_ref()
        .expect("textured cube must expose its base color texture");
    let texture = &texture_ref.texture;
    assert_eq!((texture.width, texture.height), (2, 2));
    assert_eq!(texture.rgba8.len(), 16);
    // 2x2 checker: green at (0,0), magenta at (1,0).
    assert_eq!(&texture.rgba8[0..4], &[40, 220, 60, 255]);
    assert_eq!(&texture.rgba8[4..8], &[220, 40, 200, 255]);
    // The asset requests NEAREST filtering (A1: sampler modes honored).
    assert!(texture_ref.sampler.mag_nearest && texture_ref.sampler.min_nearest);
    assert!(texture_ref.srgb);
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

    // Corner: procedural sky (blue-dominant gradient), untouched by the cube.
    let corner = pixel(2, 2);
    assert!(
        corner[2] > corner[0] && corner[2] > 60,
        "corner pixel should be sky (blue-dominant), got {corner:?}"
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
        if neutral && r > 180 {
            lit_plane += 1;
        } else if r < 110 && b > r + 15 && b < 180 {
            // Sky-lit shadow: with analytic IBL the shadowed plane only
            // receives blue hemisphere irradiance.
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
fn alpha_modes_blend_and_mask() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_alpha.gltf");
    let scene = load_gltf(&path).expect("cube_alpha.gltf must load");
    assert_eq!(scene.primitives.len(), 4);

    use kataglyphis_webgpu_renderer::scene::AlphaMode;
    let modes: Vec<AlphaMode> = scene
        .primitives
        .iter()
        .map(|p| p.material.alpha_mode)
        .collect();
    assert!(modes.contains(&AlphaMode::Blend));
    assert!(modes.iter().any(|m| matches!(m, AlphaMode::Mask(c) if (c - 0.5).abs() < 1e-6)));

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);

    // Look down so the translucent green quad overlaps the red cube.
    let camera = OrbitCamera {
        radius: 5.0,
        pitch_deg: 60.0,
        ..OrbitCamera::default()
    };
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");

    let mut blended_over_cube = 0usize;
    let mut yellowish = 0usize;
    for p in pixels.chunks_exact(4) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        // Green blend quad over the bright white plane: green-tinted but
        // clearly translucent (red/blue still present from the white below).
        if g > 140 && g > r + 25 && g > b + 25 && r > 70 && b > 60 {
            blended_over_cube += 1;
        }
        // The MASK quad (yellow, alpha 0.3 < cutoff 0.5) must be fully
        // discarded: no yellow-dominant pixels anywhere.
        if r > 140 && g > 140 && b < 90 {
            yellowish += 1;
        }
    }
    assert!(
        blended_over_cube > 200,
        "expected the green BLEND quad composited over the red cube, got {blended_over_cube} pixels"
    );
    assert_eq!(
        yellowish, 0,
        "MASK quad below cutoff must be discarded entirely, got {yellowish} yellow pixels"
    );
}

#[test]
fn punctual_lights_pool_on_plane() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/point_light.gltf");
    let scene = load_gltf(&path).expect("point_light.gltf must load");
    assert_eq!(scene.lights.len(), 2);
    use kataglyphis_webgpu_renderer::scene::CpuLightKind;
    assert!(scene
        .lights
        .iter()
        .any(|l| matches!(l.kind, CpuLightKind::Point)));
    assert!(scene
        .lights
        .iter()
        .any(|l| matches!(l.kind, CpuLightKind::Spot { .. })));

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    // Dim the sun so the punctual pools dominate.
    renderer.light_color_intensity.w = 0.4;

    let camera = OrbitCamera {
        radius: 7.0,
        pitch_deg: 65.0,
        ..OrbitCamera::default()
    };
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");

    let mut red_pool = 0usize;
    let mut green_pool = 0usize;
    for p in pixels.chunks_exact(4) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        if r > 120 && r > g + 40 && r > b + 40 {
            red_pool += 1;
        } else if g > 120 && g > r + 40 && g > b + 40 {
            green_pool += 1;
        }
    }
    assert!(
        red_pool > 200,
        "expected a red point-light pool on the plane, got {red_pool} pixels"
    );
    assert!(
        green_pool > 100,
        "expected a green spot-light pool on the plane, got {green_pool} pixels"
    );
}

#[test]
fn bloom_adds_energy_around_bright_sources() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/point_light.gltf");
    let scene = load_gltf(&path).expect("point_light.gltf must load");
    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    renderer.light_color_intensity.w = 0.4;

    let camera = OrbitCamera {
        radius: 7.0,
        pitch_deg: 65.0,
        ..OrbitCamera::default()
    };

    let total = |pixels: &[u8]| -> u64 { pixels.iter().map(|&b| b as u64).sum() };

    renderer.bloom_strength = 0.0;
    let without = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render without bloom");
    renderer.bloom_strength = 1.5;
    let with = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render with bloom");

    let (sum_without, sum_with) = (total(&without), total(&with));
    assert!(
        sum_with > sum_without + 50_000,
        "bloom should add visible energy: {sum_without} -> {sum_with}"
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
