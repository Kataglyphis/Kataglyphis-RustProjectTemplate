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
    // KHR_texture_transform: offset (0.25, 0), scale (2, 2), no rotation.
    let t = material.base_uv_transform;
    assert!((t[0][0] - 2.0).abs() < 1e-5 && (t[1][1] - 2.0).abs() < 1e-5);
    assert!((t[0][2] - 0.25).abs() < 1e-5 && t[1][2].abs() < 1e-5);
    assert!(t[0][1].abs() < 1e-5 && t[1][0].abs() < 1e-5);
}

#[test]
fn gltf_loader_applies_emissive_strength() {
    // The asset declares emissiveFactor [0.5, 0.4, 0.3] and a
    // KHR_materials_emissive_strength of 3.0, so the loaded factor must be the
    // product (HDR emitters exceed the [0,1] glTF factor range).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets/cube_emissive_strength.gltf");
    let scene = load_gltf(path).expect("cube_emissive_strength.gltf must load");
    let e = scene.primitives[0].material.emissive_factor;
    assert!((e[0] - 1.5).abs() < 1e-5, "r: {}", e[0]);
    assert!((e[1] - 1.2).abs() < 1e-5, "g: {}", e[1]);
    assert!((e[2] - 0.9).abs() < 1e-5, "b: {}", e[2]);
}

#[test]
fn gltf_loader_reads_morph_target_and_default_weight() {
    // cube_morph.gltf carries one POSITION morph target (every vertex +Y) and a
    // mesh-level default weight of 1.0. The loader must parse the deltas AND
    // apply the mesh default weight (not leave it at zero).
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_morph.gltf");
    let scene = load_gltf(path).expect("cube_morph.gltf must load");
    let prim = &scene.primitives[0];
    assert_eq!(prim.morph_targets.len(), 1, "one morph target expected");
    assert_eq!(
        prim.morph_targets[0].position_deltas.len(),
        24,
        "delta count must match the cube's vertices"
    );
    // Every delta is (0, 1, 0).
    for d in &prim.morph_targets[0].position_deltas {
        assert!(d.x.abs() < 1e-6 && (d.y - 1.0).abs() < 1e-6 && d.z.abs() < 1e-6);
    }
    // The mesh default weight must be honored.
    assert_eq!(
        prim.morph_weights,
        vec![1.0],
        "mesh default weight [1.0] must be applied at load"
    );
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

/// Golden coverage for the morph-target GPU apply path: a weight channel that
/// ramps 0 -> 1 must visibly lift the cube on screen. This is the render-path
/// counterpart to the CPU `blend_morph_targets`/`sample_morph_weights` unit
/// tests — it proves `set_animation_time` -> `apply_morph_targets` re-blends
/// and re-uploads the vertex buffer so the rendered silhouette actually moves.
#[test]
fn morph_weight_lifts_the_silhouette() {
    use glam::{Quat, Vec3};
    use kataglyphis_webgpu_renderer::scene::{
        ChannelValues, CpuAnimation, CpuAnimationChannel, CpuNode, Interpolation, MorphTarget,
    };

    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // Bundled cube + one morph target that lifts every vertex +Y, driven by a
    // linear weight channel on the cube's node (0 at t=0, 1 at t=1).
    let mut scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut prim = scene.primitives[0].clone();
    let vcount = prim.vertices.len();
    prim.node_index = Some(0);
    prim.morph_targets = vec![MorphTarget {
        position_deltas: vec![Vec3::new(0.0, 0.6, 0.0); vcount],
        normal_deltas: vec![Vec3::ZERO; vcount],
    }];
    prim.morph_weights = vec![0.0];
    scene.primitives = vec![prim];
    scene.nodes = vec![CpuNode {
        parent: None,
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    }];
    scene.animations = vec![CpuAnimation {
        // Duration deliberately longer than the last keyframe: sampling at
        // t=1.0 must land at full weight, not wrap (t % duration) back to 0.
        name: "morph".into(),
        duration: 2.0,
        channels: vec![CpuAnimationChannel {
            node: 0,
            times: vec![0.0, 1.0],
            values: ChannelValues::MorphWeights(vec![0.0, 1.0]),
            interpolation: Interpolation::Linear,
        }],
    }];

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    let camera = OrbitCamera::default();

    // Red-dominant pixels are the (red-ish) cube; measure how many there are
    // and their vertical centroid. Row index grows downward in the readback,
    // so lifting the cube in world space lowers the mean row.
    let cube_stats = |pixels: &[u8]| -> (usize, f64) {
        let mut count = 0usize;
        let mut sum_y = 0f64;
        for (i, p) in pixels.chunks_exact(4).enumerate() {
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if r > 110 && r > g + 40 && r > b + 40 {
                count += 1;
                sum_y += (i as u32 / width) as f64;
            }
        }
        (count, if count > 0 { sum_y / count as f64 } else { 0.0 })
    };

    renderer.set_animation_time(0.0);
    let base = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render at weight 0");
    let (lit0, cy0) = cube_stats(&base);

    renderer.set_animation_time(1.0);
    let morphed = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render at weight 1");
    let (lit1, cy1) = cube_stats(&morphed);

    eprintln!("morph golden: lit {lit0}->{lit1}, centroid_y {cy0:.1}->{cy1:.1}");

    // The cube stays clearly visible in both poses...
    let total = (width * height) as usize;
    assert!(
        lit0 > total / 40 && lit1 > total / 40,
        "cube must be visible at both weights, got {lit0}/{lit1} lit"
    );
    // ...the frame actually changes (the re-blend + re-upload happened)...
    assert!(
        base != morphed,
        "weight 1 must produce a different frame than weight 0"
    );
    // ...and the +Y morph lifts the silhouette (centroid rises, i.e. smaller row).
    assert!(
        cy1 + 6.0 < cy0,
        "the +Y morph should raise the cube: centroid_y {cy0:.1} -> {cy1:.1}"
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
    assert!(modes
        .iter()
        .any(|m| matches!(m, AlphaMode::Mask(c) if (c - 0.5).abs() < 1e-6)));

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
        // The MASK quad (saturated yellow 0.9/0.9/0.1, alpha 0.3 < cutoff
        // 0.5) must be fully discarded. Require STRONG yellow so darkened
        // olive blend-mix tones never trip the detector.
        if r > 140 && g > 140 && b * 3 < r {
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
fn ssao_darkens_geometry() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_alpha.gltf");
    let scene = load_gltf(&path).expect("cube_alpha.gltf must load");
    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    renderer.bloom_strength = 0.0;

    let camera = OrbitCamera {
        radius: 5.0,
        pitch_deg: 60.0,
        ..OrbitCamera::default()
    };
    let total = |pixels: &[u8]| -> u64 { pixels.iter().map(|&b| b as u64).sum() };

    renderer.ssao_strength = 0.0;
    let without = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render without ssao");
    renderer.ssao_strength = 1.0;
    let with = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render with ssao");

    let (sum_without, sum_with) = (total(&without), total(&with));
    assert!(
        sum_with + 50_000 < sum_without,
        "SSAO should remove energy near geometry: {sum_without} -> {sum_with}"
    );
}

#[test]
fn animation_moves_the_cube() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_animated.gltf");
    let scene = load_gltf(&path).expect("cube_animated.gltf must load");
    assert_eq!(scene.animations.len(), 1);
    assert!((scene.animations[0].duration - 2.0).abs() < 1e-5);

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    let camera = OrbitCamera {
        radius: 6.0,
        pitch_deg: 20.0,
        yaw_deg: 90.0, // look along -z so x maps to screen x
        ..OrbitCamera::default()
    };

    let red_centroid_x = |pixels: &[u8]| -> f32 {
        let (mut sum_x, mut count) = (0.0f32, 0u32);
        for (i, p) in pixels.chunks_exact(4).enumerate() {
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if r > 110 && r > g + 40 && r > b + 40 {
                sum_x += (i % width as usize) as f32;
                count += 1;
            }
        }
        assert!(count > 200, "cube not found (only {count} red pixels)");
        sum_x / count as f32
    };

    renderer.set_animation_time(0.05);
    let start = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render at t=0");
    renderer.set_animation_time(1.95);
    let end = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render at t=1.95");

    let (x0, x1) = (red_centroid_x(&start), red_centroid_x(&end));
    assert!(
        (x1 - x0).abs() > 30.0,
        "animated cube should move across the frame: centroid {x0:.1} -> {x1:.1}"
    );
}

#[test]
fn skinning_bends_the_bar() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/skinned_bar.gltf");
    let scene = load_gltf(&path).expect("skinned_bar.gltf must load");
    assert_eq!(scene.skins.len(), 1);
    assert_eq!(scene.skins[0].joints.len(), 2);
    assert_eq!(scene.skins[0].inverse_bind_matrices.len(), 2);
    // Vertices must carry non-zero skin weights.
    assert!(scene.primitives[0]
        .vertices
        .iter()
        .any(|v| v.weights.iter().sum::<f32>() > 0.5));

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    let camera = OrbitCamera {
        radius: 5.0,
        pitch_deg: 5.0,
        yaw_deg: 90.0,
        target: glam::Vec3::new(0.0, 1.0, 0.0),
        ..OrbitCamera::default()
    };

    // Mean x of the bar's red pixels in the TOP half of the frame: bending
    // joint 1 swings the upper half sideways.
    let upper_centroid_x = |pixels: &[u8]| -> f32 {
        let (mut sum, mut count) = (0.0f32, 0u32);
        for (i, p) in pixels.chunks_exact(4).enumerate() {
            let (x, y) = (i % width as usize, i / width as usize);
            if y > (height as usize) / 2 {
                continue;
            }
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if r > 90 && r > g + 30 && r > b + 30 {
                sum += x as f32;
                count += 1;
            }
        }
        assert!(count > 50, "bar not visible in upper half ({count} px)");
        sum / count as f32
    };

    renderer.set_animation_time(0.0);
    let straight = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render straight");
    renderer.set_animation_time(1.0);
    let bent = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render bent");

    let (x0, x1) = (upper_centroid_x(&straight), upper_centroid_x(&bent));
    assert!(
        (x1 - x0).abs() > 8.0,
        "skinned bar should bend: upper centroid {x0:.1} -> {x1:.1}"
    );
}

#[test]
fn loads_binary_glb() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.glb");
    let bytes = std::fs::read(&path).expect("cube.glb must exist");
    // GLB magic + version.
    assert_eq!(&bytes[0..4], b"glTF");

    let scene = kataglyphis_webgpu_renderer::asset::gltf_loader::load_gltf_slice(&bytes)
        .expect("cube.glb must load from memory");
    assert_eq!(scene.primitives.len(), 1);
    assert_eq!(scene.vertex_count(), 24);
    assert_eq!(scene.triangle_count(), 12);
}

#[test]
fn reads_gltf_cameras() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_animated.gltf");
    let scene = load_gltf(&path).expect("cube_animated.gltf must load");
    assert_eq!(scene.cameras.len(), 1);
    let camera = &scene.cameras[0];
    assert_eq!(camera.name.as_deref(), Some("demo_cam"));
    // The asset authors yfov as 0.7854 rad (45 deg).
    assert!((camera.yfov_rad - std::f32::consts::FRAC_PI_4).abs() < 1e-3);
    assert!((camera.znear - 0.1).abs() < 1e-6);
    assert_eq!(camera.zfar, Some(100.0));
    // Its node must exist and sit where the asset places it.
    let world = kataglyphis_webgpu_renderer::CpuScene::compute_world_transforms(&scene.nodes);
    let position = world[camera.node].transform_point3(glam::Vec3::ZERO);
    assert!((position - glam::Vec3::new(0.0, 1.0, 5.0)).length() < 1e-4);
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

/// The web sRGB fix, asserted rather than assumed.
///
/// Native swapchains expose an sRGB format and the hardware gamma-encodes the
/// tonemap output. WebGPU canvases do not: the browser hands back something
/// like `Bgra8Unorm`, and writing linear values there displays them
/// uncorrected - the "slightly dark web demo" that
/// docs/webgpu-srgb-audit.md carried as the single known deviation.
///
/// With the shader-side encode in place, both targets must end up holding
/// approximately the SAME sRGB-encoded bytes. Without it the non-sRGB buffer
/// is dramatically darker: linear 0.05 stores as byte ~13 instead of ~63.
#[test]
fn non_srgb_target_is_gamma_encoded_like_an_srgb_one() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (128, 128);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    let camera = OrbitCamera::default();

    let srgb = renderer
        .render_to_pixels_with_format(
            &gpu,
            width,
            height,
            &camera,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .expect("sRGB render must succeed");
    let unorm = renderer
        .render_to_pixels_with_format(
            &gpu,
            width,
            height,
            &camera,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("non-sRGB render must succeed");

    assert_eq!(srgb.len(), unorm.len());

    let mean = |px: &[u8]| px.iter().map(|&b| b as f64).sum::<f64>() / px.len() as f64;
    let mean_srgb = mean(&srgb);
    let mean_unorm = mean(&unorm);

    // Hardware encode and the shader's transfer function are the same curve,
    // so the two differ only by rounding. A tolerance of 2 levels is far
    // tighter than the gap the bug produced (tens of levels) while leaving
    // room for per-driver rounding of the hardware path.
    assert!(
        (mean_srgb - mean_unorm).abs() < 2.0,
        "non-sRGB target should be gamma-encoded to match the sRGB one; \
         mean was {mean_srgb:.2} (sRGB) vs {mean_unorm:.2} (non-sRGB). \
         A much darker non-sRGB mean means the shader-side encode is not running."
    );

    // Guard against the assertion above being satisfied by a black frame.
    assert!(
        mean_srgb > 20.0,
        "reference render looks blank (mean {mean_srgb:.2}); the comparison above would prove nothing"
    );
}

/// Auto-exposure, end to end through the real frame path.
///
/// The unit and compute tests cover the maths and the passes in isolation.
/// This covers the wiring, which is where it would silently do nothing: the
/// tonemap reading a buffer nobody writes, the passes never encoded, or the
/// exposure never reaching the pixels.
#[test]
fn auto_exposure_brightens_a_dark_scene_over_successive_frames() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (128, 128);

    let mean_of = |pixels: &[u8]| -> f64 {
        pixels.iter().map(|&b| b as f64).sum::<f64>() / pixels.len() as f64
    };

    // A deliberately underlit scene: dim sun, almost no ambient. Manual
    // exposure leaves it dark; auto-exposure should pull it up.
    let render = |auto: bool, frames: usize| -> f64 {
        let mut renderer = ForwardRenderer::new(&gpu, width, height);
        renderer.upload_scene(&gpu, &scene);
        renderer.light_dir_ambient = glam::Vec4::new(-0.4, 1.0, 0.3, 0.01);
        renderer.light_color_intensity = glam::Vec4::new(1.0, 1.0, 1.0, 0.05);
        renderer.auto_exposure = auto;
        renderer.exposure_ev = 0.0;
        // Large steps so adaptation converges within a few frames rather than
        // needing hundreds - this tests the wiring, not the rate constant.
        renderer.frame_delta_seconds = 0.5;

        let camera = OrbitCamera::default();
        let mut pixels = Vec::new();
        for _ in 0..frames {
            pixels = renderer
                .render_to_pixels(&gpu, width, height, &camera)
                .expect("headless render must succeed");
        }
        mean_of(&pixels)
    };

    let manual = render(false, 6);
    let automatic = render(true, 6);

    assert!(
        manual > 1.0,
        "the reference render is essentially black ({manual}); the comparison below would prove nothing"
    );
    // Measured 182.7 with auto vs 163.1 manual, a 12% lift. The bound is 8%:
    // comfortably under what the feature actually does, comfortably over the
    // 0% a disconnected one would. The lift is this modest because the
    // procedural sky fills much of the frame and is already well exposed -
    // auto-exposure is correcting the lit geometry, not the whole image.
    assert!(
        automatic > manual * 1.08,
        "auto-exposure did not brighten an underlit scene: mean {automatic} with auto vs {manual} manual"
    );
}

/// Manual mode must keep behaving exactly as before the auto path existed.
/// The exposure now travels through the same GPU buffer, so a regression here
/// would mean the slider stopped reaching the pixels.
#[test]
fn manual_exposure_still_controls_brightness() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (128, 128);

    let render_at = |ev: f32| -> f64 {
        let mut renderer = ForwardRenderer::new(&gpu, width, height);
        renderer.upload_scene(&gpu, &scene);
        renderer.auto_exposure = false;
        renderer.exposure_ev = ev;
        let camera = OrbitCamera::default();
        let pixels = renderer
            .render_to_pixels(&gpu, width, height, &camera)
            .expect("headless render must succeed");
        pixels.iter().map(|&b| b as f64).sum::<f64>() / pixels.len() as f64
    };

    let dark = render_at(-3.0);
    let bright = render_at(2.0);

    assert!(
        bright > dark * 1.2,
        "manual exposure stopped affecting the image: EV -3 gave {dark}, EV +2 gave {bright}"
    );
}

/// GPU instancing, through the real frame path.
///
/// The failure mode this guards is not a crash: an instance transform that
/// never reaches the shader draws every copy on top of the original, which
/// looks exactly like a scene with one object. Counting covered pixels is
/// what distinguishes "three instances" from "three draws of the same place".
#[test]
fn instances_appear_at_their_own_transforms() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (192, 192);
    let camera = OrbitCamera {
        radius: 14.0,
        ..OrbitCamera::default()
    };

    // Counts pixels that are not sky. The cube is lit and red-dominant; the
    // procedural sky is blue-dominant, so "red exceeds blue" separates them
    // without depending on exact shading.
    let covered = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|p| p[0] as i32 > p[2] as i32 + 20)
            .count()
    };

    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    assert_eq!(
        renderer.instance_count(0),
        1,
        "primitives start with one identity instance"
    );

    let single = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render");
    let single_covered = covered(&single);
    assert!(
        single_covered > 100,
        "the reference cube barely rendered ({single_covered} px)"
    );

    // Three copies, spread far enough apart not to overlap on screen.
    renderer.set_instances(
        &gpu,
        0,
        &[
            glam::Mat4::from_translation(glam::Vec3::new(-4.0, 0.0, 0.0)),
            glam::Mat4::IDENTITY,
            glam::Mat4::from_translation(glam::Vec3::new(4.0, 0.0, 0.0)),
        ],
    );
    assert_eq!(renderer.instance_count(0), 3);

    let instanced = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render");
    let instanced_covered = covered(&instanced);

    assert!(
        instanced_covered > single_covered * 2,
        "three separated instances should cover far more than one cube: {instanced_covered} vs {single_covered}"
    );
}

#[test]
fn clearing_instances_restores_a_single_copy_rather_than_none() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let (width, height) = (128, 128);
    let camera = OrbitCamera::default();

    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);

    renderer.set_instances(&gpu, 0, &[glam::Mat4::IDENTITY, glam::Mat4::IDENTITY]);
    assert_eq!(renderer.instance_count(0), 2);

    // Zero instances would make the primitive vanish, which is
    // indistinguishable from a culling or upload bug when looking at a frame.
    renderer.set_instances(&gpu, 0, &[]);
    assert_eq!(
        renderer.instance_count(0),
        1,
        "an empty slice must restore one identity instance"
    );

    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("render");
    let lit = pixels
        .chunks_exact(4)
        .filter(|p| p[0] as i32 > p[2] as i32 + 20)
        .count();
    assert!(
        lit > 100,
        "the cube disappeared after clearing instances ({lit} px)"
    );
}

#[test]
fn growing_the_instance_count_reallocates_correctly() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut renderer = ForwardRenderer::new(&gpu, 96, 96);
    renderer.upload_scene(&gpu, &scene);

    // Starting buffer holds exactly one instance, so this exercises the grow
    // path; writing past a too-small buffer is a validation error, and
    // reusing the old one silently draws the wrong count.
    for count in [1usize, 4, 2, 16] {
        let transforms: Vec<glam::Mat4> = (0..count)
            .map(|i| glam::Mat4::from_translation(glam::Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        renderer.set_instances(&gpu, 0, &transforms);
        assert_eq!(renderer.instance_count(0), count as u32);

        renderer
            .render_to_pixels(&gpu, 96, 96, &OrbitCamera::default())
            .expect("render must succeed at every instance count");
    }
}

/// Per-cascade shadow-caster culling engages without eating any shadow.
///
/// Two assertions that only mean something together: the shadow image test
/// above must still pass (culling deleted nothing the camera can see), and
/// the caster counters must show drawn < considered once a caster sits far
/// outside every cascade (culling actually engaged - without this, an inert
/// cull test would pass forever).
#[test]
fn caster_culling_engages_and_shadows_survive() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_on_plane.gltf");
    let mut scene = load_gltf(path).expect("cube_on_plane.gltf must load");

    // A third primitive far outside every cascade's fitted box: clone the
    // cube and push it 500 units away. Cascades fit the camera slice, which
    // ends well before that.
    let mut far_cube = scene.primitives[0].clone();
    far_cube.transform = glam::Mat4::from_translation(glam::Vec3::new(500.0, 0.0, 500.0));
    scene.primitives.push(far_cube);

    let (width, height) = (256, 256);
    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    renderer.light_dir_ambient = glam::Vec4::new(-1.0, 0.7, -0.3, 0.15);

    let camera = OrbitCamera {
        radius: 6.0,
        pitch_deg: 55.0,
        ..OrbitCamera::default()
    };
    let pixels = renderer
        .render_to_pixels(&gpu, width, height, &camera)
        .expect("headless render must succeed");

    let (drawn, considered) = renderer.shadow_caster_stats();
    assert!(considered > 0, "no casters considered - did the pass run?");
    assert!(
        drawn < considered,
        "culling never engaged: drawn {drawn} == considered {considered} \
         with a caster 500 units outside every cascade"
    );

    // Same structural check as shadow_darkens_plane_under_cube: both lit and
    // shadowed plane populations still exist, so culling did not delete the
    // visible shadow.
    let mut lit_plane = 0usize;
    let mut shadowed_plane = 0usize;
    for p in pixels.chunks_exact(4) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        let neutral = (r - g).abs() < 24 && (g - b).abs() < 24 && (r - b).abs() < 24;
        if !neutral {
            continue;
        }
        let luma = (r + g + b) / 3;
        if luma > 110 {
            lit_plane += 1;
        } else if luma > 8 {
            shadowed_plane += 1;
        }
    }
    assert!(lit_plane > 500, "lit plane vanished: {lit_plane}");
    assert!(
        shadowed_plane > 50,
        "the cube's shadow vanished - culling ate a visible caster ({shadowed_plane} shadowed px)"
    );
}
