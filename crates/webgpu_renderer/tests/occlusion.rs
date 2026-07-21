//! GPU occlusion DETECTION against a real adapter.
//!
//! The ring/readback plumbing has no pure part worth unit-testing in isolation;
//! it is a mirror of `gpu_timing`, whose maths is covered there. What only a
//! GPU can answer is the thing this feature exists for: that a primitive hidden
//! behind other geometry reads back 0 occlusion samples and a visible one reads
//! back more than 0. Both tests print the actual sample counts they measured.

use glam::{Mat4, Vec3};
use kataglyphis_webgpu_renderer::{load_gltf, CpuScene, ForwardRenderer, GpuContext, OrbitCamera};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

/// The bundled unit cube (positions in [-0.5, 0.5]) transformed by `transform`.
fn cube_with_transform(transform: Mat4) -> kataglyphis_webgpu_renderer::CpuPrimitive {
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut prim = scene.primitives[0].clone();
    prim.transform = transform;
    prim
}

/// A camera on +Z looking down -Z at the origin (yaw 90, pitch 0).
fn looking_down_neg_z() -> OrbitCamera {
    OrbitCamera {
        radius: 8.0,
        yaw_deg: 90.0,
        pitch_deg: 0.0,
        ..OrbitCamera::default()
    }
}

/// Renders enough frames for the asynchronous occlusion readback to land. The
/// readback lags the frame it measures by one or more frames (it is mapped
/// after submit and never waited on), so a single frame would read nothing -
/// the same reason `gpu_timing`'s test loops 64 times.
fn render_until_readback_lands(
    renderer: &mut ForwardRenderer,
    gpu: &GpuContext,
    camera: &OrbitCamera,
) {
    for _ in 0..64 {
        renderer
            .render_to_pixels(gpu, 256, 256, camera)
            .expect("headless render must succeed");
    }
}

#[test]
fn a_cube_hidden_behind_another_reads_back_zero_samples() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // A large occluder near the camera (index 0) and a small cube directly
    // behind it (index 1). From +Z looking down -Z, the occluder's world AABB
    // spans x,y in [-2, 2] at z in [0, 4] and the hidden cube sits at z ~= -3,
    // fully behind the occluder in both screen coverage and depth.
    let occluder = cube_with_transform(
        Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0)) * Mat4::from_scale(Vec3::splat(4.0)),
    );
    let hidden = cube_with_transform(Mat4::from_translation(Vec3::new(0.0, 0.0, -3.0)));

    let scene = CpuScene {
        primitives: vec![occluder, hidden],
        ..Default::default()
    };

    let mut renderer = ForwardRenderer::new(&gpu, 256, 256);
    renderer.upload_scene(&gpu, &scene);
    renderer.occlusion_queries_enabled = true;

    let camera = looking_down_neg_z();
    render_until_readback_lands(&mut renderer, &gpu, &camera);

    let samples = renderer.occlusion_samples();
    let visibility = renderer.occlusion_visibility();
    eprintln!("occluded-scene samples: {samples:?}");
    assert_eq!(
        visibility.len(),
        2,
        "expected one visibility entry per primitive, got {visibility:?} (samples {samples:?})"
    );

    assert!(
        visibility[0],
        "the occluder should be visible (> 0 samples), measured {}",
        samples[0]
    );
    assert!(
        !visibility[1],
        "the hidden cube should be occluded (0 samples), measured {}",
        samples[1]
    );
    assert_eq!(
        samples[1], 0,
        "a fully occluded box must count exactly zero fragments, got {}",
        samples[1]
    );
}

#[test]
fn two_side_by_side_cubes_are_both_visible() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // Two cubes offset along x, both fully in view and occluding nothing.
    let left = cube_with_transform(Mat4::from_translation(Vec3::new(-2.0, 0.0, 0.0)));
    let right = cube_with_transform(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)));

    let scene = CpuScene {
        primitives: vec![left, right],
        ..Default::default()
    };

    let mut renderer = ForwardRenderer::new(&gpu, 256, 256);
    renderer.upload_scene(&gpu, &scene);
    renderer.occlusion_queries_enabled = true;

    let camera = looking_down_neg_z();
    render_until_readback_lands(&mut renderer, &gpu, &camera);

    let samples = renderer.occlusion_samples();
    let visibility = renderer.occlusion_visibility();
    eprintln!("side-by-side samples: {samples:?}");
    assert_eq!(visibility.len(), 2, "expected two visibility entries");
    assert!(
        visibility[0] && visibility[1],
        "both visible cubes should report > 0 samples, measured {samples:?}"
    );
}

#[test]
fn detection_is_off_by_default() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);

    // Default-off is the contract: the frame renders exactly as before and no
    // occlusion is measured.
    assert!(!renderer.occlusion_queries_enabled);
    let camera = OrbitCamera::default();
    for _ in 0..4 {
        renderer
            .render_to_pixels(&gpu, 128, 128, &camera)
            .expect("headless render must succeed");
    }
    assert!(
        renderer.occlusion_visibility().is_empty(),
        "no visibility should be recorded while detection is off"
    );
}
