//! `fs_main`'s IBL ambient block, restored after the Slang port (`2a4ae68`)
//! kept the entry point but dropped what it computed: a metal with no
//! environment bound rendered fully black (no specular term at all in the
//! analytic fallback), and both `light_dir_ambient.w` (the ambient slider)
//! and `ibl_params.enabled_maxmip_intensity.z` (environment intensity) were
//! computed and never read. See `Resources/ShadersSlang/forward/forward.slang`
//! and `docs/shader-sharing.md` for how this shader feeds this renderer.

use kataglyphis_webgpu_renderer::{load_gltf, EquirectImage, ForwardRenderer, GpuContext, OrbitCamera};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

fn render(renderer: &mut ForwardRenderer, gpu: &GpuContext) -> Vec<u8> {
    renderer
        .render_to_pixels(gpu, 128, 128, &OrbitCamera::default())
        .expect("headless render must succeed")
}

/// Mean R+G+B over the cube's pixels only. The cube is red-dominant and the
/// procedural sky behind it is blue-dominant (see `sky_radiance` in
/// `forward.slang`), so `pixel[0] > pixel[2]` separates them the same way
/// `ibl.rs`'s `setting_an_environment_actually_changes_the_rendered_frame`
/// does.
fn cube_luma(pixels: &[u8]) -> f64 {
    let mut total = 0u64;
    let mut count = 0u64;
    for pixel in pixels.chunks_exact(4) {
        if pixel[0] > pixel[2] {
            total += pixel[0] as u64 + pixel[1] as u64 + pixel[2] as u64;
            count += 1;
        }
    }
    assert!(count > 100, "found only {count} cube pixels to compare");
    total as f64 / count as f64
}

#[test]
fn a_metal_without_an_environment_still_gets_ambient_specular() {
    let Some(gpu) = GpuContext::headless_or_skip() else {
        return;
    };

    let mut scene = load_gltf(cube_path()).expect("cube.gltf must load");
    // Full metal, low roughness. Under the old fallback
    // (`hemisphere_irradiance(n) * albedo * (1 - metallic)`) this renders
    // exactly black: `1 - metallic` is zero and the no-environment path had
    // no specular term at all.
    scene.primitives[0].material.metallic_factor = 1.0;
    scene.primitives[0].material.roughness_factor = 0.05;

    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    assert!(!renderer.environment_enabled());
    // Kill the direct sun and any punctual lights so the only possible
    // contribution left in the frame is ambient.
    renderer.light_color_intensity.w = 0.0;
    renderer.light_dir_ambient.w = 1.0;

    let pixels = render(&mut renderer, &gpu);
    let luma = cube_luma(&pixels);
    assert!(
        luma > 10.0,
        "a metal cube lit only by analytic ambient specular should not render black, got mean luma {luma:.2}"
    );
}

#[test]
fn environment_intensity_scales_the_ambient_term() {
    let Some(gpu) = GpuContext::headless_or_skip() else {
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");

    // `set_environment` bakes `ibl_intensity` into the uniform buffer at
    // call time, so the intensity must be set beforehand.
    let mut dim = ForwardRenderer::new(&gpu, 128, 128);
    dim.upload_scene(&gpu, &scene);
    dim.ibl_intensity = 1.0;
    dim.set_environment(&gpu, &EquirectImage::constant(64, 32, [3.0, 3.0, 3.0]));
    let dim_luma = cube_luma(&render(&mut dim, &gpu));

    let mut bright = ForwardRenderer::new(&gpu, 128, 128);
    bright.upload_scene(&gpu, &scene);
    bright.ibl_intensity = 2.0;
    bright.set_environment(&gpu, &EquirectImage::constant(64, 32, [3.0, 3.0, 3.0]));
    let bright_luma = cube_luma(&render(&mut bright, &gpu));

    eprintln!("cube mean luma: ibl_intensity 1.0 -> {dim_luma:.2}, 2.0 -> {bright_luma:.2}");
    assert!(
        bright_luma > dim_luma + 5.0,
        "doubling environment intensity should brighten the ambient term: {dim_luma:.2} -> {bright_luma:.2}"
    );
}

#[test]
fn ambient_strength_scales_the_ambient_term() {
    let Some(gpu) = GpuContext::headless_or_skip() else {
        return;
    };

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");

    let render_at_strength = |strength: f32| -> Vec<u8> {
        let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
        renderer.upload_scene(&gpu, &scene);
        // Isolate the ambient term from the direct sun.
        renderer.light_color_intensity.w = 0.0;
        renderer.light_dir_ambient.w = strength;
        render(&mut renderer, &gpu)
    };

    let dim_luma = cube_luma(&render_at_strength(0.2));
    let bright_luma = cube_luma(&render_at_strength(1.0));

    eprintln!("cube mean luma: ambient 0.2 -> {dim_luma:.2}, 1.0 -> {bright_luma:.2}");
    assert!(
        bright_luma > dim_luma + 5.0,
        "raising light_dir_ambient.w should brighten the ambient term: {dim_luma:.2} -> {bright_luma:.2}"
    );
}
