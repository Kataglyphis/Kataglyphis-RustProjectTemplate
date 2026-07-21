//! The LOD system ON THE RENDER PATH.
//!
//! `lod.rs` and `qem.rs` were fully implemented and fully tested while being
//! called by nothing: the chain builder, the selector and both simplifiers all
//! passed their unit tests, and no frame ever drew a simplified triangle.
//! These tests exist to make that failure mode impossible to reintroduce, so
//! every one of them asserts on an INDEX COUNT taken from the same accessor
//! the draw loop uses - never on a configuration bool, which the inert version
//! would have satisfied.

use glam::Vec3;
use kataglyphis_webgpu_renderer::{
    build_lod_chain_with, load_gltf, ForwardRenderer, GpuContext, OrbitCamera, Simplifier,
};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

/// The bundled cube uploaded into a renderer with LOD in the requested state.
///
/// Measured on this asset (12 triangles, 36 indices): quadric level 0 comes
/// out at 18 indices and level 1 at 6 - level 1 asks for 25% of 12 triangles
/// and lands on 2 rather than 3, because the last collapse available to it
/// removes a pair.
fn renderer_with_lod(gpu: &GpuContext, enabled: bool) -> ForwardRenderer {
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut renderer = ForwardRenderer::new(gpu, 128, 128);
    renderer.lod_enabled = enabled;
    // The cube spans roughly two units, so these bracket "up close" and
    // "across the room" for it.
    renderer.lod_switch_distances = vec![8.0, 24.0];
    renderer.upload_scene(gpu, &scene);
    renderer
}

#[test]
fn chains_are_built_at_upload_and_not_per_frame() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let renderer = renderer_with_lod(&gpu, true);

    // No frame has been rendered at this point. If the levels only appeared
    // once something asked to draw them, the renderer would be simplifying
    // inside the frame - the design this feature explicitly rejected.
    assert_eq!(
        renderer.lod_level_count(0),
        2,
        "both levels must exist immediately after upload"
    );

    let full = renderer
        .selected_index_count(0, Vec3::ZERO)
        .expect("primitive 0 exists");
    let level_0 = renderer.lod_level_index_count(0, 0).unwrap();
    let level_1 = renderer.lod_level_index_count(0, 1).unwrap();
    assert!(
        level_0 < full && level_1 < level_0,
        "levels must get strictly coarser: full {full}, l0 {level_0}, l1 {level_1}"
    );
}

#[test]
fn a_distant_primitive_draws_strictly_fewer_indices() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let renderer = renderer_with_lod(&gpu, true);

    // The cube sits at the origin, so the eye position IS the distance.
    let near = renderer
        .selected_index_count(0, Vec3::new(0.0, 0.0, 3.0))
        .unwrap();
    let middle = renderer
        .selected_index_count(0, Vec3::new(0.0, 0.0, 12.0))
        .unwrap();
    let far = renderer
        .selected_index_count(0, Vec3::new(0.0, 0.0, 60.0))
        .unwrap();

    assert!(
        far < middle && middle < near,
        "index count must fall with distance: near {near}, middle {middle}, far {far}"
    );
    assert_eq!(near % 3, 0, "the drawn range must stay whole triangles");
    assert_eq!(far % 3, 0, "the drawn range must stay whole triangles");
}

#[test]
fn lod_disabled_draws_full_detail_at_every_distance() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let renderer = renderer_with_lod(&gpu, false);

    assert_eq!(
        renderer.lod_level_count(0),
        0,
        "disabled LOD must not build or upload any levels"
    );

    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let full = scene.primitives[0].indices.len() as u32;
    for distance in [0.5f32, 3.0, 12.0, 60.0, 10_000.0] {
        assert_eq!(
            renderer
                .selected_index_count(0, Vec3::new(0.0, 0.0, distance))
                .unwrap(),
            full,
            "with LOD off the full-detail buffer must be drawn at distance {distance}"
        );
    }
}

#[test]
fn an_lod_frame_still_renders() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let mut renderer = renderer_with_lod(&gpu, true);

    // Binding a level's buffers with the other level's index count would draw
    // out of range; wgpu's validation catches that here rather than in a
    // reviewer's eye.
    let pixels = renderer
        .render_to_pixels(&gpu, 128, 128, &OrbitCamera::default())
        .expect("a frame with LOD enabled must render");
    assert_eq!(pixels.len(), 128 * 128 * 4);
    assert!(
        pixels.chunks_exact(4).any(|p| p[0] != pixels[0]),
        "the LOD frame came out uniformly flat"
    );
}

#[test]
fn morphed_primitives_are_excluded_from_lod() {
    // A morphed primitive must keep drawing its full-res buffer at every
    // distance even with LOD enabled: `apply_morph_targets` re-blends only the
    // full-res vertex buffer, so a simplified LOD level would draw the
    // un-morphed neutral pose and the object would pop to its rest shape.
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let morph_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_morph.gltf");
    let scene = load_gltf(&morph_path).expect("cube_morph.gltf must load");
    assert_eq!(
        scene.primitives[0].morph_targets.len(),
        1,
        "the fixture must actually carry a morph target"
    );

    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.lod_enabled = true;
    renderer.lod_switch_distances = vec![8.0, 24.0];
    renderer.upload_scene(&gpu, &scene);

    assert_eq!(
        renderer.lod_level_count(0),
        0,
        "a morphed primitive must not build LOD levels even with LOD enabled"
    );

    let full = scene.primitives[0].indices.len() as u32;
    for distance in [0.5f32, 12.0, 60.0, 10_000.0] {
        assert_eq!(
            renderer
                .selected_index_count(0, Vec3::new(0.0, 0.0, distance))
                .unwrap(),
            full,
            "a morphed primitive must draw full detail at distance {distance}"
        );
    }
}

#[test]
fn quadric_level_zero_differs_from_full_detail() {
    // The trap that made the whole feature inert-by-default: clustering's
    // first level uses a cell ratio of 0.02, which on a low-poly mesh puts
    // every vertex in its own cell and returns the input unchanged. A chain
    // built that way looks wired and draws identical geometry forever.
    // Quadric's ratio is a triangle BUDGET, so it halves regardless of density.
    let scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let prim = &scene.primitives[0];
    let full_triangles = prim.indices.len() / 3;

    let clustered = build_lod_chain_with(prim, &[8.0], Simplifier::VertexClustering);
    assert_eq!(
        clustered[0].primitive.indices.len() / 3,
        full_triangles,
        "documenting the measured no-op: clustering at 0.02 does not touch a \
         low-poly mesh, which is why the render path uses Quadric"
    );

    let quadric = build_lod_chain_with(prim, &[8.0], Simplifier::Quadric);
    let level_0 = quadric[0].primitive.indices.len() / 3;
    assert!(
        level_0 < full_triangles,
        "quadric level 0 must actually simplify: {full_triangles} -> {level_0}"
    );
}

/// `world_center` drives BOTH the LOD switch distance and the transparent sort
/// order, so its definition must not change underneath them. It used to be the
/// vertex centroid at upload and the AABB centre afterwards, so the first
/// `set_animation_time` - even one that moves nothing - could flip the selected
/// LOD level and reorder blending on any unevenly tessellated mesh.
#[test]
fn lod_selection_is_stable_across_a_no_op_animation_update() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };
    let mut renderer = renderer_with_lod(&gpu, true);

    // Sample a spread of distances so at least one sits near a switch boundary.
    let eyes = [
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.0, 0.0, 8.0),
        Vec3::new(0.0, 0.0, 12.0),
        Vec3::new(0.0, 0.0, 24.0),
        Vec3::new(0.0, 0.0, 60.0),
    ];
    let before: Vec<u32> = eyes
        .iter()
        .map(|e| renderer.selected_index_count(0, *e).unwrap())
        .collect();

    // Advancing to t=0 changes no pose whatsoever.
    renderer.set_animation_time(0.0);

    let after: Vec<u32> = eyes
        .iter()
        .map(|e| renderer.selected_index_count(0, *e).unwrap())
        .collect();
    assert_eq!(
        before, after,
        "a no-op animation update must not change LOD selection"
    );
}
