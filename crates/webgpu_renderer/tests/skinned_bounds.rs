//! Culling bounds must follow SKINNED deformation.
//!
//! A skinned vertex ignores the node/model matrix entirely: `skin_matrix` in
//! forward.wgsl returns the weighted joint blend and only falls back to
//! `uniforms.model` when the weights are zero. So bounds derived from the node
//! transform alone describe the bind pose, not the pose being drawn, and an
//! animated limb gets frustum-culled while it is plainly on screen.
//!
//! These tests assert on `primitive_world_aabb`, the same value the frustum
//! test reads, rather than on a recomputed copy.

use glam::{Mat4, Quat, Vec3};
use kataglyphis_webgpu_renderer::scene::{CpuNode, CpuSkin};
use kataglyphis_webgpu_renderer::{load_gltf, ForwardRenderer, GpuContext};

fn cube_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
}

/// The bundled cube, fully weighted to joint 0 of a one-joint skin. Node 0 is
/// the mesh node (identity); node 1 is the joint, placed by `joint_translation`.
fn skinned_cube_scene(joint_translation: Vec3) -> kataglyphis_webgpu_renderer::CpuScene {
    let mut scene = load_gltf(cube_path()).expect("cube.gltf must load");
    let mut prim = scene.primitives[0].clone();
    for v in &mut prim.vertices {
        v.joints = [0.0, 0.0, 0.0, 0.0];
        v.weights = [1.0, 0.0, 0.0, 0.0];
    }
    prim.node_index = Some(0);
    prim.skin_index = Some(0);
    prim.transform = Mat4::IDENTITY;
    scene.primitives = vec![prim];
    scene.nodes = vec![
        CpuNode {
            parent: None,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        CpuNode {
            parent: None,
            translation: joint_translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
    ];
    scene.skins = vec![CpuSkin {
        joints: vec![1],
        inverse_bind_matrices: vec![Mat4::IDENTITY],
    }];
    scene.animations = Vec::new();
    scene
}

#[test]
fn bounds_follow_the_joint_not_the_bind_pose() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // The joint lifts the cube 10 units. The cube itself spans [-0.5, 0.5], so
    // bind-pose bounds would stop at y = 0.5 and the drawn geometry (y ~= 10)
    // would sit entirely outside its own AABB.
    let scene = skinned_cube_scene(Vec3::new(0.0, 10.0, 0.0));
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    renderer.set_animation_time(0.0);

    let (min, max) = renderer
        .primitive_world_aabb(0)
        .expect("primitive 0 exists");

    assert!(
        max.y >= 9.5,
        "bounds must reach the joint-driven pose (~10.5), got max.y={} (min.y={})",
        max.y,
        min.y
    );
    // X/Z are untouched by the joint, so the box must not balloon sideways.
    assert!(
        max.x <= 1.0 && min.x >= -1.0,
        "bounds must stay tight where the joint does not move, got x=[{}, {}]",
        min.x,
        max.x
    );
}

#[test]
fn an_identity_joint_leaves_bounds_at_the_bind_pose() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    // Guard against over-widening: with the joint at the origin the skinned
    // bounds must still be the cube's own box, not something inflated.
    let scene = skinned_cube_scene(Vec3::ZERO);
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    renderer.set_animation_time(0.0);

    let (min, max) = renderer
        .primitive_world_aabb(0)
        .expect("primitive 0 exists");
    assert!(
        (max.y - 0.5).abs() < 1e-4 && (min.y + 0.5).abs() < 1e-4,
        "identity joint must leave the bind-pose box, got y=[{}, {}]",
        min.y,
        max.y
    );
}

/// Instance transforms apply on top of the posed box (`instance_matrix *
/// skin_matrix * v`), so bounds that ignore them cull every instance the moment
/// the BASE position leaves the view - even with the instances on screen.
#[test]
fn bounds_span_every_instance() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = skinned_cube_scene(Vec3::ZERO);
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);

    let (base_min, base_max) = renderer.primitive_world_aabb(0).expect("primitive 0");

    // Two instances, 50 units apart on X, neither at the origin.
    renderer.set_instances(
        &gpu,
        0,
        &[
            Mat4::from_translation(Vec3::new(-50.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        ],
    );
    let (min, max) = renderer.primitive_world_aabb(0).expect("primitive 0");
    assert!(
        min.x <= -50.0 && max.x >= 50.0,
        "bounds must span both instances, got x=[{}, {}]",
        min.x,
        max.x
    );

    // Resetting to the default identity instance must collapse them back, not
    // leave the primitive permanently over-sized.
    renderer.set_instances(&gpu, 0, &[]);
    let (rmin, rmax) = renderer.primitive_world_aabb(0).expect("primitive 0");
    assert!(
        (rmin.x - base_min.x).abs() < 1e-4 && (rmax.x - base_max.x).abs() < 1e-4,
        "clearing instances must restore the base box, got x=[{}, {}] vs [{}, {}]",
        rmin.x,
        rmax.x,
        base_min.x,
        base_max.x
    );
}

/// scene_bounds is the ONLY input to shadow-cascade fitting, so it has to track
/// instances too. Before this was wired, cascades stayed fitted to the
/// un-instanced scene and distant instances neither received nor cast shadows.
#[test]
fn scene_bounds_track_instances() {
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available in this environment");
        return;
    };

    let scene = skinned_cube_scene(Vec3::ZERO);
    let mut renderer = ForwardRenderer::new(&gpu, 128, 128);
    renderer.upload_scene(&gpu, &scene);
    let (base_min, base_max) = renderer.scene_bounds().expect("bounds after upload");

    renderer.set_instances(&gpu, 0, &[Mat4::from_translation(Vec3::new(40.0, 0.0, 0.0))]);
    let (min, max) = renderer.scene_bounds().expect("bounds after instancing");
    assert!(
        max.x >= 39.0,
        "scene bounds must follow the instance out to x=40, got x=[{}, {}]",
        min.x,
        max.x
    );

    // And collapse back when the instances are cleared.
    renderer.set_instances(&gpu, 0, &[]);
    let (rmin, rmax) = renderer.scene_bounds().expect("bounds after reset");
    assert!(
        (rmax.x - base_max.x).abs() < 1e-4 && (rmin.x - base_min.x).abs() < 1e-4,
        "clearing instances must restore scene bounds, got x=[{}, {}] vs [{}, {}]",
        rmin.x,
        rmax.x,
        base_min.x,
        base_max.x
    );
}
