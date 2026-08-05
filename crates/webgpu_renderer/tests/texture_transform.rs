//! KHR_texture_transform is scoped per `textureInfo` (glTF 2.0 spec): a
//! transform authored on one texture slot must not leak into another. These
//! tests guard against the "fix" of copying one slot's transform to all five.

use kataglyphis_webgpu_renderer::load_gltf;

const IDENTITY: [[f32; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

fn assert_identity(t: [[f32; 3]; 2], slot: &str) {
    assert!(
        (t[0][0] - IDENTITY[0][0]).abs() < 1e-6
            && (t[0][1] - IDENTITY[0][1]).abs() < 1e-6
            && (t[0][2] - IDENTITY[0][2]).abs() < 1e-6
            && (t[1][0] - IDENTITY[1][0]).abs() < 1e-6
            && (t[1][1] - IDENTITY[1][1]).abs() < 1e-6
            && (t[1][2] - IDENTITY[1][2]).abs() < 1e-6,
        "{slot} slot must stay identity, got {t:?}"
    );
}

fn assert_non_identity(t: [[f32; 3]; 2], slot: &str) {
    assert!(
        (t[0][0] - IDENTITY[0][0]).abs() > 1e-3 || (t[0][2] - IDENTITY[0][2]).abs() > 1e-3,
        "{slot} slot must carry its authored transform, got {t:?}"
    );
}

#[test]
fn texture_transform_on_normal_slot_does_not_leak_to_other_slots() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets/cube_normal_transform.gltf");
    let scene = load_gltf(path).expect("cube_normal_transform.gltf must load");
    let material = &scene.primitives[0].material;

    assert_non_identity(material.normal_uv_transform, "normal");
    assert_identity(material.base_uv_transform, "base color");
    assert_identity(material.mr_uv_transform, "metallic-roughness");
    assert_identity(material.emissive_uv_transform, "emissive");
    assert_identity(material.occlusion_uv_transform, "occlusion");
}

#[test]
fn texture_transform_on_base_color_slot_does_not_leak_to_other_slots() {
    // cube_textured.gltf carries KHR_texture_transform on baseColorTexture
    // only (offset (0.25, 0), scale (2, 2)).
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_textured.gltf");
    let scene = load_gltf(path).expect("cube_textured.gltf must load");
    let material = &scene.primitives[0].material;

    assert_non_identity(material.base_uv_transform, "base color");
    assert_identity(material.mr_uv_transform, "metallic-roughness");
    assert_identity(material.normal_uv_transform, "normal");
    assert_identity(material.emissive_uv_transform, "emissive");
    assert_identity(material.occlusion_uv_transform, "occlusion");
}
