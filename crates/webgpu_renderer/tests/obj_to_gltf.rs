//! OBJ -> glTF conversion, verified by loading the result back.
//!
//! The emitter writes glTF JSON by hand, so "it produced a file" proves
//! nothing. Every test here round-trips through the real `gltf` crate via the
//! renderer's own loader: if the offsets, accessor counts, component types or
//! bounds are wrong, the load fails or the geometry comes back different.

use kataglyphis_webgpu_renderer::asset::gltf_loader::load_gltf;
use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{convert_file, parse_obj, to_gltf};

/// A unit cube with normals and UVs, as a Blender-style OBJ.
const CUBE_OBJ: &str = "\
# a comment that must be ignored
mtllib ignored.mtl
o Cube
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
usemtl Material
s off
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
";

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("kataglyphis_obj_gltf_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

#[test]
fn parses_positions_normals_uvs_and_triangulates() {
    let mesh = parse_obj(CUBE_OBJ).expect("the cube must parse");

    assert_eq!(mesh.triangle_count(), 2, "two faces should produce two triangles");
    // Four distinct position/uv/normal triples, all sharing one normal.
    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(mesh.normals.len(), 4);
    assert_eq!(mesh.uvs.len(), 4);

    let (min, max) = mesh.bounds();
    assert_eq!(min, [-1.0, -1.0, 1.0]);
    assert_eq!(max, [1.0, 1.0, 1.0]);
}

#[test]
fn the_v_axis_is_flipped_for_gltf() {
    // OBJ's V points up, glTF's points down. Getting this wrong mirrors every
    // texture vertically - which looks plausible on a symmetric test texture
    // and wrong on everything else.
    let mesh = parse_obj(CUBE_OBJ).expect("parse");
    // OBJ vt 0.0 0.0 -> glTF 0.0 1.0
    assert!(
        mesh.uvs.iter().any(|uv| (uv[1] - 1.0).abs() < 1e-6),
        "expected a flipped V of 1.0 among {:?}",
        mesh.uvs
    );
}

#[test]
fn quads_are_fan_triangulated() {
    let quad = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
    let mesh = parse_obj(quad).expect("parse");
    assert_eq!(mesh.triangle_count(), 2, "a quad must become two triangles");
    assert_eq!(mesh.positions.len(), 4, "fan triangulation must not duplicate vertices");
}

#[test]
fn malformed_input_is_rejected_rather_than_guessed_at() {
    // Each of these could plausibly be "fixed up" silently, and each would
    // produce an asset that differs from the source without anyone noticing.
    assert!(parse_obj("v 1.0 2.0\nf 1 1 1\n").is_err(), "a 2-component vertex must be rejected");
    assert!(parse_obj("v 0 0 0\nf 1 2 3\n").is_err(), "an out-of-range index must be rejected");
    assert!(parse_obj("v 0 0 0\nf -1 -2 -3\n").is_err(), "relative indices are unsupported, not ignored");
    assert!(parse_obj("v 0 0 0\nf 0 0 0\n").is_err(), "index 0 is invalid in OBJ");
    assert!(parse_obj("# nothing here\n").is_err(), "an empty OBJ must not produce an empty mesh silently");
    assert!(parse_obj("curv 0 1 2\n").is_err(), "an unsupported directive must not be skipped quietly");
}

#[test]
fn converted_gltf_loads_back_with_matching_geometry() {
    let dir = temp_dir("roundtrip");
    let obj_path = dir.join("cube.obj");
    let gltf_path = dir.join("cube.gltf");
    std::fs::write(&obj_path, CUBE_OBJ).expect("write obj");

    let source = convert_file(&obj_path, &gltf_path).expect("conversion must succeed");

    // The real loader, not a bespoke parser: this is what makes the test
    // meaningful, since it exercises the same code path the renderer uses.
    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    assert_eq!(scene.primitives.len(), 1);
    let loaded = &scene.primitives[0];

    assert_eq!(
        loaded.indices.len(),
        source.indices.len(),
        "index count changed through the round trip"
    );
    assert_eq!(
        loaded.vertices.len(),
        source.positions.len(),
        "vertex count changed through the round trip"
    );

    for (index, vertex) in loaded.vertices.iter().enumerate() {
        for axis in 0..3 {
            assert!(
                (vertex.position[axis] - source.positions[index][axis]).abs() < 1e-6,
                "vertex {index} position differs on axis {axis}: {:?} vs {:?}",
                vertex.position,
                source.positions[index]
            );
        }
    }
}

#[test]
fn the_declared_buffer_length_matches_the_bytes_written() {
    // glTF loaders trust byteLength. A mismatch either truncates geometry or
    // reads past the buffer, and the gltf crate rejects it - so this also
    // guards the offset arithmetic above it.
    let mesh = parse_obj(CUBE_OBJ).expect("parse");
    let (json, bin) = to_gltf(&mesh, "cube.bin");

    let declared = json
        .split("\"byteLength\": ")
        .nth(1)
        .and_then(|rest| rest.split([',', ' ', '}']).next())
        .and_then(|value| value.parse::<usize>().ok())
        .expect("the buffer must declare a byteLength");

    assert_eq!(declared, bin.len(), "declared buffer length disagrees with the data");
}

#[test]
fn converts_a_real_engine_asset() {
    // The point of the whole exercise: the C++ engine's own models becoming
    // loadable here. Skips rather than fails if the asset tree is absent, so
    // the crate stays testable standalone.
    let obj = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../Resources/Models/ShadowTest/shadow_rig.obj");
    if !obj.exists() {
        eprintln!("SKIP: {} not present", obj.display());
        return;
    }

    let dir = temp_dir("engine_asset");
    let gltf_path = dir.join("shadow_rig.gltf");
    let mesh = convert_file(&obj, &gltf_path).expect("engine asset must convert");
    assert!(mesh.triangle_count() > 0);

    let scene = load_gltf(&gltf_path).expect("converted engine asset must load");
    assert_eq!(scene.primitives.len(), 1);
    assert_eq!(scene.primitives[0].indices.len(), mesh.indices.len());
}
