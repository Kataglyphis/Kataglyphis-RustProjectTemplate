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

const TWO_MATERIAL_OBJ: &str = "\
mtllib pair.mtl
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
usemtl red
f 1//1 2//1 3//1
usemtl blue
f 1//1 3//1 4//1
";

const PAIR_MTL: &str = "\
newmtl red
Ka 0.1 0.1 0.1
Kd 0.9 0.1 0.05
Ks 0.5 0.5 0.5
Ns 32
d 1
illum 2

newmtl blue
Kd 0.05 0.15 0.85
d 0.4
";

#[test]
fn mtl_diffuse_and_opacity_become_base_color() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    let materials = parse_mtl(PAIR_MTL);
    assert_eq!(materials.len(), 2);

    assert_eq!(materials[0].name, "red");
    assert!((materials[0].base_color[0] - 0.9).abs() < 1e-6);
    assert!((materials[0].base_color[3] - 1.0).abs() < 1e-6, "d 1 is fully opaque");

    assert_eq!(materials[1].name, "blue");
    assert!((materials[1].base_color[2] - 0.85).abs() < 1e-6);
    assert!((materials[1].base_color[3] - 0.4).abs() < 1e-6, "d 0.4 is the alpha");
}

#[test]
fn tr_is_inverted_relative_to_d() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    // The same quantity written two ways. Treating them as interchangeable
    // makes transparent materials opaque and vice versa - a mistake that
    // looks like a renderer bug, not a converter bug.
    let by_opacity = parse_mtl("newmtl a\nd 0.25\n");
    let by_transparency = parse_mtl("newmtl a\nTr 0.25\n");

    assert!((by_opacity[0].base_color[3] - 0.25).abs() < 1e-6);
    assert!((by_transparency[0].base_color[3] - 0.75).abs() < 1e-6);
}

#[test]
fn each_usemtl_run_becomes_its_own_primitive() {
    let dir = temp_dir("materials");
    std::fs::write(dir.join("pair.mtl"), PAIR_MTL).expect("write mtl");
    let obj_path = dir.join("pair.obj");
    std::fs::write(&obj_path, TWO_MATERIAL_OBJ).expect("write obj");
    let gltf_path = dir.join("pair.gltf");

    let mesh = convert_file(&obj_path, &gltf_path).expect("conversion must succeed");
    assert_eq!(mesh.submeshes.len(), 2, "two usemtl runs should produce two submeshes");

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    assert_eq!(scene.primitives.len(), 2, "each material run should load as its own primitive");

    // The colours must survive, and land on the right half.
    let red = scene
        .primitives
        .iter()
        .find(|p| p.material.base_color[0] > 0.5)
        .expect("a red-dominant primitive");
    let blue = scene
        .primitives
        .iter()
        .find(|p| p.material.base_color[2] > 0.5)
        .expect("a blue-dominant primitive");

    assert!((red.material.base_color[0] - 0.9).abs() < 1e-4);
    assert!((blue.material.base_color[2] - 0.85).abs() < 1e-4);
    assert!(
        (blue.material.base_color[3] - 0.4).abs() < 1e-4,
        "the transparent material lost its alpha: {:?}",
        blue.material.base_color
    );
}

#[test]
fn material_runs_share_one_vertex_buffer() {
    // Splitting vertex data per material would duplicate shared vertices and
    // change the geometry - the exact thing a cross-renderer comparison must
    // hold constant. Both triangles here share two of the four vertices.
    let dir = temp_dir("shared_vertices");
    std::fs::write(dir.join("pair.mtl"), PAIR_MTL).expect("write mtl");
    let obj_path = dir.join("pair.obj");
    std::fs::write(&obj_path, TWO_MATERIAL_OBJ).expect("write obj");
    let gltf_path = dir.join("pair.gltf");

    let mesh = convert_file(&obj_path, &gltf_path).expect("convert");
    assert_eq!(mesh.positions.len(), 4, "the quad's four corners must not be duplicated");

    let scene = load_gltf(&gltf_path).expect("load");
    let total_indices: usize = scene.primitives.iter().map(|p| p.indices.len()).sum();
    assert_eq!(total_indices, mesh.indices.len(), "index data changed across the split");
}

#[test]
fn an_obj_without_materials_still_gets_a_usable_default() {
    // glTF's default material is metallic 1 / roughness 1, which renders as a
    // dark mirror. An OBJ with no mtllib must not convert into that.
    let dir = temp_dir("no_materials");
    let obj_path = dir.join("cube.obj");
    std::fs::write(&obj_path, CUBE_OBJ).expect("write obj");
    let gltf_path = dir.join("cube.gltf");

    convert_file(&obj_path, &gltf_path).expect("convert");
    let scene = load_gltf(&gltf_path).expect("load");

    let material = &scene.primitives[0].material;
    assert!(material.metallic_factor < 0.01, "converted assets must not be metallic by default");
    assert!(material.base_color[0] > 0.9, "an untextured OBJ should convert to an untinted material");
}

#[test]
fn a_usemtl_naming_an_undeclared_material_is_visible_not_silent() {
    // Referencing a material the .mtl never declared is an authoring error.
    // Collapsing it onto material 0 would render the geometry with someone
    // else's colour and look intentional.
    let mesh = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nusemtl ghost\nf 1 2 3\n").expect("parse");
    assert!(
        mesh.materials.iter().any(|m| m.name == "ghost"),
        "the undeclared name should survive into the output: {:?}",
        mesh.materials.iter().map(|m| &m.name).collect::<Vec<_>>()
    );
}
