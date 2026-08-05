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

    assert_eq!(
        mesh.triangle_count(),
        2,
        "two faces should produce two triangles"
    );
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
    assert_eq!(
        mesh.positions.len(),
        4,
        "fan triangulation must not duplicate vertices"
    );
}

#[test]
fn malformed_input_is_rejected_rather_than_guessed_at() {
    // Each of these could plausibly be "fixed up" silently, and each would
    // produce an asset that differs from the source without anyone noticing.
    assert!(
        parse_obj("v 1.0 2.0\nf 1 1 1\n").is_err(),
        "a 2-component vertex must be rejected"
    );
    assert!(
        parse_obj("v 0 0 0\nf 1 2 3\n").is_err(),
        "an out-of-range index must be rejected"
    );
    assert!(
        parse_obj("v 0 0 0\nf -1 -2 -3\n").is_err(),
        "relative indices are unsupported, not ignored"
    );
    assert!(
        parse_obj("v 0 0 0\nf 0 0 0\n").is_err(),
        "index 0 is invalid in OBJ"
    );
    assert!(
        parse_obj("# nothing here\n").is_err(),
        "an empty OBJ must not produce an empty mesh silently"
    );
    assert!(
        parse_obj("curv 0 1 2\n").is_err(),
        "an unsupported directive must not be skipped quietly"
    );
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

    assert_eq!(
        declared,
        bin.len(),
        "declared buffer length disagrees with the data"
    );
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
    assert!(
        (materials[0].base_color[3] - 1.0).abs() < 1e-6,
        "d 1 is fully opaque"
    );

    assert_eq!(materials[1].name, "blue");
    assert!((materials[1].base_color[2] - 0.85).abs() < 1e-6);
    assert!(
        (materials[1].base_color[3] - 0.4).abs() < 1e-6,
        "d 0.4 is the alpha"
    );
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
fn mtl_emissive_becomes_emissive_factor() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nKe 0.5 0.25 0.0\n");
    assert_eq!(materials[0].emissive, [0.5, 0.25, 0.0]);

    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");
    assert!(
        json.contains(r#""emissiveFactor": [0.5, 0.25, 0]"#),
        "expected an emissiveFactor entry, got: {json}"
    );
}

#[test]
fn mtl_without_ke_emits_no_emissive_factor() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nKd 1 0 0\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");
    assert!(
        !json.contains("emissiveFactor"),
        "a material with no Ke must not emit emissiveFactor: {json}"
    );
}

#[test]
fn an_hdr_ke_emits_emissive_strength() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    // The C++ engine keeps an HDR `Ke` verbatim (Src/GraphicsEngineVulkan/
    // scene/ObjLoader.cpp), so this converter must carry the same magnitude
    // via KHR_materials_emissive_strength instead of clamping it to 1 and
    // rendering four times dimmer than the C++ side.
    let materials = parse_mtl("newmtl a\nKe 4 4 4\n");
    assert_eq!(
        materials[0].emissive,
        [4.0, 4.0, 4.0],
        "Ke must be stored unclamped"
    );

    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");
    assert!(
        json.contains(r#""emissiveFactor": [1, 1, 1]"#),
        "expected a normalised emissiveFactor of [1, 1, 1], got: {json}"
    );
    assert!(
        json.contains(
            r#""extensions": { "KHR_materials_emissive_strength": { "emissiveStrength": 4 } }"#
        ),
        "expected a KHR_materials_emissive_strength extension, got: {json}"
    );
    assert!(
        json.contains(r#""extensionsUsed": ["KHR_materials_emissive_strength"]"#),
        "expected extensionsUsed to name the extension, got: {json}"
    );
}

#[test]
fn a_non_hdr_ke_emits_no_emissive_strength_extension() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    // The byte-identical case: every document converted before this
    // extension existed must not gain an unconditional extensions/
    // extensionsUsed array.
    let materials = parse_mtl("newmtl a\nKe 0.5 0.5 0.5\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");
    assert!(
        !json.contains("extensions"),
        "a non-HDR Ke must not emit an extensions block: {json}"
    );
    assert!(
        !json.contains("extensionsUsed"),
        "a non-HDR Ke must not emit extensionsUsed: {json}"
    );
}

#[test]
fn an_hdr_ke_round_trips_through_the_real_gltf_loader() {
    // That is the property that actually matters: the loader folds
    // emissiveFactor * emissiveStrength back into the HDR value, so this is
    // what makes the C++ and Rust renderers agree on brightness.
    let dir = temp_dir("hdr_emissive");
    std::fs::write(dir.join("hdr.mtl"), "newmtl painted\nKd 1 1 1\nKe 4 4 4\n")
        .expect("mtl");
    let obj_path = dir.join("hdr.obj");
    std::fs::write(
        &obj_path,
        "mtllib hdr.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nusemtl painted\nf 1 2 3\n",
    )
    .expect("obj");
    let gltf_path = dir.join("hdr.gltf");

    convert_file(&obj_path, &gltf_path).expect("conversion must succeed");
    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    let emissive = scene.primitives[0].material.emissive_factor;
    for channel in emissive {
        assert!(
            (channel - 4.0).abs() < 1e-4,
            "expected the HDR emissive factor to survive the round trip as 4.0, got {emissive:?}"
        );
    }
}

#[test]
fn each_usemtl_run_becomes_its_own_primitive() {
    let dir = temp_dir("materials");
    std::fs::write(dir.join("pair.mtl"), PAIR_MTL).expect("write mtl");
    let obj_path = dir.join("pair.obj");
    std::fs::write(&obj_path, TWO_MATERIAL_OBJ).expect("write obj");
    let gltf_path = dir.join("pair.gltf");

    let mesh = convert_file(&obj_path, &gltf_path).expect("conversion must succeed");
    assert_eq!(
        mesh.submeshes.len(),
        2,
        "two usemtl runs should produce two submeshes"
    );

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    assert_eq!(
        scene.primitives.len(),
        2,
        "each material run should load as its own primitive"
    );

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
    assert_eq!(
        mesh.positions.len(),
        4,
        "the quad's four corners must not be duplicated"
    );

    let scene = load_gltf(&gltf_path).expect("load");
    let total_indices: usize = scene.primitives.iter().map(|p| p.indices.len()).sum();
    assert_eq!(
        total_indices,
        mesh.indices.len(),
        "index data changed across the split"
    );
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
    assert!(
        material.metallic_factor < 0.01,
        "converted assets must not be metallic by default"
    );
    assert!(
        material.base_color[0] > 0.9,
        "an untextured OBJ should convert to an untinted material"
    );
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

const TEXTURED_OBJ: &str = "\
mtllib textured.mtl
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vn 0.0 0.0 1.0
usemtl painted
f 1/1/1 2/2/1 3/3/1
";

/// Writes a 2x2 PNG with a distinctive corner so the round trip can prove the
/// image data survived, not merely that a file was referenced.
fn write_test_png(path: &std::path::Path) {
    use image::{ImageBuffer, Rgba};
    let mut buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(2, 2);
    buffer.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
    buffer.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
    buffer.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
    buffer.put_pixel(1, 1, Rgba([255, 255, 0, 255]));
    buffer.save(path).expect("write png");
}

#[test]
fn map_kd_takes_the_filename_not_the_first_option() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    // MTL allows options before the path. Taking the first token turns any
    // option-carrying map into a texture named "-s", which fails to load with
    // a message about a file nobody wrote.
    let plain = parse_mtl("newmtl a\nmap_Kd wood.png\n");
    assert_eq!(plain[0].base_color_texture.as_deref(), Some("wood.png"));

    let with_options = parse_mtl("newmtl a\nmap_Kd -s 1 1 1 -o 0 0 0 wood.png\n");
    assert_eq!(
        with_options[0].base_color_texture.as_deref(),
        Some("wood.png"),
        "the filename must be taken from the end, past the options"
    );
}

#[test]
fn a_textured_obj_converts_to_a_gltf_with_a_loadable_texture() {
    let dir = temp_dir("textured");
    write_test_png(&dir.join("paint.png"));
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl painted\nKd 1 1 1\nmap_Kd paint.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");

    let out = temp_dir("textured_out");
    let gltf_path = out.join("textured.gltf");
    convert_file(&obj_path, &gltf_path).expect("conversion must succeed");

    // The texture must have been copied next to the output, or the document
    // only loads on the machine that produced it.
    assert!(
        out.join("paint.png").exists(),
        "the referenced texture was not copied next to the glTF"
    );

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    let material = &scene.primitives[0].material;
    let texture = material
        .base_color_texture
        .as_ref()
        .expect("the material must carry a base colour texture");

    assert_eq!((texture.texture.width, texture.texture.height), (2, 2));
    assert!(texture.srgb, "base colour is authored in sRGB");
    // Top-left red, proving pixel data survived rather than just a reference.
    assert_eq!(&texture.texture.rgba8[0..4], &[255, 0, 0, 255]);
}

#[test]
fn a_textures_subdirectory_layout_is_resolved() {
    // Every shipped engine OBJ (crytek-sponza, Pillum, Sulo/WolfStahl,
    // VikingRoom) puts its textures in a `textures/` subdirectory next to the
    // .mtl and references them by bare filename - `map_Kd` does not name the
    // subdirectory. docs/model-loading.md's second candidate is the rule that
    // makes that layout resolve instead of converting to an untextured glTF.
    let dir = temp_dir("textures_subdir");
    let texture_dir = dir.join("textures");
    std::fs::create_dir_all(&texture_dir).expect("textures dir");
    write_test_png(&texture_dir.join("paint.png"));
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl painted\nKd 1 1 1\nmap_Kd paint.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");

    let out = temp_dir("textures_subdir_out");
    let gltf_path = out.join("textured.gltf");
    convert_file(&obj_path, &gltf_path).expect("conversion must succeed");

    // The bare filename is what gets copied and emitted as the glTF uri -
    // which candidate resolved it is not observable from the output.
    assert!(
        out.join("paint.png").exists(),
        "the textures/-subdirectory texture was not copied next to the glTF"
    );

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    let texture = scene.primitives[0]
        .material
        .base_color_texture
        .as_ref()
        .expect("the material must carry a base colour texture");
    assert_eq!((texture.texture.width, texture.texture.height), (2, 2));
}

#[test]
fn a_backslash_map_kd_under_textures_is_resolved() {
    // A Windows-authored `map_Kd textures\paint.png` must normalise to
    // `textures/paint.png` (parse_mtl) before path resolution ever runs, or
    // the literal backslash is just another filename character on Linux and
    // the candidate lookup misses. .obj and .gltf share a directory here (as
    // in `a_backslash_texture_path_does_not_corrupt_the_document` above), so
    // the copy step short-circuits on source == destination and the test
    // isolates normalisation from the separate copy-destination behaviour.
    let dir = temp_dir("backslash_subdir");
    let texture_dir = dir.join("textures");
    std::fs::create_dir_all(&texture_dir).expect("textures dir");
    write_test_png(&texture_dir.join("paint.png"));
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl painted\nKd 1 1 1\nmap_Kd textures\\paint.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");
    let gltf_path = dir.join("textured.gltf");

    convert_file(&obj_path, &gltf_path).expect("conversion must succeed");

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    assert!(
        scene.primitives[0]
            .material
            .base_color_texture
            .is_some(),
        "the backslash-authored texture under textures/ must still resolve"
    );
}

#[test]
fn materials_sharing_a_map_emit_one_image() {
    let dir = temp_dir("shared_texture");
    write_test_png(&dir.join("shared.png"));
    std::fs::write(
        dir.join("shared.mtl"),
        "newmtl a\nKd 1 0 0\nmap_Kd shared.png\n\nnewmtl b\nKd 0 0 1\nmap_Kd shared.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("shared.obj");
    std::fs::write(
        &obj_path,
        "mtllib shared.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n\
usemtl a\nf 1/1/1 2/2/1 3/3/1\nusemtl b\nf 1/1/1 3/3/1 4/4/1\n",
    )
    .expect("obj");

    let gltf_path = dir.join("shared.gltf");
    convert_file(&obj_path, &gltf_path).expect("convert");

    // Emitting one image per material makes the loader decode the same file
    // repeatedly and upload duplicate GPU textures.
    let json = std::fs::read_to_string(&gltf_path).expect("read gltf");
    let image_count = json.matches(r#""uri": "shared.png""#).count();
    assert_eq!(
        image_count, 1,
        "the shared texture should appear once, got {image_count}"
    );

    let scene = load_gltf(&gltf_path).expect("load");
    assert_eq!(scene.primitives.len(), 2);
    for primitive in &scene.primitives {
        assert!(
            primitive.material.base_color_texture.is_some(),
            "both materials should reference the shared texture"
        );
    }
}

#[test]
fn a_missing_texture_does_not_abort_the_conversion() {
    // OBJ files routinely reference maps that were never shipped with them.
    // Geometry and base colours are still worth converting.
    let dir = temp_dir("missing_texture");
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl painted\nKd 0.2 0.4 0.6\nmap_Kd absent.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");
    let gltf_path = dir.join("textured.gltf");

    let mesh = convert_file(&obj_path, &gltf_path).expect("conversion must still succeed");
    assert_eq!(mesh.triangle_count(), 1);
    assert!(
        (mesh.materials[0].base_color[2] - 0.6).abs() < 1e-6,
        "base colour must survive"
    );
}

#[test]
fn an_untextured_obj_emits_no_texture_arrays() {
    // Empty images/samplers/textures arrays are legal but noisy, and an empty
    // samplers array with a texture referencing sampler 0 would be invalid.
    let dir = temp_dir("untextured");
    let obj_path = dir.join("cube.obj");
    std::fs::write(&obj_path, CUBE_OBJ).expect("obj");
    let gltf_path = dir.join("cube.gltf");
    convert_file(&obj_path, &gltf_path).expect("convert");

    let json = std::fs::read_to_string(&gltf_path).expect("read");
    assert!(
        !json.contains("\"images\""),
        "no images array should be emitted"
    );
    assert!(
        !json.contains("\"textures\""),
        "no textures array should be emitted"
    );
    load_gltf(&gltf_path).expect("an untextured conversion must still load");
}

#[test]
fn a_material_name_with_json_metacharacters_still_produces_loadable_gltf() {
    // A quote and a trailing backslash in a material name would otherwise
    // terminate the JSON string early or escape the closing quote. Every
    // material declared in the .mtl reaches the output regardless of whether
    // any `usemtl` references it, so this does not need to match `painted`.
    let dir = temp_dir("quoted_material_name");
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl he said \"hi\"\\\\\nKd 1 0 0\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");
    let gltf_path = dir.join("quoted.gltf");

    convert_file(&obj_path, &gltf_path).expect("conversion must succeed");
    load_gltf(&gltf_path).expect("the converted glTF must load despite the quote/backslash");
}

#[test]
fn a_backslash_texture_path_does_not_corrupt_the_document() {
    // A Windows-authored `map_Kd textures\wood.png` must not be interpolated
    // as a raw JSON string, or `\w` becomes an invalid escape sequence. The
    // OBJ and glTF live in the same directory, so the referenced texture
    // resolves without needing the (separate) copy-next-to-output step.
    let dir = temp_dir("backslash_texture");
    let texture_dir = dir.join("textures");
    std::fs::create_dir_all(&texture_dir).expect("textures dir");
    write_test_png(&texture_dir.join("wood.png"));
    std::fs::write(
        dir.join("textured.mtl"),
        "newmtl painted\nKd 1 1 1\nmap_Kd textures\\wood.png\n",
    )
    .expect("mtl");
    let obj_path = dir.join("textured.obj");
    std::fs::write(&obj_path, TEXTURED_OBJ).expect("obj");
    let gltf_path = dir.join("textured.gltf");

    convert_file(&obj_path, &gltf_path).expect("conversion must still succeed");
    load_gltf(&gltf_path).expect("the converted glTF must load despite the backslash path");
}

#[test]
fn an_empty_mesh_emits_finite_accessor_bounds() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::ObjMesh;

    // `bounds()` returns +/-infinity when there is no geometry to fold over,
    // and `inf`/`NaN` are not valid JSON numbers.
    let (json, _bin) = to_gltf(&ObjMesh::default(), "empty.bin");

    assert!(!json.contains("inf"), "accessor bounds must not contain inf");
    assert!(!json.contains("NaN"), "accessor bounds must not contain NaN");
}

#[test]
fn per_vertex_colors_are_parsed_from_the_v_line() {
    let mesh = parse_obj("v 0 0 0 1 0 0\nv 1 0 0 1 0 0\nv 1 1 0 1 0 0\nf 1 2 3\n").expect("parse");
    for color in &mesh.colors {
        assert_eq!(*color, [1.0, 0.0, 0.0, 1.0]);
    }
    assert!(mesh.has_vertex_colors);
}

#[test]
fn a_seven_component_v_line_carries_alpha() {
    let mesh =
        parse_obj("v 0 0 0 0.2 0.4 0.6 0.5\nv 1 0 0 0.2 0.4 0.6 0.5\nv 1 1 0 0.2 0.4 0.6 0.5\nf 1 2 3\n")
            .expect("parse");
    for color in &mesh.colors {
        assert_eq!(*color, [0.2, 0.4, 0.6, 0.5]);
    }
}

#[test]
fn a_colorless_obj_has_no_vertex_colors_and_no_color_0() {
    let mesh = parse_obj(CUBE_OBJ).expect("parse");
    assert!(
        !mesh.has_vertex_colors,
        "a colourless OBJ must not be flagged as carrying colours"
    );

    let (json, _bin) = to_gltf(&mesh, "cube.bin");
    assert!(
        !json.contains("COLOR_0"),
        "no COLOR_0 attribute should be emitted for a colourless mesh"
    );
}

#[test]
fn a_colorless_obj_index_accessors_are_unchanged() {
    // Pins the index-accessor numbering for the common (colourless) case, so
    // adding the COLOR_0 attribute slot cannot silently repoint `indices`.
    let mesh = parse_obj(CUBE_OBJ).expect("parse");
    let (json, _bin) = to_gltf(&mesh, "cube.bin");

    assert!(
        json.contains(r#""indices": 3"#),
        "a colourless mesh's single primitive must still address index accessor 3: {json}"
    );
    assert!(
        json.contains(r#""bufferView": 3, "byteOffset": 0, "componentType": 5125"#),
        "the index accessor must still view bufferView 3: {json}"
    );
}

#[test]
fn a_vn_less_obj_gets_a_geometric_flat_normal_not_an_up_vector() {
    // Two coplanar triangles in the XZ plane, no `vn` at all. The old
    // fallback returned a fabricated (0, 1, 0) for every corner - which
    // happens to be right here by coincidence of the geometry, so this test
    // alone would not prove anything; see the XY-plane case below for the
    // one that actually distinguishes fabricated from geometric.
    let mesh = parse_obj("v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\nf 1 2 3\nf 1 3 4\n").expect("parse");
    assert!(!mesh.has_normals, "no vn line appeared in the source");
    for normal in &mesh.normals {
        assert!(
            (normal[1].abs() - 1.0).abs() < 1e-6
                && normal[0].abs() < 1e-6
                && normal[2].abs() < 1e-6,
            "expected (0, +/-1, 0), got {normal:?}"
        );
    }
}

#[test]
fn a_vn_less_obj_in_the_xy_plane_gets_a_z_normal_not_a_fabricated_up_vector() {
    // The case that red-proves the fix: the old code fabricated (0, 1, 0)
    // for every corner regardless of the triangle's actual orientation. A
    // triangle lying in the XY plane must come out normal-Z, not normal-Y.
    let mesh = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n").expect("parse");
    for normal in &mesh.normals {
        assert!(
            (normal[2].abs() - 1.0).abs() < 1e-6
                && normal[0].abs() < 1e-6
                && normal[1].abs() < 1e-6,
            "expected (0, 0, +/-1), got {normal:?}"
        );
    }
}

#[test]
fn a_mixed_obj_preserves_explicit_normals_and_fills_the_rest_geometrically() {
    // One face carries `vn`, the other does not. The explicit normal must
    // survive verbatim, and the vn-less face must get its own geometric
    // normal rather than either the other face's normal or a fabricated up
    // vector.
    let mesh = parse_obj(
        "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 0 0
v 1 0 0
v 1 0 1
vn 1.0 0.0 0.0
f 1//1 2//1 3//1
f 4 5 6
",
    )
    .expect("parse");
    assert!(mesh.has_normals);

    // First triangle: explicit normal preserved verbatim, even though it
    // does not match the triangle's actual (XY-plane) geometry - proving
    // this is the explicit `vn`, not a recomputed value.
    for i in 0..3 {
        assert_eq!(mesh.normals[i], [1.0, 0.0, 0.0]);
    }
    // Second triangle: no vn, lies in the XZ plane -> its own geometric flat
    // normal, not the first face's explicit normal and not a fabricated up
    // vector.
    for i in 3..6 {
        let n = mesh.normals[i];
        assert!(
            (n[1].abs() - 1.0).abs() < 1e-6 && n[0].abs() < 1e-6 && n[2].abs() < 1e-6,
            "expected the second face's own geometric normal (0, +/-1, 0), got {n:?}"
        );
    }
}

#[test]
fn parse_mtl_reads_map_bump_as_the_normal_texture() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    let materials = parse_mtl("newmtl a\nmap_Bump rock_normal.png\n");
    assert_eq!(
        materials[0].normal_texture.as_deref(),
        Some("rock_normal.png")
    );
}

#[test]
fn parse_mtl_prefers_norm_over_map_bump() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    // norm first, map_Bump second: the later, less-specific directive must
    // not override it.
    let norm_first = parse_mtl("newmtl a\nnorm better.png\nmap_Bump worse.png\n");
    assert_eq!(norm_first[0].normal_texture.as_deref(), Some("better.png"));

    // map_Bump first, norm second: norm always wins regardless of order.
    let bump_first = parse_mtl("newmtl a\nmap_Bump worse.png\nnorm better.png\n");
    assert_eq!(bump_first[0].normal_texture.as_deref(), Some("better.png"));
}

#[test]
fn parse_mtl_takes_the_last_token_of_an_option_carrying_map_bump() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    // MTL allows options before the path, and a bump directive's `-bm` option
    // sets glTF's normalTexture.scale.
    let materials = parse_mtl("newmtl a\nmap_Bump -bm 0.5 rock_normal.png\n");
    assert_eq!(
        materials[0].normal_texture.as_deref(),
        Some("rock_normal.png"),
        "the filename must be taken from the end, past the -bm option"
    );
    assert!(
        (materials[0].normal_scale - 0.5).abs() < 1e-6,
        "the -bm option must set normal_scale, got {}",
        materials[0].normal_scale
    );
}

#[test]
fn to_gltf_emits_a_normal_texture_pointing_at_a_distinct_image_when_the_maps_differ() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nmap_Kd base.png\nmap_Bump normal.png\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");

    assert!(
        json.contains(r#""baseColorTexture": { "index": 0 }"#),
        "expected the base colour texture at image index 0, got: {json}"
    );
    assert!(
        json.contains(r#""normalTexture": { "index": 1, "scale": 1 }"#),
        "expected a distinct normalTexture image index, got: {json}"
    );
}

#[test]
fn to_gltf_shares_one_image_when_map_kd_and_map_bump_name_the_same_file() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nmap_Kd shared.png\nmap_Bump shared.png\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");

    assert!(
        json.contains(r#""normalTexture": { "index": 0, "scale": 1 }"#),
        "expected the base colour and normal maps to share image index 0, got: {json}"
    );
    assert_eq!(
        json.matches(r#""uri": "shared.png""#).count(),
        1,
        "the shared file must appear once in the images array, got: {json}"
    );
}

#[test]
fn parse_mtl_takes_the_last_token_of_an_option_carrying_map_ke() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::parse_mtl;

    // Same option-skipping rule as map_Kd/map_Bump: the path is the LAST
    // token, not the first.
    let materials = parse_mtl("newmtl a\nmap_Ke -s 1 1 1 glow.png\n");
    assert_eq!(
        materials[0].emissive_texture.as_deref(),
        Some("glow.png"),
        "the filename must be taken from the end, past the options"
    );
}

#[test]
fn to_gltf_emits_an_emissive_texture_pointing_at_the_right_image() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nmap_Kd base.png\nmap_Ke glow.png\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");

    assert!(
        json.contains(r#""baseColorTexture": { "index": 0 }"#),
        "expected the base colour texture at image index 0, got: {json}"
    );
    assert!(
        json.contains(r#""emissiveTexture": { "index": 1 }"#),
        "expected a distinct emissiveTexture image index, got: {json}"
    );
}

#[test]
fn a_map_ke_without_ke_gets_a_normalised_emissive_factor() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    // glTF's default emissiveFactor is [0,0,0] - a material with a map_Ke and
    // no Ke line would render black without this.
    let materials = parse_mtl("newmtl a\nmap_Ke glow.png\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");

    assert!(
        json.contains(r#""emissiveFactor": [1, 1, 1]"#),
        "expected a normalised emissiveFactor of [1, 1, 1], got: {json}"
    );
}

#[test]
fn to_gltf_shares_one_image_when_map_kd_and_map_ke_name_the_same_file() {
    use kataglyphis_webgpu_renderer::asset::obj_to_gltf::{parse_mtl, ObjMesh};

    let materials = parse_mtl("newmtl a\nmap_Kd wood.png\nmap_Ke wood.png\n");
    let mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let (json, _bin) = to_gltf(&mesh, "a.bin");

    assert_eq!(
        json.matches(r#""uri": "wood.png""#).count(),
        1,
        "the shared file must appear once in the images array, got: {json}"
    );
}

#[test]
fn vertex_colors_round_trip_through_the_real_gltf_loader() {
    let dir = temp_dir("vertex_colors");
    let obj_path = dir.join("triangle.obj");
    std::fs::write(
        &obj_path,
        "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 1 1 0 0 0 1\nf 1 2 3\n",
    )
    .expect("write obj");
    let gltf_path = dir.join("triangle.gltf");

    let source = convert_file(&obj_path, &gltf_path).expect("conversion must succeed");
    assert!(source.has_vertex_colors);

    let scene = load_gltf(&gltf_path).expect("the converted glTF must load");
    let loaded = &scene.primitives[0];
    assert_eq!(loaded.vertices.len(), source.colors.len());
    for (index, vertex) in loaded.vertices.iter().enumerate() {
        for channel in 0..4 {
            assert!(
                (vertex.color[channel] - source.colors[index][channel]).abs() < 1e-6,
                "vertex {index} colour differs on channel {channel}: {:?} vs {:?}",
                vertex.color,
                source.colors[index]
            );
        }
    }
}
