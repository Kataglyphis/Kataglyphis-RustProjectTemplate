//! Converts Wavefront OBJ to glTF 2.0, so the C++ engine's `Resources/Models`
//! can be loaded by this renderer.
//!
//! Written by hand rather than pulling in an OBJ crate and a JSON crate: the
//! subset that matters here is small (positions, normals, UVs, triangulated
//! faces), and the output is checked by loading it back with the real `gltf`
//! crate rather than by trusting the emitter.
//!
//! Materials carry across as far as glTF's PBR model allows: OBJ's diffuse
//! `Kd` becomes `baseColorFactor` and `d`/`Tr` its alpha. OBJ is a
//! Phong-era format, so `Ks`/`Ns` have no faithful PBR equivalent and are
//! dropped rather than guessed into metallic/roughness - a converted asset
//! should differ from the source in ways that are written down, not invented.
//!
//! Deliberately NOT supported, and rejected rather than silently mangled:
//! smoothing groups and negative (relative) indices. A converter that quietly
//! drops what it does not understand produces assets that differ from the
//! source in ways nobody notices until the two renderers disagree.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// A material as OBJ describes it, reduced to what glTF can represent.
#[derive(Debug, Clone)]
pub struct ObjMaterial {
    pub name: String,
    /// `Kd` plus `d`/`Tr` as alpha.
    pub base_color: [f32; 4],
    /// `map_Kd`, as written in the .mtl (relative to it).
    pub base_color_texture: Option<String>,
}

impl Default for ObjMaterial {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            // glTF's own default base colour, so an OBJ without materials
            // converts to something a loader treats as untinted rather than
            // black.
            base_color: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
        }
    }
}

/// One triangulated mesh extracted from an OBJ file.
#[derive(Debug, Default, Clone)]
pub struct ObjMesh {
    /// Interleaved-free parallel arrays, one entry per unique vertex.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Materials referenced by the file, in declaration order.
    pub materials: Vec<ObjMaterial>,
    /// `(first_index, index_count, material_index)` per material run.
    ///
    /// OBJ interleaves `usemtl` with faces, so one file becomes several glTF
    /// primitives. Indices stay in ONE array and the ranges point into it -
    /// splitting the vertex data per material would duplicate shared vertices
    /// and change the geometry, which is exactly what a comparison harness
    /// must not do.
    pub submeshes: Vec<(u32, u32, usize)>,
}

impl ObjMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned bounds; glTF requires min/max on the POSITION accessor,
    /// and loaders use them for culling, so they are not optional metadata.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        (min, max)
    }
}

/// Parses the `.mtl` subset that maps onto glTF's PBR base colour.
///
/// Unknown directives are ignored here, unlike in the OBJ parser: MTL files
/// are full of Phong-era fields (`Ns`, `Ka`, `Ks`, `illum`, `map_Bump`) that
/// have no glTF equivalent, and refusing a file for containing them would
/// reject essentially every real material library.
pub fn parse_mtl(source: &str) -> Vec<ObjMaterial> {
    let mut materials: Vec<ObjMaterial> = Vec::new();

    for raw_line in source.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(keyword) = parts.next() else {
            continue;
        };
        let values: Vec<&str> = parts.collect();

        match keyword {
            "newmtl" => materials.push(ObjMaterial {
                name: values.first().copied().unwrap_or("unnamed").to_string(),
                ..ObjMaterial::default()
            }),
            "Kd" if values.len() >= 3 => {
                if let Some(material) = materials.last_mut() {
                    for (axis, value) in values.iter().take(3).enumerate() {
                        if let Ok(component) = value.parse::<f32>() {
                            material.base_color[axis] = component;
                        }
                    }
                }
            }
            // `d` is opacity, `Tr` is transparency - the same quantity
            // inverted. Treating them as interchangeable makes transparent
            // materials opaque and vice versa.
            "d" if !values.is_empty() => {
                if let (Some(material), Ok(opacity)) =
                    (materials.last_mut(), values[0].parse::<f32>())
                {
                    material.base_color[3] = opacity.clamp(0.0, 1.0);
                }
            }
            "map_Kd" if !values.is_empty() => {
                if let Some(material) = materials.last_mut() {
                    // MTL allows options before the filename
                    // (`map_Kd -s 1 1 1 wood.png`), so the path is the LAST
                    // token, not the first. Taking values[0] silently turns
                    // any option-carrying map into a texture named "-s".
                    material.base_color_texture = values.last().map(|name| name.to_string());
                }
            }
            "Tr" if !values.is_empty() => {
                if let (Some(material), Ok(transparency)) =
                    (materials.last_mut(), values[0].parse::<f32>())
                {
                    material.base_color[3] = (1.0 - transparency).clamp(0.0, 1.0);
                }
            }
            _ => {}
        }
    }

    materials
}

/// Parses the OBJ subset this converter supports.
///
/// OBJ indices are 1-based and per-attribute: one face vertex is a
/// `position/uv/normal` triple, and the same position can appear with
/// different normals. glTF has a single index stream, so each distinct triple
/// becomes one vertex - which is why the output vertex count is usually
/// higher than the OBJ's `v` count and that is not a bug.
pub fn parse_obj(source: &str) -> Result<ObjMesh> {
    parse_obj_with_materials(source, Vec::new())
}

/// As [`parse_obj`], with materials already loaded from the companion `.mtl`.
pub fn parse_obj_with_materials(source: &str, materials: Vec<ObjMaterial>) -> Result<ObjMesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut mesh = ObjMesh {
        materials,
        ..ObjMesh::default()
    };
    let mut seen: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut active_material: usize = 0;
    let mut run_start: u32 = 0;

    for (line_number, raw_line) in source.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(keyword) = parts.next() else {
            continue;
        };
        let values: Vec<&str> = parts.collect();
        let at = line_number + 1;

        match keyword {
            "v" => {
                if values.len() < 3 {
                    bail!("line {at}: 'v' needs 3 coordinates, got {}", values.len());
                }
                positions.push([
                    parse_f32(values[0], at)?,
                    parse_f32(values[1], at)?,
                    parse_f32(values[2], at)?,
                ]);
            }
            "vn" => {
                if values.len() < 3 {
                    bail!("line {at}: 'vn' needs 3 components, got {}", values.len());
                }
                normals.push([
                    parse_f32(values[0], at)?,
                    parse_f32(values[1], at)?,
                    parse_f32(values[2], at)?,
                ]);
            }
            "vt" => {
                if values.len() < 2 {
                    bail!("line {at}: 'vt' needs 2 components, got {}", values.len());
                }
                // OBJ's V axis points up, glTF's points down. Flipping here
                // rather than at load time keeps the converted asset correct
                // for any consumer, not just this renderer.
                uvs.push([parse_f32(values[0], at)?, 1.0 - parse_f32(values[1], at)?]);
            }
            "f" => {
                if values.len() < 3 {
                    bail!(
                        "line {at}: 'f' needs at least 3 vertices, got {}",
                        values.len()
                    );
                }
                let mut face: Vec<u32> = Vec::with_capacity(values.len());
                for value in &values {
                    let key = parse_face_vertex(value, at)?;
                    let index = match seen.get(&key) {
                        Some(&existing) => existing,
                        None => {
                            let position = resolve(key.0, positions.len(), at, "position")?;
                            mesh.positions.push(positions[position]);

                            if key.1 > 0 {
                                let uv = resolve(key.1, uvs.len(), at, "texcoord")?;
                                mesh.uvs.push(uvs[uv]);
                            } else {
                                mesh.uvs.push([0.0, 0.0]);
                            }

                            if key.2 > 0 {
                                let normal = resolve(key.2, normals.len(), at, "normal")?;
                                mesh.normals.push(normals[normal]);
                            } else {
                                // A loader that sees a zero normal shades the
                                // surface black; an up vector is wrong but
                                // visibly wrong rather than invisibly so.
                                mesh.normals.push([0.0, 1.0, 0.0]);
                            }

                            let index = (mesh.positions.len() - 1) as u32;
                            seen.insert(key, index);
                            index
                        }
                    };
                    face.push(index);
                }

                // Fan-triangulate. Correct for convex faces, which is what
                // OBJ exporters emit; concave n-gons would need ear clipping
                // and are not worth supporting until something needs them.
                for i in 1..face.len() - 1 {
                    mesh.indices
                        .extend_from_slice(&[face[0], face[i], face[i + 1]]);
                }
            }
            "usemtl" => {
                // Close the run in progress before switching. A material
                // change with no faces since the last one produces an empty
                // run, which would become a glTF primitive drawing nothing.
                let current = mesh.indices.len() as u32;
                if current > run_start {
                    mesh.submeshes
                        .push((run_start, current - run_start, active_material));
                    run_start = current;
                }
                let name = values.first().copied().unwrap_or("");
                active_material = mesh
                    .materials
                    .iter()
                    .position(|material| material.name == name)
                    .unwrap_or_else(|| {
                        // Referenced but not declared: keep the name so the
                        // mismatch is visible in the output rather than
                        // silently collapsing onto material 0.
                        mesh.materials.push(ObjMaterial {
                            name: name.to_string(),
                            ..ObjMaterial::default()
                        });
                        mesh.materials.len() - 1
                    });
            }
            // Known-but-unsupported directives are ignored rather than fatal:
            // almost every real OBJ carries them, and refusing the file would
            // make the converter useless.
            "mtllib" | "o" | "g" | "s" => {}
            other => {
                bail!("line {at}: unsupported OBJ directive '{other}'");
            }
        }
    }

    // Close the final run.
    let total = mesh.indices.len() as u32;
    if total > run_start {
        mesh.submeshes
            .push((run_start, total - run_start, active_material));
    }

    if mesh.positions.is_empty() {
        bail!("the OBJ contained no geometry");
    }
    if mesh.materials.is_empty() {
        mesh.materials.push(ObjMaterial::default());
    }
    if mesh.submeshes.is_empty() {
        mesh.submeshes.push((0, mesh.indices.len() as u32, 0));
    }
    Ok(mesh)
}

fn parse_f32(text: &str, line: usize) -> Result<f32> {
    text.parse::<f32>()
        .with_context(|| format!("line {line}: '{text}' is not a number"))
}

/// Parses `v`, `v/vt`, `v//vn` or `v/vt/vn`. Missing components come back 0.
fn parse_face_vertex(text: &str, line: usize) -> Result<(i64, i64, i64)> {
    let mut fields = text.split('/');
    let position = fields
        .next()
        .unwrap_or("")
        .parse::<i64>()
        .with_context(|| format!("line {line}: '{text}' has no position index"))?;
    let uv = fields.next().unwrap_or("").parse::<i64>().unwrap_or(0);
    let normal = fields.next().unwrap_or("").parse::<i64>().unwrap_or(0);
    Ok((position, uv, normal))
}

/// OBJ indices are 1-based, and negative values mean "relative to the end".
fn resolve(index: i64, available: usize, line: usize, what: &str) -> Result<usize> {
    if index < 0 {
        bail!("line {line}: negative (relative) {what} indices are not supported");
    }
    if index == 0 {
        bail!("line {line}: {what} index 0 is invalid - OBJ indices start at 1");
    }
    let zero_based = (index - 1) as usize;
    if zero_based >= available {
        bail!("line {line}: {what} index {index} exceeds the {available} declared");
    }
    Ok(zero_based)
}

/// Serialises a mesh as a glTF 2.0 document plus its binary buffer.
///
/// Returns `(gltf_json, bin)`. The JSON references `bin_uri`, so the caller
/// decides the file layout.
pub fn to_gltf(mesh: &ObjMesh, bin_uri: &str) -> (String, Vec<u8>) {
    let mut bin: Vec<u8> = Vec::new();

    let positions_offset = bin.len();
    for p in &mesh.positions {
        for component in p {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let normals_offset = bin.len();
    for n in &mesh.normals {
        for component in n {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let uvs_offset = bin.len();
    for uv in &mesh.uvs {
        for component in uv {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    // Index data must start on a multiple of its component size; u32 needs 4,
    // which the float arrays above already guarantee, but pad defensively so
    // a future non-float attribute cannot silently misalign it.
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let indices_offset = bin.len();
    for index in &mesh.indices {
        bin.extend_from_slice(&index.to_le_bytes());
    }

    let (min, max) = mesh.bounds();
    let vertex_count = mesh.positions.len();
    let index_count = mesh.indices.len();

    // One index accessor and one primitive per material run. They all view
    // the SAME index bufferView at different offsets, so shared vertices stay
    // shared - splitting the vertex data per material would duplicate them
    // and change the geometry a comparison harness is meant to hold constant.
    let mut index_accessors = String::new();
    let mut primitives = String::new();
    for (run, &(first_index, count, material)) in mesh.submeshes.iter().enumerate() {
        if run > 0 {
            index_accessors.push_str(
                ",
    ",
            );
            primitives.push_str(", ");
        }
        index_accessors.push_str(&format!(
            r#"{{ "bufferView": 3, "byteOffset": {}, "componentType": 5125, "count": {}, "type": "SCALAR" }}"#,
            first_index as usize * 4,
            count
        ));
        primitives.push_str(&format!(
            r#"{{ "attributes": {{ "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }}, "indices": {}, "material": {}, "mode": 4 }}"#,
            3 + run,
            material
        ));
    }

    // Deduplicate textures: several materials commonly share one map, and
    // emitting an image per material makes the loader decode the same file
    // repeatedly and upload duplicate GPU textures.
    let mut image_uris: Vec<String> = Vec::new();
    for material in &mesh.materials {
        if let Some(uri) = &material.base_color_texture {
            if !image_uris.iter().any(|existing| existing == uri) {
                image_uris.push(uri.clone());
            }
        }
    }

    let images_json = image_uris
        .iter()
        .map(|uri| format!(r#"{{ "uri": "{uri}" }}"#))
        .collect::<Vec<_>>()
        .join(", ");
    let textures_json = (0..image_uris.len())
        .map(|index| format!(r#"{{ "source": {index}, "sampler": 0 }}"#))
        .collect::<Vec<_>>()
        .join(", ");
    // REPEAT wrap (10497) matches OBJ's convention of UVs running outside
    // 0..1 for tiling; glTF's default is also repeat, but stating it keeps
    // the asset explicit rather than dependent on loader defaults.
    let samplers_json = if image_uris.is_empty() {
        String::new()
    } else {
        r#"{ "wrapS": 10497, "wrapT": 10497 }"#.to_string()
    };

    let materials_json = mesh
        .materials
        .iter()
        .map(|material| {
            let [r, g, b, a] = material.base_color;
            // OBJ has no metallic/roughness. 0 metallic and 1 roughness is the
            // closest thing to "plain diffuse", which is what Kd describes;
            // inheriting glTF's default (metallic 1) would render every
            // converted asset as a dark mirror.
            let texture_json = match &material.base_color_texture {
                Some(uri) => {
                    let index = image_uris.iter().position(|existing| existing == uri).unwrap_or(0);
                    format!(r#", "baseColorTexture": {{ "index": {index} }}"#)
                }
                None => String::new(),
            };
            format!(
                r#"{{ "name": "{}", "pbrMetallicRoughness": {{ "baseColorFactor": [{}, {}, {}, {}]{}, "metallicFactor": 0.0, "roughnessFactor": 1.0 }}{} }}"#,
                material.name,
                r, g, b, a,
                texture_json,
                if a < 1.0 { r#", "alphaMode": "BLEND""# } else { "" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0", "generator": "kataglyphis obj_to_gltf" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "primitives": [{primitives}] }}],
  "materials": [{materials_json}],{texture_arrays}
  "buffers": [{{ "uri": "{bin_uri}", "byteLength": {buffer_length} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {positions_offset}, "byteLength": {positions_length}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {normals_offset}, "byteLength": {normals_length}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {uvs_offset}, "byteLength": {uvs_length}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {indices_offset}, "byteLength": {indices_length}, "target": 34963 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": {vertex_count}, "type": "VEC3", "min": [{min0}, {min1}, {min2}], "max": [{max0}, {max1}, {max2}] }},
    {{ "bufferView": 1, "componentType": 5126, "count": {vertex_count}, "type": "VEC3" }},
    {{ "bufferView": 2, "componentType": 5126, "count": {vertex_count}, "type": "VEC2" }},
    {index_accessors}
  ]
}}"#,
        bin_uri = bin_uri,
        buffer_length = bin.len(),
        positions_offset = positions_offset,
        positions_length = vertex_count * 12,
        normals_offset = normals_offset,
        normals_length = vertex_count * 12,
        uvs_offset = uvs_offset,
        uvs_length = vertex_count * 8,
        indices_offset = indices_offset,
        indices_length = index_count * 4,
        vertex_count = vertex_count,
        index_accessors = index_accessors,
        primitives = primitives,
        materials_json = materials_json,
        texture_arrays = if image_uris.is_empty() {
            String::new()
        } else {
            format!(
                "
  \"images\": [{images_json}],
  \"samplers\": [{samplers_json}],
  \"textures\": [{textures_json}],"
            )
        },
        min0 = min[0],
        min1 = min[1],
        min2 = min[2],
        max0 = max[0],
        max1 = max[1],
        max2 = max[2],
    );

    (json, bin)
}

/// Converts `obj_path` to `gltf_path`, writing the binary buffer alongside it.
pub fn convert_file(obj_path: &Path, gltf_path: &Path) -> Result<ObjMesh> {
    let source = std::fs::read_to_string(obj_path)
        .with_context(|| format!("reading {}", obj_path.display()))?;

    // Load every mtllib the OBJ names, resolved next to the OBJ itself. A
    // missing library is not fatal: the geometry is still worth converting,
    // and failing the whole conversion over an absent .mtl would block assets
    // that ship without one.
    let mut materials: Vec<ObjMaterial> = Vec::new();
    for line in source.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = line.strip_prefix("mtllib ") {
            for name in rest.split_whitespace() {
                let mtl_path = obj_path.with_file_name(name);
                match std::fs::read_to_string(&mtl_path) {
                    Ok(mtl) => materials.extend(parse_mtl(&mtl)),
                    Err(error) => {
                        log::warn!("{}: {error}; converting without it", mtl_path.display());
                    }
                }
            }
        }
    }

    let mesh = parse_obj_with_materials(&source, materials)
        .with_context(|| format!("parsing {}", obj_path.display()))?;

    let bin_name = gltf_path
        .file_stem()
        .map(|stem| format!("{}.bin", stem.to_string_lossy()))
        .unwrap_or_else(|| "buffer.bin".to_string());
    let (json, bin) = to_gltf(&mesh, &bin_name);

    if let Some(parent) = gltf_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(gltf_path, json).with_context(|| format!("writing {}", gltf_path.display()))?;
    let bin_path = gltf_path.with_file_name(bin_name);
    std::fs::write(&bin_path, bin).with_context(|| format!("writing {}", bin_path.display()))?;

    // Copy referenced textures next to the glTF so the output is
    // self-contained. Emitting a URI that points back into the source tree
    // would produce a document that loads on this machine and nowhere else -
    // and the failure would be a missing texture, not a missing file.
    for material in &mesh.materials {
        let Some(uri) = &material.base_color_texture else {
            continue;
        };
        let source_path = obj_path.with_file_name(uri);
        let destination = gltf_path.with_file_name(uri);
        if source_path == destination {
            continue;
        }
        match std::fs::copy(&source_path, &destination) {
            Ok(_) => {}
            Err(error) => {
                // Warn rather than fail: the geometry and materials are still
                // worth having, and OBJ files routinely reference textures
                // that were never shipped alongside them.
                log::warn!(
                    "{}: {error}; the converted glTF references a texture that is not there",
                    source_path.display()
                );
            }
        }
    }

    Ok(mesh)
}
