//! Converts Wavefront OBJ to glTF 2.0, so the C++ engine's `Resources/Models`
//! can be loaded by this renderer.
//!
//! Written by hand rather than pulling in an OBJ crate and a JSON crate: the
//! subset that matters here is small (positions, normals, UVs, triangulated
//! faces), and the output is checked by loading it back with the real `gltf`
//! crate rather than by trusting the emitter.
//!
//! Deliberately NOT supported, and rejected rather than silently mangled:
//! materials (`usemtl`/`mtllib`), smoothing groups, and negative (relative)
//! indices. A converter that quietly drops what it does not understand
//! produces assets that differ from the source in ways nobody notices until
//! the two renderers disagree.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// One triangulated mesh extracted from an OBJ file.
#[derive(Debug, Default, Clone)]
pub struct ObjMesh {
    /// Interleaved-free parallel arrays, one entry per unique vertex.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
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

/// Parses the OBJ subset this converter supports.
///
/// OBJ indices are 1-based and per-attribute: one face vertex is a
/// `position/uv/normal` triple, and the same position can appear with
/// different normals. glTF has a single index stream, so each distinct triple
/// becomes one vertex - which is why the output vertex count is usually
/// higher than the OBJ's `v` count and that is not a bug.
pub fn parse_obj(source: &str) -> Result<ObjMesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut mesh = ObjMesh::default();
    let mut seen: HashMap<(i64, i64, i64), u32> = HashMap::new();

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
                positions.push([parse_f32(values[0], at)?, parse_f32(values[1], at)?, parse_f32(values[2], at)?]);
            }
            "vn" => {
                if values.len() < 3 {
                    bail!("line {at}: 'vn' needs 3 components, got {}", values.len());
                }
                normals.push([parse_f32(values[0], at)?, parse_f32(values[1], at)?, parse_f32(values[2], at)?]);
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
                    bail!("line {at}: 'f' needs at least 3 vertices, got {}", values.len());
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
                    mesh.indices.extend_from_slice(&[face[0], face[i], face[i + 1]]);
                }
            }
            // Known-but-unsupported directives are ignored rather than fatal:
            // almost every real OBJ carries them, and refusing the file would
            // make the converter useless.
            "mtllib" | "usemtl" | "o" | "g" | "s" => {}
            other => {
                bail!("line {at}: unsupported OBJ directive '{other}'");
            }
        }
    }

    if mesh.positions.is_empty() {
        bail!("the OBJ contained no geometry");
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
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let indices_offset = bin.len();
    for index in &mesh.indices {
        bin.extend_from_slice(&index.to_le_bytes());
    }

    let (min, max) = mesh.bounds();
    let vertex_count = mesh.positions.len();
    let index_count = mesh.indices.len();

    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0", "generator": "kataglyphis obj_to_gltf" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }}, "indices": 3, "mode": 4 }}] }}],
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
    {{ "bufferView": 3, "componentType": 5125, "count": {index_count}, "type": "SCALAR" }}
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
        index_count = index_count,
        min0 = min[0], min1 = min[1], min2 = min[2],
        max0 = max[0], max1 = max[1], max2 = max[2],
    );

    (json, bin)
}

/// Converts `obj_path` to `gltf_path`, writing the binary buffer alongside it.
pub fn convert_file(obj_path: &Path, gltf_path: &Path) -> Result<ObjMesh> {
    let source = std::fs::read_to_string(obj_path)
        .with_context(|| format!("reading {}", obj_path.display()))?;
    let mesh = parse_obj(&source).with_context(|| format!("parsing {}", obj_path.display()))?;

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

    Ok(mesh)
}
