//! glTF 2.0 -> `CpuScene`. Covers milestone 2 of the plan: positions,
//! normals, UVs, indices, flattened node transforms, and per-primitive
//! base-color materials. Textures/PBR maps arrive with milestone 3.

use std::path::Path;

use anyhow::Context as _;
use glam::{Mat4, Vec3};

use crate::scene::{CpuMaterial, CpuPrimitive, CpuScene, Vertex};

pub fn load_gltf(path: impl AsRef<Path>) -> anyhow::Result<CpuScene> {
    let path = path.as_ref();
    let (document, buffers, _images) = gltf::import(path)
        .with_context(|| format!("Failed to import glTF file: {}", path.display()))?;

    let mut scene = CpuScene::default();

    let gltf_scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF file contains no scenes")?;

    for node in gltf_scene.nodes() {
        visit_node(&node, Mat4::IDENTITY, &buffers, &mut scene)?;
    }

    anyhow::ensure!(
        !scene.primitives.is_empty(),
        "glTF file contains no triangle primitives: {}",
        path.display()
    );

    Ok(scene)
}

fn visit_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    scene: &mut CpuScene,
) -> anyhow::Result<()> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "Skipping non-triangle primitive (mode {:?}) in mesh {:?}",
                    primitive.mode(),
                    mesh.name().unwrap_or("<unnamed>")
                );
                continue;
            }
            if let Some(cpu) = load_primitive(&primitive, world, buffers)? {
                scene.primitives.push(cpu);
            }
        }
    }

    for child in node.children() {
        visit_node(&child, world, buffers, scene)?;
    }
    Ok(())
}

fn load_primitive(
    primitive: &gltf::Primitive,
    transform: Mat4,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<Option<CpuPrimitive>> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|b| &b.0[..]));

    let Some(positions) = reader.read_positions() else {
        return Ok(None);
    };
    let positions: Vec<[f32; 3]> = positions.collect();

    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(iter) => iter.collect(),
        None => vec![[0.0, 0.0, 0.0]; positions.len()],
    };
    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(iter) => iter.into_f32().collect(),
        None => vec![[0.0, 0.0]; positions.len()],
    };

    anyhow::ensure!(
        normals.len() == positions.len() && uvs.len() == positions.len(),
        "Attribute count mismatch: {} positions, {} normals, {} uvs",
        positions.len(),
        normals.len(),
        uvs.len()
    );

    let mut vertices: Vec<Vertex> = positions
        .iter()
        .zip(normals.iter())
        .zip(uvs.iter())
        .map(|((p, n), t)| Vertex {
            position: *p,
            normal: *n,
            uv: *t,
        })
        .collect();

    let indices: Vec<u32> = match reader.read_indices() {
        Some(iter) => iter.into_u32().collect(),
        None => (0..vertices.len() as u32).collect(),
    };

    // Missing normals: derive flat face normals so lighting stays sane.
    if reader.read_normals().is_none() {
        compute_flat_normals(&mut vertices, &indices);
    }

    let base_color = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_factor();

    Ok(Some(CpuPrimitive {
        vertices,
        indices,
        transform,
        material: CpuMaterial { base_color },
    }))
}

fn compute_flat_normals(vertices: &mut [Vertex], indices: &[u32]) {
    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p0 = Vec3::from_array(vertices[i0].position);
        let p1 = Vec3::from_array(vertices[i1].position);
        let p2 = Vec3::from_array(vertices[i2].position);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array();
        for i in [i0, i1, i2] {
            vertices[i].normal = n;
        }
    }
}
