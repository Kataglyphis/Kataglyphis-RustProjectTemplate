//! glTF 2.0 -> `CpuScene`. Covers milestone 2 of the plan: positions,
//! normals, UVs, indices, flattened node transforms, and per-primitive
//! base-color materials. Textures/PBR maps arrive with milestone 3.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use glam::{Mat4, Vec3};

use crate::scene::{CpuMaterial, CpuPrimitive, CpuScene, CpuTexture, Vertex};

pub fn load_gltf(path: impl AsRef<Path>) -> anyhow::Result<CpuScene> {
    let path = path.as_ref();
    let (document, buffers, images) = gltf::import(path)
        .with_context(|| format!("Failed to import glTF file: {}", path.display()))?;

    let textures: Vec<Arc<CpuTexture>> = images
        .into_iter()
        .map(|img| to_rgba8(img).map(Arc::new))
        .collect::<anyhow::Result<_>>()?;

    let mut scene = CpuScene::default();

    let gltf_scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF file contains no scenes")?;

    for node in gltf_scene.nodes() {
        visit_node(&node, Mat4::IDENTITY, &buffers, &textures, &mut scene)?;
    }

    anyhow::ensure!(
        !scene.primitives.is_empty(),
        "glTF file contains no triangle primitives: {}",
        path.display()
    );

    Ok(scene)
}

/// Converts a decoded glTF image (any of the formats the `gltf` crate emits)
/// into tightly packed RGBA8.
fn to_rgba8(img: gltf::image::Data) -> anyhow::Result<CpuTexture> {
    use gltf::image::Format;

    let pixel_count = (img.width * img.height) as usize;
    let rgba8 = match img.format {
        Format::R8G8B8A8 => img.pixels,
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for rgb in img.pixels.chunks_exact(3) {
                out.extend_from_slice(rgb);
                out.push(255);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &r in &img.pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for rg in img.pixels.chunks_exact(2) {
                out.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            out
        }
        other => anyhow::bail!("Unsupported glTF image format: {other:?}"),
    };

    anyhow::ensure!(
        rgba8.len() == pixel_count * 4,
        "Image byte count mismatch after RGBA8 conversion"
    );

    Ok(CpuTexture {
        width: img.width,
        height: img.height,
        rgba8,
    })
}

fn visit_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &[Arc<CpuTexture>],
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
            if let Some(cpu) = load_primitive(&primitive, world, buffers, textures)? {
                scene.primitives.push(cpu);
            }
        }
    }

    for child in node.children() {
        visit_node(&child, world, buffers, textures, scene)?;
    }
    Ok(())
}

fn load_primitive(
    primitive: &gltf::Primitive,
    transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &[Arc<CpuTexture>],
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

    let pbr = primitive.material().pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();
    let base_color_texture = pbr
        .base_color_texture()
        .and_then(|info| textures.get(info.texture().source().index()).cloned());

    Ok(Some(CpuPrimitive {
        vertices,
        indices,
        transform,
        material: CpuMaterial {
            base_color,
            base_color_texture,
        },
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
