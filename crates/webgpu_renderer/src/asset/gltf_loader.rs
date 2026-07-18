//! glTF 2.0 -> `CpuScene`. Positions, normals, UVs, tangents (generated when
//! absent), indices, flattened node transforms, and the metallic-roughness
//! material model: factors + base color / metallic-roughness / normal /
//! emissive / occlusion textures with their glTF sampler settings.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use glam::{Mat4, Vec2, Vec3};

use crate::scene::{
    AlphaMode, CpuLight, CpuLightKind, CpuMaterial, CpuPrimitive, CpuSampler, CpuScene, CpuTexture,
    CpuTextureRef, CpuWrap, Vertex,
};

pub fn load_gltf(path: impl AsRef<Path>) -> anyhow::Result<CpuScene> {
    let path = path.as_ref();
    let (document, buffers, images) = gltf::import(path)
        .with_context(|| format!("Failed to import glTF file: {}", path.display()))?;
    build_scene(document, buffers, images)
        .with_context(|| format!("Failed to build scene from {}", path.display()))
}

/// In-memory variant (embedded assets, wasm32 where there is no filesystem).
/// Buffers must be embedded (data URIs or GLB binary chunk).
pub fn load_gltf_slice(bytes: &[u8]) -> anyhow::Result<CpuScene> {
    let (document, buffers, images) =
        gltf::import_slice(bytes).context("Failed to import glTF from memory")?;
    build_scene(document, buffers, images)
}

fn build_scene(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
) -> anyhow::Result<CpuScene> {
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
        "glTF file contains no triangle primitives"
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

fn to_cpu_sampler(sampler: &gltf::texture::Sampler) -> CpuSampler {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};

    let wrap = |mode: WrappingMode| match mode {
        WrappingMode::ClampToEdge => CpuWrap::ClampToEdge,
        WrappingMode::MirroredRepeat => CpuWrap::MirroredRepeat,
        WrappingMode::Repeat => CpuWrap::Repeat,
    };

    let (min_nearest, mip_nearest) = match sampler.min_filter() {
        Some(MinFilter::Nearest | MinFilter::NearestMipmapNearest) => (true, true),
        Some(MinFilter::NearestMipmapLinear) => (true, false),
        Some(MinFilter::LinearMipmapNearest) => (false, true),
        Some(MinFilter::Linear | MinFilter::LinearMipmapLinear) | None => (false, false),
    };

    CpuSampler {
        mag_nearest: matches!(sampler.mag_filter(), Some(MagFilter::Nearest)),
        min_nearest,
        mip_nearest,
        wrap_u: wrap(sampler.wrap_s()),
        wrap_v: wrap(sampler.wrap_t()),
    }
}

fn texture_ref(
    info: &gltf::Texture,
    textures: &[Arc<CpuTexture>],
    srgb: bool,
) -> Option<CpuTextureRef> {
    textures
        .get(info.source().index())
        .cloned()
        .map(|texture| CpuTextureRef {
            texture,
            sampler: to_cpu_sampler(&info.sampler()),
            srgb,
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

    if let Some(light) = node.light() {
        use gltf::khr_lights_punctual::Kind;
        let kind = match light.kind() {
            Kind::Point => CpuLightKind::Point,
            Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => CpuLightKind::Spot {
                cos_inner: inner_cone_angle.cos(),
                cos_outer: outer_cone_angle.cos(),
            },
            Kind::Directional => CpuLightKind::Directional,
        };
        // glTF lights point down their node's -Z axis.
        let direction = world.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
        scene.lights.push(CpuLight {
            kind,
            color: light.color(),
            intensity: light.intensity(),
            range: light.range().unwrap_or(0.0),
            position: world.transform_point3(Vec3::ZERO).to_array(),
            direction: direction.to_array(),
        });
    }

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
    let tangents: Vec<[f32; 4]> = match reader.read_tangents() {
        Some(iter) => iter.collect(),
        None => vec![[0.0, 0.0, 0.0, 0.0]; positions.len()],
    };
    let had_tangents = reader.read_tangents().is_some();

    anyhow::ensure!(
        normals.len() == positions.len()
            && uvs.len() == positions.len()
            && tangents.len() == positions.len(),
        "Attribute count mismatch: {} positions, {} normals, {} uvs, {} tangents",
        positions.len(),
        normals.len(),
        uvs.len(),
        tangents.len()
    );

    let mut vertices: Vec<Vertex> = positions
        .iter()
        .zip(normals.iter())
        .zip(uvs.iter())
        .zip(tangents.iter())
        .map(|(((p, n), t), tan)| Vertex {
            position: *p,
            normal: *n,
            uv: *t,
            tangent: *tan,
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
    if !had_tangents {
        compute_tangents(&mut vertices, &indices);
    }

    let material = primitive.material();
    let pbr = material.pbr_metallic_roughness();

    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(material.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    // KHR_texture_transform (base color slot): T * R * S per spec.
    let base_uv_transform = pbr
        .base_color_texture()
        .and_then(|info| info.texture_transform())
        .map(|t| {
            let offset = glam::Vec2::from_array(t.offset());
            let scale = glam::Vec2::from_array(t.scale());
            let m = glam::Mat3::from_translation(offset)
                * glam::Mat3::from_angle(-t.rotation())
                * glam::Mat3::from_scale(scale);
            [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y],
            ]
        })
        .unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

    let cpu_material = CpuMaterial {
        base_color: pbr.base_color_factor(),
        base_uv_transform,
        alpha_mode,
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        emissive_factor: material.emissive_factor(),
        occlusion_strength: material
            .occlusion_texture()
            .map_or(1.0, |occ| occ.strength()),
        normal_scale: material.normal_texture().map_or(1.0, |nrm| nrm.scale()),
        double_sided: material.double_sided(),
        base_color_texture: pbr
            .base_color_texture()
            .and_then(|info| texture_ref(&info.texture(), textures, true)),
        metallic_roughness_texture: pbr
            .metallic_roughness_texture()
            .and_then(|info| texture_ref(&info.texture(), textures, false)),
        normal_texture: material
            .normal_texture()
            .and_then(|info| texture_ref(&info.texture(), textures, false)),
        emissive_texture: material
            .emissive_texture()
            .and_then(|info| texture_ref(&info.texture(), textures, true)),
        occlusion_texture: material
            .occlusion_texture()
            .and_then(|info| texture_ref(&info.texture(), textures, false)),
    };

    Ok(Some(CpuPrimitive {
        vertices,
        indices,
        transform,
        material: cpu_material,
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

/// Per-vertex tangent accumulation from triangle UV gradients (Lengyel-style,
/// not full MikkTSpace — sufficient until baked assets demand exact parity).
pub(crate) fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut accum = vec![Vec3::ZERO; vertices.len()];

    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p0 = Vec3::from_array(vertices[i0].position);
        let p1 = Vec3::from_array(vertices[i1].position);
        let p2 = Vec3::from_array(vertices[i2].position);
        let u0 = Vec2::from_array(vertices[i0].uv);
        let u1 = Vec2::from_array(vertices[i1].uv);
        let u2 = Vec2::from_array(vertices[i2].uv);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let d1 = u1 - u0;
        let d2 = u2 - u0;

        let det = d1.x * d2.y - d2.x * d1.y;
        if det.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / det;
        let tangent = (e1 * d2.y - e2 * d1.y) * r;
        for i in [i0, i1, i2] {
            accum[i] += tangent;
        }
    }

    for (vertex, tangent) in vertices.iter_mut().zip(accum) {
        let n = Vec3::from_array(vertex.normal);
        // Gram-Schmidt against the normal; fall back to any perpendicular
        // axis for degenerate UVs.
        let mut t = (tangent - n * n.dot(tangent)).normalize_or_zero();
        if t == Vec3::ZERO {
            t = n.cross(Vec3::Y).normalize_or_zero();
            if t == Vec3::ZERO {
                t = n.cross(Vec3::X).normalize_or_zero();
            }
        }
        vertex.tangent = [t.x, t.y, t.z, 1.0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangents_follow_uv_gradient() {
        // Unit quad in the XY plane, UVs aligned with X/Y: tangent must be +X.
        let mut vertices: Vec<Vertex> = [
            ([0.0, 0.0, 0.0], [0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0]),
            ([1.0, 1.0, 0.0], [1.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0]),
        ]
        .iter()
        .map(|(p, uv)| Vertex {
            position: *p,
            normal: [0.0, 0.0, 1.0],
            uv: *uv,
            tangent: [0.0; 4],
        })
        .collect();
        let indices = [0u32, 1, 2, 0, 2, 3];

        compute_tangents(&mut vertices, &indices);

        for vertex in &vertices {
            assert!(
                (vertex.tangent[0] - 1.0).abs() < 1e-4
                    && vertex.tangent[1].abs() < 1e-4
                    && vertex.tangent[2].abs() < 1e-4,
                "tangent should be +X, got {:?}",
                vertex.tangent
            );
        }
    }
}

