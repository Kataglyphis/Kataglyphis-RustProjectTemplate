//! glTF 2.0 -> `CpuScene`. Positions, normals, UVs, tangents (generated when
//! absent), indices, flattened node transforms, and the metallic-roughness
//! material model: factors + base color / metallic-roughness / normal /
//! emissive / occlusion textures with their glTF sampler settings.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use glam::{Mat4, Vec2, Vec3};

use crate::scene::{
    AlphaMode, ChannelValues, CpuAnimation, CpuAnimationChannel, CpuCamera, CpuLight, CpuLightKind,
    CpuMaterial, CpuNode, CpuPrimitive, CpuSampler, CpuScene, CpuSkin, CpuTexture, CpuTextureRef,
    CpuWrap, Interpolation, Vertex,
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

    // Full node table (index-aligned with glTF node indices).
    let mut nodes: Vec<CpuNode> = document
        .nodes()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            CpuNode {
                parent: None,
                translation: glam::Vec3::from_array(t),
                rotation: glam::Quat::from_array(r),
                scale: glam::Vec3::from_array(s),
            }
        })
        .collect();
    for node in document.nodes() {
        for child in node.children() {
            nodes[child.index()].parent = Some(node.index());
        }
    }

    let mut scene = CpuScene {
        nodes,
        ..CpuScene::default()
    };

    // Animations (linear/step; cubic spline collapses to linear on the
    // in-between values for now).
    for animation in document.animations() {
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(|b| &b.0[..]));
            let Some(times) = reader.read_inputs() else {
                continue;
            };
            let times: Vec<f32> = times.collect();
            if let Some(&last) = times.last() {
                duration = duration.max(last);
            }
            let Some(outputs) = reader.read_outputs() else {
                continue;
            };
            use gltf::animation::util::ReadOutputs;
            let values = match outputs {
                ReadOutputs::Translations(iter) => {
                    ChannelValues::Translation(iter.map(glam::Vec3::from_array).collect())
                }
                ReadOutputs::Rotations(rotations) => ChannelValues::Rotation(
                    rotations.into_f32().map(glam::Quat::from_array).collect(),
                ),
                ReadOutputs::Scales(iter) => {
                    ChannelValues::Scale(iter.map(glam::Vec3::from_array).collect())
                }
                ReadOutputs::MorphTargetWeights(w) => {
                    ChannelValues::MorphWeights(w.into_f32().collect())
                }
            };
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };
            channels.push(CpuAnimationChannel {
                node: channel.target().node().index(),
                times,
                values,
                interpolation,
            });
        }
        if !channels.is_empty() {
            scene.animations.push(CpuAnimation {
                name: animation.name().unwrap_or("animation").to_string(),
                duration,
                channels,
            });
        }
    }

    // Skins: joint node indices + inverse bind matrices.
    for skin in document.skins() {
        let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|b| &b.0[..]));
        let inverse_bind_matrices = reader
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_default();
        scene.skins.push(CpuSkin {
            joints: skin.joints().map(|j| j.index()).collect(),
            inverse_bind_matrices,
        });
    }

    // Cameras authored in the file (pose = their node's world transform).
    for node in document.nodes() {
        let Some(camera) = node.camera() else {
            continue;
        };
        if let gltf::camera::Projection::Perspective(perspective) = camera.projection() {
            scene.cameras.push(CpuCamera {
                name: camera.name().map(str::to_string),
                node: node.index(),
                yfov_rad: perspective.yfov(),
                znear: perspective.znear(),
                zfar: perspective.zfar(),
            });
        }
    }

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
        compressed: None,
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
            if let Some(mut cpu) = load_primitive(&primitive, world, buffers, textures)? {
                cpu.node_index = Some(node.index());
                cpu.skin_index = node.skin().map(|s| s.index());
                // Default morph weights: node-level overrides mesh-level (glTF
                // spec), applied before any animation channel drives them. Kept
                // only when the count matches this primitive's target count.
                if !cpu.morph_targets.is_empty() {
                    if let Some(weights) = node.weights().or_else(|| mesh.weights()) {
                        if weights.len() == cpu.morph_weights.len() {
                            cpu.morph_weights = weights.to_vec();
                        }
                    }
                }
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
    let joints: Vec<[f32; 4]> = match reader.read_joints(0) {
        Some(iter) => iter
            .into_u16()
            .map(|j| [j[0] as f32, j[1] as f32, j[2] as f32, j[3] as f32])
            .collect(),
        None => vec![[0.0; 4]; positions.len()],
    };
    let weights: Vec<[f32; 4]> = match reader.read_weights(0) {
        Some(iter) => iter.into_f32().collect(),
        None => vec![[0.0; 4]; positions.len()],
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
        .enumerate()
        .map(|(i, (((p, n), t), tan))| Vertex {
            position: *p,
            normal: *n,
            uv: *t,
            tangent: *tan,
            joints: joints.get(i).copied().unwrap_or([0.0; 4]),
            weights: weights.get(i).copied().unwrap_or([0.0; 4]),
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
    // Opt-in: regenerate the generated tangents with MikkTSpace (the reference
    // basis DCC tools bake normal maps against). Off by default - the Lengyel
    // path above stays the default - and only for meshes whose tangents WE
    // generated; a file that shipped its own tangents keeps them. MikkTSpace may
    // split vertices at UV seams, so it hands back fresh buffers.
    let (vertices, indices) = if !had_tangents && mikktspace_tangents_enabled() {
        match generate_tangents_mikktspace(&vertices, &indices) {
            Some(rebuilt) => rebuilt,
            None => (vertices, indices),
        }
    } else {
        (vertices, indices)
    };

    // Morph targets: per-target POSITION/NORMAL deltas. Weights start at zero
    // here; mesh/node default weights are applied by the caller (which has the
    // mesh/node), and a WEIGHTS animation channel can drive them per frame.
    let morph_targets: Vec<crate::scene::MorphTarget> = reader
        .read_morph_targets()
        .map(|(pos, norm, tan)| crate::scene::MorphTarget {
            position_deltas: pos
                .map(|it| it.map(Vec3::from_array).collect())
                .unwrap_or_default(),
            normal_deltas: norm
                .map(|it| it.map(Vec3::from_array).collect())
                .unwrap_or_default(),
            // glTF morph TANGENT deltas are vec3 (direction only); the base
            // tangent's w handedness is never morphed.
            tangent_deltas: tan
                .map(|it| it.map(Vec3::from_array).collect())
                .unwrap_or_default(),
        })
        .collect();
    let morph_weights = vec![0.0f32; morph_targets.len()];

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
        // KHR_materials_emissive_strength scales the emissive contribution
        // past the [0,1] glTF factor range (for HDR emitters). Fold it into
        // the factor so the shader path stays unchanged; default 1.0 when the
        // extension is absent.
        emissive_factor: {
            let ef = material.emissive_factor();
            let strength = material.emissive_strength().unwrap_or(1.0);
            [ef[0] * strength, ef[1] * strength, ef[2] * strength]
        },
        occlusion_strength: material
            .occlusion_texture()
            .map_or(1.0, |occ| occ.strength()),
        normal_scale: material.normal_texture().map_or(1.0, |nrm| nrm.scale()),
        double_sided: material.double_sided(),
        unlit: material.unlit(),
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
        node_index: None,
        skin_index: None,
        material: cpu_material,
        morph_targets,
        morph_weights,
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

/// Per-vertex tangent frame from triangle UV gradients (Lengyel's method).
///
/// Accumulates BOTH the tangent and the bitangent per vertex, then stores the
/// tangent with a handedness sign in `.w`. Handedness is the part that
/// matters and that the earlier version got wrong: it hard-coded `w = 1.0`,
/// so every mirrored UV island (the norm on a symmetric mesh - a face, a
/// character) sampled its normal map with the bitangent flipped the wrong way,
/// lighting the mirrored half as if lit from the opposite side. `w` now
/// carries `sign(dot(cross(N, T), B))`, which the shader multiplies into its
/// reconstructed bitangent - the glTF convention.
///
/// Still not full MikkTSpace: it does not split vertices across hard UV seams,
/// so a vertex shared by islands of opposite handedness gets one averaged
/// frame. That needs the welding pass MikkTSpace does and is a separate step;
/// this fixes the handedness error, which is the visible one.
pub(crate) fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tan_accum = vec![Vec3::ZERO; vertices.len()];
    let mut bitan_accum = vec![Vec3::ZERO; vertices.len()];

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
        let bitangent = (e2 * d1.x - e1 * d2.x) * r;
        for i in [i0, i1, i2] {
            tan_accum[i] += tangent;
            bitan_accum[i] += bitangent;
        }
    }

    for ((vertex, tangent), bitangent) in vertices.iter_mut().zip(tan_accum).zip(bitan_accum) {
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
        // Handedness: negative when the UV chart is mirrored relative to the
        // geometric bitangent. Default to +1 for degenerate accumulation
        // rather than 0, which would zero the shader's bitangent entirely.
        let w = if n.cross(t).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        vertex.tangent = [t.x, t.y, t.z, w];
    }
}

/// Whether to regenerate generated tangents with MikkTSpace instead of the
/// Lengyel default. Off unless `KATAGLYPHIS_MIKKTSPACE_TANGENTS` is set in the
/// environment (mirrors the C++ engine's asset-loading override style). On
/// wasm32 there is no environment, so it is always off and the browser demo
/// keeps the default basis.
fn mikktspace_tangents_enabled() -> bool {
    std::env::var_os("KATAGLYPHIS_MIKKTSPACE_TANGENTS").is_some()
}

/// MikkTSpace reference tangent generation (opt-in; [`compute_tangents`] stays
/// the default).
///
/// The default Lengyel path accumulates ONE averaged tangent per vertex. It is
/// adequate, but it is not the per-face-corner basis that DCC tools (Blender,
/// the glTF reference exporter) bake normal maps against, so a normal-mapped
/// surface authored elsewhere can show subtle seams. MikkTSpace is that standard.
///
/// MikkTSpace computes a tangent per face-corner, and where a shared vertex's
/// corners disagree - a hard UV seam or a mirrored island - the vertex must be
/// SPLIT. So this returns fresh vertex+index buffers rather than writing tangents
/// in place. Corners MikkTSpace considers identical weld back together (it emits
/// bit-identical tangents for them), so only genuine seams add vertices.
///
/// Returns `None` if MikkTSpace reports failure (e.g. no triangles), so the
/// caller can keep its existing buffers.
pub(crate) fn generate_tangents_mikktspace(
    vertices: &[Vertex],
    indices: &[u32],
) -> Option<(Vec<Vertex>, Vec<u32>)> {
    if indices.len() < 3 || !indices.len().is_multiple_of(3) {
        return None;
    }

    struct MikkMesh<'a> {
        vertices: &'a [Vertex],
        indices: &'a [u32],
        /// Per-corner tangent, length `num_faces * 3`, filled by the algorithm.
        tangents: Vec<[f32; 4]>,
    }
    impl MikkMesh<'_> {
        fn corner(&self, face: usize, vert: usize) -> usize {
            self.indices[face * 3 + vert] as usize
        }
    }
    impl bevy_mikktspace::Geometry for MikkMesh<'_> {
        fn num_faces(&self) -> usize {
            self.indices.len() / 3
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.vertices[self.corner(face, vert)].position
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.vertices[self.corner(face, vert)].normal
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            self.vertices[self.corner(face, vert)].uv
        }
        fn set_tangent(
            &mut self,
            tangent_space: Option<bevy_mikktspace::TangentSpace>,
            face: usize,
            vert: usize,
        ) {
            // `tangent_encoded()` is [x, y, z, w] with w carrying the handedness
            // sign - the glTF convention `Vertex::tangent` stores. A `None` space
            // (degenerate corner) keeps the initialized default.
            if let Some(ts) = tangent_space {
                self.tangents[face * 3 + vert] = ts.tangent_encoded();
            }
        }
    }

    let num_corners = indices.len();
    let mut mesh = MikkMesh {
        vertices,
        indices,
        tangents: vec![[0.0, 0.0, 0.0, 1.0]; num_corners],
    };
    if bevy_mikktspace::generate_tangents(&mut mesh).is_err() {
        return None;
    }

    // Weld corners back into vertices, splitting only where a vertex's corners
    // carry different tangents. Key on (original index, exact tangent bits):
    // MikkTSpace emits bit-identical tangents for corners it treats as shared,
    // so identical corners collapse and real seams split.
    let mut remap: std::collections::HashMap<(u32, [u32; 4]), u32> = std::collections::HashMap::new();
    let mut out_vertices: Vec<Vertex> = Vec::with_capacity(vertices.len());
    let mut out_indices: Vec<u32> = Vec::with_capacity(indices.len());
    for (corner, &orig) in indices.iter().enumerate() {
        let tangent = mesh.tangents[corner];
        let key = (
            orig,
            [
                tangent[0].to_bits(),
                tangent[1].to_bits(),
                tangent[2].to_bits(),
                tangent[3].to_bits(),
            ],
        );
        let new_index = *remap.entry(key).or_insert_with(|| {
            let mut v = vertices[orig as usize];
            v.tangent = tangent;
            out_vertices.push(v);
            (out_vertices.len() - 1) as u32
        });
        out_indices.push(new_index);
    }
    Some((out_vertices, out_indices))
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
            joints: [0.0; 4],
            weights: [0.0; 4],
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
            // Right-handed UVs: handedness must be +1.
            assert_eq!(vertex.tangent[3], 1.0, "expected +1 handedness");
        }
    }

    fn quad_with_uvs(uvs: [[f32; 2]; 4]) -> (Vec<Vertex>, [u32; 6]) {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let vertices = pos
            .iter()
            .zip(uvs)
            .map(|(p, uv)| Vertex {
                position: *p,
                normal: [0.0, 0.0, 1.0],
                uv,
                tangent: [0.0; 4],
                joints: [0.0; 4],
                weights: [0.0; 4],
            })
            .collect();
        (vertices, [0u32, 1, 2, 0, 2, 3])
    }

    #[test]
    fn mirrored_uvs_flip_handedness() {
        // The headline regression this rewrite fixes. Same quad, but the V
        // axis of the UVs is mirrored (v = 1 - v). The geometry is unchanged,
        // so the tangent still points along +X, but the chart is now
        // left-handed and the handedness sign MUST flip to -1. The previous
        // implementation hard-coded +1 and would fail this.
        let (mut normal, idx) = quad_with_uvs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        compute_tangents(&mut normal, &idx);

        let (mut mirrored, idx) = quad_with_uvs([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        compute_tangents(&mut mirrored, &idx);

        for v in &normal {
            assert_eq!(v.tangent[3], 1.0, "unmirrored chart is right-handed");
        }
        for v in &mirrored {
            assert_eq!(
                v.tangent[3], -1.0,
                "mirrored chart must be left-handed, got {:?}",
                v.tangent
            );
        }
    }

    #[test]
    fn mikktspace_quad_tangent_is_x() {
        // Unit quad in XY, UVs aligned with X/Y: MikkTSpace must produce a
        // unit +X tangent, right-handed (+1), and weld the corners back to the
        // original 4 vertices (no seam -> no split).
        let (vertices, indices) =
            quad_with_uvs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let (out_v, out_i) =
            generate_tangents_mikktspace(&vertices, &indices).expect("mikktspace should succeed");
        assert_eq!(out_i.len(), indices.len(), "index count is preserved");
        assert_eq!(out_v.len(), 4, "aligned UVs have no seam, corners weld to 4 verts");
        for v in &out_v {
            let t = v.tangent;
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "tangent must be unit length, got {len}");
            assert!(
                (t[0].abs() - 1.0).abs() < 1e-3 && t[1].abs() < 1e-3 && t[2].abs() < 1e-3,
                "tangent should lie along X, got {:?}",
                t
            );
            assert_eq!(t[3], 1.0, "right-handed chart -> +1 handedness");
        }
    }

    #[test]
    fn mikktspace_mirrored_uvs_flip_handedness() {
        // Same geometry, V axis mirrored: handedness must flip to -1, matching
        // the Lengyel path's mirrored_uvs_flip_handedness and the glTF convention.
        let (vertices, indices) =
            quad_with_uvs([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        let (out_v, _) =
            generate_tangents_mikktspace(&vertices, &indices).expect("mikktspace should succeed");
        for v in &out_v {
            assert_eq!(
                v.tangent[3], -1.0,
                "mirrored chart must be left-handed, got {:?}",
                v.tangent
            );
        }
    }

    #[test]
    fn mikktspace_rejects_degenerate_input() {
        // Fewer than one triangle (or a non-multiple of 3) -> None, so the
        // caller keeps its existing buffers instead of getting empty geometry.
        let (vertices, _) = quad_with_uvs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(generate_tangents_mikktspace(&vertices, &[0, 1]).is_none());
        assert!(generate_tangents_mikktspace(&vertices, &[]).is_none());
        assert!(generate_tangents_mikktspace(&vertices, &[0, 1, 2, 3]).is_none());
    }
}
