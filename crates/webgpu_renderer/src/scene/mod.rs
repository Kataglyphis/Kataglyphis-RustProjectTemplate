//! CPU-side scene representation produced by the asset loaders and consumed
//! by the render passes.

pub mod camera;
pub mod controller;
pub mod lod;
pub mod qem;

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz: tangent, w: bitangent handedness (+1/-1), glTF convention.
    pub tangent: [f32; 4],
    /// Skin joint indices (glTF JOINTS_0), as floats for a simple layout.
    pub joints: [f32; 4],
    /// Skin weights (glTF WEIGHTS_0); all zero = unskinned.
    pub weights: [f32; 4],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
            4 => Float32x4, 5 => Float32x4
        ],
    };
}

/// A per-instance transform, uploaded as four vec4 columns.
///
/// mat4 has no vertex-attribute format, so it travels as four Float32x4 at
/// consecutive locations and is reassembled in the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
}

impl InstanceRaw {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        // The whole point: advance once per INSTANCE, not per vertex.
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4],
    };

    pub const IDENTITY: Self = Self {
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };
}

/// Texture filtering/wrapping requested by the glTF sampler.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct CpuSampler {
    pub mag_nearest: bool,
    pub min_nearest: bool,
    pub mip_nearest: bool,
    pub wrap_u: CpuWrap,
    pub wrap_v: CpuWrap,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum CpuWrap {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

/// GPU block-compressed formats we can upload straight through (no
/// transcode). Basis ETC1S/UASTC supercompression is not handled yet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompressedFormat {
    Bc1RgbaUnorm,
    Bc3RgbaUnorm,
    Bc5RgUnorm,
    Bc7RgbaUnorm,
}

impl CompressedFormat {
    /// Bytes per 4x4 block.
    pub fn block_bytes(self) -> u32 {
        match self {
            CompressedFormat::Bc1RgbaUnorm => 8,
            _ => 16,
        }
    }
}

/// Pre-compressed mip chain straight from a KTX2 container.
#[derive(Clone, Debug)]
pub struct CompressedTexture {
    pub format: CompressedFormat,
    /// Block data per mip level, level 0 first.
    pub mips: Vec<Vec<u8>>,
}

/// Decoded RGBA8 texture (or a compressed payload). `srgb` decides the GPU
/// format: color data (base color, emissive) is sRGB; data maps (normal,
/// metallic-roughness, occlusion) are linear.
#[derive(Clone, Debug)]
pub struct CpuTexture {
    pub width: u32,
    pub height: u32,
    /// Empty when `compressed` is set.
    pub rgba8: Vec<u8>,
    pub compressed: Option<CompressedTexture>,
}

/// A texture reference as a material uses it: image + sampler + color space.
#[derive(Clone, Debug)]
pub struct CpuTextureRef {
    pub texture: Arc<CpuTexture>,
    pub sampler: CpuSampler,
    pub srgb: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    /// Fragments with alpha below the cutoff are discarded.
    Mask(f32),
    /// Sorted back-to-front, alpha-blended, no depth writes.
    Blend,
}

#[derive(Clone, Debug)]
pub struct CpuMaterial {
    pub base_color: [f32; 4],
    /// KHR_texture_transform for the base color UV set, as two affine rows
    /// [m00, m01, tx], [m10, m11, ty]. Identity when absent. Applied to the
    /// base color slot only (other slots: roadmap refinement).
    pub base_uv_transform: [[f32; 3]; 2],
    pub alpha_mode: AlphaMode,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub double_sided: bool,
    pub base_color_texture: Option<CpuTextureRef>,
    pub metallic_roughness_texture: Option<CpuTextureRef>,
    pub normal_texture: Option<CpuTextureRef>,
    pub emissive_texture: Option<CpuTextureRef>,
    pub occlusion_texture: Option<CpuTextureRef>,
}

impl Default for CpuMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            base_uv_transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            alpha_mode: AlphaMode::Opaque,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            double_sided: false,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            emissive_texture: None,
            occlusion_texture: None,
        }
    }
}

/// Punctual light (KHR_lights_punctual), world-space.
#[derive(Copy, Clone, Debug)]
pub enum CpuLightKind {
    Point,
    /// Cosines of the inner/outer cone angles.
    Spot {
        cos_inner: f32,
        cos_outer: f32,
    },
    Directional,
}

#[derive(Copy, Clone, Debug)]
pub struct CpuLight {
    pub kind: CpuLightKind,
    pub color: [f32; 3],
    pub intensity: f32,
    /// 0 = unbounded.
    pub range: f32,
    pub position: [f32; 3],
    /// Direction the light POINTS (spot/directional).
    pub direction: [f32; 3],
}

/// A scene-graph node with its local TRS (animation targets these).
#[derive(Clone, Debug)]
pub struct CpuNode {
    pub parent: Option<usize>,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Clone, Debug)]
pub enum ChannelValues {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
}

/// glTF keyframe interpolation mode. For `CubicSpline` the value array holds
/// THREE entries per keyframe - in-tangent, value, out-tangent - so it is 3x
/// the length of `times`; `Linear`/`Step` store one value per keyframe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Interpolation {
    #[default]
    Linear,
    Step,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub struct CpuAnimationChannel {
    pub node: usize,
    /// Keyframe times (seconds), ascending.
    pub times: Vec<f32>,
    pub values: ChannelValues,
    /// How to interpolate between keyframes. Note `CubicSpline` makes `values`
    /// 3x as long as `times` (in-tangent, value, out-tangent per keyframe).
    pub interpolation: Interpolation,
}

#[derive(Clone, Debug)]
pub struct CpuAnimation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<CpuAnimationChannel>,
}

/// A camera authored in the glTF file.
#[derive(Clone, Debug)]
pub struct CpuCamera {
    pub name: Option<String>,
    /// Index into `CpuScene::nodes` (its world transform is the camera pose).
    pub node: usize,
    pub yfov_rad: f32,
    pub znear: f32,
    /// `None` for an infinite projection.
    pub zfar: Option<f32>,
}

/// A glTF skin: joint nodes plus their inverse bind matrices.
#[derive(Clone, Debug)]
pub struct CpuSkin {
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// One drawable: an indexed triangle list with a world transform and material.
#[derive(Clone, Debug)]
pub struct CpuPrimitive {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub transform: Mat4,
    /// Index into `CpuScene::nodes` when the primitive belongs to the scene
    /// graph (animation retargets its transform).
    pub node_index: Option<usize>,
    /// Index into `CpuScene::skins` for skinned primitives.
    pub skin_index: Option<usize>,
    pub material: CpuMaterial,
    /// glTF morph targets (POSITION/NORMAL deltas per target). Empty for meshes
    /// without morphing.
    pub morph_targets: Vec<MorphTarget>,
    /// Current morph weights, one per `morph_targets` entry. Seeded from the
    /// mesh's default weights and animatable via a WEIGHTS animation channel.
    pub morph_weights: Vec<f32>,
}

/// One glTF morph target: per-vertex deltas added to the base attributes,
/// scaled by the target's weight. `normal_deltas` is empty when the target
/// morphs only positions.
#[derive(Clone, Debug, Default)]
pub struct MorphTarget {
    pub position_deltas: Vec<Vec3>,
    pub normal_deltas: Vec<Vec3>,
}

/// Blend a base vertex buffer with its morph targets at the given weights:
/// `out[v] = base[v] + Σ_i weight[i] * target[i].delta[v]`. Positions always,
/// normals when the target provides them (re-normalized). Targets/weights of
/// mismatched length and zero-weight targets are skipped. Returns a fresh buffer
/// so the base stays intact for the next frame's weights.
pub fn blend_morph_targets(base: &[Vertex], targets: &[MorphTarget], weights: &[f32]) -> Vec<Vertex> {
    let mut out = base.to_vec();
    for (target, &w) in targets.iter().zip(weights) {
        if w == 0.0 {
            continue;
        }
        for (v, delta) in target.position_deltas.iter().enumerate() {
            if let Some(vert) = out.get_mut(v) {
                vert.position[0] += w * delta.x;
                vert.position[1] += w * delta.y;
                vert.position[2] += w * delta.z;
            }
        }
        for (v, delta) in target.normal_deltas.iter().enumerate() {
            if let Some(vert) = out.get_mut(v) {
                vert.normal[0] += w * delta.x;
                vert.normal[1] += w * delta.y;
                vert.normal[2] += w * delta.z;
            }
        }
    }
    // Re-normalize any normals we touched.
    if targets.iter().any(|t| !t.normal_deltas.is_empty()) {
        for vert in &mut out {
            let n = Vec3::from_array(vert.normal);
            let len = n.length();
            if len > 1e-6 {
                vert.normal = (n / len).to_array();
            }
        }
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct CpuScene {
    pub primitives: Vec<CpuPrimitive>,
    pub lights: Vec<CpuLight>,
    pub nodes: Vec<CpuNode>,
    pub animations: Vec<CpuAnimation>,
    pub skins: Vec<CpuSkin>,
    pub cameras: Vec<CpuCamera>,
}

impl CpuScene {
    /// World transforms for all nodes from their current local TRS.
    pub fn compute_world_transforms(nodes: &[CpuNode]) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; nodes.len()];
        let mut done = vec![false; nodes.len()];
        fn resolve(
            i: usize,
            nodes: &[CpuNode],
            world: &mut Vec<Mat4>,
            done: &mut Vec<bool>,
        ) -> Mat4 {
            if done[i] {
                return world[i];
            }
            let local = Mat4::from_scale_rotation_translation(
                nodes[i].scale,
                nodes[i].rotation,
                nodes[i].translation,
            );
            let m = match nodes[i].parent {
                Some(p) => resolve(p, nodes, world, done) * local,
                None => local,
            };
            world[i] = m;
            done[i] = true;
            m
        }
        for i in 0..nodes.len() {
            resolve(i, nodes, &mut world, &mut done);
        }
        world
    }
}

impl CpuScene {
    pub fn vertex_count(&self) -> usize {
        self.primitives.iter().map(|p| p.vertices.len()).sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vert(pos: [f32; 3]) -> Vertex {
        Vertex {
            position: pos,
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [0.0; 4],
            joints: [0.0; 4],
            weights: [0.0; 4],
        }
    }

    #[test]
    fn morph_zero_weight_returns_the_base() {
        let base = vec![vert([0.0, 0.0, 0.0]), vert([1.0, 0.0, 0.0])];
        let t = MorphTarget {
            position_deltas: vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 2.0, 0.0)],
            normal_deltas: vec![],
        };
        let out = blend_morph_targets(&base, &[t], &[0.0]);
        assert_eq!(out[0].position, base[0].position);
        assert_eq!(out[1].position, base[1].position);
    }

    #[test]
    fn morph_full_weight_adds_the_delta() {
        let base = vec![vert([0.0, 0.0, 0.0])];
        let t = MorphTarget {
            position_deltas: vec![Vec3::new(0.0, 10.0, 0.0)],
            normal_deltas: vec![],
        };
        let out = blend_morph_targets(&base, &[t], &[1.0]);
        assert!((out[0].position[1] - 10.0).abs() < 1e-5, "got {:?}", out[0].position);
    }

    #[test]
    fn morph_two_targets_accumulate_at_their_weights() {
        let base = vec![vert([0.0, 0.0, 0.0])];
        let t1 = MorphTarget {
            position_deltas: vec![Vec3::new(4.0, 0.0, 0.0)],
            normal_deltas: vec![],
        };
        let t2 = MorphTarget {
            position_deltas: vec![Vec3::new(0.0, 8.0, 0.0)],
            normal_deltas: vec![],
        };
        // 0.5*4 = 2.0 on x, 0.25*8 = 2.0 on y.
        let out = blend_morph_targets(&base, &[t1, t2], &[0.5, 0.25]);
        assert!(
            (out[0].position[0] - 2.0).abs() < 1e-5 && (out[0].position[1] - 2.0).abs() < 1e-5,
            "got {:?}",
            out[0].position
        );
    }

    #[test]
    fn morph_normals_are_renormalized() {
        let mut base = vert([0.0, 0.0, 0.0]);
        base.normal = [1.0, 0.0, 0.0];
        let t = MorphTarget {
            position_deltas: vec![],
            normal_deltas: vec![Vec3::new(0.0, 1.0, 0.0)],
        };
        let out = blend_morph_targets(&[base], &[t], &[1.0]);
        let n = Vec3::from_array(out[0].normal);
        assert!((n.length() - 1.0).abs() < 1e-5, "normal must be unit, got {n:?}");
    }

    fn node(parent: Option<usize>, translation: Vec3) -> CpuNode {
        CpuNode {
            parent,
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    #[test]
    fn root_node_world_equals_its_local_transform() {
        let nodes = vec![node(None, Vec3::new(1.0, 2.0, 3.0))];
        let world = CpuScene::compute_world_transforms(&nodes);
        let p = world[0].transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5, "got {p:?}");
    }

    #[test]
    fn child_transform_composes_with_its_parent() {
        // Parent translates +10 X, child translates +5 Y: the child's origin
        // lands at (10, 5, 0) in world space.
        let nodes = vec![
            node(None, Vec3::new(10.0, 0.0, 0.0)),
            node(Some(0), Vec3::new(0.0, 5.0, 0.0)),
        ];
        let world = CpuScene::compute_world_transforms(&nodes);
        let p = world[1].transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-5, "got {p:?}");
    }

    #[test]
    fn resolves_correctly_when_a_child_precedes_its_parent_in_the_array() {
        // The memoized recursion must handle out-of-order nodes: here the child
        // is index 0 and its parent index 1. A naive single forward pass would
        // compute the child before the parent and get the wrong world matrix.
        let nodes = vec![
            node(Some(1), Vec3::new(0.0, 0.0, 2.0)), // child first
            node(None, Vec3::new(1.0, 0.0, 0.0)),    // parent second
        ];
        let world = CpuScene::compute_world_transforms(&nodes);
        let child = world[0].transform_point3(Vec3::ZERO);
        assert!((child - Vec3::new(1.0, 0.0, 2.0)).length() < 1e-5, "got {child:?}");
    }

    #[test]
    fn three_level_chain_accumulates() {
        let nodes = vec![
            node(None, Vec3::new(1.0, 0.0, 0.0)),
            node(Some(0), Vec3::new(1.0, 0.0, 0.0)),
            node(Some(1), Vec3::new(1.0, 0.0, 0.0)),
        ];
        let world = CpuScene::compute_world_transforms(&nodes);
        let leaf = world[2].transform_point3(Vec3::ZERO);
        assert!((leaf - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-5, "got {leaf:?}");
    }
}
