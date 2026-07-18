//! CPU-side scene representation produced by the asset loaders and consumed
//! by the render passes.

pub mod camera;
pub mod controller;

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
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4
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

/// Decoded RGBA8 texture. `srgb` decides the GPU format: color data (base
/// color, emissive) is sRGB; data maps (normal, metallic-roughness,
/// occlusion) are linear.
#[derive(Clone, Debug)]
pub struct CpuTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
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
    Spot { cos_inner: f32, cos_outer: f32 },
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

#[derive(Clone, Debug)]
pub struct CpuAnimationChannel {
    pub node: usize,
    /// Keyframe times (seconds), ascending.
    pub times: Vec<f32>,
    pub values: ChannelValues,
}

#[derive(Clone, Debug)]
pub struct CpuAnimation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<CpuAnimationChannel>,
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
    pub material: CpuMaterial,
}

#[derive(Clone, Debug, Default)]
pub struct CpuScene {
    pub primitives: Vec<CpuPrimitive>,
    pub lights: Vec<CpuLight>,
    pub nodes: Vec<CpuNode>,
    pub animations: Vec<CpuAnimation>,
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
