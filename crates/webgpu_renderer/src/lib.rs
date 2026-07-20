//! WebGPU renderer with glTF loading (see docs/webgpu-gltf-rust-plan.md in
//! Kataglyphis-BeschleunigerBallett for the roadmap this implements).
//!
//! Milestones implemented so far:
//! 1. Context + surface lifecycle with correct resize/outdated handling.
//! 2. glTF meshes (positions/normals/UVs/indices, node transforms) rendered
//!    through a forward pass with per-material base color and a directional
//!    light; headless render-to-texture for golden tests.
//! 3. Base-color textures (sRGB, white fallback) and an HDR (Rgba16Float)
//!    render target composited through an ACES tonemap pass.

pub mod asset;
pub mod context;
pub mod render;
pub mod scene;
#[cfg(target_arch = "wasm32")]
pub mod wasm_demo;

pub use asset::gltf_loader::load_gltf;
pub use context::GpuContext;
pub use render::forward::ForwardRenderer;
pub use render::overlay::{Overlay, OverlayControls};
pub use render::tonemap::TonemapPass;
pub use scene::camera::OrbitCamera;
pub use scene::controller::OrbitController;
pub use scene::lod::{
    build_lod_chain, build_lod_chain_with, select_lod, select_lod_by_distance, simplify_primitive,
    Lod, Simplifier,
};
pub use scene::qem::simplify_primitive_qem;
pub use scene::{CpuMaterial, CpuPrimitive, CpuScene, CpuTexture};
