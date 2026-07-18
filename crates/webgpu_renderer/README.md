# kataglyphis_webgpu_renderer

A WebGPU (wgpu) renderer with glTF loading, written in Rust. The same code
runs natively on Vulkan/DX12/Metal and in the browser on the WebGPU API.

Companion to the C++ Vulkan engine in
[Kataglyphis-BeschleunigerBallett](https://github.com/Kataglyphis/Kataglyphis-BeschleunigerBallett);
its `docs/` hold the [roadmap](../../../docs/webgpu-renderer-roadmap.md),
the [sRGB audit](../../../docs/webgpu-srgb-audit.md), and the
[shader-sharing guide](../../../docs/shader-sharing.md).

## Features

**Assets** — glTF 2.0 and GLB: meshes, node hierarchy, samplers + wrap
modes, `KHR_texture_transform`, tangents (loaded or generated), skins,
animations, punctual lights, cameras. KTX2 textures with BC1/3/5/7
passthrough.

**Shading** — metallic-roughness PBR (GGX + Smith + Fresnel-Schlick) with
base color / metallic-roughness / normal / emissive / occlusion maps,
alpha OPAQUE/MASK/BLEND, double-sided materials, CPU-generated mip chains.

**Lighting** — directional sun with 3-cascade shadow maps (3×3 PCF),
`KHR_lights_punctual` point/spot lights, procedural sky with an analytic
sun, and analytic image-based lighting (hemisphere irradiance + sky
reflections via the split-sum approximation).

**Post** — HDR `Rgba16Float` target, bloom (bright-pass + separable
Gaussian), SSAO (depth reconstruction), exposure control, ACES tonemapping.

**Runtime** — GPU skinning, TRS animation playback, frustum culling, LOD
simplification, a validated render graph, hot shader reload, an egui
overlay, and screenshot capture.

## Running

```bash
# Native viewer (drop a .gltf/.glb on the window to load it)
cargo run -p kataglyphis_webgpu_renderer --example viewer [model.gltf]

# Headless + unit tests (GPU tests self-skip without an adapter)
cargo test -p kataglyphis_webgpu_renderer

# Export shaders as SPIR-V/GLSL for the C++ engine
cargo run -p kataglyphis_webgpu_renderer --example export_shaders -- out_dir

# Browser demo
cargo build -p kataglyphis_webgpu_renderer --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/kataglyphis_webgpu_renderer.wasm \
  --out-dir crates/webgpu_renderer/web/pkg --target web
python -m http.server 8931 --directory crates/webgpu_renderer/web
```

Viewer controls: drag to orbit, wheel to zoom, **S** screenshot,
**R** reload shaders, **Esc** quit.

## Testing approach

GPU tests render headlessly and assert *structural* pixel properties
(colour dominance, coverage ratios, energy deltas) rather than exact
images, so they survive driver differences. They skip themselves when no
adapter is present, which keeps CI honest without a GPU runner.
