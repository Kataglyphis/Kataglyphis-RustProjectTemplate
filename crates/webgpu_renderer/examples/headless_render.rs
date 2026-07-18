//! Renders a glTF file to a PNG without opening a window.
//!
//! ```bash
//! cargo run -p kataglyphis_webgpu_renderer --example headless_render -- model.gltf out.png [width height]
//! ```
//!
//! Useful for docs screenshots, batch turntables, and regression baselines
//! — the same path the golden tests use.

use kataglyphis_webgpu_renderer::{load_gltf, ForwardRenderer, GpuContext, OrbitCamera};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1);

    let model = args.next().unwrap_or_else(|| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets/cube_on_plane.gltf")
            .to_string_lossy()
            .into_owned()
    });
    let output = args.next().unwrap_or_else(|| "headless.png".to_string());
    let width: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1280);
    let height: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(720);

    let gpu = GpuContext::new_headless()?;
    let scene = load_gltf(&model)?;
    println!(
        "loaded {model}: {} primitives, {} triangles",
        scene.primitives.len(),
        scene.triangle_count()
    );

    let mut renderer = ForwardRenderer::new(&gpu, width, height);
    renderer.upload_scene(&gpu, &scene);
    if renderer.has_animations() {
        renderer.set_animation_time(0.0);
    }

    let camera = OrbitCamera {
        radius: 6.0,
        pitch_deg: 25.0,
        ..OrbitCamera::default()
    };
    let pixels = renderer.render_to_pixels(&gpu, width, height, &camera)?;
    image::save_buffer(&output, &pixels, width, height, image::ColorType::Rgba8)?;
    println!("wrote {output} ({width}x{height})");
    Ok(())
}
