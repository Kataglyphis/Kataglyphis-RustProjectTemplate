//! KTX2 (block-compressed texture) loading.

use kataglyphis_webgpu_renderer::asset::ktx2_loader::load_ktx2;
use kataglyphis_webgpu_renderer::scene::CompressedFormat;

const RED_BC1: &[u8] = include_bytes!("assets/red_bc1.ktx2");

#[test]
fn loads_bc1_ktx2() {
    let texture = load_ktx2(RED_BC1).expect("red_bc1.ktx2 must load");
    assert_eq!((texture.width, texture.height), (8, 8));
    assert!(texture.rgba8.is_empty(), "compressed payload stays packed");

    let compressed = texture
        .compressed
        .as_ref()
        .expect("BC1 file must produce a compressed payload");
    assert_eq!(compressed.format, CompressedFormat::Bc1RgbaUnorm);
    assert_eq!(compressed.format.block_bytes(), 8);
    assert_eq!(compressed.mips.len(), 1);
    // 8x8 at 4x4 blocks = 4 blocks * 8 bytes.
    assert_eq!(compressed.mips[0].len(), 32);
}

#[test]
fn rejects_non_ktx2_input() {
    assert!(load_ktx2(b"definitely not a ktx2 file").is_err());
}

#[test]
fn uploads_bc1_when_supported() {
    use kataglyphis_webgpu_renderer::GpuContext;
    let Ok(gpu) = GpuContext::new_headless() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };
    if !gpu.supports_bc {
        eprintln!("SKIP: adapter lacks TEXTURE_COMPRESSION_BC");
        return;
    }
    // Building a scene with the compressed texture must not panic or trip
    // wgpu validation (block layout/rows-per-image are easy to get wrong).
    let texture = load_ktx2(RED_BC1).expect("load");
    let scene = kataglyphis_webgpu_renderer::CpuScene {
        primitives: vec![],
        ..Default::default()
    };
    let mut renderer = kataglyphis_webgpu_renderer::ForwardRenderer::new(&gpu, 64, 64);
    renderer.upload_scene(&gpu, &scene);
    assert!(texture.compressed.is_some());
}
