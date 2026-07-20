//! Converts a Wavefront OBJ (with materials and textures) to glTF.
//!
//! Usage: `cargo run --example obj2gltf -- <in.obj> <out.gltf>`
//!
//! This is the CLI face of `asset::obj_to_gltf::convert_file`, which the
//! comparison harness uses to push the C++ engine's OBJ scenes through the
//! glTF-only Rust renderer - the "same scene in both renderers" bridge.

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [obj, gltf] = args.as_slice() else {
        eprintln!("usage: obj2gltf <in.obj> <out.gltf>");
        std::process::exit(2);
    };
    match kataglyphis_webgpu_renderer::asset::obj_to_gltf::convert_file(
        Path::new(obj),
        Path::new(gltf),
    ) {
        Ok(mesh) => eprintln!(
            "wrote {gltf}: {} positions, {} indices",
            mesh.positions.len(),
            mesh.indices.len()
        ),
        Err(err) => {
            eprintln!("conversion failed: {err:#}");
            std::process::exit(1);
        }
    }
}
