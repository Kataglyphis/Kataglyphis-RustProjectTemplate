//! Exports the crate's WGSL shaders as SPIR-V (and GLSL 450) so the C++
//! Vulkan engine can consume the same shader code.
//!
//! Usage:
//!   cargo run -p kataglyphis_webgpu_renderer --example export_shaders -- [out_dir]
//!
//! Default out_dir: `target/shader-export`. Each WGSL entry point becomes
//! `<shader>.<entry>.spv` plus `<shader>.<entry>.glsl`.
//!
//! Why this direction (WGSL as the source of truth): WGSL is the stricter
//! language (uniformity analysis, no implicit conversions), so what
//! validates here also compiles for Vulkan — the reverse is not true. See
//! docs/shader-sharing.md in Kataglyphis-BeschleunigerBallett.

use std::path::{Path, PathBuf};

use naga::back::{glsl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::ShaderStage;

const SHADERS: &[(&str, &str)] = &[
    ("forward", include_str!("../src/shaders/forward.wgsl")),
    ("sky", include_str!("../src/shaders/sky.wgsl")),
    ("tonemap", include_str!("../src/shaders/tonemap.wgsl")),
    ("bloom", include_str!("../src/shaders/bloom.wgsl")),
    ("ssao", include_str!("../src/shaders/ssao.wgsl")),
    ("ibl", include_str!("../src/shaders/ibl.wgsl")),
];

fn main() -> anyhow::Result<()> {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new("target").join("shader-export"));
    std::fs::create_dir_all(&out_dir)?;

    let mut exported = 0usize;
    for (name, source) in SHADERS {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|e| anyhow::anyhow!("{name}.wgsl parse failed: {e:?}"))?;
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| anyhow::anyhow!("{name}.wgsl validation failed: {e:?}"))?;

        // One SPIR-V module per file carries every entry point.
        let spv_words = spv::write_vec(&module, &info, &spv::Options::default(), None)
            .map_err(|e| anyhow::anyhow!("{name}: SPIR-V emit failed: {e:?}"))?;
        let mut spv_bytes = Vec::with_capacity(spv_words.len() * 4);
        for word in &spv_words {
            spv_bytes.extend_from_slice(&word.to_le_bytes());
        }
        let spv_path = out_dir.join(format!("{name}.spv"));
        std::fs::write(&spv_path, &spv_bytes)?;
        println!("wrote {} ({} bytes)", spv_path.display(), spv_bytes.len());
        exported += 1;

        // GLSL is emitted per entry point (the GLSL backend is single-stage).
        for entry in &module.entry_points {
            let stage: ShaderStage = entry.stage;
            let mut glsl_source = String::new();
            // Desktop GLSL 450 to match the C++ Vulkan engine's shaders.
            let options = glsl::Options {
                version: glsl::Version::Desktop(450),
                ..Default::default()
            };
            let pipeline_options = glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.name.clone(),
                multiview: None,
            };
            match glsl::Writer::new(
                &mut glsl_source,
                &module,
                &info,
                &options,
                &pipeline_options,
                Default::default(),
            )
            .and_then(|mut writer| writer.write())
            {
                Ok(_) => {
                    let path = out_dir.join(format!("{name}.{}.glsl", entry.name));
                    std::fs::write(&path, glsl_source)?;
                    println!("wrote {}", path.display());
                    exported += 1;
                }
                Err(err) => {
                    // Not every WGSL construct maps to GLSL 450 (e.g. some
                    // texture array forms); SPIR-V remains the portable path.
                    println!("skip {name}.{} (GLSL): {err:?}", entry.name);
                }
            }
        }
    }

    println!("exported {exported} artifacts to {}", out_dir.display());
    Ok(())
}
