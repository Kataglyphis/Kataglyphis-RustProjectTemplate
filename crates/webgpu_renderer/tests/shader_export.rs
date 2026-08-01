//! Guards the WGSL -> SPIR-V path used to share shader code with the C++
//! Vulkan engine: every shader must parse, validate, and emit SPIR-V.

use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

const SHADERS: &[(&str, &str)] = &[
    ("forward", include_str!("../src/shaders/forward.wgsl")),
    ("sky", include_str!("../src/shaders/sky.wgsl")),
    ("tonemap", include_str!("../src/shaders/tonemap.wgsl")),
    ("bloom", include_str!("../src/shaders/bloom.wgsl")),
    ("ssao", include_str!("../src/shaders/ssao.wgsl")),
    ("ibl", include_str!("../src/shaders/ibl.wgsl")),
    (
        "occlusion_bbox",
        include_str!("../src/shaders/occlusion_bbox.wgsl"),
    ),
    (
        "depth_resolve",
        include_str!("../src/shaders/depth_resolve.wgsl"),
    ),
    ("gpu_cull", include_str!("../src/shaders/gpu_cull.wgsl")),
    ("histogram", include_str!("../src/shaders/histogram.wgsl")),
];

#[test]
fn all_shaders_export_to_spirv() {
    for (name, source) in SHADERS {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}.wgsl must parse: {e:?}"));
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}.wgsl must validate: {e:?}"));
        let words = spv::write_vec(&module, &info, &spv::Options::default(), None)
            .unwrap_or_else(|e| panic!("{name}.wgsl must emit SPIR-V: {e:?}"));

        assert!(!words.is_empty(), "{name}: empty SPIR-V");
        // SPIR-V magic number.
        assert_eq!(words[0], 0x0723_0203, "{name}: bad SPIR-V magic");
        assert!(
            !module.entry_points.is_empty(),
            "{name}: no entry points exported"
        );
    }
}

/// `SHADERS` above is a hand-maintained list; nothing previously enforced that
/// every `.wgsl` file in `src/shaders/` actually appears in it, which is
/// exactly how `depth_resolve` (and `gpu_cull`/`histogram`) went unchecked.
/// `histogram.wgsl` is deliberately hand-written rather than Slang-generated
/// (see `compile-slang-shaders.ps1:105-109`), so it belongs in this export
/// gate but is exempt from any Slang-source staleness gate.
#[test]
fn every_shader_file_is_covered() {
    let shaders_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("failed to read {shaders_dir}: {e:?}"))
    {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read entry in {shaders_dir}: {e:?}"))
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("wgsl") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("non-UTF8 shader filename: {path:?}"))
            .to_string();
        if !SHADERS.iter().any(|(name, _)| *name == stem) {
            missing.push(stem);
        }
    }
    assert!(
        missing.is_empty(),
        "src/shaders/ has .wgsl file(s) not covered by SHADERS in shader_export.rs: {missing:?}"
    );
}

/// The depth-resolve pass has zero colour attachments (see the pipeline in
/// `forward.rs`) and must write the depth aspect via `@builtin(frag_depth)`.
/// Slang's WGSL backend has no depth-write control, so it emits a plain
/// `@location(0)` colour output that gets silently dropped by wgpu, and the
/// rasterizer's own fragment z is written instead (the fullscreen triangle's
/// NDC z is 0.0, so every pixel resolves to 0.0). `compile-slang-shaders.*`
/// patches this after emit; this test pins the patched result.
#[test]
fn depth_resolve_fragment_writes_frag_depth() {
    let source = include_str!("../src/shaders/depth_resolve.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("depth_resolve.wgsl must parse: {e:?}"));
    let fs_main = module
        .entry_points
        .iter()
        .find(|ep| ep.name == "fs_main")
        .expect("depth_resolve.wgsl must export fs_main");
    let result = fs_main
        .function
        .result
        .as_ref()
        .expect("fs_main must return a value");
    let members = match &module.types[result.ty].inner {
        naga::TypeInner::Struct { members, .. } => members,
        other => panic!("fs_main result type must be a struct, got {other:?}"),
    };
    assert!(
        members
            .iter()
            .any(|m| matches!(
                m.binding,
                Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth))
            )),
        "fs_main must write @builtin(frag_depth); result members are {members:?}"
    );
    assert!(
        !members
            .iter()
            .any(|m| matches!(m.binding, Some(naga::Binding::Location { .. }))),
        "fs_main must not emit a colour @location output; result members are {members:?}"
    );
}
