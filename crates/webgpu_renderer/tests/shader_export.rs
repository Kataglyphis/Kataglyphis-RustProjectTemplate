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
