//! `wgpu::BindGroupLayoutEntry` constructors for the five shapes that cover
//! all 47 call sites across the render passes. 24 of those entries repeated
//! `has_dynamic_offset: false, min_binding_size: None` verbatim, which is
//! exactly where the one field that actually differs (the binding, the
//! stage, read-only-ness) got lost. One helper per shape keeps that field
//! visible at the call site instead of buried in a six-line literal.

/// A uniform buffer binding.
pub const fn uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A storage buffer binding, read-only or read-write.
pub const fn storage_buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The general texture binding: every `sample_type`/`view_dimension`/
/// `multisampled` combination goes through this one.
pub const fn texture(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        },
        count: None,
    }
}

/// A non-multisampled 2D float texture - the majority shape among the
/// texture bindings. Thin wrapper over [`texture`], not a new concept.
pub const fn texture_2d(
    binding: u32,
    visibility: wgpu::ShaderStages,
    filterable: bool,
) -> wgpu::BindGroupLayoutEntry {
    texture(
        binding,
        visibility,
        wgpu::TextureSampleType::Float { filterable },
        wgpu::TextureViewDimension::D2,
        false,
    )
}

/// A sampler binding.
pub const fn sampler(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_builds_the_expected_entry() {
        let e = uniform(3, wgpu::ShaderStages::FRAGMENT);
        assert_eq!(e.binding, 3);
        assert_eq!(e.visibility, wgpu::ShaderStages::FRAGMENT);
        assert_eq!(e.count, None);
        assert!(matches!(
            e.ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        ));
    }

    #[test]
    fn storage_buffer_read_only_true_vs_false() {
        let ro = storage_buffer(1, wgpu::ShaderStages::COMPUTE, true);
        assert!(matches!(
            ro.ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        ));

        let rw = storage_buffer(2, wgpu::ShaderStages::COMPUTE, false);
        assert!(matches!(
            rw.ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        ));
        assert_eq!(rw.binding, 2);
        assert_eq!(rw.visibility, wgpu::ShaderStages::COMPUTE);
        assert_eq!(rw.count, None);
    }

    #[test]
    fn texture_builds_the_general_entry() {
        let e = texture(
            0,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
            false,
        );
        assert_eq!(e.binding, 0);
        assert_eq!(e.visibility, wgpu::ShaderStages::FRAGMENT);
        assert_eq!(e.count, None);
        assert!(matches!(
            e.ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            }
        ));
    }

    #[test]
    fn texture_2d_is_a_thin_wrapper_over_texture() {
        let e = texture_2d(4, wgpu::ShaderStages::FRAGMENT, true);
        assert_eq!(e.binding, 4);
        assert_eq!(e.visibility, wgpu::ShaderStages::FRAGMENT);
        assert_eq!(e.count, None);
        assert!(matches!(
            e.ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        ));
    }

    #[test]
    fn sampler_builds_the_expected_entry() {
        let e = sampler(
            1,
            wgpu::ShaderStages::FRAGMENT,
            wgpu::SamplerBindingType::Comparison,
        );
        assert_eq!(e.binding, 1);
        assert_eq!(e.visibility, wgpu::ShaderStages::FRAGMENT);
        assert_eq!(e.count, None);
        assert!(matches!(
            e.ty,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
        ));
    }

    /// Regression guard: every `BindGroupLayoutEntry {` literal outside this
    /// module is a copy that escaped the conversion (or a new one added
    /// after this landed). Fails RED before the conversion, GREEN after.
    #[test]
    fn bind_group_layout_entries_go_through_the_helpers() {
        let src_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let mut offenders = Vec::new();
        walk(std::path::Path::new(src_dir), &mut offenders);
        assert!(
            offenders.is_empty(),
            "found longhand BindGroupLayoutEntry literals outside bind_layout.rs:\n{}",
            offenders.join("\n")
        );
    }

    fn walk(dir: &std::path::Path, offenders: &mut Vec<String>) {
        let entries = std::fs::read_dir(dir).expect("read_dir on crate src tree");
        for entry in entries {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                walk(&path, offenders);
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) == Some("bind_layout.rs") {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let contents = std::fs::read_to_string(&path).expect("read source file");
            for (i, line) in contents.lines().enumerate() {
                if line.contains("BindGroupLayoutEntry {") {
                    offenders.push(format!("{}:{}", path.display(), i + 1));
                }
            }
        }
    }
}
