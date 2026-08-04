//! Regression guard for "Give the crate's single-layer 2D
//! `wgpu::TextureDescriptor` one definition": every `wgpu::TextureDescriptor {`
//! literal outside `render::texture::create_2d_texture` is either a copy that
//! escaped the conversion, or a genuinely different shape (cube texture,
//! depth array) marked with a trailing `// TEXTURE_2D_SHAPE_OK: <reason>`
//! comment on the same line.
//!
//! Pure CPU, no adapter: this only inspects source text, so it runs
//! everywhere, including environments with no adapter.

const SOURCES: &[(&str, &str, bool)] = &[
    ("render/bloom.rs", include_str!("../src/render/bloom.rs"), false),
    ("render/ssao.rs", include_str!("../src/render/ssao.rs"), false),
    ("render/forward.rs", include_str!("../src/render/forward.rs"), false),
    ("render/ibl.rs", include_str!("../src/render/ibl.rs"), false),
    ("render/texture.rs", include_str!("../src/render/texture.rs"), true),
];

const NEEDLE: &str = "wgpu::TextureDescriptor {";
const MARKER: &str = "// TEXTURE_2D_SHAPE_OK:";

#[test]
fn texture_descriptor_literals_are_the_single_definition_or_marked_non_goals() {
    let mut unmarked = Vec::new();
    let mut marked_count = 0usize;
    let mut definition_count = 0usize;

    for (path, contents, is_texture_module) in SOURCES.iter().copied() {
        for (i, line) in contents.lines().enumerate() {
            if !line.contains(NEEDLE) {
                continue;
            }
            if line.contains(MARKER) {
                marked_count += 1;
                continue;
            }
            if is_texture_module {
                definition_count += 1;
                continue;
            }
            unmarked.push(format!("{path}:{}", i + 1));
        }
    }

    assert!(
        unmarked.is_empty(),
        "found longhand wgpu::TextureDescriptor literals that should go through \
         render::texture::create_2d_texture/create_2d_view, or be marked \
         `{MARKER}` if they are a genuinely different shape:\n{}",
        unmarked.join("\n")
    );
    assert_eq!(
        marked_count, 3,
        "expected exactly 3 marked non-goal TextureDescriptor literals (two cube \
         textures, one depth array); found {marked_count} - update this count if a \
         new non-goal shape is added, or investigate if one disappeared"
    );
    assert_eq!(
        definition_count, 1,
        "expected exactly 1 TextureDescriptor literal in render/texture.rs, \
         inside create_2d_texture; found {definition_count}"
    );
}
