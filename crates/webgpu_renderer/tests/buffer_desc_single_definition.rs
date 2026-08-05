//! Regression guard for "Give the Rust crate's 26 `wgpu::BufferDescriptor`
//! literals five named constructors": every `wgpu::BufferDescriptor {`
//! literal outside `render::buffer_desc` is either a copy that escaped the
//! conversion, or one of the two genuinely different shapes (the exposure
//! state buffer in `histogram.rs`, the vertex-usage instance buffer in
//! `occlusion.rs`) that stay literal because folding them into
//! `buffer_desc` would turn five named shapes back into one anonymous one.
//!
//! Pure CPU, no adapter: this only inspects source text, so it runs
//! everywhere, including environments with no adapter.

const NEEDLE: &str = "wgpu::BufferDescriptor {";

/// The part of `line` that is code: everything before the first `//`.
///
/// This guard matches source TEXT, so it used to count prose. `buffer_desc.rs`
/// documents, in its own module comment, that two call sites stay literal
/// `wgpu::BufferDescriptor { .. }` calls - and a plain `matches()` counted
/// that sentence as a sixth constructor. The test has been wrong since the
/// commit that introduced it (2a56786 wrote the five constructors, the
/// sentence and the test together); it went red the first time this repo's
/// Rust step actually ran it, on 2026-08-05, without a single buffer having
/// been added. The same blindness applies to the per-file scan below, where a
/// comment mentioning the needle would have been reported as a stray literal.
///
/// A `//` inside a string literal would truncate this early, which can only
/// make the guard blinder, never noisier - and there is no such string in the
/// files it reads.
fn code_of(line: &str) -> &str {
    match line.find("//") {
        Some(comment_start) => &line[..comment_start],
        None => line,
    }
}

#[test]
fn buffer_descriptor_literals_are_the_single_definition_or_the_two_named_outliers() {
    let buffer_desc_src = include_str!("../src/render/buffer_desc.rs");
    let definition_count: usize = buffer_desc_src
        .lines()
        .map(|line| code_of(line).matches(NEEDLE).count())
        .sum();
    assert_eq!(
        definition_count, 5,
        "expected exactly 5 wgpu::BufferDescriptor literals in render/buffer_desc.rs \
         (uniform, storage_dst, storage_src, readback, query_resolve); found {definition_count}"
    );

    let sources: &[(&str, &str)] = &[
        (
            "render/forward.rs",
            include_str!("../src/render/forward.rs"),
        ),
        (
            "render/occlusion.rs",
            include_str!("../src/render/occlusion.rs"),
        ),
        (
            "render/gpu_timing.rs",
            include_str!("../src/render/gpu_timing.rs"),
        ),
        (
            "render/tonemap.rs",
            include_str!("../src/render/tonemap.rs"),
        ),
        ("render/ibl.rs", include_str!("../src/render/ibl.rs")),
        (
            "render/histogram.rs",
            include_str!("../src/render/histogram.rs"),
        ),
        (
            "render/gpu_occlusion.rs",
            include_str!("../src/render/gpu_occlusion.rs"),
        ),
        ("render/ssao.rs", include_str!("../src/render/ssao.rs")),
    ];

    let mut unmarked = Vec::new();
    for (path, contents) in sources {
        for (i, line) in contents.lines().enumerate() {
            if code_of(line).contains(NEEDLE) {
                unmarked.push(format!("{path}:{}", i + 1));
            }
        }
    }

    assert_eq!(
        unmarked,
        vec![
            "render/occlusion.rs:472".to_string(),
            "render/histogram.rs:110".to_string(),
        ],
        "found unexpected wgpu::BufferDescriptor literals outside render/buffer_desc.rs - \
         route new buffer sites through the helpers, or add a genuinely new outlier here \
         if the shape does not fit"
    );

    let occlusion_block = descriptor_block(include_str!("../src/render/occlusion.rs"), NEEDLE);
    assert!(
        occlusion_block.contains("BufferUsages::VERTEX")
            && occlusion_block.contains("BufferUsages::COPY_DST"),
        "occlusion.rs's instance buffer outlier should use VERTEX | COPY_DST:\n{occlusion_block}"
    );

    let histogram_block = descriptor_block(include_str!("../src/render/histogram.rs"), NEEDLE);
    assert!(
        histogram_block.contains("BufferUsages::STORAGE")
            && histogram_block.contains("BufferUsages::COPY_SRC")
            && histogram_block.contains("BufferUsages::COPY_DST"),
        "histogram.rs's exposure state buffer outlier should use \
         STORAGE | COPY_SRC | COPY_DST:\n{histogram_block}"
    );
}

/// Returns the text of the first `needle { ... }` block found in `src`, from
/// the needle up to (and including) its matching closing brace.
fn descriptor_block<'a>(src: &'a str, needle: &str) -> &'a str {
    let start = src.find(needle).expect("needle present in source");
    let rest = &src[start..];
    let end = rest
        .find('}')
        .expect("descriptor block has a closing brace");
    &rest[..=end]
}
