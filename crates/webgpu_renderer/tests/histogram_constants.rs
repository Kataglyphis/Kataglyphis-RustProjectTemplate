//! Pins `histogram.wgsl`'s hand-duplicated constants against the Rust copies
//! in `render::auto_exposure`.
//!
//! `histogram.wgsl` is the one shader in this crate Slang cannot generate (no
//! `InterlockedAdd` on `RWStructuredBuffer` for the WGSL target, per
//! AGENTS.md), so none of the generated-shader gates
//! (`SlangCompileManifestsAgree`, `CheckedInWgslIsNotOlderThanItsSlangSource`,
//! `CheckedInWgslHasNoHandEdits`) cover it. A drifted `MIN_LOG_LUMINANCE` here
//! does not crash anything - it just makes the CPU oracle in
//! `tests/histogram.rs` validate the GPU against a mapping the GPU is no
//! longer using.
//!
//! Deliberately separate from `tests/histogram.rs`: this one is pure CPU (no
//! `GpuContext`) so it runs everywhere, including environments with no
//! adapter.

use kataglyphis_webgpu_renderer::render::auto_exposure::{
    BLACK_THRESHOLD, BUILD_WORKGROUP, CLEAR_WORKGROUP, EXPOSURE_KEY, HISTOGRAM_BINS,
    MAX_LOG_LUMINANCE, MIN_LOG_LUMINANCE,
};

const SHADER_SOURCE: &str = include_str!("../src/shaders/histogram.wgsl");

/// Extracts the value literal from a `const <name>: <type> = <value>[u];`
/// declaration line in [`SHADER_SOURCE`].
fn extract_const(name: &str) -> &'static str {
    let needle = format!("const {name}:");
    let line = SHADER_SOURCE
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("histogram.wgsl no longer declares `{name}`"));

    let value = line
        .split('=')
        .nth(1)
        .unwrap_or_else(|| panic!("`{name}` declaration has no `=`: {line}"));
    value
        .trim()
        .trim_end_matches(';')
        .trim_end_matches('u')
        .trim()
}

/// Finds `fn <entry_point>(` in [`SHADER_SOURCE`] and returns the x dimension
/// of the nearest preceding `@workgroup_size(x, y, z)`.
fn extract_workgroup_x(entry_point: &str) -> u32 {
    let fn_needle = format!("fn {entry_point}(");
    let fn_pos = SHADER_SOURCE
        .find(&fn_needle)
        .unwrap_or_else(|| panic!("histogram.wgsl no longer declares `fn {entry_point}`"));
    let preceding = &SHADER_SOURCE[..fn_pos];

    let attr_needle = "@workgroup_size(";
    let attr_pos = preceding
        .rfind(attr_needle)
        .unwrap_or_else(|| panic!("no @workgroup_size(...) precedes `fn {entry_point}`"));
    let args_start = &preceding[attr_pos + attr_needle.len()..];
    let close = args_start
        .find(')')
        .unwrap_or_else(|| panic!("unterminated @workgroup_size(...) before `fn {entry_point}`"));

    args_start[..close]
        .split(',')
        .next()
        .unwrap_or_else(|| panic!("empty @workgroup_size(...) before `fn {entry_point}`"))
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("non-numeric @workgroup_size x for `fn {entry_point}`"))
}

#[test]
fn histogram_bins_matches_the_shader() {
    let shader_value: u32 = extract_const("HISTOGRAM_BINS")
        .parse()
        .expect("HISTOGRAM_BINS is not a u32 literal");
    assert_eq!(
        shader_value, HISTOGRAM_BINS as u32,
        "histogram.wgsl's HISTOGRAM_BINS ({shader_value}) disagrees with \
         render::auto_exposure::HISTOGRAM_BINS ({HISTOGRAM_BINS}) - edit both together"
    );
}

#[test]
fn min_log_luminance_matches_the_shader() {
    let shader_value: f32 = extract_const("MIN_LOG_LUMINANCE")
        .parse()
        .expect("MIN_LOG_LUMINANCE is not an f32 literal");
    assert_eq!(
        shader_value, MIN_LOG_LUMINANCE,
        "histogram.wgsl's MIN_LOG_LUMINANCE ({shader_value}) disagrees with \
         render::auto_exposure::MIN_LOG_LUMINANCE ({MIN_LOG_LUMINANCE}) - edit both together"
    );
}

#[test]
fn max_log_luminance_matches_the_shader() {
    let shader_value: f32 = extract_const("MAX_LOG_LUMINANCE")
        .parse()
        .expect("MAX_LOG_LUMINANCE is not an f32 literal");
    assert_eq!(
        shader_value, MAX_LOG_LUMINANCE,
        "histogram.wgsl's MAX_LOG_LUMINANCE ({shader_value}) disagrees with \
         render::auto_exposure::MAX_LOG_LUMINANCE ({MAX_LOG_LUMINANCE}) - edit both together"
    );
}

#[test]
fn exposure_key_matches_the_shader() {
    let shader_value: f32 = extract_const("EXPOSURE_KEY")
        .parse()
        .expect("EXPOSURE_KEY is not an f32 literal");
    assert_eq!(
        shader_value, EXPOSURE_KEY,
        "histogram.wgsl's EXPOSURE_KEY ({shader_value}) disagrees with \
         render::auto_exposure::EXPOSURE_KEY ({EXPOSURE_KEY}) - edit both together"
    );
}

#[test]
fn black_threshold_matches_the_shader() {
    let shader_value: f32 = extract_const("BLACK_THRESHOLD")
        .parse()
        .expect("BLACK_THRESHOLD is not an f32 literal");
    assert_eq!(
        shader_value, BLACK_THRESHOLD,
        "histogram.wgsl's BLACK_THRESHOLD ({shader_value}) disagrees with \
         render::auto_exposure::BLACK_THRESHOLD ({BLACK_THRESHOLD}) - edit both together"
    );
}

#[test]
fn build_workgroup_size_matches_the_shader() {
    let shader_value = extract_workgroup_x("cs_build_histogram");
    assert_eq!(
        shader_value, BUILD_WORKGROUP,
        "histogram.wgsl's cs_build_histogram @workgroup_size ({shader_value}) disagrees with \
         render::auto_exposure::BUILD_WORKGROUP ({BUILD_WORKGROUP}) - edit both together"
    );
}

#[test]
fn clear_workgroup_size_matches_the_shader() {
    let shader_value = extract_workgroup_x("cs_clear_histogram");
    assert_eq!(
        shader_value, CLEAR_WORKGROUP,
        "histogram.wgsl's cs_clear_histogram @workgroup_size ({shader_value}) disagrees with \
         render::auto_exposure::CLEAR_WORKGROUP ({CLEAR_WORKGROUP}) - edit both together"
    );
}
