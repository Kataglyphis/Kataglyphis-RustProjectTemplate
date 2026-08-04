//! Pins `sky.wgsl`'s analytic sky gradient (emitted from Slang's
//! `common/sky_model.slang`) against the Rust copy in `render::ibl` that
//! `EnvironmentImage::sky` panoramises into the IBL fallback environment.
//!
//! The two feed each other's ambient (see `docs/shader-sharing.md`'s "Give
//! the analytic sky one definition" entry): a drift here would not crash
//! anything, it would just make the sky the camera sees disagree with the
//! sky baked into the reflections around it.
//!
//! Deliberately separate from any `GpuContext`-backed test: this one is pure
//! CPU so it runs everywhere, including environments with no adapter.

use kataglyphis_webgpu_renderer::render::ibl::{SKY_GROUND, SKY_HORIZON, SKY_ZENITH};

const SHADER_SOURCE: &str = include_str!("../src/shaders/sky.wgsl");

/// Finds the body of `fn <name>(...)` in [`SHADER_SOURCE`] (from its opening
/// `{` to the matching `}`) and returns it.
fn extract_fn_body(name: &str) -> &'static str {
    let fn_needle = format!("fn {name}(");
    let fn_pos = SHADER_SOURCE
        .find(&fn_needle)
        .unwrap_or_else(|| panic!("sky.wgsl no longer declares `fn {name}`"));
    let after_sig = &SHADER_SOURCE[fn_pos..];
    let body_start = after_sig
        .find('{')
        .unwrap_or_else(|| panic!("`fn {name}` has no body"));

    let mut depth = 0i32;
    for (i, c) in after_sig[body_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &after_sig[body_start..body_start + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("`fn {name}`'s body is never closed");
}

/// Extracts every flat `vec3<f32>(a, b, c)` literal (three comma-separated
/// numbers, no nested calls) from `source`, in order of appearance. Skips
/// `vec3<f32>(pow(...))`-style splats, which nest a call and so never match.
fn extract_flat_vec3_literals(source: &str) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    let needle = "vec3<f32>(";
    let mut search_from = 0usize;
    while let Some(rel) = source[search_from..].find(needle) {
        let open = search_from + rel + needle.len();
        let mut depth = 1i32;
        let mut close = open;
        for (i, c) in source[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = &source[open..close];
        if !inner.contains('(') {
            let parts: Vec<f32> = inner
                .split(',')
                .map(|p| p.trim().trim_end_matches('f').parse())
                .collect::<Result<_, _>>()
                .unwrap_or_default();
            if let [a, b, c] = parts[..] {
                out.push([a, b, c]);
            }
        }
        search_from = close + 1;
    }
    out
}

#[test]
fn sky_gradient_matches_the_shader() {
    let literals = extract_flat_vec3_literals(extract_fn_body("sky_gradient_0"));
    assert_eq!(
        literals.len(),
        4,
        "sky_gradient_0 in sky.wgsl should inline exactly 4 flat vec3<f32> \
         literals (horizon, zenith, horizon, ground); found {literals:?} - \
         did the gradient shape change?"
    );
    let (horizon, zenith, ground) = (literals[0], literals[1], literals[3]);

    assert_eq!(
        horizon, SKY_HORIZON,
        "sky.wgsl's horizon literal ({horizon:?}) disagrees with \
         render::ibl::SKY_HORIZON ({SKY_HORIZON:?}) - edit both together"
    );
    assert_eq!(
        zenith, SKY_ZENITH,
        "sky.wgsl's zenith literal ({zenith:?}) disagrees with \
         render::ibl::SKY_ZENITH ({SKY_ZENITH:?}) - edit both together"
    );
    assert_eq!(
        ground, SKY_GROUND,
        "sky.wgsl's ground literal ({ground:?}) disagrees with \
         render::ibl::SKY_GROUND ({SKY_GROUND:?}) - edit both together"
    );
}
