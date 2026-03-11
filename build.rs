// build.rs — cfg aliases + CXX bridge (when applicable).

/// Returns `true` if the Cargo feature `name` is enabled for the *crate* being
/// built.  Inside a build script `cfg!()` reflects the build-script's own
/// compilation, **not** the crate's features.  Checking `CARGO_FEATURE_*`
/// environment variables is the correct mechanism.
fn has_feature(name: &str) -> bool {
    // Cargo upper-cases the feature name and replaces `-` with `_`.
    let var = format!("CARGO_FEATURE_{}", name.to_uppercase().replace('-', "_"));
    std::env::var_os(&var).is_some()
}

fn main() {
    // ── cfg aliases ────────────────────────────────────────────────
    // Emit `cfg(onnx)` when *any* ONNX backend is enabled so that
    // source files can write `#[cfg(onnx)]` instead of the verbose
    // `#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]`.
    if has_feature("onnx_tract") || has_feature("onnxruntime") {
        println!("cargo:rustc-cfg=onnx");
    }

    // Emit `cfg(gui_wgpu_backend)` when either gui_windows or gui_linux
    // enables the WGPU-based GUI, regardless of host OS.
    if has_feature("gui_windows") || has_feature("gui_linux") {
        println!("cargo:rustc-cfg=gui_wgpu_backend");
    }

    // ── CXX bridge ─────────────────────────────────────────────────
    // NOTE: `#[cfg(not(target_arch = "wasm32"))]` checks the *host* triple
    // inside a build script, NOT the crate's target.  Use the `CARGO_CFG_*`
    // environment variable so that cross-compiling to wasm32 correctly skips
    // the CXX bridge build.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "wasm32" {
        cxx_build::bridge("src/lib.rs")
            .flag_if_supported("-std=c++17")
            .compile("kataglyphis_cxx");
    }
}
