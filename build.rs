// build.rs — cfg aliases + CXX bridge (when applicable).

fn main() {
    // ── cfg aliases ────────────────────────────────────────────────
    // Emit `cfg(onnx)` when *any* ONNX backend is enabled so that
    // source files can write `#[cfg(onnx)]` instead of the verbose
    // `#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]`.
    let has_onnx = cfg!(feature = "onnx_tract") || cfg!(feature = "onnxruntime");
    if has_onnx {
        println!("cargo:rustc-cfg=onnx");
    }

    // Emit `cfg(gui_wgpu_backend)` when either gui_windows or gui_linux
    // enables the WGPU-based GUI, regardless of host OS.
    let has_gui_wgpu = cfg!(feature = "gui_windows") || cfg!(feature = "gui_linux");
    if has_gui_wgpu {
        println!("cargo:rustc-cfg=gui_wgpu_backend");
    }

    // ── CXX bridge ─────────────────────────────────────────────────
    #[cfg(not(target_arch = "wasm32"))]
    {
        cxx_build::bridge("src/lib.rs")
            .flag_if_supported("-std=c++17")
            .compile("kataglyphis_cxx");
    }
}
