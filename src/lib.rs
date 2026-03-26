#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "../logo.png")]

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn rusty_cxxbridge_integer() -> i32;
    }
}

pub mod api;
pub mod config;
pub mod detection;
pub mod logging;
pub mod resource_monitor;
pub mod utils;

#[cfg(target_os = "windows")]
pub(crate) mod gpu_wmi;

#[cfg(feature = "onnxruntime")]
pub(crate) mod ort_ext;

#[cfg(feature = "burn_demos")]
pub mod burn_demos;

mod frb_generated;
#[cfg(onnx)]
pub(crate) mod person_detection;

#[cfg(all(feature = "gui_unix", not(windows)))]
pub mod gui;
#[cfg(feature = "gui_windows")]
pub mod gui_wgpu;

/// CXX bridge demo stub — returns a fixed integer to verify the Rust-C++ bridge works.
#[cfg(all(feature = "with_cxxbridge", not(target_arch = "wasm32")))]
pub fn rusty_cxxbridge_integer() -> i32 {
    322
}

/// wasm fallback when building for wasm32 (or when the bridge feature is not enabled for wasm).
#[cfg(target_arch = "wasm32")]
pub fn rusty_cxxbridge_integer() -> i32 {
    0
}

/// C FFI demo stub — returns a fixed integer to verify `extern "C"` linkage works.
///
/// # Safety
/// Exported with `#[no_mangle]` for C interop. Callers must ensure this is invoked
/// according to the C calling convention.
#[unsafe(no_mangle)]
pub extern "C" fn rusty_extern_c_integer() -> i32 {
    322
}
