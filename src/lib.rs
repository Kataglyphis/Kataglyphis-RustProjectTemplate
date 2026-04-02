//! Main entry point for Kataglyphis.

#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "../logo.png")]

#[cfg(not(target_arch = "wasm32"))]
mod native_only;

pub mod api;
pub use kataglyphis_core::config;
pub use kataglyphis_core::detection::Detection;
pub use kataglyphis_core::logging;
pub use kataglyphis_telemetry::resource_monitor;
pub mod platform;
pub mod utils;

#[cfg(target_os = "windows")]
pub(crate) use kataglyphis_telemetry::gpu_wmi;

#[cfg(feature = "onnxruntime")]
pub use kataglyphis_inference::ort_ext;

#[cfg(feature = "burn_demos")]
pub mod burn_demos;

mod frb_generated;
#[cfg(onnx)]
pub use kataglyphis_inference::person_detection;

/// C FFI demo stub — returns a fixed integer to verify `extern "C"` linkage works.
///
/// # Safety
/// Exported with `#[no_mangle]` for C interop. Callers must ensure this is invoked
/// according to the C calling convention.
#[unsafe(no_mangle)]
pub extern "C" fn rusty_extern_c_integer() -> i32 {
    322
}
