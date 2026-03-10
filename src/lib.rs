#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "../logo.png")]

#[cfg(not(target_arch = "wasm32"))]
#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn rusty_cxxbridge_integer() -> i32;
    }
}

pub mod api;
pub mod detection;

#[cfg(feature = "burn_demos")]
pub mod burn_demos;

mod frb_generated;
#[cfg(onnx)]
pub mod person_detection;

/// CXX bridge demo stub — returns a fixed integer to verify the Rust-C++ bridge works.
pub fn rusty_cxxbridge_integer() -> i32 {
    322
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
