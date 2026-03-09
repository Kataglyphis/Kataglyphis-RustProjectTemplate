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

#[cfg(feature = "burn_demos")]
pub mod burn_demos;

mod frb_generated;
#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
mod person_detection;

pub fn rusty_cxxbridge_integer() -> i32 {
    322
}

#[unsafe(no_mangle)]
pub extern "C" fn rusty_extern_c_integer() -> i32 {
    322
}
