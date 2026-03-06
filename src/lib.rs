#![doc = include_str!("../docs/_static/getting-started.md")]

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
