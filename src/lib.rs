#![doc = include_str!("../docs/_static/getting-started.md")]

pub mod api;

#[cfg(feature = "burn_demos")]
pub mod burn_demos;

mod frb_generated;
#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
mod person_detection;
