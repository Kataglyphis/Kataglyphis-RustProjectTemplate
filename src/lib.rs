#![doc = include_str!("../docs/_static/getting-started.md")]

pub mod api;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
mod person_detection;
mod frb_generated;
