//! Machine learning and inference models.

#[cfg(feature = "onnxruntime")]
pub mod ort_ext;

#[cfg(feature = "onnx")]
pub mod person_detection;
