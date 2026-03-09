use std::sync::Arc;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use crate::person_detection::Detection;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
pub struct InferRequest {
    pub frame_id: u64,
    pub rgba: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub score_threshold: f32,
}

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
pub struct InferResult {
    pub frame_id: u64,
    pub detections: Vec<Detection>,
    pub error: Option<String>,
}
