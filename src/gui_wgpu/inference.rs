use std::sync::Arc;

#[cfg(onnx)]
use crate::detection::Detection;

#[cfg(onnx)]
pub struct InferRequest {
    pub frame_id: u64,
    pub rgba: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub score_threshold: f32,
}

#[cfg(onnx)]
pub struct InferResult {
    pub frame_id: u64,
    pub detections: Vec<Detection>,
    pub error: Option<String>,
}
