use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(onnx)]
use std::sync::mpsc::{Receiver, SyncSender};

#[cfg(onnx)]
use crate::detection::Detection;

#[cfg(onnx)]
pub struct InferRequest {
    pub frame_id: u64,
    pub rgba: Arc<[u8]>,
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

/// State for the background inference thread and its results.
#[cfg(onnx)]
pub(crate) struct InferenceState {
    pub detector_error: Option<String>,
    pub model_path: Option<String>,
    pub score_threshold: f32,
    pub infer_every: Duration,
    pub last_infer: Instant,
    pub last_detections: Vec<Detection>,
    pub last_detections_frame_id: Option<u64>,
    pub infer_tx: Option<SyncSender<InferRequest>>,
    pub infer_rx: Option<Receiver<InferResult>>,
    pub infer_in_flight: bool,
    pub infer_enabled: bool,
}

#[cfg(onnx)]
impl InferenceState {
    pub fn poll(&mut self) {
        if let Some(rx) = self.infer_rx.as_ref() {
            while let Ok(res) = rx.try_recv() {
                self.infer_in_flight = false;
                self.last_detections = res.detections;
                self.last_detections_frame_id = Some(res.frame_id);
                self.detector_error = res.error;
            }
        }
    }

    pub fn maybe_infer(&mut self, frame: &super::pipeline::Frame) {
        let Some(tx) = self.infer_tx.as_ref() else {
            return;
        };

        if !self.infer_enabled {
            return;
        }

        if self.last_infer.elapsed() < self.infer_every {
            return;
        }

        if self.infer_in_flight {
            return;
        }

        if tx
            .try_send(InferRequest {
                frame_id: frame.id,
                rgba: Arc::clone(&frame.data),
                width: frame.width,
                height: frame.height,
                score_threshold: self.score_threshold,
            })
            .is_ok()
        {
            self.last_infer = Instant::now();
            self.infer_in_flight = true;
        }
    }
}
