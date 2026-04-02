#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use ::std::sync::Arc;
#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use ::std::time::{Duration, Instant};

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use std::sync::mpsc::{Receiver, SyncSender};

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use kataglyphis_core::detection::Detection;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use super::inference_bridge::{InferRequest, InferResult};
#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use super::pipeline::Frame;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
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

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
impl InferenceState {
    pub(crate) fn poll(&mut self) {
        if let Some(rx) = self.infer_rx.as_ref() {
            while let Ok(res) = rx.try_recv() {
                self.infer_in_flight = false;
                self.last_detections = res.detections;
                self.last_detections_frame_id = Some(res.frame_id);
                self.detector_error = res.error;
            }
        }
    }

    pub(crate) fn maybe_infer(&mut self, frame: &Frame) {
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
