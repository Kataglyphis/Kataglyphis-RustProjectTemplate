use std::sync::Arc;
use std::time::Instant;

#[cfg(onnx)]
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

#[cfg(onnx)]
use crate::detection::Detection;

#[cfg(onnx)]
use super::pipeline::Frame;

#[cfg(onnx)]
pub(crate) struct InferRequest {
    pub frame_id: u64,
    pub rgba: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
    pub score_threshold: f32,
}

#[cfg(onnx)]
pub(crate) struct InferResult {
    pub frame_id: u64,
    pub detections: Vec<Detection>,
    pub error: Option<String>,
}

#[cfg(onnx)]
pub(crate) fn spawn_inference_thread() -> (
    Option<SyncSender<InferRequest>>,
    Option<Receiver<InferResult>>,
    Option<String>,
    Option<String>,
) {
    use crate::config;
    use crate::person_detection::{PersonDetector, default_model_path};
    use crate::resource_monitor;

    let path = config::onnx_model_override()
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| default_model_path().to_string_lossy().to_string());

    let (req_tx, req_rx) = sync_channel::<InferRequest>(1);
    let (res_tx, res_rx) = sync_channel::<InferResult>(1);

    let detector = match PersonDetector::new(&path) {
        Ok(detector) => Some(detector),
        Err(e) => {
            let err_msg = format!("Failed to load model '{path}': {e:#}");
            log::error!("{err_msg}");
            let _ = res_tx.try_send(InferResult {
                frame_id: 0,
                detections: Vec::new(),
                error: Some(err_msg),
            });
            None
        }
    };

    std::thread::spawn(move || {
        let Some(mut detector) = detector else {
            return;
        };

        while let Ok(req) = req_rx.recv() {
            let infer_start = Instant::now();
            let result = match detector.infer_persons_rgba(
                &req.rgba,
                req.width,
                req.height,
                req.score_threshold,
            ) {
                Ok(dets) => InferResult {
                    frame_id: req.frame_id,
                    detections: dets,
                    error: None,
                },
                Err(e) => {
                    let err_msg = format!("Inference failed: {e:#}");
                    log::warn!("{err_msg}");
                    InferResult {
                        frame_id: req.frame_id,
                        detections: Vec::new(),
                        error: Some(err_msg),
                    }
                }
            };

            resource_monitor::record_inference_duration(infer_start.elapsed());
            resource_monitor::record_inference_completion();

            if let Err(e) = res_tx.try_send(result) {
                log::warn!("Inference result channel full, dropping result: {e}");
            }
        }
    });

    (Some(req_tx), Some(res_rx), None, Some(path))
}
