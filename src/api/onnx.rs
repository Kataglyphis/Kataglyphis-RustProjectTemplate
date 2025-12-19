use log::error;

/// Person detection result in *original image pixel coordinates*.
#[derive(Clone, Debug)]
pub struct PersonDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i64,
}

#[flutter_rust_bridge::frb(sync)]
pub fn detect_persons_rgba(
    model_path: String,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    score_threshold: f32,
) -> Vec<PersonDetection> {
    let resolved_model_path = if model_path.trim().is_empty() {
        crate::person_detection::default_model_path()
            .to_string_lossy()
            .to_string()
    } else {
        model_path
    };

    match detect_persons_rgba_impl(&resolved_model_path, &rgba, width, height, score_threshold) {
        Ok(v) => v,
        Err(e) => {
            error!("detect_persons_rgba failed: {e:#}");
            Vec::new()
        }
    }
}

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
fn detect_persons_rgba_impl(
    model_path: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    score_threshold: f32,
) -> anyhow::Result<Vec<PersonDetection>> {
    use std::sync::{Mutex, OnceLock};

    use crate::person_detection::PersonDetector;

    struct Cached {
        model_path: String,
        detector: PersonDetector,
    }

    static DETECTOR: OnceLock<Mutex<Option<Cached>>> = OnceLock::new();
    let mutex = DETECTOR.get_or_init(|| Mutex::new(None));

    let mut guard = mutex.lock().expect("Detector mutex poisoned");

    let needs_reload = guard
        .as_ref()
        .map(|c| c.model_path != model_path)
        .unwrap_or(true);

    if needs_reload {
        let detector = PersonDetector::new(model_path)?;
        *guard = Some(Cached {
            model_path: model_path.to_string(),
            detector,
        });
    }

    let detector = guard.as_ref().expect("Detector missing");
    let dets = detector
        .detector
        .infer_persons_rgba(rgba, width, height, score_threshold)?;

    Ok(dets
        .into_iter()
        .map(|d| PersonDetection {
            x1: d.x1,
            y1: d.y1,
            x2: d.x2,
            y2: d.y2,
            score: d.score,
            class_id: d.class_id,
        })
        .collect())
}

#[cfg(not(any(feature = "onnx_tract", feature = "onnxruntime")))]
fn detect_persons_rgba_impl(
    _model_path: &str,
    _rgba: &[u8],
    _width: u32,
    _height: u32,
    _score_threshold: f32,
) -> anyhow::Result<Vec<PersonDetection>> {
    anyhow::bail!("ONNX inference is disabled. Build with --features onnx_tract (or onnxruntime*)")
}
