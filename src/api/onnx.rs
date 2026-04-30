use log::error;

pub use crate::Detection;

#[flutter_rust_bridge::frb(sync)]
pub fn detect_persons_rgba(
    model_path: String,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    score_threshold: f32,
) -> Result<Vec<Detection>, String> {
    // Note: `person_detection` is feature-gated; keep this function compilable even when
    // ONNX features are disabled (e.g. for WASM builds).
    let resolved_model_path = {
        #[cfg(onnx)]
        {
            crate::person_detection::resolve_model_path(Some(&model_path))
        }

        #[cfg(not(onnx))]
        {
            model_path
        }
    };

    detect_persons_rgba_impl(&resolved_model_path, &rgba, width, height, score_threshold).map_err(
        |e| {
            error!("detect_persons_rgba failed: {e:#}");
            format!("detect_persons_rgba failed: {e:#}")
        },
    )
}

#[cfg(onnx)]
fn detect_persons_rgba_impl(
    model_path: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    score_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    use std::sync::{Mutex, OnceLock};

    use crate::person_detection::PersonDetector;

    struct Cached {
        model_path: String,
        detector: PersonDetector,
    }

    static DETECTOR: OnceLock<Mutex<Option<Cached>>> = OnceLock::new();
    let mutex = DETECTOR.get_or_init(|| Mutex::new(None));

    let lock_guard = || {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };

    // Check whether a reload is needed while holding the lock briefly.
    let needs_reload = {
        let guard = lock_guard();
        guard
            .as_ref()
            .map(|c| c.model_path != model_path)
            .unwrap_or(true)
    };

    // Load the model *outside* the lock so concurrent callers aren't blocked
    // for the (potentially multi-second) model load.
    if needs_reload {
        let detector = PersonDetector::new(model_path)?;
        let mut guard = lock_guard();
        // Re-check: another thread may have loaded the same model while we
        // were loading ours.
        let still_needs = guard
            .as_ref()
            .map(|c| c.model_path != model_path)
            .unwrap_or(true);
        if still_needs {
            *guard = Some(Cached {
                model_path: model_path.to_string(),
                detector,
            });
        }
    }

    let mut guard = lock_guard();
    let Some(cached) = guard.as_mut() else {
        return Ok(Vec::new());
    };
    cached
        .detector
        .infer_persons_rgba(rgba, width, height, score_threshold)
}

#[cfg(not(onnx))]
fn detect_persons_rgba_impl(
    _model_path: &str,
    _rgba: &[u8],
    _width: u32,
    _height: u32,
    _score_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    anyhow::bail!("ONNX inference is disabled. Build with --features onnx_tract (or onnxruntime*)")
}
