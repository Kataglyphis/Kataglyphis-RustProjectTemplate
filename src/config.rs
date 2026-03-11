// src/config.rs — Centralised environment-variable configuration.
//
// All `KATAGLYPHIS_*` env vars are read here so that the configuration surface
// is discoverable and documented in a single place.
//
// Each accessor caches the parsed value in a `OnceLock` so that the env var is
// read at most once per process lifetime.

use std::sync::OnceLock;

use log::warn;

// ── Preprocessing ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreprocessMode {
    Letterbox,
    Stretch,
}

/// `KATAGLYPHIS_PREPROCESS`: `"letterbox"` | `"stretch"` (default `"stretch"`).
pub fn preprocess_mode() -> PreprocessMode {
    static CACHE: OnceLock<PreprocessMode> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = std::env::var("KATAGLYPHIS_PREPROCESS").unwrap_or_else(|_| "stretch".to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "letterbox" | "boxed" | "pad" => PreprocessMode::Letterbox,
            "stretch" | "resize" => PreprocessMode::Stretch,
            other => {
                warn!("Unknown preprocess mode '{other}', defaulting to stretch");
                PreprocessMode::Stretch
            }
        }
    })
}

/// `KATAGLYPHIS_SWAP_XY`: set to `"1"` to swap X/Y in YOLO output coords.
pub fn swap_xy_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("KATAGLYPHIS_SWAP_XY").ok().as_deref() == Some("1"))
}

// ── ONNX backend selection ─────────────────────────────────────────

/// `KATAGLYPHIS_ONNX_BACKEND`: `"ort"` / `"onnxruntime"` | `"tract"` | unset (auto).
pub fn onnx_backend() -> Option<String> {
    static CACHE: OnceLock<Option<String>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var("KATAGLYPHIS_ONNX_BACKEND")
                .ok()
                .map(|v| v.to_lowercase())
        })
        .clone()
}

/// `KATAGLYPHIS_ORT_DEVICE`: `"cpu"` (default) | `"cuda"` | `"auto"`.
pub fn ort_device() -> String {
    static CACHE: OnceLock<String> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var("KATAGLYPHIS_ORT_DEVICE")
                .unwrap_or_else(|_| "cpu".to_string())
                .to_ascii_lowercase()
        })
        .clone()
}

// ── Inference / GUI ────────────────────────────────────────────────

/// `KATAGLYPHIS_ONNX_MODEL`: override path to the ONNX model file.
pub fn onnx_model_override() -> Option<String> {
    static CACHE: OnceLock<Option<String>> = OnceLock::new();
    CACHE
        .get_or_init(|| std::env::var("KATAGLYPHIS_ONNX_MODEL").ok())
        .clone()
}

/// `KATAGLYPHIS_SCORE_THRESHOLD`: detection confidence threshold (default `0.5`).
pub fn score_threshold() -> f32 {
    static CACHE: OnceLock<f32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KATAGLYPHIS_SCORE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5)
    })
}

/// `KATAGLYPHIS_INFER_EVERY_MS`: minimum interval between inference requests (default `0`).
pub fn infer_every_ms() -> u64 {
    static CACHE: OnceLock<u64> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KATAGLYPHIS_INFER_EVERY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

// ── Logging ────────────────────────────────────────────────────────

/// `KATAGLYPHIS_LOG_LEVEL`: `error` | `warn` | `info` (default) | `debug` | `trace`.
pub fn log_level() -> log::LevelFilter {
    static CACHE: OnceLock<log::LevelFilter> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KATAGLYPHIS_LOG_LEVEL")
            .ok()
            .and_then(|v| match v.to_ascii_lowercase().as_str() {
                "error" => Some(log::LevelFilter::Error),
                "warn" | "warning" => Some(log::LevelFilter::Warn),
                "info" => Some(log::LevelFilter::Info),
                "debug" => Some(log::LevelFilter::Debug),
                "trace" => Some(log::LevelFilter::Trace),
                _ => None,
            })
            .unwrap_or(log::LevelFilter::Info)
    })
}
