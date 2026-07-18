//! Webcam live-inference API: camera enumeration + a detection event stream.
//!
//! Video frames never cross the bridge — they go straight from the capture
//! pipeline into the Flutter texture via the native plugin's C ABI. Only
//! detection metadata streams to Dart, where the UI overlays boxes.

use crate::frb_generated::StreamSink;

pub struct CameraDesc {
    pub index: u32,
    pub name: String,
}

/// A detection in source-frame pixel coordinates. Field-accessible Dart
/// mirror of `kataglyphis_core::detection::Detection` (which the bridge
/// would otherwise expose as an opaque type).
pub struct DetectionBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i64,
}

#[cfg(all(feature = "gstreamer", onnx))]
impl From<crate::Detection> for DetectionBox {
    fn from(d: crate::Detection) -> Self {
        Self {
            x1: d.x1,
            y1: d.y1,
            x2: d.x2,
            y2: d.y2,
            score: d.score,
            class_id: d.class_id,
        }
    }
}

pub struct WebcamStreamConfig {
    /// `videotestsrc` instead of a real camera (containers/CI have none).
    pub use_test_source: bool,
    /// Camera enumeration index; `None` = first available.
    pub device_index: Option<u32>,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub model_path: String,
    pub score_threshold: f32,
    /// Texture id from the native plugin's `create` call; `0` = headless.
    pub texture_id: i64,
    /// Override for the DLL exporting `knt_push_frame`.
    pub texture_library: Option<String>,
}

pub struct DetectionEvent {
    pub detections: Vec<DetectionBox>,
    pub frame_sequence: u64,
    pub pts_ms: Option<u64>,
    pub inference_ms: f64,
    pub fps: f32,
}

#[flutter_rust_bridge::frb(sync)]
pub fn list_cameras() -> Result<Vec<CameraDesc>, String> {
    #[cfg(feature = "gstreamer")]
    {
        kataglyphis_media::list_cameras()
            .map(|cams| {
                cams.into_iter()
                    .map(|c| CameraDesc {
                        index: c.index,
                        name: c.display_name,
                    })
                    .collect()
            })
            .map_err(|e| format!("camera enumeration failed: {e:#}"))
    }
    #[cfg(not(feature = "gstreamer"))]
    {
        Err("webcam capture is disabled. Build with --features gstreamer".into())
    }
}

/// Starts capture + inference; events arrive on `sink` until
/// [`stop_webcam_inference`] is called. Errors if already running.
pub fn start_webcam_inference(
    config: WebcamStreamConfig,
    sink: StreamSink<DetectionEvent>,
) -> Result<(), String> {
    #[cfg(all(feature = "gstreamer", onnx))]
    {
        enabled::start(config, sink).map_err(|e| {
            log::error!("start_webcam_inference failed: {e:#}");
            format!("start_webcam_inference failed: {e:#}")
        })
    }
    #[cfg(not(all(feature = "gstreamer", onnx)))]
    {
        let (_, _) = (config, sink);
        Err("webcam inference is disabled. Build with --features gstreamer,onnxruntime*".into())
    }
}

/// Stops the running session (no-op when idle).
#[flutter_rust_bridge::frb(sync)]
pub fn stop_webcam_inference() {
    #[cfg(all(feature = "gstreamer", onnx))]
    enabled::stop();
}

#[cfg(all(feature = "gstreamer", onnx))]
mod enabled {
    use std::sync::{Mutex, OnceLock};

    use anyhow::Result;
    use kataglyphis_media::CameraSource;

    use super::{DetectionEvent, StreamSink, WebcamStreamConfig};
    use crate::webcam_engine::{EngineConfig, TextureTarget, WebcamEngine};

    static ENGINE: OnceLock<Mutex<Option<WebcamEngine>>> = OnceLock::new();

    fn engine_slot() -> &'static Mutex<Option<WebcamEngine>> {
        ENGINE.get_or_init(|| Mutex::new(None))
    }

    fn lock() -> std::sync::MutexGuard<'static, Option<WebcamEngine>> {
        engine_slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn start(
        config: WebcamStreamConfig,
        sink: StreamSink<DetectionEvent>,
    ) -> Result<()> {
        let mut slot = lock();
        if slot.is_some() {
            anyhow::bail!("webcam inference is already running; stop it first");
        }

        let source = if config.use_test_source {
            CameraSource::Test
        } else {
            match config.device_index {
                Some(index) => CameraSource::Device(index),
                None => CameraSource::Auto,
            }
        };

        let texture = (config.texture_id != 0).then(|| TextureTarget {
            library: config
                .texture_library
                .clone()
                .unwrap_or_else(default_texture_library),
            texture_id: config.texture_id,
        });

        let engine = WebcamEngine::start(
            EngineConfig {
                source,
                width: config.width,
                height: config.height,
                framerate: config.framerate,
                model_path: config.model_path,
                score_threshold: config.score_threshold,
                texture,
            },
            Box::new(move |event| {
                let _ = sink.add(DetectionEvent {
                    detections: event.detections.into_iter().map(Into::into).collect(),
                    frame_sequence: event.frame_sequence,
                    pts_ms: event.pts_ms,
                    inference_ms: event.inference_ms,
                    fps: event.fps,
                });
            }),
        )?;

        *slot = Some(engine);
        Ok(())
    }

    pub(super) fn stop() {
        let engine = lock().take();
        if let Some(mut engine) = engine {
            engine.stop();
        }
    }

    fn default_texture_library() -> String {
        if cfg!(windows) {
            "kataglyphis_native_inference_plugin.dll".into()
        } else {
            "libkataglyphis_native_inference_plugin.so".into()
        }
    }
}
