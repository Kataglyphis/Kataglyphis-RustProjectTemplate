//! Webcam → ONNX person-detection engine (native only).
//!
//! Owns a [`CaptureSession`] plus a worker thread that pushes every processed
//! frame into the Flutter texture (via the native plugin's `knt_push_frame`
//! C ABI) and runs person detection on it, emitting metadata through a
//! callback that the frb layer forwards to Dart as a stream.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use kataglyphis_media::{CameraSource, CaptureConfig, CaptureSession};

use crate::person_detection::PersonDetector;
use crate::Detection;

pub struct EngineEvent {
    pub detections: Vec<Detection>,
    pub frame_sequence: u64,
    pub pts_ms: Option<u64>,
    pub inference_ms: f64,
    pub fps: f32,
}

pub struct EngineConfig {
    pub source: CameraSource,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    pub model_path: String,
    pub score_threshold: f32,
    /// Flutter texture to feed; `None` runs headless (container CI, tests).
    pub texture: Option<TextureTarget>,
}

pub struct TextureTarget {
    /// DLL exporting the `knt_*` C ABI (the native inference plugin).
    pub library: String,
    pub texture_id: i64,
}

type PushFrameFn =
    unsafe extern "C" fn(texture_id: i64, rgba: *const u8, width: u32, height: u32) -> i32;

struct TexturePusher {
    /// Keeps the plugin DLL pinned while `push` is callable.
    _lib: libloading::Library,
    push: PushFrameFn,
    texture_id: i64,
}

impl TexturePusher {
    fn new(target: &TextureTarget) -> Result<Self> {
        // The plugin DLL is already loaded in-process by the Flutter engine;
        // this bumps its refcount and resolves the symbol.
        unsafe {
            let lib = libloading::Library::new(&target.library)
                .with_context(|| format!("failed to open texture library `{}`", target.library))?;
            let sym: libloading::Symbol<PushFrameFn> = lib
                .get(b"knt_push_frame\0")
                .context("`knt_push_frame` not exported by texture library")?;
            let push = *sym;
            Ok(Self {
                _lib: lib,
                push,
                texture_id: target.texture_id,
            })
        }
    }

    fn push(&self, rgba: &[u8], width: u32, height: u32) -> bool {
        unsafe { (self.push)(self.texture_id, rgba.as_ptr(), width, height) == 0 }
    }
}

pub struct WebcamEngine {
    capture: CaptureSession,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl WebcamEngine {
    pub fn start(
        config: EngineConfig,
        on_event: Box<dyn Fn(EngineEvent) + Send + 'static>,
    ) -> Result<Self> {
        resolve_ort_dylib();

        let model_path = crate::person_detection::resolve_model_path(Some(&config.model_path));
        let mut detector = PersonDetector::new(&model_path)?;

        let pusher = config
            .texture
            .as_ref()
            .map(TexturePusher::new)
            .transpose()?;

        let capture = CaptureSession::start(CaptureConfig {
            source: config.source,
            width: config.width,
            height: config.height,
            framerate: config.framerate,
        })?;
        let frames = capture.frames();

        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = stop.clone();
        let score_threshold = config.score_threshold;

        let worker = std::thread::Builder::new()
            .name("webcam-inference".into())
            .spawn(move || {
                let mut recent: VecDeque<Instant> = VecDeque::with_capacity(32);
                while !stop_worker.load(Ordering::Acquire) {
                    let Some(frame) = frames.take_latest(Duration::from_millis(200)) else {
                        if frames.is_closed() {
                            break;
                        }
                        continue;
                    };

                    if let Some(pusher) = &pusher {
                        if !pusher.push(&frame.rgba, frame.width, frame.height) {
                            log::warn!("texture push rejected frame {}", frame.sequence);
                        }
                    }

                    let started = Instant::now();
                    let detections = match detector.infer_persons_rgba(
                        &frame.rgba,
                        frame.width,
                        frame.height,
                        score_threshold,
                    ) {
                        Ok(d) => d,
                        Err(e) => {
                            log::error!("inference failed on frame {}: {e:#}", frame.sequence);
                            continue;
                        }
                    };
                    let inference_ms = started.elapsed().as_secs_f64() * 1000.0;

                    recent.push_back(Instant::now());
                    while recent.len() > 30 {
                        recent.pop_front();
                    }
                    let fps = match (recent.front(), recent.back()) {
                        (Some(first), Some(last)) if recent.len() >= 2 => {
                            let span = last.duration_since(*first).as_secs_f32();
                            if span > 0.0 {
                                (recent.len() - 1) as f32 / span
                            } else {
                                0.0
                            }
                        }
                        _ => 0.0,
                    };

                    on_event(EngineEvent {
                        detections,
                        frame_sequence: frame.sequence,
                        pts_ms: frame.pts_ms,
                        inference_ms,
                        fps,
                    });
                }
            })
            .context("failed to spawn webcam inference worker")?;

        Ok(Self {
            capture,
            stop,
            worker: Some(worker),
        })
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.capture.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for WebcamEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

/// With ort's `load-dynamic`, the DLL location must be known before the first
/// ort call. Resolution: explicit env → next to the exe → dev-image prefix.
/// A no-op (unused env var) for statically linked ort builds.
fn resolve_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("onnxruntime.dll"));
        }
    }
    #[cfg(windows)]
    candidates.push(r"C:\runtime\lib\onnxruntime-source\bin\onnxruntime.dll".into());
    if let Some(path) = candidates.into_iter().find(|p| p.is_file()) {
        log::info!("ORT_DYLIB_PATH not set, using {}", path.display());
        std::env::set_var("ORT_DYLIB_PATH", &path);
    }
}
