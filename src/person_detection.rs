use anyhow::{Context, Result, bail};
use log::{debug, info, warn};

#[cfg(feature = "onnxruntime")]
use crate::ort_ext::{OrtResultExt, extract_first_f32_output};

use crate::config::{self, PreprocessMode};

pub fn default_model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("models")
        .join("yolov10m.onnx")
}

// `tvec!` is imported via `tract_onnx::prelude::*` in the tract code paths.

use crate::detection::Detection;

/// Minimal YOLO-style person detector.
///
/// Notes:
/// - This intentionally keeps post-processing simple and heuristic-based because
///   ONNX model output formats differ (e.g. Nx6, 1xNx6, end-to-end models).
/// - For YOLOv10 end-to-end exports, a common output shape is `[1, N, 6]` with
///   `xyxy + score + class`.
pub struct PersonDetector {
    backend: Backend,
    input_w: u32,
    input_h: u32,
    preprocess: PreprocessMode,
    swap_xy: bool,
    /// Reusable buffer for NCHW preprocessing — avoids a fresh allocation per frame.
    preprocess_buf: Vec<f32>,
}

#[cfg(feature = "onnx_tract")]
type TractPlan = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::TypedModel,
>;

enum Backend {
    #[cfg(feature = "onnx_tract")]
    Tract { model: Box<TractPlan> },

    #[cfg(feature = "onnxruntime")]
    Ort {
        session: std::sync::Mutex<ort::session::Session>,
    },
}

// ── ORT helpers are in crate::ort_ext ──────────────────────────────

// ── PersonDetector construction helpers ────────────────────────────

impl PersonDetector {
    /// Build a `PersonDetector` from an already-loaded backend and input dims.
    fn from_parts(
        backend: Backend,
        input_w: u32,
        input_h: u32,
        preprocess: PreprocessMode,
        swap_xy: bool,
    ) -> Self {
        let buf_len = 3 * (input_w as usize) * (input_h as usize);
        Self {
            backend,
            input_w,
            input_h,
            preprocess,
            swap_xy,
            preprocess_buf: vec![0.0; buf_len],
        }
    }

    pub fn new(model_path: &str) -> Result<Self> {
        info!("Loading ONNX model: {model_path}");

        let preprocess = config::preprocess_mode();
        let swap_xy = config::swap_xy_enabled();
        debug!("Preprocess mode: {:?}, swap_xy: {}", preprocess, swap_xy);

        let requested_backend = config::onnx_backend();

        match requested_backend.as_deref() {
            Some("ort") | Some("onnxruntime") => {
                #[cfg(feature = "onnxruntime")]
                {
                    let (session, (input_w, input_h)) = load_ort_session(model_path)?;
                    info!("ONNX backend: ort ({input_w}x{input_h})");
                    Ok(Self::from_parts(
                        Backend::Ort {
                            session: std::sync::Mutex::new(session),
                        },
                        input_w,
                        input_h,
                        preprocess,
                        swap_xy,
                    ))
                }

                #[cfg(not(feature = "onnxruntime"))]
                {
                    bail!("Requested ONNX backend 'ort', but feature 'onnxruntime' is not enabled")
                }
            }
            Some("tract") => {
                #[cfg(feature = "onnx_tract")]
                {
                    let (model, (input_w, input_h)) = load_tract_model(model_path)?;
                    info!("ONNX backend: tract ({input_w}x{input_h})");
                    Ok(Self::from_parts(
                        Backend::Tract {
                            model: Box::new(model),
                        },
                        input_w,
                        input_h,
                        preprocess,
                        swap_xy,
                    ))
                }

                #[cfg(not(feature = "onnx_tract"))]
                {
                    bail!("Requested ONNX backend 'tract', but feature 'onnx_tract' is not enabled")
                }
            }
            Some(other) => {
                bail!("Unsupported KATAGLYPHIS_ONNX_BACKEND='{other}'. Use 'tract' or 'ort'.")
            }
            None => {
                // Default: prefer ORT when available, fallback to tract.
                #[cfg(feature = "onnxruntime")]
                {
                    let device = config::ort_device();
                    let require_ort = device == "cuda" || device == "auto";

                    match load_ort_session(model_path) {
                        Ok((session, (input_w, input_h))) => {
                            info!("ONNX backend: ort ({input_w}x{input_h})");
                            return Ok(Self::from_parts(
                                Backend::Ort {
                                    session: std::sync::Mutex::new(session),
                                },
                                input_w,
                                input_h,
                                preprocess,
                                swap_xy,
                            ));
                        }
                        Err(err) => {
                            if require_ort {
                                return Err(err);
                            }
                            warn!("ORT session unavailable, falling back to tract: {err:#}");
                        }
                    }
                }

                #[cfg(feature = "onnx_tract")]
                {
                    let (model, (input_w, input_h)) = load_tract_model(model_path)?;
                    info!("ONNX backend: tract ({input_w}x{input_h})");
                    Ok(Self::from_parts(
                        Backend::Tract {
                            model: Box::new(model),
                        },
                        input_w,
                        input_h,
                        preprocess,
                        swap_xy,
                    ))
                }

                #[cfg(not(feature = "onnx_tract"))]
                {
                    bail!(
                        "No ONNX backend available. \
                         Build with --features onnx_tract to enable the tract fallback, \
                         or ensure the ORT session can be created."
                    )
                }
            }
        }
    }

    pub fn infer_persons_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        score_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let mapping = match self.preprocess {
            PreprocessMode::Letterbox => rgba_to_nchw_f32_letterboxed(
                rgba,
                width,
                height,
                self.input_w,
                self.input_h,
                &mut self.preprocess_buf,
            )?,
            PreprocessMode::Stretch => rgba_to_nchw_f32_stretched(
                rgba,
                width,
                height,
                self.input_w,
                self.input_h,
                &mut self.preprocess_buf,
            )?,
        };

        let (shape, data) = self
            .infer_raw_nchw_f32(&self.preprocess_buf)
            .context("Failed to run ONNX model")?;

        let mut detections =
            parse_yolo_like_detections(&shape, &data, score_threshold, mapping, self.swap_xy)?;

        // Keep only class 0 (= person) by convention.
        detections.retain(|d| d.class_id == 0);
        Ok(detections)
    }

    fn infer_raw_nchw_f32(&self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        match &self.backend {
            #[cfg(feature = "onnx_tract")]
            Backend::Tract { model } => {
                use tract_onnx::prelude::*;

                let shape = [1usize, 3usize, self.input_h as usize, self.input_w as usize];
                let tensor = Tensor::from_shape(&shape, input)?;
                let outputs = model
                    .run(tvec!(tensor.into()))
                    .context("Failed to run ONNX model (tract)")?;

                let out = outputs.first().context("Model returned no outputs")?;

                let shape = out.shape().to_vec();
                let data = out
                    .as_slice::<f32>()
                    .context("Model output is not contiguous f32")?;

                Ok((shape, data.to_vec()))
            }

            #[cfg(feature = "onnxruntime")]
            Backend::Ort { session } => {
                let mut session = session.lock().expect("ORT session mutex poisoned");
                // ORT's safe `Tensor::from_array` requires owned data; a zero-copy
                // view API is not available, so the copy from the reusable
                // preprocess buffer is unavoidable.
                let input_tensor = ort::value::Tensor::from_array((
                    [1usize, 3usize, self.input_h as usize, self.input_w as usize],
                    input.to_vec().into_boxed_slice(),
                ))
                .context("Failed to create ORT input tensor")?;

                let outputs = session
                    .run(ort::inputs![input_tensor])
                    .context("Failed to run ONNX model (ort)")?;

                let (shape, data) = extract_first_f32_output(&outputs)?;
                Ok((shape, data))
            }
        }
    }
}

/// Describes how an image was mapped into the model input.
#[derive(Clone, Copy, Debug)]
struct ImageMapping {
    src_w: u32,
    src_h: u32,
    scale_x: f32,
    scale_y: f32,
    pad_x: f32,
    pad_y: f32,
}

#[cfg(feature = "onnx_tract")]
fn load_tract_model(model_path: &str) -> Result<(TractPlan, (u32, u32))> {
    use tract_onnx::prelude::*;

    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .with_context(|| format!("Failed to load ONNX model from '{model_path}'"))?;

    // Try to infer a fixed input resolution from the first input fact.
    // If that fails, default to 640x640 (common YOLO exports).
    let (input_w, input_h) = {
        let fact = model
            .input_fact(0)
            .context("Model has no input[0]")?
            .clone();
        // Common layouts: NCHW => [1, 3, H, W]
        if let Some(shape) = fact.shape.as_concrete_finite()? {
            let shape: Vec<usize> = shape.to_vec();
            if shape.len() == 4 {
                let h = shape[2].max(1) as u32;
                let w = shape[3].max(1) as u32;
                (w.max(1), h.max(1))
            } else {
                (640, 640)
            }
        } else {
            (640, 640)
        }
    };

    // Force NCHW f32 input with fixed shape.
    model.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec!(1, 3, input_h as usize, input_w as usize),
        ),
    )?;

    let runnable = model.into_optimized()?.into_runnable()?;
    Ok((runnable, (input_w, input_h)))
}

#[cfg(feature = "onnxruntime")]
fn load_ort_session(model_path: &str) -> Result<(ort::session::Session, (u32, u32))> {
    use ort::session::Session;

    let mut builder = Session::builder().with_ort_context("Failed to create ORT SessionBuilder")?;

    #[cfg(feature = "onnxruntime_cuda")]
    {
        use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
        let device = config::ort_device();

        info!("ORT device request: {device}");

        if device == "cuda" || device == "auto" {
            #[cfg(all(feature = "onnxruntime_cuda", windows))]
            ensure_ort_cuda_provider_dylibs_next_to_exe()?;

            let cuda = CUDAExecutionProvider::default();
            match cuda.is_available() {
                Ok(true) => {}
                Ok(false) => {
                    bail!("ORT was built without CUDA support (CUDA EP unavailable).");
                }
                Err(err) => {
                    bail!("Failed to query CUDA EP availability: {err}");
                }
            }

            let cuda_result = builder
                .with_execution_providers([cuda.build().error_on_failure()])
                .with_ort_context("Failed to configure ORT CUDA execution provider");

            match cuda_result {
                Ok(b) => {
                    info!("ORT CUDA execution provider enabled");
                    builder = b;
                }
                Err(err) => {
                    if device == "cuda" {
                        return Err(err);
                    }
                    warn!("ORT CUDA provider unavailable, falling back to CPU: {err:#}");
                    builder = Session::builder()
                        .with_ort_context("Failed to recreate ORT SessionBuilder")?;
                }
            }
        }
    }

    #[cfg(all(feature = "onnxruntime_directml", windows))]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .with_ort_context("Failed to configure ORT DirectML execution provider")?;
        info!("ORT DirectML execution provider enabled");
    }

    let session = builder
        .commit_from_file(model_path)
        .with_ort_context("Failed to load ONNX model (ort)")?;
    info!("ORT session created from model file: {model_path}");

    // Try to extract NCHW input resolution from the first input's type info.
    // Falls back to 640x640 when the shape is dynamic or unavailable.
    let (input_w, input_h) = extract_ort_input_dims(&session);

    Ok((session, (input_w, input_h)))
}

/// Attempt to read `(W, H)` from the first input's tensor shape (`[N, C, H, W]`).
///
/// Returns `(640, 640)` when the shape is missing, non-tensor, or has dynamic
/// (negative) dimensions.
#[cfg(feature = "onnxruntime")]
fn extract_ort_input_dims(session: &ort::session::Session) -> (u32, u32) {
    const DEFAULT: (u32, u32) = (640, 640);

    let Some(first) = session.inputs().first() else {
        return DEFAULT;
    };
    let Some(shape) = first.dtype().tensor_shape() else {
        return DEFAULT;
    };
    // Expect NCHW: [batch, channels, height, width]
    if shape.len() != 4 {
        return DEFAULT;
    }
    let h = shape[2];
    let w = shape[3];
    if h <= 0 || w <= 0 {
        // Dynamic dimensions are represented as -1.
        return DEFAULT;
    }
    (w as u32, h as u32)
}

#[cfg(all(feature = "onnxruntime", feature = "onnxruntime_cuda", windows))]
fn ensure_ort_cuda_provider_dylibs_next_to_exe() -> Result<()> {
    use std::ffi::OsStr;
    use std::time::SystemTime;

    fn cache_root() -> Result<std::path::PathBuf> {
        if let Some(p) = std::env::var_os("ORT_CACHE_DIR") {
            return Ok(std::path::PathBuf::from(p));
        }

        let local_appdata = std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .context("LOCALAPPDATA is not set")?;

        Ok(local_appdata.join("ort.pyke.io"))
    }

    fn find_latest_file(root: &std::path::Path, file_name: &OsStr) -> Result<std::path::PathBuf> {
        let mut stack = vec![root.to_path_buf()];
        let mut best: Option<(SystemTime, std::path::PathBuf)> = None;

        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .with_context(|| format!("Failed to read dir '{}'", dir.display()))?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name() == Some(file_name) {
                    let meta = entry.metadata()?;
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
                        best = Some((mtime, path));
                    }
                }
            }
        }

        best.map(|(_, p)| p).with_context(|| {
            format!(
                "Could not locate '{}' under '{}'",
                file_name.to_string_lossy(),
                root.display()
            )
        })
    }

    let exe_dir = std::env::current_exe()
        .context("current_exe failed")?
        .parent()
        .map(|p| p.to_path_buf())
        .context("current_exe has no parent directory")?;

    let required = [
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];

    let already_present = required.iter().all(|name| exe_dir.join(name).is_file());
    if already_present {
        return Ok(());
    }

    let dfbin_dir = cache_root()?.join("dfbin");
    let shared_path = find_latest_file(&dfbin_dir, OsStr::new("onnxruntime_providers_shared.dll"))?;
    let src_dir = shared_path
        .parent()
        .map(|p| p.to_path_buf())
        .context("providers_shared.dll has no parent directory")?;

    for name in required {
        let dst = exe_dir.join(name);
        if dst.is_file() {
            continue;
        }

        let src = src_dir.join(name);
        if !src.is_file() {
            bail!("Required ORT CUDA DLL not found: '{}'", src.display());
        }

        std::fs::copy(&src, &dst).with_context(|| {
            format!("Failed to copy '{}' -> '{}'", src.display(), dst.display())
        })?;
        info!("Copied '{}' -> '{}'", src.display(), dst.display());
    }

    for name in required {
        if !exe_dir.join(name).is_file() {
            bail!(
                "ORT CUDA DLL missing after copy: '{}'",
                exe_dir.join(name).display()
            );
        }
    }

    Ok(())
}

// ── RGBA → NCHW preprocessing ──────────────────────────────────────

/// Validate that `rgba` has the expected length for `src_w * src_h * 4`.
#[inline]
fn validate_rgba(rgba: &[u8], src_w: u32, src_h: u32) -> Result<()> {
    let expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|p| p.checked_mul(4))
        .context("Image dimensions overflow")?;
    if rgba.len() != expected {
        bail!(
            "RGBA buffer length mismatch: got {}, expected {} ({}x{}x4)",
            rgba.len(),
            expected,
            src_w,
            src_h
        );
    }
    Ok(())
}

/// Write a single RGB pixel into a planar CHW buffer.
#[inline]
fn write_pixel_chw(chw: &mut [f32], plane: usize, dst_idx: usize, r: f32, g: f32, b: f32) {
    chw[dst_idx] = r;
    chw[plane + dst_idx] = g;
    chw[2 * plane + dst_idx] = b;
}

/// Sample an RGBA pixel from `src` using nearest-neighbour and normalise to [0,1].
#[inline]
fn sample_rgba_normalised(src: &[u8], sx: u32, sy: u32, src_w: u32) -> (f32, f32, f32) {
    let idx = ((sy * src_w + sx) * 4) as usize;
    let r = src[idx] as f32 / 255.0;
    let g = src[idx + 1] as f32 / 255.0;
    let b = src[idx + 2] as f32 / 255.0;
    (r, g, b)
}

fn rgba_to_nchw_f32_letterboxed(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    buf: &mut Vec<f32>,
) -> Result<ImageMapping> {
    validate_rgba(rgba, src_w, src_h)?;

    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (src_h as f32 * scale).round().max(1.0) as u32;
    let pad_x_i = (dst_w - new_w) / 2;
    let pad_y_i = (dst_h - new_h) / 2;

    let plane = (dst_w * dst_h) as usize;
    let total = 3 * plane;
    let fill = 114.0 / 255.0;

    buf.clear();
    buf.resize(total, fill);

    // Content region only — skip the pad rows/columns entirely.
    let content_x_end = pad_x_i + new_w;
    let content_y_end = pad_y_i + new_h;

    for dy in pad_y_i..content_y_end {
        let sy = ((dy - pad_y_i) * src_h) / new_h;
        for dx in pad_x_i..content_x_end {
            let sx = ((dx - pad_x_i) * src_w) / new_w;
            let dst_idx = (dy * dst_w + dx) as usize;
            let (r, g, b) = sample_rgba_normalised(rgba, sx, sy, src_w);
            write_pixel_chw(buf, plane, dst_idx, r, g, b);
        }
    }

    let mapping = ImageMapping {
        src_w,
        src_h,
        scale_x: scale,
        scale_y: scale,
        pad_x: (dst_w - new_w) as f32 / 2.0,
        pad_y: (dst_h - new_h) as f32 / 2.0,
    };

    Ok(mapping)
}

fn rgba_to_nchw_f32_stretched(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    buf: &mut Vec<f32>,
) -> Result<ImageMapping> {
    validate_rgba(rgba, src_w, src_h)?;

    let plane = (dst_w * dst_h) as usize;
    let total = 3 * plane;

    buf.clear();
    buf.resize(total, 0.0);

    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            let dst_idx = (dy * dst_w + dx) as usize;
            let (r, g, b) = sample_rgba_normalised(rgba, sx, sy, src_w);
            write_pixel_chw(buf, plane, dst_idx, r, g, b);
        }
    }

    let mapping = ImageMapping {
        src_w,
        src_h,
        scale_x: dst_w as f32 / src_w.max(1) as f32,
        scale_y: dst_h as f32 / src_h.max(1) as f32,
        pad_x: 0.0,
        pad_y: 0.0,
    };

    Ok(mapping)
}

// ── YOLO output parsing ────────────────────────────────────────────

fn parse_yolo_like_detections(
    output_shape: &[usize],
    data: &[f32],
    score_threshold: f32,
    mapping: ImageMapping,
    swap_xy: bool,
) -> Result<Vec<Detection>> {
    let (n, stride) = match output_shape.len() {
        2 => {
            let (n, s) = (output_shape[0], output_shape[1]);
            if s < 6 {
                bail!(
                    "Unexpected output shape {:?}; need at least 6 columns",
                    output_shape
                );
            }
            (n, s)
        }
        3 => {
            if output_shape[0] != 1 {
                bail!(
                    "Unexpected output shape {:?}; expected batch=1",
                    output_shape
                );
            }
            let (n, s) = (output_shape[1], output_shape[2]);
            if s < 6 {
                bail!(
                    "Unexpected output shape {:?}; need at least 6 columns",
                    output_shape
                );
            }
            (n, s)
        }
        _ => bail!("Unsupported output rank: {:?}", output_shape),
    };

    let mut detections = Vec::with_capacity(n.min(64));

    for i in 0..n {
        let row = i * stride;
        if row + 6 > data.len() {
            break;
        }

        let mut x1 = data[row];
        let mut y1 = data[row + 1];
        let mut x2 = data[row + 2];
        let mut y2 = data[row + 3];
        let score = data[row + 4];
        let class_id = data[row + 5] as i64;

        if !score.is_finite() || score < score_threshold {
            continue;
        }

        if swap_xy {
            std::mem::swap(&mut x1, &mut y1);
            std::mem::swap(&mut x2, &mut y2);
        }

        let (x1, y1) = unmap_point(x1, y1, mapping);
        let (x2, y2) = unmap_point(x2, y2, mapping);

        detections.push(Detection {
            x1: x1.clamp(0.0, mapping.src_w as f32),
            y1: y1.clamp(0.0, mapping.src_h as f32),
            x2: x2.clamp(0.0, mapping.src_w as f32),
            y2: y2.clamp(0.0, mapping.src_h as f32),
            score,
            class_id,
        });
    }

    Ok(detections)
}

#[inline]
fn unmap_point(x: f32, y: f32, mapping: ImageMapping) -> (f32, f32) {
    let x = (x - mapping.pad_x) / mapping.scale_x.max(1e-6);
    let y = (y - mapping.pad_y) / mapping.scale_y.max(1e-6);
    (x, y)
}
