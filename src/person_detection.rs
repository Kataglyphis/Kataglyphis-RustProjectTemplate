use anyhow::{bail, Context, Result};
use log::{debug, info, warn};

pub fn default_model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("models")
        .join("yolov10m.onnx")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreprocessMode {
    Letterbox,
    Stretch,
}

fn preprocess_mode() -> PreprocessMode {
    let raw = std::env::var("KATAGLYPHIS_PREPROCESS")
        .unwrap_or_else(|_| "stretch".to_string());

    match raw.trim().to_ascii_lowercase().as_str() {
        "letterbox" | "boxed" | "pad" => PreprocessMode::Letterbox,
        "stretch" | "resize" => PreprocessMode::Stretch,
        other => {
            warn!("Unknown preprocess mode '{other}', defaulting to stretch");
            PreprocessMode::Stretch
        }
    }
}

fn swap_xy_enabled() -> bool {
    std::env::var("KATAGLYPHIS_SWAP_XY").ok().as_deref() == Some("1")
}

// `tvec!` is imported via `tract_onnx::prelude::*` in the tract code paths.

/// A single detection in *original image pixel coordinates*.
#[derive(Clone, Debug)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i64,
}

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
}

#[cfg(feature = "onnx_tract")]
type TractPlan = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::TypedModel,
>;

enum Backend {
    #[cfg(feature = "onnx_tract")]
    Tract { model: TractPlan },

    #[cfg(feature = "onnxruntime")]
    Ort { session: std::sync::Mutex<ort::session::Session> },
}

impl PersonDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        info!("Loading ONNX model: {model_path}");
        let requested_backend = std::env::var("KATAGLYPHIS_ONNX_BACKEND")
            .ok()
            .map(|v| v.to_lowercase());

        match requested_backend.as_deref() {
            Some("ort") | Some("onnxruntime") => {
                #[cfg(feature = "onnxruntime")]
                {
                    let (session, (input_w, input_h)) = load_ort_session(model_path)?;
                    info!("ONNX backend: ort ({}x{})", input_w, input_h);
                    debug!("Preprocess mode: {:?}", preprocess_mode());
                    return Ok(Self {
                        backend: Backend::Ort {
                            session: std::sync::Mutex::new(session),
                        },
                        input_w,
                        input_h,
                    });
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
                    info!("ONNX backend: tract ({}x{})", input_w, input_h);
                    debug!("Preprocess mode: {:?}", preprocess_mode());
                    return Ok(Self {
                        backend: Backend::Tract { model },
                        input_w,
                        input_h,
                    });
                }

                #[cfg(not(feature = "onnx_tract"))]
                {
                    bail!("Requested ONNX backend 'tract', but feature 'onnx_tract' is not enabled")
                }
            }
            Some(other) => bail!(
                "Unsupported KATAGLYPHIS_ONNX_BACKEND='{other}'. Use 'tract' or 'ort'."
            ),
            None => {
                // Default: prefer ORT when available, fallback to tract.
                #[cfg(feature = "onnxruntime")]
                {
                    let device = std::env::var("KATAGLYPHIS_ORT_DEVICE")
                        .unwrap_or_else(|_| "cpu".to_string())
                        .to_ascii_lowercase();
                    let require_ort = device == "cuda" || device == "auto";

                    match load_ort_session(model_path) {
                        Ok((session, (input_w, input_h))) => {
                            info!("ONNX backend: ort ({}x{})", input_w, input_h);
                            debug!("Preprocess mode: {:?}", preprocess_mode());
                            return Ok(Self {
                                backend: Backend::Ort {
                                    session: std::sync::Mutex::new(session),
                                },
                                input_w,
                                input_h,
                            });
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
                    info!("ONNX backend: tract ({}x{})", input_w, input_h);
                    debug!("Preprocess mode: {:?}", preprocess_mode());
                    return Ok(Self {
                        backend: Backend::Tract { model },
                        input_w,
                        input_h,
                    });
                }

                #[cfg(not(any(feature = "onnx_tract", feature = "onnxruntime")))]
                {
                    bail!("ONNX inference is disabled. Build with --features onnx_tract (or onnxruntime*)")
                }
            }
        }
    }

    pub fn infer_persons_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        score_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let (input, mapping) = match preprocess_mode() {
            PreprocessMode::Letterbox => rgba_to_nchw_f32_letterboxed(
                rgba,
                width,
                height,
                self.input_w,
                self.input_h,
            )?,
            PreprocessMode::Stretch => rgba_to_nchw_f32_stretched(
                rgba,
                width,
                height,
                self.input_w,
                self.input_h,
            )?,
        };

        let (shape, data) = self
            .infer_raw_nchw_f32(input)
            .context("Failed to run ONNX model")?;

        let detections = parse_yolo_like_detections(&shape, &data, score_threshold, mapping)?;

        // Keep only class 0 (= person) by convention.
        Ok(detections
            .into_iter()
            .filter(|d| d.class_id == 0)
            .collect())
    }

    fn infer_raw_nchw_f32(&self, input: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        match &self.backend {
            #[cfg(feature = "onnx_tract")]
            Backend::Tract { model } => {
                use tract_onnx::prelude::*;

                let shape = [1usize, 3usize, self.input_h as usize, self.input_w as usize];
                let tensor = Tensor::from_shape(&shape, &input)?;
                let outputs = model
                    .run(tvec!(tensor.into()))
                    .context("Failed to run ONNX model (tract)")?;

                let out = outputs
                    .first()
                    .context("Model returned no outputs")?;

                let shape = out.shape().to_vec();
                let data = out
                    .as_slice::<f32>()
                    .context("Model output is not contiguous f32")?;

                Ok((shape, data.to_vec()))
            }

            #[cfg(feature = "onnxruntime")]
            Backend::Ort { session } => {
                let mut session = session.lock().expect("ORT session mutex poisoned");
                let input_tensor = ort::value::Tensor::from_array((
                    [1usize, 3usize, self.input_h as usize, self.input_w as usize],
                    input.into_boxed_slice(),
                ))
                .context("Failed to create ORT input tensor")?;

                let outputs = session
                    .run(ort::inputs![input_tensor])
                    .context("Failed to run ONNX model (ort)")?;

                if outputs.len() == 0 {
                    bail!("Model returned no outputs");
                }
                let out = &outputs[0];
                let (shape, data) = out
                    .try_extract_tensor::<f32>()
                    .context("Failed to extract f32 tensor from ORT output")?;

                let shape: Vec<usize> = shape
                    .iter()
                    .map(|d| {
                        if *d < 0 {
                            bail!("ORT output had dynamic/negative dimension: {d}")
                        }
                        Ok(*d as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok((shape, data.to_vec()))
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
fn load_tract_model(
    model_path: &str,
) -> Result<(
    TractPlan,
    (u32, u32),
)> {
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
        InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, input_h as usize, input_w as usize)),
    )?;

    let runnable = model.into_optimized()?.into_runnable()?;
    Ok((runnable, (input_w, input_h)))
}

#[cfg(feature = "onnxruntime")]
fn load_ort_session(model_path: &str) -> Result<(ort::session::Session, (u32, u32))> {
    use ort::session::Session;

    let mut builder = Session::builder().context("Failed to create ORT SessionBuilder")?;

    #[cfg(feature = "onnxruntime_cuda")]
    {
        use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
        let device = std::env::var("KATAGLYPHIS_ORT_DEVICE")
            .unwrap_or_else(|_| "cpu".to_string())
            .to_ascii_lowercase();

        info!("ORT device request: {device}");

        if device == "cuda" || device == "auto" {
            #[cfg(all(feature = "onnxruntime_cuda", windows))]
            ensure_ort_cuda_provider_dylibs_next_to_exe()?;

            let cuda = CUDAExecutionProvider::default();
            match cuda.is_available() {
                Ok(true) => {}
                Ok(false) => {
                    return Err(anyhow::anyhow!(
                        "ORT was built without CUDA support (CUDA EP unavailable)."
                    ));
                }
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to query CUDA EP availability: {err}"
                    ));
                }
            }

            let cuda_result = builder
                .with_execution_providers([cuda.build().error_on_failure()])
                .context("Failed to configure ORT CUDA execution provider");

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
                        .context("Failed to recreate ORT SessionBuilder")?;
                }
            }
        }
    }

    #[cfg(all(feature = "onnxruntime_directml", windows))]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .context("Failed to configure ORT DirectML execution provider")?;
            info!("ORT DirectML execution provider enabled");
    }

    let session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from '{model_path}' (ort)"))?;
    info!("ORT session created from model file");

    let (input_w, input_h) = (640, 640);

    Ok((session, (input_w, input_h)))
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
                    if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
                        best = Some((mtime, path));
                    }
                }
            }
        }

        best.map(|(_, p)| p).with_context(|| {
            format!("Could not locate '{}' under '{}'", file_name.to_string_lossy(), root.display())
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
            return Err(anyhow::anyhow!(
                "Required ORT CUDA DLL not found: '{}'",
                src.display()
            ));
        }

        std::fs::copy(&src, &dst).with_context(|| {
            format!("Failed to copy '{}' -> '{}'", src.display(), dst.display())
        })?;
        info!("Copied '{}' -> '{}'", src.display(), dst.display());
    }

    for name in required {
        if !exe_dir.join(name).is_file() {
            return Err(anyhow::anyhow!(
                "ORT CUDA DLL missing after copy: '{}'",
                exe_dir.join(name).display()
            ));
        }
    }

    Ok(())
}

fn rgba_to_nchw_f32_letterboxed(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<(Vec<f32>, ImageMapping)> {
    let expected_len = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|p| p.checked_mul(4))
        .context("Image dimensions overflow")?;
    if rgba.len() != expected_len {
        bail!(
            "RGBA buffer length mismatch: got {}, expected {} ({}x{}x4)",
            rgba.len(),
            expected_len,
            src_w,
            src_h
        );
    }

    // Letterbox to dst size (keep aspect ratio)
    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (src_h as f32 * scale).round().max(1.0) as u32;
    let pad_x_i = (dst_w - new_w) / 2;
    let pad_y_i = (dst_h - new_h) / 2;
    let pad_x = (dst_w - new_w) as f32 / 2.0;
    let pad_y = (dst_h - new_h) as f32 / 2.0;

    // Convert to NCHW f32 normalized [0,1] using nearest-neighbor sampling.
    let mut chw = vec![0f32; 3 * (dst_w * dst_h) as usize];
    let plane = (dst_w * dst_h) as usize;
    let fill = 114.0 / 255.0;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let dst_idx = (dy * dst_w + dx) as usize;

            let (r, g, b) = if dx >= pad_x_i
                && dx < pad_x_i + new_w
                && dy >= pad_y_i
                && dy < pad_y_i + new_h
            {
                let sx = ((dx - pad_x_i) * src_w) / new_w;
                let sy = ((dy - pad_y_i) * src_h) / new_h;
                let src_idx = ((sy * src_w + sx) * 4) as usize;
                let r = rgba[src_idx] as f32 / 255.0;
                let g = rgba[src_idx + 1] as f32 / 255.0;
                let b = rgba[src_idx + 2] as f32 / 255.0;
                (r, g, b)
            } else {
                (fill, fill, fill)
            };

            chw[dst_idx] = r;
            chw[plane + dst_idx] = g;
            chw[2 * plane + dst_idx] = b;
        }
    }

    let mapping = ImageMapping {
        src_w,
        src_h,
        scale_x: scale,
        scale_y: scale,
        pad_x,
        pad_y,
    };

    Ok((chw, mapping))
}

fn rgba_to_nchw_f32_stretched(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<(Vec<f32>, ImageMapping)> {
    let expected_len = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|p| p.checked_mul(4))
        .context("Image dimensions overflow")?;
    if rgba.len() != expected_len {
        bail!(
            "RGBA buffer length mismatch: got {}, expected {} ({}x{}x4)",
            rgba.len(),
            expected_len,
            src_w,
            src_h
        );
    }

    let mut chw = vec![0f32; 3 * (dst_w * dst_h) as usize];
    let plane = (dst_w * dst_h) as usize;

    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let dst_idx = (dy * dst_w + dx) as usize;

            let r = rgba[src_idx] as f32 / 255.0;
            let g = rgba[src_idx + 1] as f32 / 255.0;
            let b = rgba[src_idx + 2] as f32 / 255.0;

            chw[dst_idx] = r;
            chw[plane + dst_idx] = g;
            chw[2 * plane + dst_idx] = b;
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

    Ok((chw, mapping))
}

fn parse_yolo_like_detections(
    output_shape: &[usize],
    data: &[f32],
    score_threshold: f32,
    mapping: ImageMapping,
) -> Result<Vec<Detection>> {
    // Accept shapes:
    // - [1, N, 6]
    // - [N, 6]
    // - [1, N, >=6] (we read first 6)
    let (n, stride) = match output_shape.len() {
        2 => {
            let n = output_shape[0];
            let stride = output_shape[1];
            if stride < 6 {
                bail!("Unexpected output shape {:?}; need at least 6 columns", output_shape);
            }
            (n, stride)
        }
        3 => {
            let b = output_shape[0];
            if b != 1 {
                bail!("Unexpected output shape {:?}; expected batch=1", output_shape);
            }
            let n = output_shape[1];
            let stride = output_shape[2];
            if stride < 6 {
                bail!("Unexpected output shape {:?}; need at least 6 columns", output_shape);
            }
            (n, stride)
        }
        _ => bail!("Unsupported output rank: {:?}", output_shape),
    };

    let base = 0;

    let mut detections = Vec::new();

    for i in 0..n {
        let row_start = base + i * stride;
        if row_start + 6 > data.len() {
            break;
        }

        let mut x1 = data[row_start + 0];
        let mut y1 = data[row_start + 1];
        let mut x2 = data[row_start + 2];
        let mut y2 = data[row_start + 3];
        let score = data[row_start + 4];
        let class_id = data[row_start + 5] as i64;

        if !score.is_finite() || score < score_threshold {
            continue;
        }

        if swap_xy_enabled() {
            std::mem::swap(&mut x1, &mut y1);
            std::mem::swap(&mut x2, &mut y2);
        }

        // Map from model input coords back to original image coords.
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

fn unmap_point(x: f32, y: f32, mapping: ImageMapping) -> (f32, f32) {
    let x = (x - mapping.pad_x) / mapping.scale_x.max(1e-6);
    let y = (y - mapping.pad_y) / mapping.scale_y.max(1e-6);
    (x, y)
}
