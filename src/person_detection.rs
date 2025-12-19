use anyhow::{bail, Context, Result};

pub fn default_model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("yolov10m.onnx")
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
        let requested_backend = std::env::var("KATAGLYPHIS_ONNX_BACKEND")
            .ok()
            .map(|v| v.to_lowercase());

        match requested_backend.as_deref() {
            Some("ort") | Some("onnxruntime") => {
                #[cfg(feature = "onnxruntime")]
                {
                    let (session, (input_w, input_h)) = load_ort_session(model_path)?;
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
                // Default: prefer tract if enabled, otherwise fall back to ORT.
                #[cfg(feature = "onnx_tract")]
                {
                    let (model, (input_w, input_h)) = load_tract_model(model_path)?;
                    return Ok(Self {
                        backend: Backend::Tract { model },
                        input_w,
                        input_h,
                    });
                }

                #[cfg(all(not(feature = "onnx_tract"), feature = "onnxruntime"))]
                {
                    let (session, (input_w, input_h)) = load_ort_session(model_path)?;
                    return Ok(Self {
                        backend: Backend::Ort {
                            session: std::sync::Mutex::new(session),
                        },
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
        let (input, letterbox) = rgba_to_nchw_f32_letterboxed(
            rgba,
            width,
            height,
            self.input_w,
            self.input_h,
        )?;

        let (shape, data) = self
            .infer_raw_nchw_f32(input)
            .context("Failed to run ONNX model")?;

        let detections = parse_yolo_like_detections(&shape, &data, score_threshold, letterbox)?;

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

/// Describes how an image was letterboxed into the model input.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    scale: f32,
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
    use ort::value::ValueType;

    let mut builder = Session::builder().context("Failed to create ORT SessionBuilder")?;

    #[cfg(feature = "onnxruntime_directml")]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .context("Failed to configure ORT DirectML execution provider")?;
    }

    let session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from '{model_path}' (ort)"))?;

    let (input_w, input_h) = match session.inputs.first() {
        Some(input) => match &input.input_type {
            ValueType::Tensor { shape, .. } => {
                // Common layouts: NCHW => [1, 3, H, W]
                if shape.len() == 4 {
                    let h = shape[2];
                    let w = shape[3];
                    if h > 0 && w > 0 {
                        (w as u32, h as u32)
                    } else {
                        (640, 640)
                    }
                } else {
                    (640, 640)
                }
            }
            _ => (640, 640),
        },
        None => (640, 640),
    };

    Ok((session, (input_w, input_h)))
}

fn rgba_to_nchw_f32_letterboxed(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<(Vec<f32>, Letterbox)> {
    use image::{ImageBuffer, Rgb, RgbImage};

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

    // RGBA -> RGB
    let mut rgb = Vec::with_capacity((src_w * src_h * 3) as usize);
    for px in rgba.chunks_exact(4) {
        rgb.push(px[0]);
        rgb.push(px[1]);
        rgb.push(px[2]);
    }
    let src_img: RgbImage = ImageBuffer::<Rgb<u8>, _>::from_raw(src_w, src_h, rgb)
        .context("Failed to build RGB image from RGBA buffer")?;

    // Letterbox to dst size (keep aspect ratio)
    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (src_h as f32 * scale).round().max(1.0) as u32;
    let pad_x = ((dst_w - new_w) as f32) / 2.0;
    let pad_y = ((dst_h - new_h) as f32) / 2.0;

    let resized = image::imageops::resize(&src_img, new_w, new_h, image::imageops::FilterType::Triangle);

    let mut canvas = RgbImage::new(dst_w, dst_h);
    // Fill with 114 (typical YOLO padding)
    for p in canvas.pixels_mut() {
        *p = Rgb([114, 114, 114]);
    }
    image::imageops::overlay(
        &mut canvas,
        &resized,
        pad_x.round() as i64,
        pad_y.round() as i64,
    );

    // Convert to NCHW f32 normalized [0,1]
    let mut chw = vec![0f32; 3 * (dst_w * dst_h) as usize];
    let plane = (dst_w * dst_h) as usize;
    for (i, px) in canvas.pixels().enumerate() {
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        chw[i] = r;
        chw[plane + i] = g;
        chw[2 * plane + i] = b;
    }

    let letterbox = Letterbox {
        src_w,
        src_h,
        dst_w,
        dst_h,
        scale,
        pad_x,
        pad_y,
    };

    Ok((chw, letterbox))
}

fn parse_yolo_like_detections(
    output_shape: &[usize],
    data: &[f32],
    score_threshold: f32,
    letterbox: Letterbox,
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

        let x1 = data[row_start + 0];
        let y1 = data[row_start + 1];
        let x2 = data[row_start + 2];
        let y2 = data[row_start + 3];
        let score = data[row_start + 4];
        let class_id = data[row_start + 5] as i64;

        if !score.is_finite() || score < score_threshold {
            continue;
        }

        // Map from letterboxed input coords back to original image coords.
        let (x1, y1) = unletterbox_point(x1, y1, letterbox);
        let (x2, y2) = unletterbox_point(x2, y2, letterbox);

        detections.push(Detection {
            x1: x1.clamp(0.0, letterbox.src_w as f32),
            y1: y1.clamp(0.0, letterbox.src_h as f32),
            x2: x2.clamp(0.0, letterbox.src_w as f32),
            y2: y2.clamp(0.0, letterbox.src_h as f32),
            score,
            class_id,
        });
    }

    Ok(detections)
}

fn unletterbox_point(x: f32, y: f32, lb: Letterbox) -> (f32, f32) {
    // Undo: x' = (x - pad_x) / scale
    let x = (x - lb.pad_x) / lb.scale;
    let y = (y - lb.pad_y) / lb.scale;
    (x, y)
}
