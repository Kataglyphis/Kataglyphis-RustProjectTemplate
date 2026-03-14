mod model_utils;
mod preprocess;

#[cfg(feature = "onnx_tract")]
mod tract_backend;

#[cfg(feature = "onnxruntime")]
mod ort_backend;

use anyhow::{Context, Result, bail};
use log::info;

#[cfg(feature = "onnxruntime")]
use log::warn;

#[cfg(feature = "onnxruntime")]
use crate::ort_ext::{OrtResultExt, extract_first_f32_output};

use crate::config::{self, PreprocessMode};
use crate::detection::Detection;
use preprocess::{ImageMapping, rgba_to_nchw_f32_letterboxed, rgba_to_nchw_f32_stretched};

pub(crate) fn default_model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("models")
        .join("yolov10m.onnx")
}

enum Backend {
    #[cfg(feature = "onnx_tract")]
    Tract { model: Box<TractPlan> },

    #[cfg(feature = "onnxruntime")]
    Ort { session: ort::session::Session },
}

#[cfg(feature = "onnx_tract")]
type TractPlan = tract_onnx::prelude::SimplePlan<
    tract_onnx::prelude::TypedFact,
    Box<dyn tract_onnx::prelude::TypedOp>,
    tract_onnx::prelude::TypedModel,
>;

pub(crate) struct PersonDetector {
    backend: Backend,
    input_w: u32,
    input_h: u32,
    preprocess: PreprocessMode,
    swap_xy: bool,
    preprocess_buf: Vec<f32>,
}

impl PersonDetector {
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

    pub(crate) fn new(model_path: &str) -> Result<Self> {
        info!("Loading ONNX model: {model_path}");

        let preprocess = config::preprocess_mode();
        let swap_xy = config::swap_xy_enabled();
        log::debug!("Preprocess mode: {:?}, swap_xy: {}", preprocess, swap_xy);

        let requested_backend = config::onnx_backend();

        match requested_backend.as_deref() {
            Some("ort") | Some("onnxruntime") => {
                #[cfg(feature = "onnxruntime")]
                {
                    let (session, (input_w, input_h)) = ort_backend::load_ort_session(model_path)?;
                    info!("ONNX backend: ort ({input_w}x{input_h})");
                    Ok(Self::from_parts(
                        Backend::Ort { session },
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
                    let (model, (input_w, input_h)) = tract_backend::load_tract_model(model_path)?;
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
                #[cfg(feature = "onnxruntime")]
                {
                    let device = config::ort_device();
                    let require_ort = device == "cuda" || device == "auto";

                    match ort_backend::load_ort_session(model_path) {
                        Ok((session, (input_w, input_h))) => {
                            info!("ONNX backend: ort ({input_w}x{input_h})");
                            return Ok(Self::from_parts(
                                Backend::Ort { session },
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
                    let (model, (input_w, input_h)) = tract_backend::load_tract_model(model_path)?;
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

    pub(crate) fn infer_persons_rgba(
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

        let (shape, data) = Self::infer_raw_nchw_f32(
            &mut self.backend,
            &self.preprocess_buf,
            self.input_h,
            self.input_w,
        )
        .context("Failed to run ONNX model")?;

        let mut detections =
            parse_yolo_like_detections(&shape, &data, score_threshold, mapping, self.swap_xy)?;

        detections.retain(|d| d.class_id == 0);
        Ok(detections)
    }

    fn infer_raw_nchw_f32(
        backend: &mut Backend,
        input: &[f32],
        input_h: u32,
        input_w: u32,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        match backend {
            #[cfg(feature = "onnx_tract")]
            Backend::Tract { model } => {
                use tract_onnx::prelude::*;

                let shape = [1usize, 3usize, input_h as usize, input_w as usize];
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
                // NOTE: ORT's `Tensor::from_array` requires owned data (Box<[f32]>).
                // The `to_vec().into_boxed_slice()` pattern performs a single allocation
                // (Vec allocates with exact capacity, then converts to Box without reallocation).
                // This is optimal for the current ort API. If ort exposes a borrowed-data
                // constructor in the future, we could eliminate this allocation entirely.
                let input_tensor = ort::value::Tensor::from_array((
                    [1usize, 3usize, input_h as usize, input_w as usize],
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

        let (x1, y1) = preprocess::unmap_point(x1, y1, mapping);
        let (x2, y2) = preprocess::unmap_point(x2, y2, mapping);

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
