mod model_utils;
pub(crate) mod postprocess;
mod preprocess;

#[cfg(feature = "onnx_tract")]
mod tract_backend;

#[cfg(feature = "onnxruntime")]
mod ort_backend;

use anyhow::{bail, Context, Result};
use log::info;

#[cfg(feature = "onnxruntime")]
use log::warn;

#[cfg(feature = "onnxruntime")]
use crate::ort_ext::extract_first_f32_output;

use kataglyphis_core::config::{self, PreprocessMode};
use kataglyphis_core::detection::Detection;
use preprocess::{rgba_to_nchw_f32_letterboxed, rgba_to_nchw_f32_stretched};

pub fn resolve_model_path(explicit: Option<&str>) -> String {
    if let Some(p) = explicit {
        if !p.trim().is_empty() {
            return p.to_string();
        }
    }
    kataglyphis_core::config::onnx_model_override()
        .clone()
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("resources")
                .join("models")
                .join("yolov10m.onnx")
                .to_string_lossy()
                .to_string()
        })
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

struct BackendLoad {
    backend: Backend,
    input_dims: (u32, u32),
}

pub struct PersonDetector {
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

    pub fn new(model_path: &str) -> Result<Self> {
        info!("Loading ONNX model: {model_path}");

        let preprocess = config::preprocess_mode();
        let swap_xy = config::swap_xy_enabled();
        log::debug!("Preprocess mode: {:?}, swap_xy: {}", preprocess, swap_xy);

        let requested_backend = config::onnx_backend();

        let BackendLoad {
            backend,
            input_dims,
        } = Self::load_backend(model_path, requested_backend.as_deref())?;
        let (input_w, input_h) = input_dims;

        Ok(Self::from_parts(
            backend, input_w, input_h, preprocess, swap_xy,
        ))
    }

    #[cfg(feature = "onnxruntime")]
    fn try_load_ort(model_path: &str, require_ort: bool) -> Result<BackendLoad> {
        match ort_backend::load_ort_session(model_path) {
            Ok((session, dims)) => {
                info!("ONNX backend: ort ({}x{})", dims.0, dims.1);
                Ok(BackendLoad {
                    backend: Backend::Ort { session },
                    input_dims: dims,
                })
            }
            Err(err) => {
                if require_ort {
                    return Err(err);
                }
                warn!("ORT session unavailable, falling back to tract: {err:#}");
                Err(err)
            }
        }
    }

    #[cfg(feature = "onnx_tract")]
    fn load_tract(model_path: &str) -> Result<BackendLoad> {
        let (model, dims) = tract_backend::load_tract_model(model_path)?;
        info!("ONNX backend: tract ({}x{})", dims.0, dims.1);
        Ok(BackendLoad {
            backend: Backend::Tract {
                model: Box::new(model),
            },
            input_dims: dims,
        })
    }

    fn load_backend(model_path: &str, requested: Option<&str>) -> Result<BackendLoad> {
        match requested {
            Some("ort") | Some("onnxruntime") => {
                #[cfg(feature = "onnxruntime")]
                {
                    Self::try_load_ort(model_path, true)
                }
                #[cfg(not(feature = "onnxruntime"))]
                {
                    let _ = model_path;
                    bail!("Requested ONNX backend 'ort', but feature 'onnxruntime' is not enabled")
                }
            }
            Some("tract") => {
                #[cfg(feature = "onnx_tract")]
                {
                    Self::load_tract(model_path)
                }
                #[cfg(not(feature = "onnx_tract"))]
                {
                    let _ = model_path;
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

                    if let Ok(load) = Self::try_load_ort(model_path, require_ort) {
                        return Ok(load);
                    }
                }

                #[cfg(feature = "onnx_tract")]
                {
                    Self::load_tract(model_path)
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

        let (shape, data) = Self::infer_raw_nchw_f32(
            &mut self.backend,
            &self.preprocess_buf,
            self.input_h,
            self.input_w,
        )
        .context("Failed to run ONNX model")?;

        let mut detections = postprocess::parse_yolo_like_detections(
            &shape,
            &data,
            score_threshold,
            mapping,
            self.swap_xy,
        )?;

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
