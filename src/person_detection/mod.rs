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
use crate::ort_ext::extract_first_f32_output;

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

struct BackendLoad {
    backend: Backend,
    input_dims: (u32, u32),
}

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
                    return Self::load_tract(model_path);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapping(src_w: u32, src_h: u32) -> ImageMapping {
        ImageMapping {
            src_w,
            src_h,
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    #[test]
    fn test_parse_yolo_like_detections_empty() {
        let shape = vec![0usize, 6];
        let data: Vec<f32> = vec![];
        let mapping = make_mapping(100, 100);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_yolo_like_detections_below_threshold() {
        let shape = vec![1usize, 6];
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 0.3, 0.0];
        let mapping = make_mapping(100, 100);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_yolo_like_detections_single() {
        let shape = vec![1usize, 6];
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 0.8, 0.0];
        let mapping = make_mapping(100, 100);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, false);
        assert!(result.is_ok());
        let detections = result.unwrap();
        assert_eq!(detections.len(), 1);
        let d = &detections[0];
        assert!((d.x1 - 10.0).abs() < 0.001);
        assert!((d.y1 - 20.0).abs() < 0.001);
        assert!((d.x2 - 30.0).abs() < 0.001);
        assert!((d.y2 - 40.0).abs() < 0.001);
        assert!((d.score - 0.8).abs() < 0.001);
        assert_eq!(d.class_id, 0);
    }

    #[test]
    fn test_parse_yolo_like_detections_3d_shape() {
        let shape = vec![1usize, 1, 6];
        let data: Vec<f32> = vec![5.0, 10.0, 15.0, 20.0, 0.9, 1.0];
        let mapping = make_mapping(50, 50);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, false);
        assert!(result.is_ok());
        let detections = result.unwrap();
        assert_eq!(detections.len(), 1);
    }

    #[test]
    fn test_parse_yolo_like_detections_swap_xy() {
        let shape = vec![1usize, 6];
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 0.7, 0.0];
        let mapping = make_mapping(100, 100);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, true);
        assert!(result.is_ok());
        let d = &result.unwrap()[0];
        assert!((d.x1 - 20.0).abs() < 0.001); // swapped
        assert!((d.y1 - 10.0).abs() < 0.001); // swapped
    }

    #[test]
    fn test_parse_yolo_like_detections_invalid_shape() {
        let shape = vec![1usize, 4usize];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mapping = make_mapping(100, 100);
        let result = parse_yolo_like_detections(&shape, &data, 0.5, mapping, false);
        assert!(result.is_err());
    }
}
