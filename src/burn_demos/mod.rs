use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

pub mod onnx_yolov10;
pub mod plot;
pub mod simple;
pub mod two_moons;
pub mod yolo;

pub type InferenceBackend = NdArray<f32>;
pub type TrainingBackend = Autodiff<InferenceBackend>;
