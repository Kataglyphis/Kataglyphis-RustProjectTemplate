use burn::tensor::{Tensor, backend::Backend};

/// Manual binary cross-entropy loss (keeps external deps minimal).
///
/// `pred` and `target` should be 2-D tensors with values in `[0, 1]`.
/// Returns the mean BCE as a scalar (1-D) tensor.
pub fn binary_cross_entropy<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let eps = 1e-6;
    let pred = pred.clamp(eps, 1.0 - eps);
    let one_minus_y = target.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
    let one_minus_pred = pred.clone().mul_scalar(-1.0f32).add_scalar(1.0f32);
    -(target * pred.log() + one_minus_y * one_minus_pred.log()).mean()
}
