use anyhow::Context;

use super::model_utils::validate_model_path;

// tract 0.23 renamed `SimplePlan` to `RunnableModel` and the prelude no longer
// exports the old name at all. `TypedRunnableModel` is fully applied - it takes
// no generic argument - and stands for what used to be spelled out as
// `SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>`.
type TractPlan = tract_onnx::prelude::TypedRunnableModel;

// `into_runnable()` hands back an Arc already in 0.23, and `SimplePlan::run`
// takes `self: &Arc<Self>` - so the Arc is not ours to add or remove.
pub(crate) fn load_tract_model(
    model_path: &str,
) -> anyhow::Result<(std::sync::Arc<TractPlan>, (u32, u32))> {
    use tract_onnx::prelude::*;

    let canonical_path = validate_model_path(model_path)?;

    let mut model = tract_onnx::onnx()
        .model_for_path(&canonical_path)
        .with_context(|| {
            format!(
                "Failed to load ONNX model from '{}'",
                canonical_path.display()
            )
        })?;

    let (input_w, input_h) = {
        let fact = model
            .input_fact(0)
            .context("Model has no input[0]")?
            .clone();
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
