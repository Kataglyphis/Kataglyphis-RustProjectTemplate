//! Shared helpers for working with `ort` (ONNX Runtime) results and outputs.
//!
//! `ort::Error<R>` implements `Display` for all `R`, but only implements
//! `std::error::Error + Send + Sync` when `R` does too.  Since `SessionBuilder`
//! lacks those traits, anyhow's `.context()` cannot be used directly on
//! `Result<T, ort::Error<SessionBuilder>>`.
//!
//! This module provides:
//!
//! - [`OrtResultExt`] — converts any `Result<T, ort::Error<R>>` to
//!   `anyhow::Result<T>` via the always-available `Display` impl.
//! - [`extract_first_f32_output`] — extracts the first tensor output from a
//!   completed ORT session run as `(shape, flat_data)`.

use anyhow::{Context, Result, bail};

/// Extension trait that maps `ort::Error<R>` (for any `R`) into `anyhow::Error`
/// using the `Display` impl, which is available unconditionally.
pub(crate) trait OrtResultExt<T> {
    fn with_ort_context(self, msg: &'static str) -> Result<T>;
}

impl<T, R> OrtResultExt<T> for std::result::Result<T, ort::Error<R>> {
    #[inline]
    fn with_ort_context(self, msg: &'static str) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
    }
}

/// Extract the first output from a completed ORT run, returning owned data.
///
/// Returns `(shape, flat_f32_data)` with dimensions converted from `i64` to
/// `usize`.  Bails on negative (dynamic) dimensions.
pub(crate) fn extract_first_f32_output(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let (_, out) = outputs.iter().next().context("Model returned no outputs")?;

    let (shape, data) = out
        .try_extract_tensor::<f32>()
        .context("Failed to extract f32 tensor from ORT output")?;

    let mut dims = Vec::with_capacity(shape.len());
    for &d in shape.iter() {
        if d < 0 {
            bail!("ORT output had dynamic/negative dimension: {d}");
        }
        dims.push(d as usize);
    }

    Ok((dims, data.to_vec()))
}
