use std::path::Path;

use anyhow::{Context, Result, bail};

pub(crate) fn validate_model_path(model_path: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(model_path);
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Model path does not exist or is inaccessible: '{model_path}'"))?;
    if !canonical.is_file() {
        bail!("Model path is not a file: '{}'", canonical.display());
    }
    let ext = canonical.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext != "onnx" {
        bail!(
            "Model file should have .onnx extension, got '{}': {}",
            ext,
            canonical.display()
        );
    }
    Ok(canonical)
}
