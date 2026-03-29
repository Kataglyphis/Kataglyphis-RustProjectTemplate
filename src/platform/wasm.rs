use anyhow::Result;
use std::path::Path;

/// WASM implementation of file read using blocking std::fs (suitable for tests
/// and environments where tokio isn't available).
pub async fn read_file(path: impl AsRef<Path>) -> Result<String> {
    // run blocking sync function inside async fn, keep signature consistent
    let path = path.as_ref().to_path_buf();
    let s = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file at '{}'", path.display()))?;
    Ok(s)
}

// Re-export any other wasm-only utilities here.
pub use crate::utils::FileStats;

use anyhow::Context;

#[flutter_rust_bridge::frb(sync)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
