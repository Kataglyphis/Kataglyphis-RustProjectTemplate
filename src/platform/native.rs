use anyhow::{Context, Result};
use std::path::Path;

/// Native implementation of file read using async tokio IO.
pub async fn read_file(path: impl AsRef<Path>) -> Result<String> {
    tokio::fs::read_to_string(path.as_ref())
        .await
        .with_context(|| format!("Failed to read file at '{}'", path.as_ref().display()))
}

// Re-export any other native-only utilities here.
pub use crate::utils::FileStats;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // Default utilities - feel free to customize
    flutter_rust_bridge::setup_default_user_utils();
}
