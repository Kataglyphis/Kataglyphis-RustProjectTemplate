// src/utils.rs
use anyhow::{Context, Result};

pub struct FileStats {
    pub lines: usize,
    pub words: usize,
    pub bytes: u64,
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read_file(path: &str) -> Result<String> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read file at '{}'", path))?;
    Ok(content)
}

#[cfg(target_arch = "wasm32")]
pub async fn read_file(path: &str) -> Result<String> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file at '{}'", path))?;
    Ok(content)
}

/// Compute line, word, and byte statistics for a file.
///
/// The byte count is obtained from filesystem metadata so it reflects the
/// on-disk size without needing to measure the in-memory string length.
pub async fn file_stats(path: &str) -> Result<FileStats> {
    // Fetch byte size from metadata — cheap and works for any file size.
    #[cfg(not(target_arch = "wasm32"))]
    let bytes = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to read metadata for '{}'", path))?
        .len();

    #[cfg(target_arch = "wasm32")]
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for '{}'", path))?
        .len();

    // Lines and words still need the file content.
    let content = read_file(path).await?;
    let lines = content.lines().count();
    let words = content.split_whitespace().count();

    Ok(FileStats {
        lines,
        words,
        bytes,
    })
}
