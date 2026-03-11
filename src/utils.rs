// src/utils.rs
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug)]
pub struct FileStats {
    pub lines: usize,
    pub words: usize,
    pub bytes: u64,
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn read_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read file at '{}'", path.display()))?;
    Ok(content)
}

#[cfg(target_arch = "wasm32")]
pub async fn read_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file at '{}'", path.display()))?;
    Ok(content)
}

/// Compute line, word, and byte statistics for a file.
///
/// Uses a buffered reader to stream the file line-by-line, avoiding loading the
/// entire contents into memory at once.  The byte count comes from filesystem
/// metadata so it reflects the on-disk size.
pub async fn file_stats(path: impl AsRef<Path>) -> Result<FileStats> {
    let path = path.as_ref();

    // Fetch byte size from metadata — cheap and works for any file size.
    #[cfg(not(target_arch = "wasm32"))]
    let bytes = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to read metadata for '{}'", path.display()))?
        .len();

    #[cfg(target_arch = "wasm32")]
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for '{}'", path.display()))?
        .len();

    // Stream the file line-by-line to count lines and words without holding the
    // entire file in memory.
    let path_owned = path.to_path_buf();

    #[cfg(not(target_arch = "wasm32"))]
    let (lines, words) = tokio::task::spawn_blocking(move || -> Result<(usize, usize)> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(&path_owned)
            .with_context(|| format!("Failed to open file '{}'", path_owned.display()))?;
        let reader = BufReader::new(file);
        let mut lines = 0usize;
        let mut words = 0usize;
        for line in reader.lines() {
            let line = line
                .with_context(|| format!("Failed to read line from '{}'", path_owned.display()))?;
            lines += 1;
            words += line.split_whitespace().count();
        }
        Ok((lines, words))
    })
    .await
    .context("Blocking task panicked")??;

    #[cfg(target_arch = "wasm32")]
    let (lines, words) = {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(&path_owned)
            .with_context(|| format!("Failed to open file '{}'", path_owned.display()))?;
        let reader = BufReader::new(file);
        let mut lines = 0usize;
        let mut words = 0usize;
        for line in reader.lines() {
            let line = line
                .with_context(|| format!("Failed to read line from '{}'", path_owned.display()))?;
            lines += 1;
            words += line.split_whitespace().count();
        }
        (lines, words)
    };

    Ok(FileStats {
        lines,
        words,
        bytes,
    })
}
