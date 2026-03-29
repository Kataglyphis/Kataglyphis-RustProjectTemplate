// src/utils.rs
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStats {
    pub lines: usize,
    pub words: usize,
    pub bytes: u64,
}

/// Platform-agnostic read_file shim. Prefer using `crate::platform::read_file`
/// for platform-specific implementations; this remains as a compatibility
/// shim that delegates to the platform module.
pub async fn read_file(path: impl AsRef<Path>) -> Result<String> {
    crate::platform::read_file(path).await
}

fn count_lines_and_words(path: &Path) -> Result<(usize, usize)> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file '{}'", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = 0usize;
    let mut words = 0usize;
    for line in reader.lines() {
        let line =
            line.with_context(|| format!("Failed to read line from '{}'", path.display()))?;
        lines += 1;
        words += line.split_whitespace().count();
    }
    Ok((lines, words))
}

pub async fn file_stats(path: impl AsRef<Path>) -> Result<FileStats> {
    let path = path.as_ref();

    #[cfg(not(target_arch = "wasm32"))]
    let bytes = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Failed to read metadata for '{}'", path.display()))?
        .len();

    #[cfg(target_arch = "wasm32")]
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for '{}'", path.display()))?
        .len();

    let path_owned = path.to_path_buf();

    #[cfg(not(target_arch = "wasm32"))]
    let (lines, words) = tokio::task::spawn_blocking(move || count_lines_and_words(&path_owned))
        .await
        .context("Blocking task panicked")??;

    #[cfg(target_arch = "wasm32")]
    let (lines, words) = count_lines_and_words(&path_owned)?;

    Ok(FileStats {
        lines,
        words,
        bytes,
    })
}
