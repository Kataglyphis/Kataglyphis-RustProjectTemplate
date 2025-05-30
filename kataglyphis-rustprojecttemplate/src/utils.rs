// src/utils.rs
use anyhow::{Context, Result};
use tokio::fs;

pub struct FileStats {
    pub lines: usize,
    pub words: usize,
    pub bytes: usize,
}

pub async fn read_file(path: &str) -> Result<String> {
    let content = fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read file at '{}'", path))?;
    Ok(content)
}

pub async fn file_stats(path: &str) -> Result<FileStats> {
    let content = read_file(path).await?;
    let lines = content.lines().count();
    let words = content.split_whitespace().count();
    let bytes = content.len();

    Ok(FileStats {
        lines,
        words,
        bytes,
    })
}
