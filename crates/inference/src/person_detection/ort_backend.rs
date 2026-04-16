use anyhow::Result;
#[allow(unused_imports)]
use anyhow::bail;
use log::info;

use super::model_utils::validate_model_path;
use crate::ort_ext::OrtResultExt;
#[cfg(feature = "onnxruntime_cuda")]
use kataglyphis_core::config;

const DEFAULT_INPUT_DIMS: (u32, u32) = (640, 640);

pub(crate) fn load_ort_session(model_path: &str) -> Result<(ort::session::Session, (u32, u32))> {
    use ort::session::Session;

    let canonical_path = validate_model_path(model_path)?;

    let mut builder = Session::builder().with_ort_context("Failed to create ORT SessionBuilder")?;

    #[cfg(feature = "onnxruntime_cuda")]
    {
        use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
        let device = config::ort_device();

        info!("ORT device request: {device}");

        if device == "cuda" || device == "auto" {
            #[cfg(all(feature = "onnxruntime_cuda", windows))]
            ensure_ort_cuda_provider_dylibs_next_to_exe()?;

            let cuda = CUDAExecutionProvider::default();
            match cuda.is_available() {
                Ok(true) => {}
                Ok(false) => {
                    bail!("ORT was built without CUDA support (CUDA EP unavailable).");
                }
                Err(err) => {
                    bail!("Failed to query CUDA EP availability: {err}");
                }
            }

            let cuda_result = builder
                .with_execution_providers([cuda.build().error_on_failure()])
                .with_ort_context("Failed to configure ORT CUDA execution provider");

            match cuda_result {
                Ok(b) => {
                    info!("ORT CUDA execution provider enabled");
                    builder = b;
                }
                Err(err) => {
                    if device == "cuda" {
                        return Err(err);
                    }
                    log::warn!("ORT CUDA provider unavailable, falling back to CPU: {err:#}");
                    builder = Session::builder()
                        .with_ort_context("Failed to recreate ORT SessionBuilder")?;
                }
            }
        }
    }

    #[cfg(all(feature = "onnxruntime_directml", windows))]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .with_ort_context("Failed to configure ORT DirectML execution provider")?;
        info!("ORT DirectML execution provider enabled");
    }

    let session = builder
        .commit_from_file(&canonical_path)
        .with_ort_context("Failed to load ONNX model (ort)")?;
    info!(
        "ORT session created from model file: {}",
        canonical_path.display()
    );

    let (input_w, input_h) = extract_ort_input_dims(&session);

    Ok((session, (input_w, input_h)))
}

fn extract_ort_input_dims(session: &ort::session::Session) -> (u32, u32) {
    let Some(first) = session.inputs().first() else {
        return DEFAULT_INPUT_DIMS;
    };
    let Some(shape) = first.dtype().tensor_shape() else {
        return DEFAULT_INPUT_DIMS;
    };
    if shape.len() != 4 {
        return DEFAULT_INPUT_DIMS;
    }
    let h = shape[2];
    let w = shape[3];
    if h <= 0 || w <= 0 {
        return DEFAULT_INPUT_DIMS;
    }
    (w as u32, h as u32)
}

#[cfg(all(feature = "onnxruntime_cuda", windows))]
mod cuda_dylib {
    use super::*;

    const REQUIRED_DLLS: &[&str] = &[
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];

    fn ort_cache_dir() -> Result<std::path::PathBuf> {
        if let Some(p) = std::env::var_os("ORT_CACHE_DIR") {
            return Ok(std::path::PathBuf::from(p));
        }
        let local_appdata = std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .context("LOCALAPPDATA is not set")?;
        Ok(local_appdata.join("ort.pyke.io"))
    }

    fn exe_directory() -> Result<std::path::PathBuf> {
        std::env::current_exe()
            .context("current_exe failed")?
            .parent()
            .map(|p| p.to_path_buf())
            .context("current_exe has no parent directory")
    }

    fn find_latest_file(
        root: &std::path::Path,
        file_name: &std::ffi::OsStr,
    ) -> Result<std::path::PathBuf> {
        use std::time::SystemTime;

        let mut stack = vec![root.to_path_buf()];
        let mut best: Option<(SystemTime, std::path::PathBuf)> = None;

        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .with_context(|| format!("Failed to read dir '{}'", dir.display()))?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name() == Some(file_name) {
                    let meta = entry.metadata()?;
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
                        best = Some((mtime, path));
                    }
                }
            }
        }

        best.map(|(_, p)| p).with_context(|| {
            format!(
                "Could not locate '{}' under '{}'",
                file_name.to_string_lossy(),
                root.display()
            )
        })
    }

    fn copy_dlls_from_cache(exe_dir: &std::path::Path, src_dir: &std::path::Path) -> Result<()> {
        for name in REQUIRED_DLLS {
            let dst = exe_dir.join(name);
            if dst.is_file() {
                continue;
            }

            let src = src_dir.join(name);
            if !src.is_file() {
                bail!("Required ORT CUDA DLL not found: '{}'", src.display());
            }

            std::fs::copy(&src, &dst).with_context(|| {
                format!("Failed to copy '{}' -> '{}'", src.display(), dst.display())
            })?;
            info!("Copied '{}' -> '{}'", src.display(), dst.display());
        }
        Ok(())
    }

    pub(crate) fn ensure_ort_cuda_provider_dylibs_next_to_exe() -> Result<()> {
        let exe_dir = exe_directory()?;

        if REQUIRED_DLLS
            .iter()
            .all(|name| exe_dir.join(name).is_file())
        {
            return Ok(());
        }

        let dfbin_dir = ort_cache_dir()?.join("dfbin");
        let shared_path = find_latest_file(
            &dfbin_dir,
            std::ffi::OsStr::new("onnxruntime_providers_shared.dll"),
        )?;
        let src_dir = shared_path
            .parent()
            .map(|p| p.to_path_buf())
            .context("providers_shared.dll has no parent directory")?;

        copy_dlls_from_cache(&exe_dir, &src_dir)
    }
}

#[cfg(all(feature = "onnxruntime_cuda", windows))]
pub(crate) use cuda_dylib::ensure_ort_cuda_provider_dylibs_next_to_exe;
