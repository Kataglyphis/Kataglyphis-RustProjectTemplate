//! One-time GStreamer runtime initialization.

use std::path::PathBuf;
use std::sync::OnceLock;

static GST_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initializes GStreamer exactly once, resolving the plugin directory first.
///
/// `GST_PLUGIN_PATH` must be set before `gst::init()` reads the registry, so
/// this is the only supported entry point; all capture/device APIs call it.
pub fn ensure_gst_initialized() -> anyhow::Result<()> {
    let result = GST_INIT.get_or_init(|| {
        if std::env::var_os("GST_PLUGIN_PATH").is_none() {
            if let Some(dir) = find_plugin_dir() {
                log::info!("GST_PLUGIN_PATH not set, using {}", dir.display());
                // Runs before any GStreamer threads exist; the frb init path
                // invokes this before other API calls can race it.
                std::env::set_var("GST_PLUGIN_PATH", &dir);
            }
        }
        gstreamer::init().map_err(|e| e.to_string())
    });
    result
        .clone()
        .map_err(|e| anyhow::anyhow!("GStreamer init failed: {e}"))
}

/// Plugin dir search order: next to the executable (packaged app layouts),
/// then the container/dev-image install prefix.
fn find_plugin_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("gstreamer-1.0"));
            candidates.push(dir.join("lib").join("gstreamer-1.0"));
        }
    }
    #[cfg(windows)]
    candidates.push(PathBuf::from(r"C:\runtime\lib\gstreamer-1.0"));
    #[cfg(not(windows))]
    candidates.push(PathBuf::from("/usr/lib/gstreamer-1.0"));

    candidates.into_iter().find(|p| p.is_dir())
}
