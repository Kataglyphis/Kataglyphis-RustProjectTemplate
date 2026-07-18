//! Camera enumeration via the GStreamer device monitor.

use anyhow::{Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;

use crate::ensure_gst_initialized;

#[derive(Debug, Clone)]
pub struct CameraInfo {
    /// Enumeration order — matches the `device-index` the capture pipeline
    /// passes to `mfvideosrc`/`ksvideosrc`.
    pub index: u32,
    pub display_name: String,
    /// Provider element class, e.g. `Source/Video`.
    pub device_class: String,
}

/// Lists video capture devices. Empty inside containers (no camera devices).
pub fn list_cameras() -> Result<Vec<CameraInfo>> {
    ensure_gst_initialized()?;

    let monitor = gst::DeviceMonitor::new();
    monitor.add_filter(Some("Video/Source"), None);
    monitor
        .start()
        .context("failed to start GStreamer device monitor")?;
    let devices = monitor.devices();
    monitor.stop();

    Ok(devices
        .iter()
        .enumerate()
        .map(|(i, device)| CameraInfo {
            index: i as u32,
            display_name: device.display_name().to_string(),
            device_class: device.device_class().to_string(),
        })
        .collect())
}
