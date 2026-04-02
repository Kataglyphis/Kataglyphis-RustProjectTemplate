//! Telemetry and resource monitoring.

pub mod resource_monitor;

#[cfg(target_os = "windows")]
pub(crate) mod gpu_wmi;

pub use resource_monitor::*;
