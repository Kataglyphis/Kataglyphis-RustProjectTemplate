// src/gpu_wmi.rs — Windows GPU metrics via WMI.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::resource_monitor::GpuSample;

pub(crate) fn gpu_connect() -> Option<wmi::WMIConnection> {
    static WARNED_WMI_GPU: AtomicBool = AtomicBool::new(false);

    match wmi::WMIConnection::new() {
        Ok(wmi) => Some(wmi),
        Err(_) => {
            warn_wmi_once(
                &WARNED_WMI_GPU,
                "WMI connection failed; GPU metrics disabled",
            );
            None
        }
    }
}

pub(crate) fn gpu_sample_wmi(wmi: &wmi::WMIConnection) -> Option<GpuSample> {
    use serde::Deserialize;

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "PascalCase")]
    struct GpuAdapter {
        dedicated_usage: Option<u64>,
        shared_usage: Option<u64>,
        total_committed: Option<u64>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "PascalCase")]
    struct GpuEngine {
        utilization_percentage: Option<u64>,
    }

    let adapters: Vec<GpuAdapter> = wmi
        .raw_query(
            "SELECT DedicatedUsage, SharedUsage, TotalCommitted FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapter",
        )
        .unwrap_or_default();

    let engines: Vec<GpuEngine> = wmi
        .raw_query(
            "SELECT UtilizationPercentage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
        )
        .unwrap_or_default();

    let util_max = engines
        .iter()
        .filter_map(|e| e.utilization_percentage)
        .max()
        .map(|u| u as f64);

    let adapter = adapters
        .into_iter()
        .find(|a| a.dedicated_usage.is_some() || a.shared_usage.is_some());

    let mut sample = GpuSample {
        utilization_pct: util_max,
        ..Default::default()
    };

    if let Some(a) = adapter {
        sample.dedicated_used_bytes = a.dedicated_usage;
        sample.shared_used_bytes = a.shared_usage;
        sample.total_committed_bytes = a.total_committed;
    }

    if sample.utilization_pct.is_none()
        && sample.dedicated_used_bytes.is_none()
        && sample.shared_used_bytes.is_none()
        && sample.total_committed_bytes.is_none()
    {
        None
    } else {
        Some(sample)
    }
}

fn warn_wmi_once(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        log::warn!("{msg}");
    }
}

pub(crate) fn gpu_connect() -> Option<wmi::WMIConnection> {
    static WARNED_WMI_GPU: AtomicBool = AtomicBool::new(false);

    match wmi::WMIConnection::new() {
        Ok(wmi) => Some(wmi),
        Err(_) => {
            warn_wmi_once(
                &WARNED_WMI_GPU,
                "WMI connection failed; GPU metrics disabled",
            );
            None
        }
    }
}

pub(crate) fn gpu_sample_wmi(wmi: &wmi::WMIConnection) -> Option<GpuSample> {
    use serde::Deserialize;

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "PascalCase")]
    struct GpuAdapter {
        dedicated_usage: Option<u64>,
        shared_usage: Option<u64>,
        total_committed: Option<u64>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "PascalCase")]
    struct GpuEngine {
        utilization_percentage: Option<u64>,
    }

    let adapters: Vec<GpuAdapter> = wmi
        .raw_query(
            "SELECT DedicatedUsage, SharedUsage, TotalCommitted FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapter",
        )
        .unwrap_or_default();

    let engines: Vec<GpuEngine> = wmi
        .raw_query(
            "SELECT UtilizationPercentage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
        )
        .unwrap_or_default();

    let util_max = engines
        .iter()
        .filter_map(|e| e.utilization_percentage)
        .max()
        .map(|u| u as f64);

    let adapter = adapters
        .into_iter()
        .find(|a| a.dedicated_usage.is_some() || a.shared_usage.is_some());

    let mut sample = GpuSample {
        utilization_pct: util_max,
        ..Default::default()
    };

    if let Some(a) = adapter {
        sample.dedicated_used_bytes = a.dedicated_usage;
        sample.shared_used_bytes = a.shared_usage;
        sample.total_committed_bytes = a.total_committed;
    }

    if sample.utilization_pct.is_none()
        && sample.dedicated_used_bytes.is_none()
        && sample.shared_used_bytes.is_none()
        && sample.total_committed_bytes.is_none()
    {
        None
    } else {
        Some(sample)
    }
}

fn warn_wmi_once(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        log::warn!("{msg}");
    }
}
