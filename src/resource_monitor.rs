// src/resource_monitor.rs — Lightweight background resource monitor.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use log::info;
use sysinfo::{Pid, ProcessesToUpdate, System};

// ── Constants ──────────────────────────────────────────────────────

const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

// ── Configuration ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ResourceMonitorConfig {
    pub interval: Duration,
    pub log_file: Option<PathBuf>,
    pub include_gpu: bool,
}

// ── Global atomic counters ─────────────────────────────────────────

/// Inference-completions counter (increment once per successful inference).
pub static INFERENCE_COMPLETIONS: AtomicU64 = AtomicU64::new(0);

/// Camera frames counter (increment once per received camera frame).
pub static CAMERA_FRAMES: AtomicU64 = AtomicU64::new(0);

/// Sum of observed inference durations in nanoseconds.
pub static INFERENCE_TIME_NS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Number of inference duration samples recorded.
pub static INFERENCE_TIME_SAMPLES: AtomicU64 = AtomicU64::new(0);

// ── Counter helpers (always compiled; callers live behind feature gates) ─

/// Record a single inference completion.
#[inline]
#[allow(dead_code)]
pub fn record_inference_completion() {
    INFERENCE_COMPLETIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record a single camera frame arrival.
#[inline]
#[allow(dead_code)]
pub fn record_camera_frame() {
    CAMERA_FRAMES.fetch_add(1, Ordering::Relaxed);
}

/// Record one inference duration sample.
#[inline]
#[allow(dead_code)]
pub fn record_inference_duration(duration: Duration) {
    let ns = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    INFERENCE_TIME_NS_TOTAL.fetch_add(ns, Ordering::Relaxed);
    INFERENCE_TIME_SAMPLES.fetch_add(1, Ordering::Relaxed);
}

// ── ResourceMonitor ────────────────────────────────────────────────

pub struct ResourceMonitor {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl ResourceMonitor {
    pub fn start(config: ResourceMonitorConfig) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);

        let handle = match thread::Builder::new()
            .name("resource-monitor".to_string())
            .spawn(move || run_monitor_loop(config, stop_thread))
        {
            Ok(h) => Some(h),
            Err(e) => {
                log::warn!("Failed to spawn resource-monitor thread: {e}");
                None
            }
        };

        Self { stop, handle }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ── Unit conversions ───────────────────────────────────────────────

#[inline]
pub fn bytes_to_mib(bytes: u64) -> f64 {
    (bytes as f64) / BYTES_PER_MIB
}

// ── Monitor loop ───────────────────────────────────────────────────

fn run_monitor_loop(config: ResourceMonitorConfig, stop: Arc<AtomicBool>) {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new();

    let mut file_writer = config
        .log_file
        .as_ref()
        .and_then(|path| OpenOptions::new().create(true).append(true).open(path).ok())
        .map(BufWriter::new);

    let mut next_tick = Instant::now();
    let mut counters = CounterSnapshot::new();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let sample_instant = Instant::now();
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        sys.refresh_memory();

        let (proc_cpu_pct, proc_rss_bytes) = sys
            .process(pid)
            .map(|p| (p.cpu_usage() as f64, p.memory()))
            .unwrap_or((0.0, 0));

        let sys_total_bytes = sys.total_memory();
        let sys_used_bytes = sys.used_memory();

        let gpu = if config.include_gpu {
            gpu_sample()
        } else {
            None
        };

        let rates = counters.tick(sample_instant);

        let mut line = format!(
            "resource cpu={cpu:.1}% rss={rss:.1}MiB sys_used={sys_used:.1}MiB \
             sys_total={sys_total:.1}MiB cam_fps={cam_fps:.2} infer_fps={infer_fps:.2} \
             infer_ms={infer_ms:.2} infer_capacity_fps={infer_cap:.2}",
            cpu = proc_cpu_pct,
            rss = bytes_to_mib(proc_rss_bytes),
            sys_used = bytes_to_mib(sys_used_bytes),
            sys_total = bytes_to_mib(sys_total_bytes),
            cam_fps = rates.cam_fps,
            infer_fps = rates.infer_fps,
            infer_ms = rates.infer_latency_ms,
            infer_cap = rates.infer_capacity_fps,
        );

        if let Some(sample) = gpu {
            append_gpu_metrics(&mut line, &sample);
        }

        info!("{line}");
        write_line(&mut file_writer, &line);

        next_tick += config.interval;
        let sleep_for = next_tick.saturating_duration_since(Instant::now());
        thread::sleep(sleep_for);
    }

    if let Some(mut w) = file_writer {
        let _ = w.flush();
    }
}

// ── Counter snapshot (delta tracking) ──────────────────────────────

struct Rates {
    cam_fps: f64,
    infer_fps: f64,
    infer_latency_ms: f64,
    infer_capacity_fps: f64,
}

struct CounterSnapshot {
    at: Instant,
    cam: u64,
    infer: u64,
    infer_ns: u64,
    infer_samples: u64,
}

impl CounterSnapshot {
    fn new() -> Self {
        Self {
            at: Instant::now(),
            cam: CAMERA_FRAMES.load(Ordering::Relaxed),
            infer: INFERENCE_COMPLETIONS.load(Ordering::Relaxed),
            infer_ns: INFERENCE_TIME_NS_TOTAL.load(Ordering::Relaxed),
            infer_samples: INFERENCE_TIME_SAMPLES.load(Ordering::Relaxed),
        }
    }

    fn tick(&mut self, now: Instant) -> Rates {
        let dt_s = now.duration_since(self.at).as_secs_f64().max(0.001);
        self.at = now;

        let cam_now = CAMERA_FRAMES.load(Ordering::Relaxed);
        let cam_fps = (cam_now.wrapping_sub(self.cam) as f64) / dt_s;
        self.cam = cam_now;

        let infer_now = INFERENCE_COMPLETIONS.load(Ordering::Relaxed);
        let infer_fps = (infer_now.wrapping_sub(self.infer) as f64) / dt_s;
        self.infer = infer_now;

        let ns_now = INFERENCE_TIME_NS_TOTAL.load(Ordering::Relaxed);
        let samples_now = INFERENCE_TIME_SAMPLES.load(Ordering::Relaxed);
        let ns_delta = ns_now.wrapping_sub(self.infer_ns);
        let samples_delta = samples_now.wrapping_sub(self.infer_samples);
        self.infer_ns = ns_now;
        self.infer_samples = samples_now;

        let infer_latency_ms = if samples_delta > 0 {
            (ns_delta as f64 / samples_delta as f64) / 1_000_000.0
        } else {
            0.0
        };
        let infer_capacity_fps = if infer_latency_ms > 0.0 {
            1000.0 / infer_latency_ms
        } else {
            0.0
        };

        Rates {
            cam_fps,
            infer_fps,
            infer_latency_ms,
            infer_capacity_fps,
        }
    }
}

// ── File writer helper ─────────────────────────────────────────────

fn write_line(writer: &mut Option<BufWriter<std::fs::File>>, line: &str) {
    let Some(w) = writer.as_mut() else {
        return;
    };
    let _ = writeln!(w, "{line}");
}

// ── GPU sampling ───────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct GpuSample {
    utilization_pct: Option<f64>,
    dedicated_used_bytes: Option<u64>,
    shared_used_bytes: Option<u64>,
    total_committed_bytes: Option<u64>,
}

fn append_gpu_metrics(line: &mut String, sample: &GpuSample) {
    if let Some(util) = sample.utilization_pct {
        line.push_str(&format!(" gpu_util={util:.1}%"));
    }
    if let Some(dedicated) = sample.dedicated_used_bytes {
        line.push_str(&format!(" gpu_dedicated={:.1}MiB", bytes_to_mib(dedicated)));
    }
    if let Some(shared) = sample.shared_used_bytes {
        line.push_str(&format!(" gpu_shared={:.1}MiB", bytes_to_mib(shared)));
    }
    if let Some(total) = sample.total_committed_bytes {
        line.push_str(&format!(" gpu_total={:.1}MiB", bytes_to_mib(total)));
    }
}

fn gpu_sample() -> Option<GpuSample> {
    #[cfg(target_os = "windows")]
    {
        windows_gpu_sample()
    }

    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

#[cfg(target_os = "windows")]
fn windows_gpu_sample() -> Option<GpuSample> {
    use serde::Deserialize;
    use wmi::WMIConnection;

    use std::sync::atomic::AtomicBool;
    static WARNED_WMI_GPU: AtomicBool = AtomicBool::new(false);

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

    let wmi = match WMIConnection::new() {
        Ok(wmi) => wmi,
        Err(_) => {
            warn_wmi_once(
                &WARNED_WMI_GPU,
                "WMI connection failed; GPU metrics disabled",
            );
            return None;
        }
    };

    let adapters: Vec<GpuAdapter> = wmi.raw_query(
        "SELECT DedicatedUsage, SharedUsage, TotalCommitted FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapter",
    ).unwrap_or_default();

    let engines: Vec<GpuEngine> = wmi.raw_query(
        "SELECT UtilizationPercentage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
    ).unwrap_or_default();

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

#[cfg(target_os = "windows")]
fn warn_wmi_once(flag: &std::sync::atomic::AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        log::warn!("{msg}");
    }
}
