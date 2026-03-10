use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessesToUpdate, System};

use super::overlay::bytes_to_mib;
use crate::resource_monitor;

/// Cached overlay statistics, sampled periodically from atomic counters and sysinfo.
///
/// `update()` should be called every frame; internally it rate-limits itself to
/// one real sample every 500 ms so the cost of `sysinfo` refresh is amortised.
pub(crate) struct OverlayStats {
    sysinfo: System,
    pid: Pid,
    last_sample_at: Instant,
    last_cam_count: u64,
    last_infer_count: u64,
    last_infer_time_ns: u64,
    last_infer_time_samples: u64,
    pub cam_fps: f32,
    pub infer_fps: f32,
    pub infer_latency_ms: f32,
    pub infer_capacity_fps: f32,
    pub proc_cpu_pct: f32,
    pub proc_rss_mib: f32,
    pub cpu_history: VecDeque<f32>,
    cpu_history_cap: usize,
}

impl OverlayStats {
    pub fn new() -> Self {
        let pid = Pid::from_u32(std::process::id());
        Self {
            sysinfo: System::new(),
            pid,
            last_sample_at: Instant::now(),
            last_cam_count: resource_monitor::CAMERA_FRAMES.load(Ordering::Relaxed),
            last_infer_count: resource_monitor::INFERENCE_COMPLETIONS.load(Ordering::Relaxed),
            last_infer_time_ns: resource_monitor::INFERENCE_TIME_NS_TOTAL.load(Ordering::Relaxed),
            last_infer_time_samples: resource_monitor::INFERENCE_TIME_SAMPLES
                .load(Ordering::Relaxed),
            cam_fps: 0.0,
            infer_fps: 0.0,
            infer_latency_ms: 0.0,
            infer_capacity_fps: 0.0,
            proc_cpu_pct: 0.0,
            proc_rss_mib: 0.0,
            cpu_history: VecDeque::new(),
            cpu_history_cap: 120,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_sample_at) < Duration::from_millis(500) {
            return;
        }

        let dt_s = now
            .duration_since(self.last_sample_at)
            .as_secs_f32()
            .max(0.001);
        self.last_sample_at = now;

        self.sysinfo
            .refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        self.sysinfo.refresh_memory();

        let (cpu_pct, rss_bytes) = self
            .sysinfo
            .process(self.pid)
            .map(|p| (p.cpu_usage(), p.memory()))
            .unwrap_or((0.0, 0));

        self.proc_cpu_pct = cpu_pct;
        self.proc_rss_mib = bytes_to_mib(rss_bytes);

        let cam_count_now = resource_monitor::CAMERA_FRAMES.load(Ordering::Relaxed);
        let cam_delta = cam_count_now.wrapping_sub(self.last_cam_count);
        self.cam_fps = (cam_delta as f32) / dt_s;
        self.last_cam_count = cam_count_now;

        let infer_count_now = resource_monitor::INFERENCE_COMPLETIONS.load(Ordering::Relaxed);
        let infer_delta = infer_count_now.wrapping_sub(self.last_infer_count);
        self.infer_fps = (infer_delta as f32) / dt_s;
        self.last_infer_count = infer_count_now;

        let infer_time_ns_now = resource_monitor::INFERENCE_TIME_NS_TOTAL.load(Ordering::Relaxed);
        let infer_time_samples_now =
            resource_monitor::INFERENCE_TIME_SAMPLES.load(Ordering::Relaxed);
        let infer_time_ns_delta = infer_time_ns_now.wrapping_sub(self.last_infer_time_ns);
        let infer_time_samples_delta =
            infer_time_samples_now.wrapping_sub(self.last_infer_time_samples);
        self.last_infer_time_ns = infer_time_ns_now;
        self.last_infer_time_samples = infer_time_samples_now;

        if infer_time_samples_delta > 0 {
            self.infer_latency_ms =
                (infer_time_ns_delta as f32 / infer_time_samples_delta as f32) / 1_000_000.0;
        } else {
            self.infer_latency_ms = 0.0;
        }

        self.infer_capacity_fps = if self.infer_latency_ms > 0.0 {
            1000.0 / self.infer_latency_ms
        } else {
            0.0
        };

        self.cpu_history.push_back(cpu_pct);
        while self.cpu_history.len() > self.cpu_history_cap {
            self.cpu_history.pop_front();
        }
    }
}
