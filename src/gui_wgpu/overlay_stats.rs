use std::collections::VecDeque;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::resource_monitor::{self, CounterSnapshot};

/// Cached overlay statistics, sampled periodically from atomic counters and sysinfo.
///
/// `update()` should be called every frame; internally it rate-limits itself to
/// one real sample every 500 ms so the cost of `sysinfo` refresh is amortised.
pub(crate) struct OverlayStats {
    sysinfo: System,
    pid: Pid,
    last_sample_at: Instant,
    counters: CounterSnapshot,
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
            counters: CounterSnapshot::new(),
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
        self.last_sample_at = now;

        self.sysinfo
            .refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        // Note: refresh_memory() is intentionally omitted here — the overlay
        // only uses per-process CPU% and RSS (which come from refresh_processes),
        // not system-wide memory totals.  The background ResourceMonitor thread
        // handles the full system-memory refresh for its own logging.

        let (cpu_pct, rss_bytes) = self
            .sysinfo
            .process(self.pid)
            .map(|p| (p.cpu_usage(), p.memory()))
            .unwrap_or((0.0, 0));

        self.proc_cpu_pct = cpu_pct;
        self.proc_rss_mib = resource_monitor::bytes_to_mib(rss_bytes) as f32;

        // Delegate all counter-based rate computation to the shared snapshot.
        let rates = self.counters.tick(now);
        self.cam_fps = rates.cam_fps as f32;
        self.infer_fps = rates.infer_fps as f32;
        self.infer_latency_ms = rates.infer_latency_ms as f32;
        self.infer_capacity_fps = rates.infer_capacity_fps as f32;

        self.cpu_history.push_back(cpu_pct);
        while self.cpu_history.len() > self.cpu_history_cap {
            self.cpu_history.pop_front();
        }
    }
}
