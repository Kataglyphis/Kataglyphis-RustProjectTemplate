pub(crate) mod inference;
pub(crate) mod inference_bridge;
pub(crate) mod overlay;
pub(crate) mod overlay_stats;
pub(crate) mod pipeline;
pub(crate) mod renderer;
pub(crate) mod wgpu_init;

use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use gstreamer as gst;
use gstreamer::prelude::ElementExt;
use pollster::block_on;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use pipeline::{build_pipeline, Frame};
use renderer::WgpuState;

const REDRAW_INTERVAL_MS: u64 = 16;

/// WGPU backend selection for the GUI.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "gui_windows", derive(clap::ValueEnum))]
pub enum GpuBackend {
    /// Automatically pick the best backend for the platform.
    Auto,
    /// Vulkan.
    Vulkan,
    /// DirectX 12 (Windows only).
    Dx12,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Auto => write!(f, "auto"),
            GpuBackend::Vulkan => write!(f, "vulkan"),
            GpuBackend::Dx12 => write!(f, "dx12"),
        }
    }
}

fn backends_for(backend: &GpuBackend) -> wgpu::Backends {
    match backend {
        GpuBackend::Vulkan => wgpu::Backends::VULKAN,
        GpuBackend::Auto => wgpu::Backends::PRIMARY,
        GpuBackend::Dx12 => {
            #[cfg(target_os = "windows")]
            {
                wgpu::Backends::DX12
            }
            #[cfg(not(target_os = "windows"))]
            {
                eprintln!(
                    "--backend dx12 is only available on Windows; falling back to primary. Valid: vulkan | auto"
                );
                wgpu::Backends::PRIMARY
            }
        }
    }
}

pub fn run_with_backend(backend: &GpuBackend) -> anyhow::Result<()> {
    let label = backend.to_string();
    run_inner(backends_for(backend), &label)
}

fn run_inner(backends: wgpu::Backends, backend_label: &str) -> anyhow::Result<()> {
    gst::init().context("Failed to initialize GStreamer")?;

    let (frame_tx, frame_rx) = sync_channel::<Frame>(2);
    let pipeline = build_pipeline(frame_tx).context("Failed to build GStreamer pipeline")?;

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| anyhow::anyhow!("Failed to start pipeline: {e}"))?;

    let window_result = start_window(frame_rx, backends, backend_label);

    // Always try to stop the pipeline, even if the window loop failed.
    let _ = pipeline.set_state(gst::State::Null);

    window_result
}

// ── GuiApp ─────────────────────────────────────────────────────────

struct GuiApp {
    frame_rx: Receiver<Frame>,
    backends: wgpu::Backends,
    backend_label: String,
    window: Option<Arc<Window>>,
    state: Option<WgpuState>,
    latest_frame: Option<Frame>,
    latest_frame_id: u64,
    uploaded_frame_id: u64,
    next_redraw_deadline: Instant,
    redraw_interval: Duration,
    /// Stored init error from `resumed()` — checked on next event tick.
    init_error: Option<String>,
}

impl GuiApp {
    /// Drain all pending frames from the channel, keeping only the latest.
    /// Returns `true` if at least one new frame was received.
    fn drain_frames(&mut self) -> bool {
        let mut got_frame = false;
        while let Ok(frame) = self.frame_rx.try_recv() {
            self.latest_frame = Some(frame);
            self.latest_frame_id = self.latest_frame_id.wrapping_add(1);
            got_frame = true;
        }
        got_frame
    }
}

impl ApplicationHandler for GuiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = WindowAttributes::default()
            .with_title(format!("GStreamer + WGPU [{}]", self.backend_label))
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

        let window = match event_loop.create_window(window_attributes) {
            Ok(w) => w,
            Err(e) => {
                self.init_error = Some(format!("Failed to create window: {e}"));
                event_loop.exit();
                return;
            }
        };

        let window = Arc::new(window);
        let state = match block_on(WgpuState::new(
            Arc::clone(&window),
            event_loop,
            self.backends,
        )) {
            Ok(s) => s,
            Err(e) => {
                self.init_error = Some(format!("Failed to initialise WGPU: {e:#}"));
                event_loop.exit();
                return;
            }
        };

        self.window = Some(Arc::clone(&window));
        self.state = Some(state);

        self.next_redraw_deadline = Instant::now();

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Drain frames before borrowing window to avoid borrow conflicts
        if matches!(event, WindowEvent::RedrawRequested) {
            self.drain_frames();
        }

        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window_id != window.id() {
            return;
        }

        if let Some(s) = self.state.as_mut() {
            s.handle_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    state.resize(size);
                }
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                if let Some(state) = self.state.as_mut() {
                    state.resize(size);
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Some(frame) = self.latest_frame.as_ref() {
                    if self.uploaded_frame_id != self.latest_frame_id {
                        if let Some(state) = self.state.as_mut() {
                            state.upload_frame(frame);
                            self.uploaded_frame_id = self.latest_frame_id;
                        }
                    }
                }
                if let Some(state) = self.state.as_mut() {
                    if let Err(err) = state.render(window) {
                        eprintln!("Render error: {err:?}");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Keep the UI responsive even if the video source is low-FPS.
        // We upload new video frames only when they arrive, but we redraw at a steady cadence
        // so egui interactions feel smooth.
        let got_frame = self.drain_frames();

        let Some(window) = self.window.as_ref() else {
            return;
        };

        let now = Instant::now();
        if now >= self.next_redraw_deadline {
            self.next_redraw_deadline = now + self.redraw_interval;
            window.request_redraw();
        } else if got_frame {
            window.request_redraw();
        }

        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw_deadline));
    }
}

fn start_window(
    frame_rx: Receiver<Frame>,
    backends: wgpu::Backends,
    backend_label: &str,
) -> anyhow::Result<()> {
    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = GuiApp {
        frame_rx,
        backends,
        backend_label: backend_label.to_string(),
        window: None,
        state: None,
        latest_frame: None,
        latest_frame_id: 0,
        uploaded_frame_id: 0,
        next_redraw_deadline: Instant::now(),
        redraw_interval: Duration::from_millis(REDRAW_INTERVAL_MS),
        init_error: None,
    };

    event_loop
        .run_app(&mut app)
        .context("Failed to run event loop")?;

    if let Some(err) = app.init_error {
        anyhow::bail!("{err}");
    }

    Ok(())
}
pub(crate) mod egui_overlay;
