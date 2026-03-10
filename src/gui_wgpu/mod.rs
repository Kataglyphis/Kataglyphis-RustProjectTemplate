pub mod inference;
pub mod overlay;
pub(crate) mod overlay_stats;
pub mod pipeline;
pub mod renderer;

use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::{Duration, Instant};

use gstreamer as gst;
use gstreamer::prelude::ElementExt;
use pollster::block_on;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use pipeline::{Frame, build_pipeline};
use renderer::WgpuState;

fn parse_backends(backend: &str) -> wgpu::Backends {
    match backend.trim().to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => wgpu::Backends::VULKAN,
        "primary" | "auto" => wgpu::Backends::PRIMARY,
        "dx12" | "d3d12" => {
            #[cfg(target_os = "windows")]
            {
                wgpu::Backends::DX12
            }
            #[cfg(not(target_os = "windows"))]
            {
                eprintln!(
                    "--backend dx12 is only available on Windows; falling back to primary. Valid: vulkan | primary"
                );
                wgpu::Backends::PRIMARY
            }
        }
        other => {
            #[cfg(target_os = "windows")]
            {
                eprintln!(
                    "Unknown --backend '{other}', falling back to dx12. Valid: dx12 | vulkan | primary"
                );
                wgpu::Backends::DX12
            }
            #[cfg(not(target_os = "windows"))]
            {
                eprintln!(
                    "Unknown --backend '{other}', falling back to primary. Valid: vulkan | primary"
                );
                wgpu::Backends::PRIMARY
            }
        }
    }
}

pub fn run_with_backend(backend: &str) {
    run_inner(parse_backends(backend), backend);
}

fn run_inner(backends: wgpu::Backends, backend_label: &str) {
    gst::init().expect("Failed to initialize GStreamer");

    let (frame_tx, frame_rx) = sync_channel::<Frame>(2);
    let pipeline = build_pipeline(frame_tx).expect("Failed to build GStreamer pipeline");

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    start_window(frame_rx, backends, backend_label);

    pipeline
        .set_state(gst::State::Null)
        .expect("Failed to stop pipeline");
}

fn start_window(frame_rx: Receiver<Frame>, backends: wgpu::Backends, backend_label: &str) {
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
    }

    impl ApplicationHandler for GuiApp {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }

            let window_attributes = WindowAttributes::default()
                .with_title(format!("GStreamer + WGPU [{}]", self.backend_label))
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

            let window = event_loop
                .create_window(window_attributes)
                .expect("Failed to create window");

            let window = Arc::new(window);
            let state = block_on(WgpuState::new(
                Arc::clone(&window),
                event_loop,
                self.backends,
            ));

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
            let Some(window) = self.window.as_ref() else {
                return;
            };
            if window_id != window.id() {
                return;
            }
            let Some(state) = self.state.as_mut() else {
                return;
            };

            state.handle_window_event(window, &event);

            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    state.resize(size);
                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {
                    let size = window.inner_size();
                    state.resize(size);
                    window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    // Never block the UI thread; if there is a new frame, upload it once.
                    if let Ok(frame) = self.frame_rx.try_recv() {
                        self.latest_frame = Some(frame);
                        self.latest_frame_id = self.latest_frame_id.wrapping_add(1);
                    }

                    if let Some(frame) = self.latest_frame.as_ref()
                        && self.uploaded_frame_id != self.latest_frame_id
                    {
                        state.upload_frame(frame);
                        self.uploaded_frame_id = self.latest_frame_id;
                    }
                    if let Err(err) = state.render(window) {
                        eprintln!("Render error: {err:?}");
                        event_loop.exit();
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            // Keep the UI responsive even if the video source is low-FPS.
            // We upload new video frames only when they arrive, but we redraw at a steady cadence
            // so egui interactions feel smooth.
            let Some(window) = self.window.as_ref() else {
                return;
            };

            let mut got_frame = false;
            while let Ok(frame) = self.frame_rx.try_recv() {
                self.latest_frame = Some(frame);
                self.latest_frame_id = self.latest_frame_id.wrapping_add(1);
                got_frame = true;
            }

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

    let event_loop = EventLoop::new().expect("Failed to create event loop");
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
        redraw_interval: Duration::from_millis(16),
    };

    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop");
}
