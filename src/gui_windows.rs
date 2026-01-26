use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use pollster::block_on;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use egui::{Context as EguiContext, ViewportId};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use sysinfo::{Pid, ProcessesToUpdate, System};

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use crate::person_detection::{default_model_path, Detection, PersonDetector};
use crate::resource_monitor;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
struct InferRequest {
    frame_id: u64,
    rgba: Arc<Vec<u8>>,
    width: u32,
    height: u32,
    score_threshold: f32,
}

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
struct InferResult {
    frame_id: u64,
    detections: Vec<Detection>,
    error: Option<String>,
}

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
    let pipeline = build_pipeline(frame_tx);

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    start_window(frame_rx, backends, backend_label);

    pipeline
        .set_state(gst::State::Null)
        .expect("Failed to stop pipeline");
}

#[derive(Clone)]
struct Frame {
    id: u64,
    data: Arc<Vec<u8>>,
    width: u32,
    height: u32,
}

fn rgba_tightly_packed_copy(src: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    if width == 0 || height == 0 {
        return None;
    }

    let row_bytes = (width as usize).saturating_mul(4);
    let expected_len = row_bytes.saturating_mul(height as usize);

    if src.len() == expected_len {
        return Some(src.to_vec());
    }

    // Many GStreamer buffers are padded per row (stride). We conservatively try to
    // interpret the buffer as a single RGBA plane with a constant stride.
    if src.len() < expected_len {
        return None;
    }

    let h = height as usize;
    let mut stride = src.len() / h;
    if stride < row_bytes {
        return None;
    }

    // If there is trailing padding after the last row, prefer a stride that still
    // lets us safely read all rows.
    while stride.saturating_mul(h) > src.len() {
        if stride == 0 {
            return None;
        }
        stride -= 1;
    }

    let mut out = vec![0u8; expected_len];
    for y in 0..h {
        let src_start = y.saturating_mul(stride);
        let src_end = src_start.saturating_add(row_bytes).min(src.len());
        let dst_start = y.saturating_mul(row_bytes);
        let dst_end = dst_start.saturating_add(row_bytes).min(out.len());

        if src_end <= src_start || dst_end <= dst_start {
            return None;
        }

        out[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
    }

    Some(out)
}

fn build_pipeline(frame_tx: std::sync::mpsc::SyncSender<Frame>) -> gst::Pipeline {
    let pipeline = gst::Pipeline::new();

    let frame_id = AtomicU64::new(1);

    let src = gst::ElementFactory::make("mfvideosrc")
        .build()
        .or_else(|_| gst::ElementFactory::make("autovideosrc").build())
        .expect("Failed to create a video source");

    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Failed to create videoconvert");

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .build();

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()
        .expect("Failed to create capsfilter");

    let sink = gst::ElementFactory::make("appsink")
        .property("emit-signals", true)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()
        .expect("Failed to create appsink");

    pipeline
        .add_many([&src, &convert, &capsfilter, &sink])
        .expect("Failed to add elements");
    gst::Element::link_many([&src, &convert, &capsfilter, &sink]).expect("Failed to link elements");

    let appsink = sink
        .clone()
        .dynamic_cast::<gst_app::AppSink>()
        .expect("Sink is not an appsink");

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                if let Some(sample) = sink.pull_sample().ok() {
                    if let Some(buffer) = sample.buffer() {
                        let caps = sample
                            .caps()
                            .and_then(|c| c.structure(0))
                            .map(|s| s.to_owned());
                        if let Some(structure) = caps {
                            let width = structure.get::<i32>("width").unwrap_or(640) as u32;
                            let height = structure.get::<i32>("height").unwrap_or(480) as u32;
                            if let Ok(map) = buffer.map_readable() {
                                if let Some(data) = rgba_tightly_packed_copy(map.as_slice(), width, height) {
                                    #[cfg(target_os = "windows")]
                                    resource_monitor::record_camera_frame();

                                    let id = frame_id.fetch_add(1, Ordering::Relaxed);

                                    let _ = frame_tx.try_send(Frame {
                                        id,
                                        data: Arc::new(data),
                                        width,
                                        height,
                                    });
                                }
                            }
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    pipeline
}

fn start_window(frame_rx: Receiver<Frame>, backends: wgpu::Backends, backend_label: &str) {
    struct GuiApp {
        frame_rx: Receiver<Frame>,
        backends: wgpu::Backends,
        backend_label: String,
        window: Option<&'static Window>,
        state: Option<WgpuState<'static>>,
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
                .with_title(format!(
                    "GStreamer + WGPU [{}]",
                    self.backend_label
                ))
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

            let window = event_loop
                .create_window(window_attributes)
                .expect("Failed to create window");

            // wgpu::Surface borrows the Window by lifetime; `run_app` stores the handler for the
            // duration of the loop, so keep a stable Window reference for the app lifetime.
            let window: &'static Window = Box::leak(Box::new(window));
            let state = block_on(WgpuState::new(window, event_loop, self.backends));

            self.window = Some(window);
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
            let Some(window) = self.window else {
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

                    if let Some(frame) = self.latest_frame.as_ref() {
                        if self.uploaded_frame_id != self.latest_frame_id {
                            state.upload_frame(frame);
                            self.uploaded_frame_id = self.latest_frame_id;
                        }
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
            let Some(window) = self.window else {
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

struct WgpuState<'w> {
    surface: wgpu::Surface<'w>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    sampler: wgpu::Sampler,
    texture: Option<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup, u32, u32)>,
    egui_ctx: EguiContext,
    egui_state: EguiWinitState,
    egui_renderer: EguiRenderer,
    egui_screen: ScreenDescriptor,
    overlay_enabled: bool,

    sysinfo: System,
    pid: Pid,

    overlay_last_sample_at: Instant,
    overlay_last_cam_count: u64,
    overlay_last_infer_count: u64,
    overlay_last_infer_time_ns: u64,
    overlay_last_infer_time_samples: u64,
    overlay_cam_fps: f32,
    overlay_infer_fps: f32,
    overlay_infer_latency_ms: f32,
    overlay_infer_capacity_fps: f32,
    overlay_proc_cpu_pct: f32,
    overlay_proc_rss_mib: f32,
    overlay_cpu_history: VecDeque<f32>,
    overlay_cpu_history_cap: usize,

    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    detector_error: Option<String>,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    model_path: Option<String>,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    score_threshold: f32,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    infer_every: Duration,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    last_infer: Instant,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    last_detections: Vec<Detection>,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    last_detections_frame_id: Option<u64>,

    current_frame_id: Option<u64>,

    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    infer_tx: Option<SyncSender<InferRequest>>,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    infer_rx: Option<Receiver<InferResult>>,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    infer_in_flight: bool,
    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    infer_enabled: bool,
}

impl<'w> WgpuState<'w> {
    async fn new(
        window: &'w Window,
        display_target: &dyn wgpu::rwh::HasDisplayHandle,
        backends: wgpu::Backends,
    ) -> Self {
        let size = window.inner_size();

        // Backend selection is controlled via CLI (--backend) to make it easy to
        // switch between DX12 and Vulkan without changing code.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found on the system!");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                },
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Use FIFO to match widest support and avoid swapchain semaphore reuse issues on Vulkan.
        let present_mode = wgpu::PresentMode::Fifo;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let vertex_data = [
            // pos       // uv
            Vertex {
                position: [-1.0, -1.0],
                tex_coords: [0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0],
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: [-1.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_data: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tex_quad.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let egui_ctx = EguiContext::default();
        let egui_state = EguiWinitState::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            display_target,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = EguiRenderer::new(&device, surface_format, Default::default());
        let egui_screen = ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let (infer_tx, infer_rx, detector_error, model_path) = {
            let env_override = std::env::var("KATAGLYPHIS_ONNX_MODEL")
                .ok();
            let path = env_override
                .clone()
                .unwrap_or_else(|| default_model_path().to_string_lossy().to_string());

            let (req_tx, req_rx) = sync_channel::<InferRequest>(1);
            let (res_tx, res_rx) = sync_channel::<InferResult>(1);

            let detector = match PersonDetector::new(&path) {
                Ok(detector) => Some(detector),
                Err(e) => {
                    let _ = res_tx.try_send(InferResult {
                        frame_id: 0,
                        detections: Vec::new(),
                        error: Some(format!("Failed to load model '{path}': {e:#}")),
                    });
                    None
                }
            };

            std::thread::spawn(move || {
                let Some(detector) = detector else {
                    return;
                };

                while let Ok(req) = req_rx.recv() {
                    let infer_start = Instant::now();
                    let result = match detector.infer_persons_rgba(
                        req.rgba.as_slice(),
                        req.width,
                        req.height,
                        req.score_threshold,
                    ) {
                        Ok(dets) => InferResult {
                            frame_id: req.frame_id,
                            detections: dets,
                            error: None,
                        },
                        Err(e) => InferResult {
                            frame_id: req.frame_id,
                            detections: Vec::new(),
                            error: Some(format!("Inference failed: {e:#}")),
                        },
                    };

                    let infer_duration = infer_start.elapsed();
                    #[cfg(target_os = "windows")]
                    {
                        resource_monitor::record_inference_duration(infer_duration);
                        resource_monitor::record_inference_completion();
                    }

                    let _ = res_tx.try_send(result);
                }
            });

            (
                Some(req_tx),
                Some(res_rx),
                None,
                Some(path),
            )
        };

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let score_threshold = std::env::var("KATAGLYPHIS_SCORE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5);

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let infer_every_ms = std::env::var("KATAGLYPHIS_INFER_EVERY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let pid = Pid::from_u32(std::process::id());
        let sysinfo = System::new_all();

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: index_data.len() as u32,
            sampler,
            texture: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            egui_screen,
            overlay_enabled: true,

            sysinfo,
            pid,

            overlay_last_sample_at: Instant::now(),
            overlay_last_cam_count: resource_monitor::CAMERA_FRAMES.load(Ordering::Relaxed),
            overlay_last_infer_count: resource_monitor::INFERENCE_COMPLETIONS
                .load(Ordering::Relaxed),
            overlay_last_infer_time_ns: resource_monitor::INFERENCE_TIME_NS_TOTAL
                .load(Ordering::Relaxed),
            overlay_last_infer_time_samples: resource_monitor::INFERENCE_TIME_SAMPLES
                .load(Ordering::Relaxed),
            overlay_cam_fps: 0.0,
            overlay_infer_fps: 0.0,
            overlay_infer_latency_ms: 0.0,
            overlay_infer_capacity_fps: 0.0,
            overlay_proc_cpu_pct: 0.0,
            overlay_proc_rss_mib: 0.0,
            overlay_cpu_history: VecDeque::new(),
            overlay_cpu_history_cap: 120,

            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            detector_error,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            model_path,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            score_threshold,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            infer_every: Duration::from_millis(infer_every_ms),
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            last_infer: Instant::now() - Duration::from_secs(1),
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            last_detections: Vec::new(),
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            last_detections_frame_id: None,

            current_frame_id: None,

            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            infer_tx,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            infer_rx,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            infer_in_flight: false,
            #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
            infer_enabled: true,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.egui_screen.size_in_pixels = [new_size.width, new_size.height];
        }
    }

    fn handle_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) {
        let response = self.egui_state.on_window_event(window, event);
        if response.repaint {
            window.request_redraw();
        }
    }

    fn upload_frame(&mut self, frame: &Frame) {
        self.current_frame_id = Some(frame.id);
        self.poll_inference();
        let needs_recreate = match &self.texture {
            Some((_, _, _, w, h)) => frame.width != *w || frame.height != *h,
            None => true,
        };

        if needs_recreate {
            let size = wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            };
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("frame_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("texture_bind_group"),
                layout: &self.render_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.texture = Some((texture, view, bind_group, frame.width, frame.height));
        }

        if let Some((texture, _, _, _, _)) = &self.texture {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * frame.width),
                rows_per_image: Some(frame.height),
            };
            let size = wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            };
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &frame.data,
                layout,
                size,
            );
        }

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        self.maybe_infer(frame);
    }

    fn poll_inference(&mut self) {
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        if let Some(rx) = self.infer_rx.as_ref() {
            while let Ok(res) = rx.try_recv() {
                self.infer_in_flight = false;
                self.last_detections = res.detections;
                self.last_detections_frame_id = Some(res.frame_id);
                self.detector_error = res.error;
            }
        }
    }

    fn update_overlay_stats(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.overlay_last_sample_at) < Duration::from_millis(500) {
            return;
        }

        let dt_s = now
            .duration_since(self.overlay_last_sample_at)
            .as_secs_f32()
            .max(0.001);
        self.overlay_last_sample_at = now;

        self.sysinfo
            .refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        self.sysinfo.refresh_memory();

        let (cpu_pct, rss_bytes) = self
            .sysinfo
            .process(self.pid)
            .map(|p| (p.cpu_usage(), p.memory()))
            .unwrap_or((0.0, 0));

        self.overlay_proc_cpu_pct = cpu_pct;
        self.overlay_proc_rss_mib = bytes_to_mib(rss_bytes);

        let cam_count_now = resource_monitor::CAMERA_FRAMES.load(Ordering::Relaxed);
        let cam_delta = cam_count_now.wrapping_sub(self.overlay_last_cam_count);
        self.overlay_cam_fps = (cam_delta as f32) / dt_s;
        self.overlay_last_cam_count = cam_count_now;

        let infer_count_now = resource_monitor::INFERENCE_COMPLETIONS.load(Ordering::Relaxed);
        let infer_delta = infer_count_now.wrapping_sub(self.overlay_last_infer_count);
        self.overlay_infer_fps = (infer_delta as f32) / dt_s;
        self.overlay_last_infer_count = infer_count_now;

        let infer_time_ns_now = resource_monitor::INFERENCE_TIME_NS_TOTAL.load(Ordering::Relaxed);
        let infer_time_samples_now =
            resource_monitor::INFERENCE_TIME_SAMPLES.load(Ordering::Relaxed);
        let infer_time_ns_delta =
            infer_time_ns_now.wrapping_sub(self.overlay_last_infer_time_ns);
        let infer_time_samples_delta =
            infer_time_samples_now.wrapping_sub(self.overlay_last_infer_time_samples);
        self.overlay_last_infer_time_ns = infer_time_ns_now;
        self.overlay_last_infer_time_samples = infer_time_samples_now;

        if infer_time_samples_delta > 0 {
            self.overlay_infer_latency_ms =
                (infer_time_ns_delta as f32 / infer_time_samples_delta as f32) / 1_000_000.0;
        } else {
            self.overlay_infer_latency_ms = 0.0;
        }

        self.overlay_infer_capacity_fps = if self.overlay_infer_latency_ms > 0.0 {
            1000.0 / self.overlay_infer_latency_ms
        } else {
            0.0
        };

        self.overlay_cpu_history.push_back(cpu_pct);
        while self.overlay_cpu_history.len() > self.overlay_cpu_history_cap {
            self.overlay_cpu_history.pop_front();
        }
    }

    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
    fn maybe_infer(&mut self, frame: &Frame) {
        let Some(tx) = self.infer_tx.as_ref() else {
            return;
        };

        if !self.infer_enabled {
            return;
        }

        if self.last_infer.elapsed() < self.infer_every {
            return;
        }

        if self.infer_in_flight {
            return;
        }

        // Clone only when we actually schedule inference (decimated by `infer_every`).
        // This keeps the UI thread responsive even for heavy models.
        if tx
            .try_send(InferRequest {
                frame_id: frame.id,
                rgba: Arc::clone(&frame.data),
                width: frame.width,
                height: frame.height,
                score_threshold: self.score_threshold,
            })
            .is_ok()
        {
            self.last_infer = Instant::now();
            self.infer_in_flight = true;
        }
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        self.poll_inference();

        self.egui_screen.pixels_per_point = window.scale_factor() as f32;
        self.egui_screen.size_in_pixels = [self.config.width, self.config.height];

        let raw_input = self.egui_state.take_egui_input(window);
        self.update_overlay_stats();

        let mut overlay_enabled = self.overlay_enabled;
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let mut infer_enabled = self.infer_enabled;
        let frame_dimensions = self
            .texture
            .as_ref()
            .map(|(_, _, _, w, h)| (*w, *h))
            .unwrap_or((0, 0));

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let detections_count = if self.last_detections_frame_id.is_some() {
            self.last_detections.len()
        } else {
            0
        };
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let detector_error = self.detector_error.clone();
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let model_path = self.model_path.clone();
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let score_threshold = self.score_threshold;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if overlay_enabled {
                egui::Window::new("Overlay")
                    .default_pos(egui::pos2(12.0, 12.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("ONNX overlay");
                        ui.label(format!(
                            "Frame: {}x{}",
                            frame_dimensions.0, frame_dimensions.1
                        ));
                        ui.separator();
                        ui.label(format!("Camera FPS: {:.2}", self.overlay_cam_fps));
                        ui.label(format!("Infer FPS:  {:.2}", self.overlay_infer_fps));
                        ui.label(format!("Infer ms:   {:.2}", self.overlay_infer_latency_ms));
                        ui.label(format!(
                            "Infer cap: {:.2} fps",
                            self.overlay_infer_capacity_fps
                        ));
                        ui.label(format!(
                            "CPU: {:.1}%   RSS: {:.1} MiB",
                            self.overlay_proc_cpu_pct,
                            self.overlay_proc_rss_mib
                        ));
                        draw_cpu_history(ui, &self.overlay_cpu_history);

                        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
                        {
                            // Draw bounding boxes in the foreground over the video.
                            if frame_dimensions.0 > 0 && frame_dimensions.1 > 0 && !self.last_detections.is_empty() {
                                let screen = ctx.content_rect();
                                let sx = screen.width() / frame_dimensions.0 as f32;
                                let sy = screen.height() / frame_dimensions.1 as f32;
                                let painter = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("detections"),
                                ));

                                for d in &self.last_detections {
                                    let x1 = screen.left() + d.x1 * sx;
                                    let y1 = screen.top() + d.y1 * sy;
                                    let x2 = screen.left() + d.x2 * sx;
                                    let y2 = screen.top() + d.y2 * sy;

                                    let rect = egui::Rect::from_min_max(
                                        egui::pos2(x1, y1),
                                        egui::pos2(x2, y2),
                                    );
                                    painter.rect_stroke(
                                        rect,
                                        0.0,
                                        egui::Stroke::new(2.0, egui::Color32::LIGHT_GREEN),
                                        egui::StrokeKind::Inside,
                                    );
                                }
                            }

                            if let Some(path) = model_path.as_deref() {
                                ui.label(format!("Model: {path}"));
                            } else {
                                ui.label("Model: (not set)");
                            }
                            ui.label(format!("Score threshold: {score_threshold:.2}"));
                            ui.label(format!("Persons: {}", detections_count));
                            ui.checkbox(&mut infer_enabled, "Inference enabled");
                            if let Some(err) = detector_error.as_deref() {
                                ui.separator();
                                ui.label(err);
                            }
                        }

                        if ui.button("Toggle overlay visibility").clicked() {
                            overlay_enabled = false;
                        }
                    });
            } else {
                egui::Window::new("Overlay (hidden)")
                    .default_pos(egui::pos2(12.0, 12.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("Overlay is hidden");
                        if ui.button("Show overlay again").clicked() {
                            overlay_enabled = true;
                        }
                    });
            }
        });
        self.overlay_enabled = overlay_enabled;
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        {
            self.infer_enabled = infer_enabled;
        }

        let clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, self.egui_screen.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            if let Some((_, _, bind_group, _, _)) = &self.texture {
                rpass.set_bind_group(0, bind_group, &[]);
            }

            rpass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &self.egui_screen,
        );

        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];

            let egui_rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            })
                .forget_lifetime();

            let mut egui_rpass = egui_rpass;

            self.egui_renderer
                .render(&mut egui_rpass, &clipped_primitives, &self.egui_screen);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

fn bytes_to_mib(bytes: u64) -> f32 {
    (bytes as f32) / (1024.0 * 1024.0)
}

fn draw_cpu_history(ui: &mut egui::Ui, history: &VecDeque<f32>) {
    let desired = egui::vec2(160.0, 48.0);
    let (rect, _response) = ui.allocate_exact_size(desired, egui::Sense::hover());

    let painter = ui.painter();
    painter.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::GRAY),
        egui::StrokeKind::Inside,
    );

    if history.len() < 2 {
        return;
    }

    let max_points = history.len().max(2) as f32 - 1.0;
    let step_x = rect.width() / max_points;
    let mut prev = None;

    for (i, &value) in history.iter().enumerate() {
        let x = rect.left() + (i as f32) * step_x;
        let y_norm = (value / 100.0).clamp(0.0, 1.0);
        let y = rect.bottom() - y_norm * rect.height();
        let pos = egui::pos2(x, y);

        if let Some(prev) = prev {
            painter.line_segment([prev, pos], egui::Stroke::new(1.5, egui::Color32::LIGHT_BLUE));
        }
        prev = Some(pos);
    }
}

