#[cfg(onnx)]
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use anyhow::Context;
use wgpu::util::DeviceExt;
use winit::window::Window;

use egui::{Context as EguiContext, ViewportId};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use std::sync::Arc;

#[cfg(onnx)]
use crate::person_detection::{PersonDetector, default_model_path};
use crate::resource_monitor;

#[cfg(onnx)]
use super::inference::{InferRequest, InferResult, InferenceState};
use super::overlay::draw_cpu_history;
use super::overlay_stats::OverlayStats;
use super::pipeline::Frame;

/// Named grouping of a GPU texture, its view, bind group, and dimensions.
/// Replaces the previous anonymous 5-tuple for readability.
struct FrameTexture {
    texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct WgpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    sampler: wgpu::Sampler,
    texture: Option<FrameTexture>,
    egui_ctx: EguiContext,
    egui_state: EguiWinitState,
    egui_renderer: EguiRenderer,
    egui_screen: ScreenDescriptor,
    overlay_enabled: bool,

    overlay: OverlayStats,

    #[cfg(onnx)]
    inference: InferenceState,

    current_frame_id: Option<u64>,
}

// ── Initialisation helpers ─────────────────────────────────────────

/// Acquire a WGPU adapter, device, queue, and configured surface for the given window.
async fn init_wgpu(
    window: &Arc<Window>,
    backends: wgpu::Backends,
) -> anyhow::Result<(
    wgpu::Surface<'static>,
    wgpu::Device,
    wgpu::Queue,
    wgpu::SurfaceConfiguration,
)> {
    let size = window.inner_size();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let surface = instance
        .create_surface(Arc::clone(window))
        .context("Failed to create surface")?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .context("No suitable GPU adapters found on the system")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
        })
        .await
        .context("Failed to create device")?;

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    Ok((surface, device, queue, config))
}

/// Create the full-screen-quad geometry buffers used for video display.
fn create_quad_buffers(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let vertex_data = [
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

    (vertex_buffer, index_buffer, index_data.len() as u32)
}

/// Build the texture-sampling render pipeline (shader + layout + pipeline object).
fn create_render_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tex_quad.wgsl").into()),
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

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                format: surface_format,
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
    })
}

/// Initialise egui integration (context, platform state, renderer).
fn init_egui(
    window: &Window,
    display_target: &dyn wgpu::rwh::HasDisplayHandle,
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    config: &wgpu::SurfaceConfiguration,
) -> (EguiContext, EguiWinitState, EguiRenderer, ScreenDescriptor) {
    let egui_ctx = EguiContext::default();
    let egui_state = EguiWinitState::new(
        egui_ctx.clone(),
        ViewportId::ROOT,
        display_target,
        Some(window.scale_factor() as f32),
        None,
        None,
    );
    let egui_renderer = EguiRenderer::new(device, surface_format, Default::default());
    let egui_screen = ScreenDescriptor {
        size_in_pixels: [config.width, config.height],
        pixels_per_point: window.scale_factor() as f32,
    };
    (egui_ctx, egui_state, egui_renderer, egui_screen)
}

/// Spawn the background inference thread and return its channels + initial metadata.
#[cfg(onnx)]
fn spawn_inference_thread() -> (
    Option<std::sync::mpsc::SyncSender<InferRequest>>,
    Option<std::sync::mpsc::Receiver<InferResult>>,
    Option<String>,
    Option<String>,
) {
    let path = crate::config::onnx_model_override()
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
        let Some(mut detector) = detector else {
            return;
        };

        while let Ok(req) = req_rx.recv() {
            let infer_start = Instant::now();
            let result = match detector.infer_persons_rgba(
                &req.rgba,
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

            resource_monitor::record_inference_duration(infer_start.elapsed());
            resource_monitor::record_inference_completion();

            let _ = res_tx.try_send(result);
        }
    });

    (Some(req_tx), Some(res_rx), None, Some(path))
}

impl WgpuState {
    pub async fn new(
        window: Arc<Window>,
        display_target: &dyn wgpu::rwh::HasDisplayHandle,
        backends: wgpu::Backends,
    ) -> anyhow::Result<Self> {
        let (surface, device, queue, config) = init_wgpu(&window, backends).await?;

        let (vertex_buffer, index_buffer, num_indices) = create_quad_buffers(&device);

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

        let render_pipeline = create_render_pipeline(&device, config.format);

        let (egui_ctx, egui_state, egui_renderer, egui_screen) =
            init_egui(&window, display_target, &device, config.format, &config);

        #[cfg(onnx)]
        let (infer_tx, infer_rx, detector_error, model_path) = spawn_inference_thread();

        #[cfg(onnx)]
        let score_threshold = crate::config::score_threshold();

        #[cfg(onnx)]
        let infer_every_ms = crate::config::infer_every_ms();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            sampler,
            texture: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            egui_screen,
            overlay_enabled: true,

            overlay: OverlayStats::new(),

            #[cfg(onnx)]
            inference: InferenceState {
                detector_error,
                model_path,
                score_threshold,
                infer_every: Duration::from_millis(infer_every_ms),
                last_infer: Instant::now() - Duration::from_secs(1),
                last_detections: Vec::new(),
                last_detections_frame_id: None,
                infer_tx,
                infer_rx,
                infer_in_flight: false,
                infer_enabled: true,
            },

            current_frame_id: None,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.egui_screen.size_in_pixels = [new_size.width, new_size.height];
        }
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) {
        let response = self.egui_state.on_window_event(window, event);
        if response.repaint {
            window.request_redraw();
        }
    }

    pub fn upload_frame(&mut self, frame: &Frame) {
        self.current_frame_id = Some(frame.id);
        self.poll_inference();
        let needs_recreate = match &self.texture {
            Some(ft) => frame.width != ft.width || frame.height != ft.height,
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
            self.texture = Some(FrameTexture {
                texture,
                _view: view,
                bind_group,
                width: frame.width,
                height: frame.height,
            });
        }

        if let Some(ft) = &self.texture {
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
                    texture: &ft.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &frame.data,
                layout,
                size,
            );
        }

        #[cfg(onnx)]
        self.inference.maybe_infer(frame);
    }

    pub fn poll_inference(&mut self) {
        #[cfg(onnx)]
        self.inference.poll();
    }

    pub fn update_overlay_stats(&mut self) {
        self.overlay.update();
    }

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        self.poll_inference();

        self.egui_screen.pixels_per_point = window.scale_factor() as f32;
        self.egui_screen.size_in_pixels = [self.config.width, self.config.height];

        let raw_input = self.egui_state.take_egui_input(window);
        self.update_overlay_stats();

        let full_output = self.run_egui_overlay(raw_input);

        let clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, self.egui_screen.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.submit_frame(window, &clipped_primitives, &full_output)?;

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        Ok(())
    }

    /// Build and run the egui overlay UI, returning the full output for tessellation.
    fn run_egui_overlay(&mut self, raw_input: egui::RawInput) -> egui::FullOutput {
        let mut overlay_enabled = self.overlay_enabled;
        #[cfg(onnx)]
        let mut infer_enabled = self.inference.infer_enabled;
        let frame_dimensions = self
            .texture
            .as_ref()
            .map(|ft| (ft.width, ft.height))
            .unwrap_or((0, 0));

        #[cfg(onnx)]
        let detections_count = if self.inference.last_detections_frame_id.is_some() {
            self.inference.last_detections.len()
        } else {
            0
        };
        #[cfg(onnx)]
        let detector_error = self.inference.detector_error.clone();
        #[cfg(onnx)]
        let model_path = self.inference.model_path.clone();
        #[cfg(onnx)]
        let score_threshold = self.inference.score_threshold;

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
                        ui.label(format!("Camera FPS: {:.2}", self.overlay.cam_fps));
                        ui.label(format!("Infer FPS:  {:.2}", self.overlay.infer_fps));
                        ui.label(format!("Infer ms:   {:.2}", self.overlay.infer_latency_ms));
                        ui.label(format!(
                            "Infer cap: {:.2} fps",
                            self.overlay.infer_capacity_fps
                        ));
                        ui.label(format!(
                            "CPU: {:.1}%   RSS: {:.1} MiB",
                            self.overlay.proc_cpu_pct, self.overlay.proc_rss_mib
                        ));
                        draw_cpu_history(ui, &self.overlay.cpu_history);

                        #[cfg(onnx)]
                        {
                            Self::draw_detection_boxes(
                                ctx,
                                frame_dimensions,
                                &self.inference.last_detections,
                            );

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
        #[cfg(onnx)]
        {
            self.inference.infer_enabled = infer_enabled;
        }

        full_output
    }

    /// Draw detection bounding boxes on the egui foreground layer.
    #[cfg(onnx)]
    fn draw_detection_boxes(
        ctx: &EguiContext,
        frame_dimensions: (u32, u32),
        detections: &[crate::detection::Detection],
    ) {
        if frame_dimensions.0 == 0 || frame_dimensions.1 == 0 || detections.is_empty() {
            return;
        }

        let screen = ctx.content_rect();
        let sx = screen.width() / frame_dimensions.0 as f32;
        let sy = screen.height() / frame_dimensions.1 as f32;
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("detections"),
        ));

        for d in detections {
            let x1 = screen.left() + d.x1 * sx;
            let y1 = screen.top() + d.y1 * sy;
            let x2 = screen.left() + d.x2 * sx;
            let y2 = screen.top() + d.y2 * sy;

            let rect = egui::Rect::from_min_max(egui::pos2(x1, y1), egui::pos2(x2, y2));
            painter.rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(2.0, egui::Color32::LIGHT_GREEN),
                egui::StrokeKind::Inside,
            );
        }
    }

    /// Encode and submit the GPU commands for one frame (video quad + egui).
    fn submit_frame(
        &mut self,
        window: &Window,
        clipped_primitives: &[egui::ClippedPrimitive],
        full_output: &egui::FullOutput,
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // Video quad pass.
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

            if let Some(ft) = &self.texture {
                rpass.set_bind_group(0, &ft.bind_group, &[]);
            }

            rpass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // Egui pass.
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            clipped_primitives,
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

            let mut egui_rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui_render_pass"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                })
                // Erase the borrow lifetime so we can pass the render pass to
                // `egui_renderer.render()` which requires `'static`.  The pass
                // is dropped before the encoder is submitted, so this is safe.
                .forget_lifetime();

            self.egui_renderer
                .render(&mut egui_rpass, clipped_primitives, &self.egui_screen);
        }

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
