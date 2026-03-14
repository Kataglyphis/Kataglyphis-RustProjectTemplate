use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use wgpu::util::DeviceExt;
use winit::window::Window;

use egui::Context as EguiContext;
use egui_wgpu::ScreenDescriptor;

#[cfg(onnx)]
use crate::detection::Detection;
use crate::resource_monitor;

#[cfg(onnx)]
use super::inference::InferenceState;
#[cfg(onnx)]
use super::inference_bridge::spawn_inference_thread;
use super::overlay::draw_cpu_history;
use super::overlay_stats::OverlayStats;
use super::pipeline::Frame;
use super::wgpu_init::{create_quad_buffers, create_render_pipeline, init_egui, init_wgpu};

struct FrameTexture {
    texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub(crate) struct WgpuState {
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
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    egui_screen: ScreenDescriptor,
    overlay_enabled: bool,
    overlay: OverlayStats,
    #[cfg(onnx)]
    inference: InferenceState,
    current_frame_id: Option<u64>,
}

impl WgpuState {
    pub(crate) async fn new(
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
                last_infer: std::time::Instant::now() - Duration::from_secs(1),
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

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.egui_screen.size_in_pixels = [new_size.width, new_size.height];
        }
    }

    pub(crate) fn handle_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) {
        let response = self.egui_state.on_window_event(window, event);
        if response.repaint {
            window.request_redraw();
        }
    }

    pub(crate) fn upload_frame(&mut self, frame: &Frame) {
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

    pub(crate) fn poll_inference(&mut self) {
        #[cfg(onnx)]
        self.inference.poll();
    }

    pub(crate) fn update_overlay_stats(&mut self) {
        self.overlay.update();
    }

    pub(crate) fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
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
        let detector_error = self.inference.detector_error.as_deref();
        #[cfg(onnx)]
        let model_path = self.inference.model_path.as_deref();
        #[cfg(onnx)]
        let score_threshold = self.inference.score_threshold;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if overlay_enabled {
                egui::Window::new("Overlay")
                    .default_pos(egui::pos2(12.0, 12.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        Self::draw_system_stats(ui, frame_dimensions, &self.overlay);

                        #[cfg(onnx)]
                        {
                            Self::draw_detection_boxes(
                                ctx,
                                frame_dimensions,
                                &self.inference.last_detections,
                            );
                            Self::draw_onnx_panel(
                                ui,
                                model_path,
                                score_threshold,
                                detections_count,
                                detector_error,
                                &mut infer_enabled,
                            );
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

    fn draw_system_stats(ui: &mut egui::Ui, frame_dimensions: (u32, u32), overlay: &OverlayStats) {
        ui.label("ONNX overlay");
        ui.label(format!(
            "Frame: {}x{}",
            frame_dimensions.0, frame_dimensions.1
        ));
        ui.separator();
        ui.label(format!("Camera FPS: {:.2}", overlay.cam_fps));
        ui.label(format!("Infer FPS:  {:.2}", overlay.infer_fps));
        ui.label(format!("Infer ms:   {:.2}", overlay.infer_latency_ms));
        ui.label(format!("Infer cap: {:.2} fps", overlay.infer_capacity_fps));
        ui.label(format!(
            "CPU: {:.1}%   RSS: {:.1} MiB",
            overlay.proc_cpu_pct, overlay.proc_rss_mib
        ));
        draw_cpu_history(ui, &overlay.cpu_history);
    }

    #[cfg(onnx)]
    fn draw_onnx_panel(
        ui: &mut egui::Ui,
        model_path: Option<&str>,
        score_threshold: f32,
        detections_count: usize,
        detector_error: Option<&str>,
        infer_enabled: &mut bool,
    ) {
        if let Some(path) = model_path {
            ui.label(format!("Model: {path}"));
        } else {
            ui.label("Model: (not set)");
        }
        ui.label(format!("Score threshold: {score_threshold:.2}"));
        ui.label(format!("Persons: {detections_count}"));
        ui.checkbox(infer_enabled, "Inference enabled");
        if let Some(err) = detector_error {
            ui.separator();
            ui.label(err);
        }
    }

    #[cfg(onnx)]
    fn draw_detection_boxes(
        ctx: &EguiContext,
        frame_dimensions: (u32, u32),
        detections: &[Detection],
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
                .forget_lifetime();

            self.egui_renderer
                .render(&mut egui_rpass, clipped_primitives, &self.egui_screen);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
