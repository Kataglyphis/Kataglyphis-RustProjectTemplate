#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use std::time::Duration;

use egui::Context as EguiContext;
use egui_wgpu::ScreenDescriptor;
use winit::window::Window;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use kataglyphis_core::detection::Detection;

#[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
use super::inference::InferenceState;

use super::overlay::draw_cpu_history;
use super::overlay_stats::OverlayStats;
use super::wgpu_init::init_egui;

pub(crate) struct EguiOverlay {
    pub(crate) ctx: EguiContext,
    pub(crate) state: egui_winit::State,
    pub(crate) renderer: egui_wgpu::Renderer,
    pub(crate) screen: ScreenDescriptor,
    pub(crate) enabled: bool,
    pub(crate) stats: OverlayStats,
}

impl EguiOverlay {
    pub(crate) fn new(
        window: &Window,
        display_target: &dyn wgpu::rwh::HasDisplayHandle,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let (ctx, state, renderer, screen) =
            init_egui(window, display_target, device, surface_format, config);

        Self {
            ctx,
            state,
            renderer,
            screen,
            enabled: true,
            stats: OverlayStats::new(),
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.screen.size_in_pixels = [width, height];
    }

    pub(crate) fn handle_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.state.on_window_event(window, event);
        response.repaint
    }

    pub(crate) fn update_stats(&mut self) {
        self.stats.update();
    }

    pub(crate) fn run_overlay(
        &mut self,
        window: &Window,
        frame_dimensions: (u32, u32),
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))] inference: &mut InferenceState,
    ) -> egui::FullOutput {
        self.screen.pixels_per_point = window.scale_factor() as f32;
        let raw_input = self.state.take_egui_input(window);

        let mut overlay_enabled = self.enabled;
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let mut infer_enabled = inference.infer_enabled;

        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let detections_count = if inference.last_detections_frame_id.is_some() {
            inference.last_detections.len()
        } else {
            0
        };
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let detector_error = inference.detector_error.as_deref();
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let model_path = inference.model_path.as_deref();
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        let score_threshold = inference.score_threshold;

        let full_output = self.ctx.run(raw_input, |ctx| {
            if overlay_enabled {
                egui::Window::new("Overlay")
                    .default_pos(egui::pos2(12.0, 12.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        Self::draw_system_stats(ui, frame_dimensions, &self.stats);

                        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
                        {
                            Self::draw_detection_boxes(
                                ctx,
                                frame_dimensions,
                                &inference.last_detections,
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

        self.enabled = overlay_enabled;
        #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
        {
            inference.infer_enabled = infer_enabled;
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

    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
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

    #[cfg(any(feature = "onnx_tract", feature = "onnxruntime"))]
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
}
