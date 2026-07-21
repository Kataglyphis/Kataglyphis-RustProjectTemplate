//! egui overlay drawn on top of the tonemapped output (stats + light and
//! tonemap controls). Works native and on wasm32.

use winit::window::Window;

pub struct Overlay {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Overlay {
    pub fn new(window: &Window, device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(device, output_format, Default::default());
        Self {
            ctx,
            state,
            renderer,
        }
    }

    /// Feeds a window event; returns true when egui consumed it (pointer over
    /// a panel etc.) so camera controls should skip it.
    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Runs the UI closure and paints on `view` (expects the scene already
    /// rendered there; loads, never clears).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        run_ui: impl FnMut(&egui::Context),
    ) {
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let raw_input = self.state.take_egui_input(window);
        let output = self.ctx.run(raw_input, run_ui);
        self.state
            .handle_platform_output(window, output.platform_output);

        let clipped = self.ctx.tessellate(output.shapes, screen.pixels_per_point);

        for (id, image_delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &clipped, &screen);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui_overlay_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &clipped, &screen);
        }

        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

/// State backing the standard demo control panel.
pub struct OverlayControls {
    pub light_azimuth_deg: f32,
    pub light_elevation_deg: f32,
    pub intensity: f32,
    pub ambient: f32,
    pub bloom: f32,
    pub ssao: f32,
    pub exposure_ev: f32,
    pub occlusion_culling: bool,
    /// Last frame's (drawn, considered) opaque-primitive counts, shown next to
    /// the occlusion toggle so the cull is observable. Set by the caller each
    /// frame; `None` hides the readout.
    pub occlusion_stats: Option<(u32, u32)>,
}

impl OverlayControls {
    pub fn from_light(light_dir_ambient: glam::Vec4, light_color_intensity: glam::Vec4) -> Self {
        let dir = light_dir_ambient.truncate();
        Self {
            light_azimuth_deg: dir.z.atan2(dir.x).to_degrees(),
            light_elevation_deg: dir
                .y
                .atan2((dir.x * dir.x + dir.z * dir.z).sqrt())
                .to_degrees(),
            intensity: light_color_intensity.w,
            ambient: light_dir_ambient.w,
            bloom: 0.6,
            ssao: 0.7,
            exposure_ev: 0.0,
            occlusion_culling: false,
            occlusion_stats: None,
        }
    }

    /// Direction TOWARDS the light from azimuth/elevation.
    pub fn light_direction(&self) -> glam::Vec3 {
        let az = self.light_azimuth_deg.to_radians();
        let el = self.light_elevation_deg.to_radians();
        glam::Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
    }

    /// Draws the panel and returns true when a value changed.
    pub fn ui(&mut self, ctx: &egui::Context) -> bool {
        let mut changed = false;
        egui::Window::new("Renderer")
            .default_open(true)
            .resizable(false)
            .show(ctx, |ui| {
                let dt = ctx.input(|i| i.stable_dt).max(1e-6);
                ui.label(format!("{:.1} ms/frame ({:.0} FPS)", dt * 1000.0, 1.0 / dt));
                ui.separator();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.light_azimuth_deg, -180.0..=180.0)
                            .text("light azimuth"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.light_elevation_deg, 5.0..=89.0)
                            .text("light elevation"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.intensity, 0.0..=20.0).text("intensity"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.ambient, 0.0..=2.0).text("IBL strength"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.bloom, 0.0..=2.0).text("bloom"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.ssao, 0.0..=1.0).text("SSAO"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.exposure_ev, -4.0..=4.0).text("exposure EV"))
                    .changed();
                changed |= ui
                    .checkbox(&mut self.occlusion_culling, "occlusion culling")
                    .changed();
                if self.occlusion_culling {
                    if let Some((drawn, considered)) = self.occlusion_stats {
                        let culled = considered.saturating_sub(drawn);
                        ui.label(format!(
                            "  drew {drawn}/{considered} ({culled} culled)"
                        ));
                    }
                }
                ui.separator();
                ui.label("drag: orbit - wheel: zoom");
            });
        changed
    }
}
