//! Minimal glTF viewer: `cargo run -p kataglyphis_webgpu_renderer --example viewer [model.gltf]`
//! Drag to orbit, wheel to zoom (auto-orbit until first drag), egui overlay
//! with light/tonemap controls; Esc closes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use kataglyphis_webgpu_renderer::{
    load_gltf, ForwardRenderer, GpuContext, OrbitCamera, OrbitController, Overlay, OverlayControls,
    TonemapPass,
};

struct Viewer {
    model_path: PathBuf,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<ForwardRenderer>,
    tonemap: Option<TonemapPass>,
    overlay: Option<Overlay>,
    controls: Option<OverlayControls>,
    controller: OrbitController,
    camera: OrbitCamera,
    started: Instant,
    frame: u64,
    shader_mtimes: Vec<(PathBuf, std::time::SystemTime)>,
}

fn shader_paths() -> Vec<PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    vec![dir.join("forward.wgsl"), dir.join("sky.wgsl")]
}

fn shader_mtimes() -> Vec<(PathBuf, std::time::SystemTime)> {
    shader_paths()
        .into_iter()
        .filter_map(|p| {
            let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?;
            Some((p, mtime))
        })
        .collect()
}

impl Viewer {
    fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            window: None,
            gpu: None,
            renderer: None,
            tonemap: None,
            overlay: None,
            controls: None,
            controller: OrbitController::default(),
            camera: OrbitCamera::default(),
            started: Instant::now(),
            frame: 0,
            shader_mtimes: shader_mtimes(),
        }
    }

    /// Loads a different model at runtime (drag & drop).
    fn load_model(&mut self, path: PathBuf) {
        let (Some(gpu), Some(renderer)) = (self.gpu.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        match load_gltf(&path) {
            Ok(scene) => {
                log::info!(
                    "Loaded {}: {} primitives, {} triangles",
                    path.display(),
                    scene.primitives.len(),
                    scene.triangle_count()
                );
                renderer.upload_scene(gpu, &scene);
                self.model_path = path;
                // Frame the new model: fit the orbit radius to its bounds.
                self.camera = OrbitCamera::default();
                self.controller = OrbitController::default();
                self.started = Instant::now();
            }
            Err(err) => log::error!("Failed to load {}: {err:#}", path.display()),
        }
    }

    /// Hot shader reload: re-reads src/shaders/*.wgsl and rebuilds the
    /// pipelines; invalid WGSL keeps the previous pipelines running.
    fn reload_shaders(&mut self) {
        let (Some(gpu), Some(renderer)) = (self.gpu.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        let paths = shader_paths();
        let (Ok(forward_src), Ok(sky_src)) = (
            std::fs::read_to_string(&paths[0]),
            std::fs::read_to_string(&paths[1]),
        ) else {
            log::error!("Failed to read shader sources for reload");
            return;
        };
        match renderer.reload_shaders(gpu, &forward_src, &sky_src) {
            Ok(()) => log::info!("Shaders reloaded"),
            Err(err) => log::error!("{err:#}"),
        }
        self.shader_mtimes = shader_mtimes();
    }

    /// Polls shader file mtimes (~every 30 frames) and reloads on change.
    fn poll_shader_changes(&mut self) {
        self.frame += 1;
        if !self.frame.is_multiple_of(30) {
            return;
        }
        let current = shader_mtimes();
        if current != self.shader_mtimes {
            log::info!("Shader change detected - reloading");
            self.reload_shaders();
        }
    }

    /// S key: renders one frame offscreen and writes screenshots/NNN.png.
    fn save_screenshot(&mut self) {
        let (Some(gpu), Some(renderer)) = (self.gpu.as_mut(), self.renderer.as_mut()) else {
            return;
        };
        let (width, height) = (1920u32, 1080u32);
        match renderer.render_to_pixels(gpu, width, height, &self.camera) {
            Ok(pixels) => {
                let dir = std::path::Path::new("screenshots");
                let _ = std::fs::create_dir_all(dir);
                let mut index = 0;
                let path = loop {
                    let candidate = dir.join(format!("screenshot-{index:03}.png"));
                    if !candidate.exists() {
                        break candidate;
                    }
                    index += 1;
                };
                match image::save_buffer(&path, &pixels, width, height, image::ColorType::Rgba8) {
                    Ok(()) => log::info!("Saved {}", path.display()),
                    Err(err) => log::error!("Failed to save screenshot: {err}"),
                }
            }
            Err(err) => log::error!("Screenshot render failed: {err:#}"),
        }
    }

    fn redraw(&mut self) {
        self.poll_shader_changes();
        let (Some(window), Some(gpu), Some(renderer), Some(tonemap), Some(overlay), Some(controls)) = (
            self.window.as_ref(),
            self.gpu.as_mut(),
            self.renderer.as_mut(),
            self.tonemap.as_mut(),
            self.overlay.as_mut(),
            self.controls.as_mut(),
        ) else {
            return;
        };
        let Some(surface) = gpu.surface.as_ref() else {
            return;
        };

        if self.controller.auto_orbit {
            self.camera.yaw_deg = 45.0 + self.started.elapsed().as_secs_f32() * 30.0;
        }
        if renderer.has_animations() {
            renderer.set_animation_time(self.started.elapsed().as_secs_f32());
        }

        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                // Never abort on Outdated/Lost: reconfigure and retry next frame.
                gpu.reconfigure();
                window.request_redraw();
                return;
            }
            Err(err) => {
                log::error!("Surface error: {err:?}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Size from the ACQUIRED texture, not the window: during a resize the
        // window's client size can already differ from the still-configured
        // surface, and depth/color attachments must match exactly.
        let (width, height) = (frame.texture.width(), frame.texture.height());
        renderer.render_tonemapped(gpu, tonemap, &view, width, height, &self.camera);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlay_encoder"),
            });
        let mut changed = false;
        overlay.render(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            window,
            &view,
            width,
            height,
            |ctx| {
                changed |= controls.ui(ctx);
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        if changed {
            let dir = controls.light_direction();
            renderer.light_dir_ambient = glam::Vec4::new(dir.x, dir.y, dir.z, controls.ambient);
            renderer.light_color_intensity.w = controls.intensity;
            renderer.bloom_strength = controls.bloom;
            renderer.ssao_strength = controls.ssao;
            renderer.exposure_ev = controls.exposure_ev;
        }

        window.pre_present_notify();
        frame.present();
        window.request_redraw();
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Kataglyphis WebGPU glTF viewer (drop a .gltf/.glb to load)")
                        .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
                )
                .expect("failed to create window"),
        );

        let gpu = GpuContext::new_windowed(Arc::clone(&window)).expect("failed to init wgpu");
        let format = gpu.surface_format().expect("windowed context has a format");
        let size = window.inner_size();
        let mut renderer = ForwardRenderer::new(&gpu, size.width.max(1), size.height.max(1));
        let tonemap = TonemapPass::new(&gpu, format);
        let overlay = Overlay::new(&window, &gpu.device, format);
        let controls =
            OverlayControls::from_light(renderer.light_dir_ambient, renderer.light_color_intensity);

        let scene = load_gltf(&self.model_path)
            .unwrap_or_else(|err| panic!("failed to load {}: {err:#}", self.model_path.display()));
        log::info!(
            "Loaded {}: {} primitives, {} triangles",
            self.model_path.display(),
            scene.primitives.len(),
            scene.triangle_count()
        );
        renderer.upload_scene(&gpu, &scene);

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.tonemap = Some(tonemap);
        self.overlay = Some(overlay);
        self.controls = Some(controls);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let consumed = match (self.overlay.as_mut(), self.window.as_ref()) {
            (Some(overlay), Some(window)) => overlay.on_event(window, &event),
            _ => false,
        };
        if !consumed {
            self.controller.handle_event(&event, &mut self.camera);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit()
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed() && event.logical_key == Key::Character("s".into()) =>
            {
                self.save_screenshot();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed() && event.logical_key == Key::Character("r".into()) =>
            {
                self.reload_shaders();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::DroppedFile(path) => self.load_model(path),
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_on_plane.gltf")
        });

    let event_loop = EventLoop::new()?;
    let mut viewer = Viewer::new(model_path);
    event_loop.run_app(&mut viewer)?;
    Ok(())
}
