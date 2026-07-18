//! Minimal glTF viewer: `cargo run -p kataglyphis_webgpu_renderer --example viewer [model.gltf]`
//! Auto-orbits the camera; Esc closes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use kataglyphis_webgpu_renderer::{load_gltf, ForwardRenderer, GpuContext, OrbitCamera};

struct Viewer {
    model_path: PathBuf,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<ForwardRenderer>,
    camera: OrbitCamera,
    started: Instant,
}

impl Viewer {
    fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            window: None,
            gpu: None,
            renderer: None,
            camera: OrbitCamera::default(),
            started: Instant::now(),
        }
    }

    fn redraw(&mut self) {
        let (Some(window), Some(gpu), Some(renderer)) = (
            self.window.as_ref(),
            self.gpu.as_mut(),
            self.renderer.as_mut(),
        ) else {
            return;
        };
        let Some(surface) = gpu.surface.as_ref() else {
            return;
        };

        self.camera.yaw_deg = 45.0 + self.started.elapsed().as_secs_f32() * 30.0;

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
        renderer.render(gpu, &view, width, height, &self.camera);
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
                        .with_title("Kataglyphis WebGPU glTF viewer")
                        .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
                )
                .expect("failed to create window"),
        );

        let gpu = GpuContext::new_windowed(Arc::clone(&window)).expect("failed to init wgpu");
        let format = gpu.surface_format().expect("windowed context has a format");
        let size = window.inner_size();
        let mut renderer = ForwardRenderer::new(&gpu, format, size.width.max(1), size.height.max(1));

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
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit()
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let model_path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf")
    });

    let event_loop = EventLoop::new()?;
    let mut viewer = Viewer::new(model_path);
    event_loop.run_app(&mut viewer)?;
    Ok(())
}
