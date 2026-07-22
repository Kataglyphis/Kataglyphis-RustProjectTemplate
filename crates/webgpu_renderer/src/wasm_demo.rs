//! Browser (wasm32/WebGPU) demo entry point: renders the embedded
//! cube-on-plane shadow scene into a canvas appended to the document body.
//! Built with wasm-bindgen; see crates/webgpu_renderer/web/index.html.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowExtWebSys};
use winit::window::{Window, WindowId};

use crate::asset::gltf_loader::load_gltf_slice;
use crate::context::GpuContext;
use crate::render::forward::ForwardRenderer;
use crate::render::overlay::{Overlay, OverlayControls};
use crate::render::tonemap::TonemapPass;
use crate::scene::camera::OrbitCamera;
use crate::scene::controller::OrbitController;

const DEMO_SCENE: &[u8] = include_bytes!("../tests/assets/cube_on_plane.gltf");

/// A model dropped onto the page, parked here until the render loop picks it
/// up: the File read is async, and the GPU state may not even exist yet when
/// the drop happens.
type DroppedScene = Rc<RefCell<Option<(String, Vec<u8>)>>>;

/// Browser drag-and-drop model loading (the winit web backend never delivers
/// `WindowEvent::DroppedFile`, so this goes through the DOM File API).
/// Listeners go on the document so the whole page is the drop target.
fn install_drop_zone(document: &web_sys::Document, slot: DroppedScene) {
    // dragover must be cancelled, otherwise the browser handles the drop
    // itself and navigates away to the file.
    let on_dragover = Closure::<dyn FnMut(web_sys::DragEvent)>::new(|event: web_sys::DragEvent| {
        event.prevent_default();
    });
    let _ = document
        .add_event_listener_with_callback("dragover", on_dragover.as_ref().unchecked_ref());
    on_dragover.forget();

    let on_drop = Closure::<dyn FnMut(web_sys::DragEvent)>::new(move |event: web_sys::DragEvent| {
        event.prevent_default();
        let Some(file) = event
            .data_transfer()
            .and_then(|transfer| transfer.files())
            .and_then(|files| files.get(0))
        else {
            return;
        };
        let name = file.name();
        let slot = Rc::clone(&slot);
        wasm_bindgen_futures::spawn_local(async move {
            match wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await {
                Ok(buffer) => {
                    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                    slot.borrow_mut().replace((name, bytes));
                }
                Err(err) => log::error!("Reading the dropped file failed: {err:?}"),
            }
        });
    });
    let _ = document.add_event_listener_with_callback("drop", on_drop.as_ref().unchecked_ref());
    on_drop.forget();
}

/// Syncs the canvas backing store to its CSS layout size x devicePixelRatio
/// and returns the backing size. winit does not do this on web — an unsized
/// canvas leads to an invisible ~1x1 surface.
fn sync_canvas_backing_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
    let width = ((canvas.client_width().max(1) as f64) * dpr) as u32;
    let height = ((canvas.client_height().max(1) as f64) * dpr) as u32;
    if canvas.width() != width {
        canvas.set_width(width);
    }
    if canvas.height() != height {
        canvas.set_height(height);
    }
    (width.max(1), height.max(1))
}

struct GpuState {
    gpu: GpuContext,
    renderer: ForwardRenderer,
    tonemap: TonemapPass,
    overlay: Overlay,
    controls: OverlayControls,
}

#[derive(Default)]
struct DemoApp {
    window: Option<Arc<Window>>,
    /// Filled asynchronously once the WebGPU adapter/device resolve.
    state: Rc<RefCell<Option<GpuState>>>,
    /// Filled asynchronously by the drop-zone listeners.
    dropped_scene: DroppedScene,
    controller: OrbitController,
    camera: OrbitCamera,
    frame: u64,
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(winit::dpi::LogicalSize::new(1024, 640)),
                )
                .expect("failed to create window"),
        );

        // Attach the winit canvas to the page; CSS drives the layout size,
        // the backing store follows it (responsive, DPI-aware).
        let canvas = window.canvas().expect("winit window must expose a canvas");
        let style = canvas.style();
        let _ = style.set_property("width", "100%");
        let _ = style.set_property("height", "100%");
        let document = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document");
        let mount = document
            .get_element_by_id("demo")
            .unwrap_or_else(|| document.body().expect("no body").into());
        mount
            .append_child(&canvas)
            .expect("failed to append canvas");
        install_drop_zone(&document, Rc::clone(&self.dropped_scene));
        let (initial_width, initial_height) = sync_canvas_backing_size(&canvas);

        // WebGPU init is async-only in browsers: fill the shared state slot
        // when ready and kick the first redraw.
        let state_slot = Rc::clone(&self.state);
        let init_window = Arc::clone(&window);
        wasm_bindgen_futures::spawn_local(async move {
            let mut gpu = GpuContext::new_windowed_async(Arc::clone(&init_window))
                .await
                .expect("WebGPU init failed (does this browser support WebGPU?)");
            let format = gpu.surface_format().expect("windowed context has a format");
            // The context read inner_size before the canvas sizing propagated
            // through winit — force the surface to the real canvas size.
            gpu.resize(initial_width, initial_height);
            let mut renderer = ForwardRenderer::new(&gpu, initial_width, initial_height);
            let tonemap = TonemapPass::new(&gpu, format);

            let scene = load_gltf_slice(DEMO_SCENE).expect("embedded demo scene must load");
            renderer.upload_scene(&gpu, &scene);

            // Real split-sum IBL from the procedural sky, so the web showcase
            // shows environment-lit reflections rather than the analytic
            // fallback. Baked once from a small panorama - enough for the
            // low-frequency irradiance and prefilter.
            let sky_env = crate::render::ibl::EquirectImage::sky(256, 128);
            renderer.set_environment(&gpu, &sky_env);

            let overlay = Overlay::new(&init_window, &gpu.device, format);
            let controls = OverlayControls::from_light(
                renderer.light_dir_ambient,
                renderer.light_color_intensity,
            );

            state_slot.borrow_mut().replace(GpuState {
                gpu,
                renderer,
                tonemap,
                overlay,
                controls,
            });
            init_window.request_redraw();
        });

        self.window = Some(window);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let consumed = match (self.state.borrow_mut().as_mut(), self.window.as_ref()) {
            (Some(state), Some(window)) => state.overlay.on_event(window, &event),
            _ => false,
        };
        if !consumed {
            self.controller.handle_event(&event, &mut self.camera);
        }

        match event {
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.borrow_mut().as_mut() {
                    state.gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(window) = self.window.as_ref() else {
                    return;
                };
                let mut state_ref = self.state.borrow_mut();
                let Some(state) = state_ref.as_mut() else {
                    return;
                };

                // A dropped model swaps the scene with the same semantics as
                // the native viewer: upload, re-frame the camera, and on a
                // parse error keep the current scene running. Only .glb (or
                // .gltf with embedded buffers) can work from a single file -
                // external .bin/textures are unreachable from one File.
                if let Some((name, bytes)) = self.dropped_scene.borrow_mut().take() {
                    match load_gltf_slice(&bytes) {
                        Ok(scene) => {
                            log::info!(
                                "Loaded {name}: {} primitives, {} triangles",
                                scene.primitives.len(),
                                scene.triangle_count()
                            );
                            state.renderer.upload_scene(&state.gpu, &scene);
                            self.camera = OrbitCamera::default();
                            self.controller = OrbitController::default();
                            self.frame = 0;
                        }
                        Err(err) => log::error!(
                            "Failed to load {name}: {err:#} (external .bin/textures cannot \
                             be resolved from a drop - use a self-contained .glb)"
                        ),
                    }
                }

                // Responsive canvas: follow the CSS layout size every frame
                // and reconfigure the surface when it changes.
                if let Some(canvas) = window.canvas() {
                    let (width, height) = sync_canvas_backing_size(&canvas);
                    let configured = state
                        .gpu
                        .surface_config
                        .as_ref()
                        .map(|c| (c.width, c.height));
                    if configured != Some((width, height)) {
                        state.gpu.resize(width, height);
                    }
                }

                let Some(surface) = state.gpu.surface.as_ref() else {
                    return;
                };

                // std::time::Instant is unavailable on wasm32: animate by frame.
                self.frame += 1;
                if state.renderer.has_animations() {
                    state.renderer.set_animation_time(self.frame as f32 / 60.0);
                }
                if self.controller.auto_orbit {
                    self.camera.yaw_deg = 45.0 + self.frame as f32 * 0.5;
                    self.camera.radius = 6.0;
                    self.camera.pitch_deg = 35.0;
                }

                // wgpu 29: get_current_texture returns a CurrentSurfaceTexture
                // enum instead of Result<_, SurfaceError>. This mirrors the
                // native viewer exactly - it is the same acquire, and the two
                // drifting apart is what broke the web build unnoticed.
                let frame = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    wgpu::CurrentSurfaceTexture::Outdated
                    | wgpu::CurrentSurfaceTexture::Lost => {
                        // Never abort on Outdated/Lost: reconfigure and retry.
                        state.gpu.reconfigure();
                        window.request_redraw();
                        return;
                    }
                    other => {
                        log::error!("Surface acquire failed: {other:?}");
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let (width, height) = (frame.texture.width(), frame.texture.height());
                let GpuState {
                    gpu,
                    renderer,
                    tonemap,
                    overlay,
                    controls,
                } = state;
                renderer.render_tonemapped(gpu, tonemap, &view, width, height, &self.camera);

                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("overlay_encoder"),
                        });
                // Show this frame's cull counts next to the occlusion toggle,
                // same as the native viewer, so the web overlay reports them too.
                controls.occlusion_stats = Some(renderer.occlusion_cull_stats());
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
                    renderer.light_dir_ambient =
                        glam::Vec4::new(dir.x, dir.y, dir.z, controls.ambient);
                    renderer.light_color_intensity.w = controls.intensity;
                    renderer.bloom_strength = controls.bloom;
                    renderer.ssao_strength = controls.ssao;
                    renderer.exposure_ev = controls.exposure_ev;
                    renderer.occlusion_queries_enabled = controls.occlusion_culling;
                }

                window.pre_present_notify();
                frame.present();
                window.request_redraw();
            }
            _ => {}
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.spawn_app(DemoApp::default());
}
