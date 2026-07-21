//! Instance/adapter/device/queue management plus surface configuration.
//!
//! Surface lifecycle rules (the part most renderers get wrong):
//! - `SurfaceError::Outdated`/`Lost` never abort a frame loop — reconfigure
//!   the surface with the current size and try again next frame.
//! - A suboptimal-but-successful acquire still produces a frame; the surface
//!   is reconfigured *after* presenting, never mid-frame.

use std::sync::Arc;

use anyhow::Context as _;
use winit::window::Window;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// TEXTURE_COMPRESSION_BC was available and enabled on this device.
    pub supports_bc: bool,
    /// TIMESTAMP_QUERY was available and enabled on this device. Browsers gate
    /// it behind a flag and most do not expose it, so per-pass GPU timings are
    /// a desktop-only diagnostic in practice.
    pub supports_timestamps: bool,
    /// Present target; `None` for headless (render-to-texture) contexts.
    pub surface: Option<wgpu::Surface<'static>>,
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
}

impl GpuContext {
    /// Creates a context that presents to `window` (native: blocks).
    pub fn new_windowed(window: Arc<Window>) -> anyhow::Result<Self> {
        pollster::block_on(Self::new_windowed_async(window))
    }

    /// Async variant for platforms without block_on (wasm32/browsers).
    pub async fn new_windowed_async(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("Failed to create surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("No suitable GPU adapter found")?;

        let (device, queue, supports_bc, supports_timestamps) = request_device(&adapter).await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .context("Surface reports no supported texture formats")?;
        let alpha_mode = caps
            .alpha_modes
            .first()
            .copied()
            .context("Surface reports no supported alpha modes")?;

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Ok(Self {
            device,
            queue,
            supports_bc,
            supports_timestamps,
            surface: Some(surface),
            surface_config: Some(surface_config),
        })
    }

    /// Creates a surfaceless context for render-to-texture (tests, CI).
    pub fn new_headless() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("No GPU adapter found (headless)")?;

        // Refuse the OpenGL backend for headless work. This renderer's shaders
        // target WebGPU/Vulkan-class backends - the SSAO pass does a
        // `textureLoad` from a depth texture, which naga cannot translate to
        // GLSL ("not supported in GLSL"), so `create_render_pipeline` aborts
        // before a single test runs. CI's image ships only llvmpipe (GL
        // software) with no lavapipe (Vulkan software), so wgpu falls back to GL
        // and every GPU test dies at pipeline creation. Treating GL as "no
        // usable adapter" lets those tests skip through their
        // `let Ok(gpu) = new_headless() else { skip }` guards, exactly as when
        // no adapter exists at all. A real Vulkan/DX/Metal adapter - including
        // lavapipe if it is ever added to the image - still runs them.
        let info = adapter.get_info();
        if info.backend == wgpu::Backend::Gl {
            anyhow::bail!(
                "headless adapter '{}' is the OpenGL backend, which cannot run this \
                 renderer's WGSL (depth textureLoad has no GLSL translation); \
                 treating as no usable adapter",
                info.name
            );
        }

        let (device, queue, supports_bc, supports_timestamps) = request_device(&adapter).await?;
        Ok(Self {
            device,
            queue,
            supports_bc,
            supports_timestamps,
            surface: None,
            surface_config: None,
        })
    }

    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface_config.as_ref().map(|c| c.format)
    }

    /// Reconfigures the surface for a new size. Zero dimensions (minimized
    /// windows) are ignored — the surface keeps its last valid size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if let (Some(surface), Some(config)) = (self.surface.as_ref(), self.surface_config.as_mut())
        {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
    }

    /// Reconfigures the surface with its current size (Outdated/Lost recovery).
    pub fn reconfigure(&self) {
        if let (Some(surface), Some(config)) = (self.surface.as_ref(), self.surface_config.as_ref())
        {
            surface.configure(&self.device, config);
        }
    }
}

async fn request_device(
    adapter: &wgpu::Adapter,
) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool, bool)> {
    // Optional features are requested only where the adapter reports them:
    // naming one the adapter lacks fails the whole device request, which would
    // take the renderer down on every browser rather than degrading.
    //
    // Block-compressed textures where the adapter offers them (desktop);
    // browsers/mobile usually do not, and the loader falls back.
    let supports_bc = adapter
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut required_features = wgpu::Features::empty();
    if supports_bc {
        required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    if supports_timestamps {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("webgpu_renderer_device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
        })
        .await
        .map(|(device, queue)| (device, queue, supports_bc, supports_timestamps))
        .context("Failed to create device")
}
