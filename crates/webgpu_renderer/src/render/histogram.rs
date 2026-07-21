//! GPU luminance histogram over the HDR target, for auto-exposure.
//!
//! The first compute pass in this renderer. Everything else here is a render
//! pass, so the plumbing (storage buffer, atomics, a readback path) is new -
//! which is exactly why it lands on its own, verified against the CPU binning
//! in [`crate::render::auto_exposure`], before anything depends on its output.

use crate::context::GpuContext;
use crate::render::auto_exposure::HISTOGRAM_BINS;
use crate::render::gpu_timing::PassScope;

/// Workgroup size of `cs_build_histogram`; must match the shader.
const BUILD_WORKGROUP: u32 = 16;
/// Workgroup size of `cs_clear_histogram`; must match the shader.
const CLEAR_WORKGROUP: u32 = 64;

/// Per-frame inputs to the reduction pass.
#[derive(Copy, Clone, Debug)]
pub struct ExposureSettings {
    pub delta_time_seconds: f32,
    /// Adaptation rate constant; 0 snaps straight to the target.
    pub speed: f32,
    pub auto_enabled: bool,
    /// Used when `auto_enabled` is false.
    pub manual_ev: f32,
}

impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            delta_time_seconds: 1.0 / 60.0,
            speed: 3.0,
            auto_enabled: true,
            manual_ev: 0.0,
        }
    }
}

pub struct HistogramPass {
    build_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    histogram_buffer: wgpu::Buffer,
    /// [adapted EV, target EV]. Lives on the GPU for the tonemap to read;
    /// reading it back per frame would serialise the pipeline.
    exposure_buffer: wgpu::Buffer,
    exposure_params_buffer: wgpu::Buffer,
    exposure_readback_buffer: wgpu::Buffer,
    /// MAP_READ staging target. Kept alive rather than allocated per read so a
    /// diagnostic readback does not churn allocations.
    readback_buffer: wgpu::Buffer,
}

impl HistogramPass {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histogram_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/histogram.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // textureLoad, not textureSample: no filtering, and
                        // sampling an HDR target with a filtering sampler is
                        // not guaranteed on every backend.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histogram_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_build_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_build_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_reduce_exposure"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_clear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_clear_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        let byte_size = (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let exposure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure_state"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let exposure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let exposure_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure_readback"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            build_pipeline,
            clear_pipeline,
            reduce_pipeline,
            exposure_buffer,
            exposure_params_buffer,
            exposure_readback_buffer,
            bind_group_layout,
            bind_group: None,
            histogram_buffer,
            readback_buffer,
        }
    }

    /// (Re)binds the HDR source. Call whenever the HDR target is recreated,
    /// e.g. on resize - a stale view here reads a destroyed texture.
    pub fn set_input(&mut self, gpu: &GpuContext, hdr_view: &wgpu::TextureView) {
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.exposure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exposure_params_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Clears and rebuilds the histogram for a `width` x `height` HDR target.
    ///
    /// Both passes go into one encoder in order; wgpu inserts the barrier
    /// between them, so the build cannot observe a partially cleared buffer.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        scope: PassScope<'_>,
    ) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_clear"),
                timestamp_writes: scope.compute_writes(0, 2),
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                HISTOGRAM_BINS.div_ceil(CLEAR_WORKGROUP as usize) as u32,
                1,
                1,
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_build"),
                timestamp_writes: scope.compute_writes(1, 2),
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            // Round UP: a truncating division leaves the right and bottom
            // edges of the image unsampled, which biases exposure toward
            // whatever is in the middle of the frame.
            pass.dispatch_workgroups(
                width.div_ceil(BUILD_WORKGROUP),
                height.div_ceil(BUILD_WORKGROUP),
                1,
            );
        }
    }

    /// Uploads this frame's adaptation inputs. Call before [`Self::encode_reduce`].
    pub fn set_exposure_settings(&self, queue: &wgpu::Queue, settings: ExposureSettings) {
        queue.write_buffer(
            &self.exposure_params_buffer,
            0,
            bytemuck::bytes_of(&[
                settings.delta_time_seconds,
                settings.speed,
                if settings.auto_enabled {
                    1.0f32
                } else {
                    0.0f32
                },
                settings.manual_ev,
            ]),
        );
    }

    /// Reduces the histogram to an adapted exposure, in the same encoder and
    /// after [`Self::encode`] so the barrier between them is wgpu's problem.
    pub fn encode_reduce(&self, encoder: &mut wgpu::CommandEncoder, scope: PassScope<'_>) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exposure_reduce"),
            timestamp_writes: scope.compute_writes(0, 1),
        });
        pass.set_pipeline(&self.reduce_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// The buffer the tonemap should read the adapted exposure from.
    pub fn exposure_buffer(&self) -> &wgpu::Buffer {
        &self.exposure_buffer
    }

    /// Resets the adaptation state. Without this a test (or a scene change)
    /// inherits whatever exposure the previous frames converged to, which
    /// makes "did it adapt?" unanswerable.
    pub fn reset_exposure(&self, queue: &wgpu::Queue, ev: f32) {
        queue.write_buffer(&self.exposure_buffer, 0, bytemuck::bytes_of(&[ev, ev]));
    }

    /// Copies the exposure state out for tests and diagnostics. Same caveat as
    /// [`Self::read_back`]: it stalls the queue and must not be on the frame
    /// path.
    pub fn encode_exposure_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.exposure_buffer,
            0,
            &self.exposure_readback_buffer,
            0,
            8,
        );
    }

    /// Returns (adapted EV, target EV) from the last copied exposure state.
    pub fn read_back_exposure(&self, gpu: &GpuContext) -> (f32, f32) {
        let slice = self.exposure_readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let _ = receiver.recv();

        let values = {
            let data = slice.get_mapped_range();
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (floats[0], floats[1])
        };
        self.exposure_readback_buffer.unmap();
        values
    }

    /// Copies the histogram into the mappable staging buffer. Must be encoded
    /// after [`Self::encode`] and submitted before [`Self::read_back`].
    pub fn encode_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.histogram_buffer,
            0,
            &self.readback_buffer,
            0,
            self.histogram_buffer.size(),
        );
    }

    /// Blocking readback of the last copied histogram.
    ///
    /// For tests and diagnostics only: it stalls the queue. The real
    /// auto-exposure path reduces the histogram on the GPU and never reads it
    /// back - a per-frame CPU readback would serialise the pipeline.
    pub fn read_back(&self, gpu: &GpuContext) -> Vec<u32> {
        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let _ = receiver.recv();

        let counts = {
            let data = slice.get_mapped_range();
            data.chunks_exact(4)
                .map(|bytes| u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect::<Vec<u32>>()
        };
        self.readback_buffer.unmap();
        counts
    }
}
