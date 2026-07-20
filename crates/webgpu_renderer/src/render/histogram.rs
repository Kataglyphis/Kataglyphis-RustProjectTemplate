//! GPU luminance histogram over the HDR target, for auto-exposure.
//!
//! The first compute pass in this renderer. Everything else here is a render
//! pass, so the plumbing (storage buffer, atomics, a readback path) is new -
//! which is exactly why it lands on its own, verified against the CPU binning
//! in [`crate::render::auto_exposure`], before anything depends on its output.

use crate::context::GpuContext;
use crate::render::auto_exposure::HISTOGRAM_BINS;

/// Workgroup size of `cs_build_histogram`; must match the shader.
const BUILD_WORKGROUP: u32 = 16;
/// Workgroup size of `cs_clear_histogram`; must match the shader.
const CLEAR_WORKGROUP: u32 = 64;

pub struct HistogramPass {
    build_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    histogram_buffer: wgpu::Buffer,
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histogram_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_build_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_build_histogram"),
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

        Self {
            build_pipeline,
            clear_pipeline,
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
            ],
        }));
    }

    /// Clears and rebuilds the histogram for a `width` x `height` HDR target.
    ///
    /// Both passes go into one encoder in order; wgpu inserts the barrier
    /// between them, so the build cannot observe a partially cleared buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let Some(bind_group) = self.bind_group.as_ref() else {
            return;
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(HISTOGRAM_BINS.div_ceil(CLEAR_WORKGROUP as usize) as u32, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram_build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            // Round UP: a truncating division leaves the right and bottom
            // edges of the image unsampled, which biases exposure toward
            // whatever is in the middle of the frame.
            pass.dispatch_workgroups(width.div_ceil(BUILD_WORKGROUP), height.div_ceil(BUILD_WORKGROUP), 1);
        }
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
