//! GPU-driven occlusion culling: compute shader tests primitives against the
//! depth buffer, writes per-primitive visibility flags to a storage buffer.
//! Read back to CPU → drives the forward draw loop's skip logic.
//!
//! This replaces the hardware occlusion query approach (`OcclusionQueries`)
//! with a compute-shader-based test that evaluates every primitive each frame.

use crate::context::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CullParams {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    width: f32,
    height: f32,
    primitive_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AabbGpu {
    min: [f32; 4],
    max: [f32; 4],
}

pub struct GpuCulling {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    aabb_buffer: wgpu::Buffer,
    visibility_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    max_primitives: usize,
    pub visibility: Vec<bool>,
}

impl GpuCulling {
    pub fn new(gpu: &GpuContext, max_primitives: usize) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gpu_cull.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_cull_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buf_size = (max_primitives as u64) * 32; // AabbGpu = 32 bytes
        let vis_size = (max_primitives as u64) * 4;

        let aabb_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_cull_aabbs"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_cull_visibility"),
            size: vis_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_cull_readback"),
            size: vis_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params = CullParams {
            view_proj: [[0.0; 4]; 4],
            inv_view_proj: [[0.0; 4]; 4],
            width: 0.0,
            height: 0.0,
            primitive_count: 0,
            _pad: [0; 3],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_cull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_binding = params_buf.as_entire_binding();

        Self {
            pipeline,
            bind_group_layout: bgl,
            params_buffer: params_buf,
            aabb_buffer: aabb_buf,
            visibility_buffer: vis_buf,
            readback_buffer: readback,
            max_primitives,
            visibility: Vec::new(),
        }
    }

    /// Upload AABBs, run culling compute, dispatch readback.
    #[allow(clippy::too_many_arguments)]
    pub fn cull(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        view_proj: &[[f32; 4]; 4],
        inv_view_proj: &[[f32; 4]; 4],
        aabbs: &[(glam::Vec3, glam::Vec3)],
    ) {
        let count = aabbs.len().min(self.max_primitives);
        if count == 0 {
            return;
        }

        // Upload AABBs
        let mut gpu_aabbs: Vec<AabbGpu> = Vec::with_capacity(count);
        for (mn, mx) in aabbs.iter().take(count) {
            gpu_aabbs.push(AabbGpu {
                min: [mn.x, mn.y, mn.z, 0.0],
                max: [mx.x, mx.y, mx.z, 0.0],
            });
        }
        gpu.queue
            .write_buffer(&self.aabb_buffer, 0, bytemuck::cast_slice(&gpu_aabbs));

        // Update params
        let params = CullParams {
            view_proj: *view_proj,
            inv_view_proj: *inv_view_proj,
            width: width as f32,
            height: height as f32,
            primitive_count: count as u32,
            _pad: [0; 3],
        };
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Create ad-hoc bind group (changes per frame: depth view)
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_cull_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.aabb_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.visibility_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_cull"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            // round up to 64 (workgroup_size)
            let wg_count = (count as u32 + 63) / 64;
            cpass.dispatch_workgroups(wg_count, 1, 1);
        }

        // Copy to readback
        encoder.copy_buffer_to_buffer(
            &self.visibility_buffer,
            0,
            &self.readback_buffer,
            0,
            (count as u64) * 4,
        );
    }

    /// Read back visibility results (blocking). Call after GPU work completes.
    pub fn readback(&mut self, gpu: &GpuContext, count: usize) {
        let read_size = (count.min(self.max_primitives) as u64) * 4;
        let slice = self.readback_buffer.slice(..read_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().ok() != Some(Ok(())) {
            return;
        }
        let data = slice.get_mapped_range();
        let vis: &[u32] = bytemuck::cast_slice(&data);
        self.visibility = vis.iter().take(count).map(|&v| v != 0).collect();
        drop(data);
        self.readback_buffer.unmap();
    }
}
