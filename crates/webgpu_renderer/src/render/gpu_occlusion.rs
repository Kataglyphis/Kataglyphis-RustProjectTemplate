//! GPU-driven occlusion culling: compute shader tests primitives against the
//! depth buffer, writes per-primitive visibility flags to a storage buffer.
//! Read back to CPU → drives the forward draw loop's skip logic.
//!
//! This replaces the hardware occlusion query approach (`OcclusionQueries`)
//! with a compute-shader-based test that evaluates every primitive each frame.
//! The readback follows the same non-blocking ring as [`crate::render::occlusion`]:
//! `map_async` is polled, never waited on, so a frame whose result is not yet
//! ready simply keeps drawing with the last-known visibility.

use crate::context::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
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

/// Readback ring depth, same reasoning as `occlusion::SLOT_COUNT`: two slots
/// cover `desired_maximum_frame_latency: 2`, and a third spare means a frame
/// whose poll comes up empty still finds a free slot instead of the frame
/// being dropped outright.
const SLOT_COUNT: usize = 3;

/// Map state of one readback slot, shared with the `map_async` callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

struct Slot {
    readback: wgpu::Buffer,
    /// `Some` while a map is outstanding; the callback flips it to READY/FAILED
    /// and the next `end_frame` drains it.
    map_state: Option<Arc<AtomicU8>>,
    /// Primitives actually recorded into this slot (this frame's count).
    count: u32,
    /// The scene generation this slot was recorded under; see
    /// [`crate::render::occlusion::OcclusionQueries`]'s field of the same name.
    generation: u64,
}

pub struct GpuCulling {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    aabb_buffer: wgpu::Buffer,
    visibility_buffer: wgpu::Buffer,
    max_primitives: usize,
    slots: Vec<Slot>,
    /// Slot the current frame records into.
    current: usize,
    /// True when this frame claimed a free slot and recorded a dispatch.
    recording: bool,
    /// Bumped by [`Self::reset`] on a scene change; stamped onto a slot when
    /// its readback is kicked off and checked when it lands, so results from a
    /// replaced scene are discarded rather than applied to the new one.
    generation: u64,
    /// Per-primitive visibility from the most recent completed readback,
    /// `true` when the primitive's centre passed the depth test.
    visibility: Vec<bool>,
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

        Self {
            pipeline,
            bind_group_layout: bgl,
            params_buffer: params_buf,
            aabb_buffer: aabb_buf,
            visibility_buffer: vis_buf,
            max_primitives,
            slots: create_slots(device, max_primitives),
            current: 0,
            recording: false,
            generation: 0,
            visibility: Vec::new(),
        }
    }

    /// Upload AABBs, run the culling compute pass, and kick off a copy into
    /// this frame's readback slot - or do nothing when every slot is still
    /// mapping from a prior frame, the same trade `OcclusionQueries::record`
    /// makes rather than stalling. Call [`Self::end_frame`] after submit.
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
        self.recording = false;
        let count = aabbs.len().min(self.max_primitives);
        if count == 0 {
            return;
        }

        // Claim a free slot; if every slot is still mapping, skip this frame
        // rather than stall.
        let Some(slot_index) = self.free_slot() else {
            return;
        };
        self.current = slot_index;

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

        // Copy to this frame's readback slot
        let slot = &mut self.slots[slot_index];
        slot.count = count as u32;
        encoder.copy_buffer_to_buffer(
            &self.visibility_buffer,
            0,
            &slot.readback,
            0,
            (count as u64) * 4,
        );
        self.recording = true;
    }

    /// Starts this frame's readback and consumes any slot whose readback has
    /// since completed. Never blocks; call after submit, alongside
    /// [`crate::render::occlusion::OcclusionQueries::end_frame`].
    pub fn end_frame(&mut self, device: &wgpu::Device) {
        if self.recording {
            let state = Arc::new(AtomicU8::new(MAP_PENDING));
            let callback_state = Arc::clone(&state);
            let bytes = u64::from(self.slots[self.current].count) * 4;
            self.slots[self.current].readback.slice(..bytes).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    let code = if result.is_ok() {
                        MAP_READY
                    } else {
                        MAP_FAILED
                    };
                    callback_state.store(code, Ordering::Release);
                },
            );
            self.slots[self.current].map_state = Some(state);
            self.slots[self.current].generation = self.generation;
            self.current = (self.current + 1) % SLOT_COUNT;
            self.recording = false;
        }

        // Non-blocking: only lets already-finished callbacks run. A waiting
        // poll here would reintroduce the stall the ring exists to avoid.
        let _ = device.poll(wgpu::PollType::Poll);

        for index in 0..self.slots.len() {
            let Some(state) = self.slots[index].map_state.as_ref() else {
                continue;
            };
            match state.load(Ordering::Acquire) {
                MAP_READY => {
                    let count = self.slots[index].count as usize;
                    let bytes = (count as u64) * 4;
                    let view = self.slots[index].readback.slice(..bytes).get_mapped_range();
                    let raw: &[u32] = bytemuck::cast_slice(&view);
                    let visibility: Vec<bool> = raw.iter().map(|&v| v != 0).collect();
                    drop(view);
                    self.slots[index].readback.unmap();
                    self.slots[index].map_state = None;
                    // A stale generation means the scene was replaced after this
                    // slot recorded; keep the reset's empty visibility so the new
                    // scene is not culled by the old scene's samples.
                    if self.slots[index].generation == self.generation {
                        self.visibility = visibility;
                    }
                }
                MAP_FAILED => {
                    self.slots[index].map_state = None;
                }
                _ => {}
            }
        }
    }

    /// Whether primitive `i` should be drawn given the last readback.
    ///
    /// Defaults to VISIBLE (`true`) for any index the readback has not covered
    /// yet - the first frames before results land, or a primitive added since.
    /// Defaulting to visible is the safe direction: a never-culled primitive
    /// costs a draw, a wrongly-culled one pops out of existence.
    pub fn visible(&self, i: usize) -> bool {
        self.visibility.get(i).copied().unwrap_or(true)
    }

    /// Forgets all readback state; call when the scene changes. Per-index
    /// visibility is only meaningful for the primitive list it was measured
    /// against - see [`crate::render::occlusion::OcclusionQueries::reset`] for
    /// why carrying it across a scene change is unsafe. Clearing makes
    /// [`Self::visible`] default to true again until fresh results land, and
    /// the bumped generation discards any readback still in flight from the
    /// old scene.
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.visibility.clear();
    }

    /// Index of a slot with no outstanding map, preferring `current` so the
    /// ring advances in order when nothing is in flight.
    fn free_slot(&self) -> Option<usize> {
        (0..SLOT_COUNT)
            .map(|offset| (self.current + offset) % SLOT_COUNT)
            .find(|&candidate| self.slots[candidate].map_state.is_none())
    }
}

fn create_slots(device: &wgpu::Device, max_primitives: usize) -> Vec<Slot> {
    let bytes = (max_primitives as u64) * 4;
    (0..SLOT_COUNT)
        .map(|i| Slot {
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("gpu_cull_readback_{i}")),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            map_state: None,
            count: 0,
            generation: 0,
        })
        .collect()
}
