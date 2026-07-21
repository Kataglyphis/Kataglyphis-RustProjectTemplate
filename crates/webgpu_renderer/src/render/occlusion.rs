//! GPU occlusion culling, DETECTION half (increment 1).
//!
//! After the forward pass fills the depth buffer, this pass draws each opaque
//! primitive's world-space AABB as a box - depth-test LessEqual, depth-write
//! OFF, no color attachments - wrapped in a hardware occlusion query. The query
//! counts the box fragments that pass the depth test, so a primitive fully
//! behind other geometry reads back 0 samples and a visible one reads > 0.
//!
//! Three deliberate choices, all mirroring [`crate::render::gpu_timing`]:
//!
//! 1. **Hardware occlusion queries, not a Hi-Z depth pyramid.** `QueryType::
//!    Occlusion` is WebGPU-core and works on the web backend; a mip-reduced
//!    depth pyramid is not portable there. The query is the only occlusion
//!    primitive that ships everywhere this renderer runs.
//!
//! 2. **Reading results must not stall the frame.** Occlusion counts land in a
//!    ring of [`SLOT_COUNT`] slots, each mapped asynchronously after submit and
//!    consumed whenever it happens to be ready - one or more frames later. A
//!    slot still in flight is skipped for that frame rather than waited on, so
//!    the frame path never blocks. Detection lagging the frame it measured is
//!    fine: increment 2 (skipping draws) reads last-known visibility, exactly
//!    as a GPU-driven culling pipeline does.
//!
//! 3. **Depth is never written.** The occlusion pipeline sets
//!    `depth_write_enabled: false`; the pass loads and stores the forward
//!    depth unchanged, so the SSAO pass still reconstructs positions from the
//!    depth the forward pass wrote.
//!
//! Detection only this increment: nothing here changes what the forward pass
//! draws. The visibility it produces is exposed for tests and for a later
//! increment to consume.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::render::forward::DEPTH_FORMAT;

/// Frames of occlusion storage in flight. Same reasoning as `gpu_timing`'s
/// ring: two would cover `desired_maximum_frame_latency: 2`, and one spare
/// means a frame whose poll came up empty still finds a free slot instead of
/// dropping the sample.
const SLOT_COUNT: usize = 3;

/// Bytes per occlusion query result (a `u64` sample count).
const QUERY_BYTES: u64 = 8;

/// Map state of one readback slot, shared with the `map_async` callback.
const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

/// One primitive's world AABB, fed to the box vertex shader through an
/// instance-step vertex buffer. The draw for primitive `i` binds instances
/// `i..i+1`, so exactly that primitive's box is emitted.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BboxInstance {
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

impl BboxInstance {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<BboxInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

struct Slot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// `Some` while a map is outstanding; the callback flips it to READY/FAILED
    /// and the next `end_frame` drains it.
    map_state: Option<Arc<AtomicU8>>,
    /// Queries actually recorded into this slot (this frame's primitive count).
    count: u32,
}

/// Per-primitive occlusion detection over the forward depth buffer.
pub struct OcclusionQueries {
    pipeline: wgpu::RenderPipeline,
    view_proj_buffer: wgpu::Buffer,
    view_proj_bind_group: wgpu::BindGroup,
    /// One AABB per primitive, grown with the scene. Rewritten every frame.
    instance_buffer: wgpu::Buffer,
    query_set: wgpu::QuerySet,
    /// Query-set count and the element capacity of every buffer above.
    capacity: u32,
    slots: Vec<Slot>,
    /// Slot the current frame records into.
    current: usize,
    /// True when this frame claimed a free slot and recorded a pass.
    recording: bool,
    /// Latest completed readback: sample count per primitive, index-aligned to
    /// `ForwardRenderer::primitives`.
    samples: Vec<u64>,
    /// `samples[i] > 0`, cached so callers get a `&[bool]` without recomputing.
    visibility: Vec<bool>,
}

impl OcclusionQueries {
    /// Initial per-buffer capacity. Small - a scene with more primitives grows
    /// it on the first frame that needs to, exactly like the instance buffer.
    const INITIAL_CAPACITY: u32 = 32;

    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion_bbox_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/occlusion_bbox.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("occlusion_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[BboxInstance::LAYOUT],
                compilation_options: Default::default(),
            },
            // A fragment stage with ZERO color targets: the pass has no color
            // attachment, only the depth test and the occlusion count matter.
            // WebGPU still wants the stage present to complete the pipeline.
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            // Cull nothing: winding on the box is arbitrary and the camera may
            // sit inside a box, where the front faces are behind it.
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            // Test against the stored forward depth, never write it - writing
            // would corrupt the depth SSAO reads back.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let view_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_view_proj"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_proj_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("occlusion_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_proj_buffer.as_entire_binding(),
            }],
        });

        let capacity = Self::INITIAL_CAPACITY;
        Self {
            pipeline,
            view_proj_buffer,
            view_proj_bind_group,
            instance_buffer: create_instance_buffer(device, capacity),
            query_set: create_query_set(device, capacity),
            capacity,
            slots: create_slots(device, capacity),
            current: 0,
            recording: false,
            samples: Vec::new(),
            visibility: Vec::new(),
        }
    }

    /// Records the occlusion pass into `encoder` and resolves it, or does
    /// nothing when there is no scene or no free ring slot this frame.
    ///
    /// `depth_view` is the forward pass's depth buffer, `view_proj` the matrix
    /// the forward pass drew with, and `aabbs` one world AABB per primitive in
    /// primitive order. Must be recorded into the same encoder as the forward
    /// pass and before submit; call [`Self::end_frame`] after submit.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        view_proj: Mat4,
        aabbs: &[(Vec3, Vec3)],
    ) {
        self.recording = false;
        let count = aabbs.len() as u32;
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        // Claim a free slot; if every slot is still mapping, skip this frame
        // rather than stall - the same trade `gpu_timing` makes.
        let Some(slot_index) = self.free_slot() else {
            return;
        };
        self.current = slot_index;

        queue.write_buffer(
            &self.view_proj_buffer,
            0,
            bytemuck::bytes_of(&view_proj.to_cols_array_2d()),
        );
        let instances: Vec<BboxInstance> = aabbs
            .iter()
            .map(|(min, max)| BboxInstance {
                aabb_min: min.to_array(),
                aabb_max: max.to_array(),
            })
            .collect();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("occlusion_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    // Load the forward depth and store it back untouched: the
                    // pipeline writes no depth, so Store just preserves what
                    // SSAO reads next.
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: Some(&self.query_set),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.view_proj_bind_group, &[]);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            // Exactly one draw per query, indices unique and dense in 0..count,
            // all inside this single pass whose occlusion_query_set is set.
            for i in 0..count {
                pass.begin_occlusion_query(i);
                pass.draw(0..36, i..i + 1);
                pass.end_occlusion_query();
            }
        }

        let slot = &mut self.slots[slot_index];
        slot.count = count;
        let bytes = u64::from(count) * QUERY_BYTES;
        encoder.resolve_query_set(&self.query_set, 0..count, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, bytes);
        self.recording = true;
    }

    /// Starts this frame's readback and consumes any slot whose readback has
    /// since completed. Never blocks; call after submit.
    pub fn end_frame(&mut self, device: &wgpu::Device) {
        if self.recording {
            let state = Arc::new(AtomicU8::new(MAP_PENDING));
            let callback_state = Arc::clone(&state);
            let bytes = u64::from(self.slots[self.current].count) * QUERY_BYTES;
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
                    let samples = read_samples(&self.slots[index].readback, count);
                    self.slots[index].readback.unmap();
                    self.slots[index].map_state = None;
                    self.visibility = samples.iter().map(|&s| s > 0).collect();
                    self.samples = samples;
                }
                MAP_FAILED => {
                    self.slots[index].map_state = None;
                }
                _ => {}
            }
        }
    }

    /// Per-primitive visibility from the most recent completed readback: `true`
    /// when the primitive's box had > 0 fragments pass the depth test. Empty
    /// until the first readback lands (a frame or two after recording starts).
    pub fn visibility(&self) -> &[bool] {
        &self.visibility
    }

    /// Raw sample counts behind [`Self::visibility`], for tests and diagnostics
    /// that want the actual fragment counts rather than the boolean.
    pub fn samples(&self) -> &[u64] {
        &self.samples
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

    /// Index of a slot with no outstanding map, preferring `current` so the
    /// ring advances in order when nothing is in flight.
    fn free_slot(&self) -> Option<usize> {
        (0..SLOT_COUNT)
            .map(|offset| (self.current + offset) % SLOT_COUNT)
            .find(|&candidate| self.slots[candidate].map_state.is_none())
    }

    /// Grows the query set and every buffer to hold at least `count` queries.
    ///
    /// The query set's count is fixed at creation, so growing means recreating
    /// it and the slot buffers. In-flight readbacks in the old slots are
    /// dropped - a scene whose primitive count just changed re-measures over
    /// the next frames anyway, and a resize is rare.
    fn ensure_capacity(&mut self, device: &wgpu::Device, count: u32) {
        if count <= self.capacity {
            return;
        }
        // Grow generously so a slowly growing scene does not reallocate every
        // frame; matches the "double or the request" habit of Vec growth.
        let capacity = count.max(self.capacity * 2);
        self.instance_buffer = create_instance_buffer(device, capacity);
        self.query_set = create_query_set(device, capacity);
        self.slots = create_slots(device, capacity);
        self.capacity = capacity;
        self.current = 0;
        self.recording = false;
    }
}

fn create_instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occlusion_instances"),
        size: u64::from(capacity) * std::mem::size_of::<BboxInstance>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_query_set(device: &wgpu::Device, capacity: u32) -> wgpu::QuerySet {
    device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("occlusion_queries"),
        ty: wgpu::QueryType::Occlusion,
        count: capacity,
    })
}

fn create_slots(device: &wgpu::Device, capacity: u32) -> Vec<Slot> {
    let bytes = u64::from(capacity) * QUERY_BYTES;
    (0..SLOT_COUNT)
        .map(|i| Slot {
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("occlusion_resolve_{i}")),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("occlusion_readback_{i}")),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            map_state: None,
            count: 0,
        })
        .collect()
}

/// Reads `count` `u64` sample counts out of a mapped readback buffer.
fn read_samples(buffer: &wgpu::Buffer, count: usize) -> Vec<u64> {
    let bytes = count as u64 * QUERY_BYTES;
    let view = buffer.slice(..bytes).get_mapped_range();
    let samples = view
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("chunks_exact(8) yields 8 bytes")))
        .collect();
    drop(view);
    samples
}
