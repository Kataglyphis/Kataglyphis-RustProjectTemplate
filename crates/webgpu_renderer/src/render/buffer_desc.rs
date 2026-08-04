//! `wgpu::BufferDescriptor` constructors for the five shapes that cover 24 of
//! the crate's 26 `create_buffer` call sites - the sixth member of the
//! descriptor-shape family, after `bind_layout.rs`, `pipeline_desc.rs` and
//! the texture descriptor in `texture.rs`. Every site repeated
//! `mapped_at_creation: false` and one of a handful of `usage` combinations
//! around the two fields that actually differ (label, size); one helper per
//! shape keeps those two visible at the call site instead of buried in a
//! four-line literal.
//!
//! Two sites are a genuinely different shape and stay as literal
//! `wgpu::BufferDescriptor { .. }` calls rather than growing this module a
//! `usage` parameter (which would turn five named shapes back into one
//! anonymous one):
//! - `histogram.rs`'s `exposure_buffer` needs
//!   `STORAGE | COPY_SRC | COPY_DST` - it is read back *and* seeded, unlike
//!   every other storage buffer in the crate.
//! - `occlusion.rs`'s `create_instance_buffer` needs `VERTEX | COPY_DST` -
//!   it is a vertex buffer, not a uniform/storage/readback/query-resolve
//!   buffer.

/// A uniform buffer: `UNIFORM | COPY_DST`.
pub fn uniform(device: &wgpu::Device, label: &str, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A host-written storage buffer: `STORAGE | COPY_DST`.
pub fn storage_dst(device: &wgpu::Device, label: &str, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A GPU-written storage buffer that is later copied out: `STORAGE | COPY_SRC`.
pub fn storage_src(device: &wgpu::Device, label: &str, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// A host-mappable readback buffer: `COPY_DST | MAP_READ`.
pub fn readback(device: &wgpu::Device, label: &str, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

/// A timestamp/occlusion query resolve target: `QUERY_RESOLVE | COPY_SRC`.
pub fn query_resolve(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::BufferAddress,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
