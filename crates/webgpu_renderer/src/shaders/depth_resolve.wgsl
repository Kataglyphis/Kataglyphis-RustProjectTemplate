// Depth resolve: downsamples MSAA depth → single-sample via nearest-sample
// selection (minimum depth, which is the nearest fragment).
// Used after the forward MSAA render pass so that SSAO, occlusion culling
// and other single-sample consumers can read a non-MSAA depth texture.

@group(0) @binding(0) var msaa_depth: texture_depth_2d_multisampled;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> Varyings {
    // Fullscreen triangle: indices 0,1,2 → (-1,-1), (3,-1), (-1,3)
    let uv = vec2<f32>(f32(idx & 1u) * 2.0, f32((idx >> 1u) & 1u) * 2.0);
    var out: Varyings;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @builtin(frag_depth) f32 {
    let dims = vec2<f32>(textureDimensions(msaa_depth));
    let coords = vec2<i32>(uv * dims);

    // Take the minimum depth across all samples (nearest fragment wins).
    // This preserves the front-most surface, which is correct for SSAO,
    // occlusion queries, and any consumer expecting a standard depth buffer.
    var min_depth = 1.0;
    for (var i = 0u; i < 4u; i++) {
        let d = textureLoad(msaa_depth, coords, i);
        min_depth = min(min_depth, d);
    }
    return min_depth;
}
