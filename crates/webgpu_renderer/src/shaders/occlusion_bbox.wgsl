// GPU occlusion culling, DETECTION half (increment 1).
//
// Draws one primitive's WORLD-space AABB as a solid box, wrapped by the caller
// in begin/end_occlusion_query. The pass runs against the forward pass's depth
// buffer with depth-test LessEqual and depth-WRITE OFF, so the hardware
// occlusion query counts the box fragments that pass the depth test: a
// primitive fully behind other geometry yields 0 samples, a visible one > 0.
//
// The box comes from the SAME view-projection the forward pass used, so it
// lines up with where the geometry actually is. Depth is never written here -
// that would corrupt the stored depth the SSAO pass reads back.

struct OcclusionUniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: OcclusionUniforms;

// The AABB arrives per-primitive through an instance-step vertex buffer: the
// draw for primitive i binds instances i..i+1, so the box for exactly that
// primitive is fetched. This keeps each draw a single command that the query
// can bracket, with no per-primitive bind group churn.
struct BboxInstance {
    @location(0) aabb_min: vec3<f32>,
    @location(1) aabb_max: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, instance: BboxInstance) -> @builtin(position) vec4<f32> {
    // 12 triangles (36 vertices) over the 8 corners of the unit box, corner
    // c selecting min/max per axis by its bits (bit0=x, bit1=y, bit2=z). Wound
    // arbitrarily: the occlusion pipeline disables culling, so every face
    // rasterises regardless of winding and the camera may sit inside the box.
    var corners = array<u32, 36>(
        // -Z face
        0u, 2u, 1u, 1u, 2u, 3u,
        // +Z face
        4u, 5u, 6u, 5u, 7u, 6u,
        // -Y face
        0u, 1u, 4u, 1u, 5u, 4u,
        // +Y face
        2u, 6u, 3u, 3u, 6u, 7u,
        // -X face
        0u, 4u, 2u, 2u, 4u, 6u,
        // +X face
        1u, 3u, 5u, 3u, 7u, 5u,
    );
    let c = corners[vertex_index];
    let sel = vec3<f32>(
        f32(c & 1u),
        f32((c >> 1u) & 1u),
        f32((c >> 2u) & 1u),
    );
    // Expand the box by a small margin before testing. A primitive whose AABB
    // coincides with its own front face (an axis-aligned box is its own AABB)
    // otherwise z-fights the stored depth under LessEqual, and only a fragment
    // or two survive - a knife-edge that reads as visible on one GPU and
    // occluded on another. The margin pulls the box's front slightly toward the
    // camera so a genuinely visible primitive passes robustly. This is the
    // conservative direction: over-reporting visibility never wrongly skips a
    // draw, and the margin (2% of the half-extent plus 1 cm) is far smaller
    // than the depth gap a real occluder puts behind it, so it cannot make an
    // occluded primitive test visible.
    let half = (instance.aabb_max - instance.aabb_min) * 0.5;
    let margin = half * 0.02 + vec3<f32>(0.01);
    let expanded_min = instance.aabb_min - margin;
    let expanded_max = instance.aabb_max + margin;
    let corner = mix(expanded_min, expanded_max, sel);
    return uniforms.view_proj * vec4<f32>(corner, 1.0);
}

// No color output: the pass has zero color attachments and only the depth test
// and occlusion count matter. WebGPU still wants a fragment stage to complete
// the pipeline, so this is an empty entry point.
@fragment
fn fs_main() {
}
