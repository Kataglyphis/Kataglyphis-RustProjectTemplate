// GPU occlusion culling: tests each primitive's projected AABB against the
// depth buffer. Writes 1 (visible) or 0 (occluded) to a storage buffer.
//
// The test: project the AABB's center to UV+depth, sample depth buffer at
// that UV, compare. If the primitive's depth > sampled depth (behind surface)
// it's occluded. Conservative - favors visible when uncertain.
//
// Dispatch: NUM_PRIMITIVES workgroups of size 64, each WG processes 64 prims.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> params: CullParams;
@group(0) @binding(2) var<storage, read> aabb_data: array<Aabb>;
@group(0) @binding(3) var<storage, read_write> visibility: array<u32>;

struct CullParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    width: f32,
    height: f32,
    primitive_count: u32,
}

struct Aabb {
    min: vec4<f32>,
    max: vec4<f32>,
}

fn project_center(min: vec3<f32>, max: vec3<f32>, vp: mat4x4<f32>) -> (vec2<f32>, f32) {
    let center = (min + max) * 0.5;
    let clip = vp * vec4<f32>(center, 1.0);
    let ndc = clip.xy / clip.w;
    let depth = clip.z / clip.w;
    let uv = ndc * 0.5 + 0.5;
    return (uv, depth);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.primitive_count {
        return;
    }

    let aabb = aabb_data[idx];
    let (uv, prim_depth) = project_center(aabb.min.xyz, aabb.max.xyz, params.view_proj);

    // Behind camera or beyond far plane → not visible
    if prim_depth < 0.0 || prim_depth > 1.0 {
        visibility[idx] = 0u;
        return;
    }

    // Off-screen → not drawn
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        visibility[idx] = 0u;
        return;
    }

    // Sample depth buffer at projected UV
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let coords = vec2<i32>(uv * dims);
    let sampled_depth = textureLoad(depth_tex, coords, 0);

    // If the primitive's nearest point is behind the sampled depth → occluded
    // Add a small bias to avoid z-fighting
    if prim_depth > sampled_depth + 0.001 {
        visibility[idx] = 0u;
    } else {
        visibility[idx] = 1u;
    }
}
