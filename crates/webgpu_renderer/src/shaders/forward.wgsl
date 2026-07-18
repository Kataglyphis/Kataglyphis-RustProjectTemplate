// Forward pass: single directional light + ambient, per-primitive base color.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    // Assumes uniform scaling; a normal matrix arrives with milestone 3.
    out.world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(uniforms.light_dir_ambient.xyz);
    let ambient = uniforms.light_dir_ambient.w;
    let n_dot_l = max(dot(normalize(in.world_normal), light_dir), 0.0);
    let lit = ambient + (1.0 - ambient) * n_dot_l;
    return vec4<f32>(uniforms.base_color.rgb * lit, uniforms.base_color.a);
}
