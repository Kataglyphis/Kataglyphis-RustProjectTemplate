// Forward pass into an HDR (Rgba16Float) target: directional light + ambient,
// base color factor x sampled base color texture (1x1 white when untextured).
// Output stays linear HDR; the tonemap pass compresses to display range.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
    // rgb: light color, w: intensity multiplier
    light_color_intensity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var base_color_tex: texture_2d<f32>;
@group(0) @binding(2) var base_color_sampler: sampler;

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
    // Assumes uniform scaling; a full normal matrix arrives with skinning.
    out.world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let albedo = uniforms.base_color
        * textureSample(base_color_tex, base_color_sampler, in.uv);

    let light_dir = normalize(uniforms.light_dir_ambient.xyz);
    let ambient = uniforms.light_dir_ambient.w;
    let n_dot_l = max(dot(normalize(in.world_normal), light_dir), 0.0);

    let light = uniforms.light_color_intensity.rgb * uniforms.light_color_intensity.w;
    let color = albedo.rgb * (vec3<f32>(ambient) + light * n_dot_l);
    return vec4<f32>(color, albedo.a);
}
