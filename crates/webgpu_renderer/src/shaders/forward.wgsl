// Forward pass into an HDR (Rgba16Float) target: directional light + ambient,
// base color factor x sampled base color texture (1x1 white when untextured),
// and a comparison-sampled directional shadow map.
// Output stays linear HDR; the tonemap pass compresses to display range.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    // World -> light clip space (directional shadow map).
    light_space: mat4x4<f32>,
    base_color: vec4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
    // rgb: light color, w: intensity multiplier
    light_color_intensity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var base_color_tex: texture_2d<f32>;
@group(0) @binding(2) var base_color_sampler: sampler;
@group(0) @binding(3) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) light_space_pos: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    // Assumes uniform scaling; a full normal matrix arrives with skinning.
    out.world_normal = normalize((uniforms.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    out.light_space_pos = uniforms.light_space * world_pos;
    return out;
}

// Depth-only variant for the shadow pass (no fragment stage).
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    return uniforms.light_space * uniforms.model * vec4<f32>(in.position, 1.0);
}

fn shadow_factor(light_space_pos: vec4<f32>, n_dot_l: f32) -> f32 {
    let proj = light_space_pos.xyz / light_space_pos.w;
    // Light clip -> UV (y flipped); depth is already 0..1 in WebGPU.
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    // Slope-scaled bias against acne.
    let bias = clamp(0.002 * (1.0 - n_dot_l) + 0.0005, 0.0005, 0.004);

    // 3x3 PCF.
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, proj.z - bias);
        }
    }
    return sum / 9.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let albedo = uniforms.base_color
        * textureSample(base_color_tex, base_color_sampler, in.uv);

    let light_dir = normalize(uniforms.light_dir_ambient.xyz);
    let ambient = uniforms.light_dir_ambient.w;
    let n_dot_l = max(dot(normalize(in.world_normal), light_dir), 0.0);
    let shadow = shadow_factor(in.light_space_pos, n_dot_l);

    let light = uniforms.light_color_intensity.rgb * uniforms.light_color_intensity.w;
    let color = albedo.rgb * (vec3<f32>(ambient) + light * n_dot_l * shadow);
    return vec4<f32>(color, albedo.a);
}
