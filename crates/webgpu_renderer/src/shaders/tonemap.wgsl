// Fullscreen tonemap pass: HDR (Rgba16Float, linear) -> display target.
// ACES filmic approximation (Narkowicz 2015). The sRGB output format handles
// gamma encoding in hardware, so the shader emits linear values.

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var ao_tex: texture_2d<f32>;
struct TonemapUniforms {
    // x: bloom strength, y: SSAO strength, z: exposure multiplier
    params: vec4<f32>,
};
@group(0) @binding(3) var<uniform> tonemap_uniforms: TonemapUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // Single fullscreen triangle from the vertex index; no vertex buffer.
    var out: VsOut;
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) % 2) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_tex, hdr_sampler, in.uv).rgb;
    let ao_raw = textureSample(ao_tex, hdr_sampler, in.uv).r;
    let ao = mix(1.0, ao_raw, tonemap_uniforms.params.y);
    let exposure = tonemap_uniforms.params.z;
    return vec4<f32>(aces((hdr * ao + bloom * tonemap_uniforms.params.x) * exposure), 1.0);
}
