// Bloom chain: bright-pass extraction and separable 9-tap Gaussian blur,
// all fullscreen triangles at half resolution on Rgba16Float targets.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var out: VsOut;
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) % 2) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

const THRESHOLD: f32 = 1.0;

@fragment
fn fs_brightpass(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(src_tex, src_sampler, in.uv).rgb;
    // Keep only energy above the threshold (HDR values > 1).
    let bright = max(hdr - vec3<f32>(THRESHOLD), vec3<f32>(0.0));
    return vec4<f32>(bright, 1.0);
}

const W0: f32 = 0.227027;
const W1: f32 = 0.1945946;
const W2: f32 = 0.1216216;
const W3: f32 = 0.054054;
const W4: f32 = 0.016216;

fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec3<f32> {
    let texel = dir / vec2<f32>(textureDimensions(src_tex));
    var color = textureSample(src_tex, src_sampler, uv).rgb * W0;
    color += textureSample(src_tex, src_sampler, uv + texel * 1.0).rgb * W1;
    color += textureSample(src_tex, src_sampler, uv - texel * 1.0).rgb * W1;
    color += textureSample(src_tex, src_sampler, uv + texel * 2.0).rgb * W2;
    color += textureSample(src_tex, src_sampler, uv - texel * 2.0).rgb * W2;
    color += textureSample(src_tex, src_sampler, uv + texel * 3.0).rgb * W3;
    color += textureSample(src_tex, src_sampler, uv - texel * 3.0).rgb * W3;
    color += textureSample(src_tex, src_sampler, uv + texel * 4.0).rgb * W4;
    color += textureSample(src_tex, src_sampler, uv - texel * 4.0).rgb * W4;
    return color;
}

@fragment
fn fs_blur_h(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(blur(in.uv, vec2<f32>(1.0, 0.0)), 1.0);
}

@fragment
fn fs_blur_v(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(blur(in.uv, vec2<f32>(0.0, 1.0)), 1.0);
}
