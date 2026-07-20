// Fullscreen tonemap pass: HDR (Rgba16Float, linear) -> display target.
// ACES filmic approximation (Narkowicz 2015). The sRGB output format handles
// gamma encoding in hardware, so the shader emits linear values.

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var ao_tex: texture_2d<f32>;
struct TonemapUniforms {
    // x: bloom strength, y: SSAO strength, z: unused (exposure now comes
    // from exposure_state), w: 1.0 when this shader must gamma-encode its own
    // output
    params: vec4<f32>,
};
@group(0) @binding(3) var<uniform> tonemap_uniforms: TonemapUniforms;
// [adapted EV, target EV], written by cs_reduce_exposure. Read here rather
// than passed through params.z so there is ONE source of truth: manual mode
// writes the slider value into this same buffer, so switching modes cannot
// leave the tonemap reading a stale value from the other path.
@group(0) @binding(5) var<storage, read> exposure_state: array<f32, 2>;

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

// The exact IEC 61966-2-1 transfer function, matching what an sRGB target's
// hardware encode does - NOT a pow(x, 1/2.2) approximation, which is visibly
// wrong in the darks and would make the web build differ from native.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c < vec3<f32>(0.0031308);
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, cutoff);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_tex, hdr_sampler, in.uv).rgb;
    let ao_raw = textureSample(ao_tex, hdr_sampler, in.uv).r;
    let ao = mix(1.0, ao_raw, tonemap_uniforms.params.y);
    let exposure = exp2(exposure_state[0]);
    let mapped = aces((hdr * ao + bloom * tonemap_uniforms.params.x) * exposure);

    // An sRGB target encodes in hardware and gets linear values. WebGPU
    // canvases expose no sRGB surface format, so on web the encode has to
    // happen here or the image displays uncorrected - the "slightly dark web
    // demo" recorded in docs/webgpu-srgb-audit.md.
    if (tonemap_uniforms.params.w > 0.5) {
        return vec4<f32>(linear_to_srgb(mapped), 1.0);
    }
    return vec4<f32>(mapped, 1.0);
}
