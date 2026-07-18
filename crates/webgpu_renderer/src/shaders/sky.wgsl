// Procedural sky: horizon/zenith gradient + analytic sun disk derived from
// the directional light, so the overlay's light sliders move the sun.
// Drawn as a fullscreen triangle at far depth (z = 1) with LessEqual
// compare and no depth writes — only background pixels survive.

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    // xyz: direction TOWARDS the light/sun, w: light intensity
    light_dir_intensity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sky: SkyUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var out: VsOut;
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) % 2) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let near = sky.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = sky.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let zenith = vec3<f32>(0.09, 0.16, 0.35);
    let horizon = vec3<f32>(0.55, 0.62, 0.72);
    let ground = vec3<f32>(0.18, 0.16, 0.15);

    var color: vec3<f32>;
    if (dir.y >= 0.0) {
        color = mix(horizon, zenith, pow(clamp(dir.y, 0.0, 1.0), 0.7));
    } else {
        color = mix(horizon, ground, clamp(-dir.y * 3.0, 0.0, 1.0));
    }

    let l = normalize(sky.light_dir_intensity.xyz);
    let cos_sun = max(dot(dir, l), 0.0);
    // Sharp HDR sun disk + soft haze around it.
    let sun = pow(cos_sun, 1200.0) * 24.0 + pow(cos_sun, 48.0) * 0.5;
    color += vec3<f32>(1.0, 0.95, 0.85) * sun * (sky.light_dir_intensity.w * 0.4);

    return vec4<f32>(color, 1.0);
}
