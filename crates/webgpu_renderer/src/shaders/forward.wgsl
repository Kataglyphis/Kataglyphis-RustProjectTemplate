// Forward PBR pass into an HDR (Rgba16Float) target.
// Metallic-roughness BRDF (GGX + Smith + Fresnel-Schlick), TBN normal
// mapping, emissive + occlusion, one directional light with a
// comparison-sampled shadow map. Output stays linear HDR; the tonemap pass
// compresses to display range.
//
// WGSL web notes: all textureSample* calls happen in uniform control flow at
// the top of fs_main (Chrome's validator is strict); the shadow lookup uses
// textureSampleCompareLevel, which is legal anywhere.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    // Inverse-transpose of model (upper 3x3 meaningful).
    normal_matrix: mat4x4<f32>,
    // World -> light clip space (directional shadow map).
    light_space: mat4x4<f32>,
    base_color: vec4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
    // rgb: light color, w: intensity multiplier
    light_color_intensity: vec4<f32>,
    // x: metallic factor, y: roughness factor, z: occlusion strength, w: normal scale
    material_factors: vec4<f32>,
    // rgb: emissive factor, w: camera-space unused
    emissive_factor: vec4<f32>,
    // xyz: world-space camera position, w: unused
    camera_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var base_color_tex: texture_2d<f32>;
@group(0) @binding(4) var base_color_sampler: sampler;
@group(0) @binding(5) var metal_rough_tex: texture_2d<f32>;
@group(0) @binding(6) var metal_rough_sampler: sampler;
@group(0) @binding(7) var normal_tex: texture_2d<f32>;
@group(0) @binding(8) var normal_sampler: sampler;
@group(0) @binding(9) var emissive_tex: texture_2d<f32>;
@group(0) @binding(10) var emissive_sampler: sampler;
@group(0) @binding(11) var occlusion_tex: texture_2d<f32>;
@group(0) @binding(12) var occlusion_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) light_space_pos: vec4<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) world_position: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    out.world_normal = normalize((uniforms.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.world_tangent = vec4<f32>(
        normalize((uniforms.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );
    out.uv = in.uv;
    out.light_space_pos = uniforms.light_space * world_pos;
    out.world_position = world_pos.xyz;
    return out;
}

// Depth-only variant for the shadow pass (no fragment stage).
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    return uniforms.light_space * uniforms.model * vec4<f32>(in.position, 1.0);
}

fn shadow_factor(light_space_pos: vec4<f32>, n_dot_l: f32) -> f32 {
    let proj = light_space_pos.xyz / light_space_pos.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    let bias = clamp(0.002 * (1.0 - n_dot_l) + 0.0005, 0.0005, 0.004);

    // 3x3 PCF; textureSampleCompareLevel is valid in non-uniform control flow.
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, proj.z - bias);
        }
    }
    return sum / 9.0;
}

const PI: f32 = 3.14159265359;

// ---- Analytic environment (kept in sync with sky.wgsl) ---------------------
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.09, 0.16, 0.35);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.55, 0.62, 0.72);
const SKY_GROUND: vec3<f32> = vec3<f32>(0.18, 0.16, 0.15);

fn sky_radiance(dir: vec3<f32>, with_sun: bool) -> vec3<f32> {
    var color: vec3<f32>;
    if (dir.y >= 0.0) {
        color = mix(SKY_HORIZON, SKY_ZENITH, pow(clamp(dir.y, 0.0, 1.0), 0.7));
    } else {
        color = mix(SKY_HORIZON, SKY_GROUND, clamp(-dir.y * 3.0, 0.0, 1.0));
    }
    if (with_sun) {
        let l = normalize(uniforms.light_dir_ambient.xyz);
        let cos_sun = max(dot(dir, l), 0.0);
        let sun = pow(cos_sun, 1200.0) * 24.0 + pow(cos_sun, 48.0) * 0.5;
        color += vec3<f32>(1.0, 0.95, 0.85) * sun
            * (uniforms.light_color_intensity.w * 0.4);
    }
    return color;
}

/// Cosine-weighted hemisphere estimate of the analytic sky (cheap
/// irradiance: sky above, bounced ground below, no convolution needed).
fn hemisphere_irradiance(n: vec3<f32>) -> vec3<f32> {
    let sky = mix(SKY_HORIZON, SKY_ZENITH, pow(clamp(n.y, 0.0, 1.0), 0.7));
    return mix(SKY_GROUND * 0.7, sky, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
}

/// Karis/Lazarov split-sum environment BRDF approximation (no LUT).
fn env_brdf_approx(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = vec4<f32>(roughness) * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}
// ----------------------------------------------------------------------------

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-6);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let g1 = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
    let g2 = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
    return g1 * g2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // All implicit-derivative samples up front, in uniform control flow.
    let base_sample = textureSample(base_color_tex, base_color_sampler, in.uv);
    let mr_sample = textureSample(metal_rough_tex, metal_rough_sampler, in.uv);
    let normal_sample = textureSample(normal_tex, normal_sampler, in.uv);
    let emissive_sample = textureSample(emissive_tex, emissive_sampler, in.uv);
    let occlusion_sample = textureSample(occlusion_tex, occlusion_sampler, in.uv);

    let albedo = uniforms.base_color * base_sample;
    // glTF: metallic in B, roughness in G.
    let metallic = clamp(uniforms.material_factors.x * mr_sample.b, 0.0, 1.0);
    let roughness = clamp(uniforms.material_factors.y * mr_sample.g, 0.045, 1.0);
    let occlusion = mix(1.0, occlusion_sample.r, uniforms.material_factors.z);
    let emissive = uniforms.emissive_factor.rgb * emissive_sample.rgb;

    // TBN normal mapping (glTF convention: +Z out of the surface).
    let n_geom = normalize(in.world_normal);
    let t = normalize(in.world_tangent.xyz - n_geom * dot(n_geom, in.world_tangent.xyz));
    let b = cross(n_geom, t) * in.world_tangent.w;
    var n_ts = normal_sample.xyz * 2.0 - 1.0;
    n_ts = vec3<f32>(n_ts.xy * uniforms.material_factors.w, n_ts.z);
    let n = normalize(mat3x3<f32>(t, b, n_geom) * n_ts);

    let l = normalize(uniforms.light_dir_ambient.xyz);
    let v = normalize(uniforms.camera_position.xyz - in.world_position);
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo.rgb, metallic);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = k_d * albedo.rgb / PI;

    let shadow = shadow_factor(in.light_space_pos, n_dot_l);
    let radiance = uniforms.light_color_intensity.rgb * uniforms.light_color_intensity.w;

    // Analytic IBL: hemisphere irradiance for diffuse, roughness-blended sky
    // reflection (sun included -> metals catch sun glints) for specular.
    let ibl_strength = uniforms.light_dir_ambient.w;
    let k_s_ibl = fresnel_schlick(n_dot_v, f0);
    let diffuse_ibl =
        (vec3<f32>(1.0) - k_s_ibl) * (1.0 - metallic) * albedo.rgb * hemisphere_irradiance(n);
    let reflected = reflect(-v, n);
    let env = mix(
        sky_radiance(reflected, true),
        hemisphere_irradiance(n),
        roughness * roughness,
    );
    let specular_ibl = env * env_brdf_approx(f0, roughness, n_dot_v);
    let ambient = ibl_strength * occlusion * (diffuse_ibl + specular_ibl);

    let color = (diffuse + specular) * radiance * n_dot_l * shadow + ambient + emissive;
    return vec4<f32>(color, albedo.a);
}
