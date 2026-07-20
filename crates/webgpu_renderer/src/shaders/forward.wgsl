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
    // World -> light clip space per cascade (CASCADE_COUNT used).
    light_space: mat4x4<f32>,
    light_space_1: mat4x4<f32>,
    light_space_2: mat4x4<f32>,
    base_color: vec4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
    // rgb: light color, w: intensity multiplier
    light_color_intensity: vec4<f32>,
    // x: metallic factor, y: roughness factor, z: occlusion strength, w: normal scale
    material_factors: vec4<f32>,
    // rgb: emissive factor, w: camera-space unused
    emissive_factor: vec4<f32>,
    // xyz: world-space camera position, w: active punctual light count
    camera_position: vec4<f32>,
    // Per light: [pos.xyz, kind], [color*intensity.rgb, range],
    // [dir.xyz, cos_inner], [cos_outer, 0, 0, 0]. kind: 1=point 2=spot 3=dir.
    punctual_lights: array<vec4<f32>, 16>,
    // KHR_texture_transform affine rows for the base color UV.
    base_uv_row0: vec4<f32>,
    base_uv_row1: vec4<f32>,
    // x,y: cascade split distances (view depth), z: cascade count
    cascade_splits: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
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
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
    // Per-instance transform, four columns of a mat4. Every draw binds an
    // instance buffer - unbatched primitives get a single identity instance -
    // so there is one code path rather than two pipelines to keep in step.
    @location(6) instance0: vec4<f32>,
    @location(7) instance1: vec4<f32>,
    @location(8) instance2: vec4<f32>,
    @location(9) instance3: vec4<f32>,
};

fn instance_matrix(in: VsIn) -> mat4x4<f32> {
    return mat4x4<f32>(in.instance0, in.instance1, in.instance2, in.instance3);
}

// Joint matrices for skinned primitives (identity-filled when unskinned).
@group(0) @binding(13) var<storage, read> joint_matrices: array<mat4x4<f32>>;

/// Linear blend skinning; returns the model matrix to use for this vertex.
fn skin_matrix(in: VsIn) -> mat4x4<f32> {
    let w = in.weights;
    let total = w.x + w.y + w.z + w.w;
    if (total <= 0.0001) {
        return uniforms.model;
    }
    var m = joint_matrices[u32(in.joints.x)] * w.x;
    m += joint_matrices[u32(in.joints.y)] * w.y;
    m += joint_matrices[u32(in.joints.z)] * w.z;
    m += joint_matrices[u32(in.joints.w)] * w.w;
    return m * (1.0 / total);
}

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) light_space_pos: vec4<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) view_depth: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // Instance transform applies in world space, AFTER the model matrix, so
    // an instanced primitive keeps its own authored transform and is then
    // placed by the instance.
    let model = instance_matrix(in) * skin_matrix(in);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    // Skinned normals use the skinning matrix (uniform scale assumed);
    // unskinned vertices keep the precomputed normal matrix.
    if (in.weights.x + in.weights.y + in.weights.z + in.weights.w > 0.0001) {
        out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    } else {
        // The instance matrix must rotate the normal too, or instanced copies
        // keep the un-instanced orientation's lighting.
        out.world_normal =
            normalize((instance_matrix(in) * uniforms.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    }
    out.world_tangent = vec4<f32>(
        normalize((model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );
    out.uv = in.uv;
    out.light_space_pos = uniforms.light_space * world_pos;
    out.world_position = world_pos.xyz;
    out.view_depth = distance(world_pos.xyz, uniforms.camera_position.xyz);
    return out;
}

// Depth-only variant for the shadow pass (no fragment stage).
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    // Instanced casters must shadow from their instance position, not the
    // authored one - otherwise every copy casts the original's shadow.
    let world = instance_matrix(in) * skin_matrix(in) * vec4<f32>(in.position, 1.0);
    let cascade = i32(uniforms.cascade_splits.w);
    if (cascade == 1) {
        return uniforms.light_space_1 * world;
    }
    if (cascade == 2) {
        return uniforms.light_space_2 * world;
    }
    return uniforms.light_space * world;
}

fn shadow_factor(view_depth: f32, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    // Cascade selection by view distance.
    var cascade = 0;
    if (view_depth > uniforms.cascade_splits.x) {
        cascade = 1;
    }
    if (view_depth > uniforms.cascade_splits.y) {
        cascade = 2;
    }
    let count = i32(uniforms.cascade_splits.z);
    if (cascade > count - 1) {
        cascade = count - 1;
    }

    var light_space_pos: vec4<f32>;
    if (cascade == 1) {
        light_space_pos = uniforms.light_space_1 * vec4<f32>(world_pos, 1.0);
    } else if (cascade == 2) {
        light_space_pos = uniforms.light_space_2 * vec4<f32>(world_pos, 1.0);
    } else {
        light_space_pos = uniforms.light_space * vec4<f32>(world_pos, 1.0);
    }

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
            sum += textureSampleCompareLevel(
                shadow_map,
                shadow_sampler,
                uv + offset,
                cascade,
                proj.z - bias,
            );
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

fn brdf_direct(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
    radiance: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

/// KHR_lights_punctual accumulation (no shadows for these lights yet).
fn punctual_lighting(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    var total = vec3<f32>(0.0);
    let count = i32(uniforms.camera_position.w);
    for (var i = 0; i < 4; i = i + 1) {
        if (i >= count) {
            break;
        }
        let a = uniforms.punctual_lights[i * 4];
        let b = uniforms.punctual_lights[i * 4 + 1];
        let cvec = uniforms.punctual_lights[i * 4 + 2];
        let dvec = uniforms.punctual_lights[i * 4 + 3];
        let kind = a.w;

        var l: vec3<f32>;
        var attenuation = 1.0;
        if (kind > 2.5) {
            // Directional: light points down cvec.xyz.
            l = normalize(-cvec.xyz);
        } else {
            let to_light = a.xyz - world_pos;
            let dist = max(length(to_light), 1e-4);
            l = to_light / dist;
            // Inverse-square with the KHR range window.
            attenuation = 1.0 / (dist * dist);
            let range = b.w;
            if (range > 0.0) {
                let k = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
                attenuation *= k * k;
            }
            if (kind > 1.5) {
                // Spot cone falloff between outer and inner cosines.
                let cos_angle = dot(normalize(cvec.xyz), -l);
                attenuation *= smoothstep(dvec.x, cvec.w, cos_angle);
            }
        }
        if (attenuation <= 0.0) {
            continue;
        }
        total += brdf_direct(n, v, l, albedo, metallic, roughness, f0, b.rgb * attenuation);
    }
    return total;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // All implicit-derivative samples up front, in uniform control flow.
    let base_uv = vec2<f32>(
        uniforms.base_uv_row0.x * in.uv.x + uniforms.base_uv_row0.y * in.uv.y
            + uniforms.base_uv_row0.z,
        uniforms.base_uv_row1.x * in.uv.x + uniforms.base_uv_row1.y * in.uv.y
            + uniforms.base_uv_row1.z,
    );
    let base_sample = textureSample(base_color_tex, base_color_sampler, base_uv);
    let mr_sample = textureSample(metal_rough_tex, metal_rough_sampler, in.uv);
    let normal_sample = textureSample(normal_tex, normal_sampler, in.uv);
    let emissive_sample = textureSample(emissive_tex, emissive_sampler, in.uv);
    let occlusion_sample = textureSample(occlusion_tex, occlusion_sampler, in.uv);

    let albedo = uniforms.base_color * base_sample;
    // MASK alpha mode: emissive_factor.w carries the cutoff (0 = keep all).
    if (albedo.a < uniforms.emissive_factor.w) {
        discard;
    }
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

    let shadow = shadow_factor(in.view_depth, in.world_position, n_dot_l);
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

    let punctual =
        punctual_lighting(in.world_position, n, v, albedo.rgb, metallic, roughness, f0);

    let color =
        (diffuse + specular) * radiance * n_dot_l * shadow + punctual + ambient + emissive;
    return vec4<f32>(color, albedo.a);
}
