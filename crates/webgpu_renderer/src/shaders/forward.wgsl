// Forward PBR pass into an HDR (Rgba16Float) target.
// Metallic-roughness BRDF (GGX + Smith + Fresnel-Schlick), TBN normal
// mapping, emissive + occlusion, one directional light with a
// comparison-sampled shadow map. Output stays linear HDR; the tonemap pass
// compresses to display range.
//
// WGSL web notes: all textureSample* calls happen in uniform control flow at
// the top of fs_main (Chrome's validator is strict); the shadow lookup uses
// textureSampleCompareLevel, which is legal anywhere.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    // World -> light clip space per cascade (CASCADE_COUNT used).
    light_space: mat4x4<f32>,
    light_space_1: mat4x4<f32>,
    light_space_2: mat4x4<f32>,
    // xyz: direction TOWARDS the light, w: ambient strength
    light_dir_ambient: vec4<f32>,
    // rgb: light color, w: intensity multiplier
    light_color_intensity: vec4<f32>,
    // xyz: world-space camera position, w: active punctual light count
    camera_position: vec4<f32>,
    // x,y: cascade split distances (view depth), z: cascade count
    cascade_splits: vec4<f32>,
};

// Punctual lights in a storage buffer (group 3, forward-only).
// Each light is packed as 4 × vec4<f32>:
//   [0]: position.xyz, kind (1=point, 2=spot, 3=directional)
//   [1]: color*intensity.rgb, range
//   [2]: direction.xyz, cos_inner
//   [3]: cos_outer, 0, 0, 0
@group(3) @binding(0) var<storage, read> punctual_lights: array<vec4<f32>>;

// Tile light grid (group 4): per-tile [count, offset] + light index list.
// Screen is divided into TILE_SIZE×TILE_SIZE tiles; each fragment reads its
// tile's light list and iterates only overlapping lights.
@group(4) @binding(0) var<storage, read> tile_light_grid: array<vec2<u32>>;
@group(4) @binding(1) var<storage, read> tile_light_indices: array<u32>;

struct PrimUniforms {
    model: mat4x4<f32>,
    // Inverse-transpose of model (upper 3x3 meaningful).
    normal_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    // x: metallic factor, y: roughness factor, z: occlusion strength, w: normal scale
    material_factors: vec4<f32>,
    // rgb: emissive factor, w: MASK alpha cutoff (-1 = no discard, >=0 = threshold)
    emissive_factor: vec4<f32>,
    // KHR_texture_transform affine rows for the base color UV.
    base_uv_row0: vec4<f32>,
    base_uv_row1: vec4<f32>,
    // x: 1.0 when KHR_materials_unlit, else 0.0
    material_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> prim: PrimUniforms;
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
    // glTF COLOR_0 (linear RGBA), (1,1,1,1) when absent.
    @location(6) color: vec4<f32>,
    // glTF TEXCOORD_1 (second UV set); slots flagged in material_flags.y use it.
    @location(7) uv1: vec2<f32>,
    // Per-instance transform, four columns of a mat4. Every draw binds an
    // instance buffer - unbatched primitives get a single identity instance -
    // so there is one code path rather than two pipelines to keep in step.
    // Locations 8-11: vertex attrs occupy 0-7 (colour 6, uv1 7).
    @location(8) instance0: vec4<f32>,
    @location(9) instance1: vec4<f32>,
    @location(10) instance2: vec4<f32>,
    @location(11) instance3: vec4<f32>,
};

fn instance_matrix(in: VsIn) -> mat4x4<f32> {
    return mat4x4<f32>(in.instance0, in.instance1, in.instance2, in.instance3);
}

// Joint matrices for skinned primitives (identity-filled when unskinned).
@group(0) @binding(13) var<storage, read> joint_matrices: array<mat4x4<f32>>;

// ---- Image-based lighting (group 1, bound once per pass) -------------------
//
// A second group rather than more of group 0: these three maps are the same
// for every primitive in the frame, and group 0 is rebound per draw.
//
// When no environment is set the textures are 1x1 stand-ins and `ibl_params.x`
// is 0; fs_main still samples them (see the note at the ambient block) and
// discards the result.
struct IblParams {
    // x: 1 when an environment is bound, y: highest prefiltered mip index,
    // z: environment intensity multiplier, w: unused
    enabled_maxmip_intensity: vec4<f32>,
};

@group(1) @binding(0) var<uniform> ibl_params: IblParams;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(3) var brdf_lut: texture_2d<f32>;
@group(1) @binding(4) var ibl_sampler: sampler;

// Per-frame data shared by every primitive: view/projection, lights, camera.
// Split from the per-primitive Uniforms so the ~576 bytes of identical data
// are written ONCE per frame instead of per-primitive.
@group(2) @binding(0) var<uniform> frame: FrameUniforms;

/// Normal transform for the per-instance matrix: the COFACTOR of its upper 3x3.
///
/// The cofactor equals det(M) * inverse-transpose, and the result is normalized
/// straight afterwards, so the determinant factor drops out - which makes this
/// both cheaper than a real inverse and finite for singular instance matrices
/// (a zero-scale instance no longer produces NaN normals). Columns of the
/// cofactor are the cross products of the other two columns.
fn instance_cofactor(in: VsIn) -> mat3x3<f32> {
    let m = instance_matrix(in);
    let c0 = m[0].xyz;
    let c1 = m[1].xyz;
    let c2 = m[2].xyz;
    return mat3x3<f32>(cross(c1, c2), cross(c2, c0), cross(c0, c1));
}

/// Linear blend skinning; returns the model matrix to use for this vertex.
fn skin_matrix(in: VsIn) -> mat4x4<f32> {
    let w = in.weights;
    let total = w.x + w.y + w.z + w.w;
    if (total <= 0.0001) {
        return prim.model;
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
    @location(6) vertex_color: vec4<f32>,
    @location(7) uv1: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // Instance transform applies in world space, AFTER the model matrix, so
    // an instanced primitive keeps its own authored transform and is then
    // placed by the instance.
    let model = instance_matrix(in) * skin_matrix(in);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    out.clip_position = frame.view_proj * world_pos;
    // Skinned normals use the skinning matrix (uniform scale assumed);
    // unskinned vertices keep the precomputed normal matrix.
    if (in.weights.x + in.weights.y + in.weights.z + in.weights.w > 0.0001) {
        out.world_normal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    } else {
        // The instance matrix must transform the normal too, or instanced copies
        // keep the un-instanced orientation's lighting - but a normal is NOT
        // transformed by the matrix itself. Applying the raw instance matrix is
        // only correct for rotation plus uniform scale; under non-uniform or
        // mirrored instance scale it shears normals off the surface, which is
        // exactly the scattered/squashed foliage-and-debris case instancing
        // exists for. (The tangent path above IS right to use `model`: tangents
        // are direction vectors and do transform by the matrix.)
        let n_model = (prim.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz;
        out.world_normal = normalize(instance_cofactor(in) * n_model);
    }
    out.world_tangent = vec4<f32>(
        normalize((model * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
        in.tangent.w,
    );
    out.uv = in.uv;
    out.light_space_pos = frame.light_space * world_pos;
    out.world_position = world_pos.xyz;
    out.view_depth = distance(world_pos.xyz, frame.camera_position.xyz);
    out.vertex_color = in.color;
    out.uv1 = in.uv1;
    return out;
}

// Which cascade the CURRENT shadow pass renders. A static per-cascade buffer
// bound at group(1), NOT a field written into the shared uniforms per pass:
// Queue.write_buffer executes before every queued command at submit, so
// rewriting a shared field per cascade inside one encoder hands EVERY pass
// the last value - all three layers were being rendered with cascade 2's
// matrix while the fragment stage projected with 0/1/2. The visible symptom
// was shadows that existed but sat coarse and slightly wrong; the structural
// tests could not tell.
@group(1) @binding(0) var<uniform> shadow_cascade_index: vec4<u32>;

// Depth-only variant for the shadow pass (no fragment stage).
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    // Instanced casters must shadow from their instance position, not the
    // authored one - otherwise every copy casts the original's shadow.
    let world = instance_matrix(in) * skin_matrix(in) * vec4<f32>(in.position, 1.0);
    let cascade = shadow_cascade_index.x;
    if (cascade == 1u) {
        return frame.light_space_1 * world;
    }
    if (cascade == 2u) {
        return frame.light_space_2 * world;
    }
    return frame.light_space * world;
}

// Per-pixel alpha-tested shadow variant for MASK materials. The depth-only
// vs_shadow casts the shadow of the full geometry a cut-out is modelled as
// (a foliage card shadows as its quad); this pair samples the base-color
// alpha and discards below the cutoff, so the cut-out's SHAPE shadows.
// Reuses the main pass's binding slots - the masked shadow bind group binds
// only 0 (uniforms), 3/4 (base color), 13 (joints).
struct VsShadowMaskedOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow_masked(in: VsIn) -> VsShadowMaskedOut {
    var out: VsShadowMaskedOut;
    let world = instance_matrix(in) * skin_matrix(in) * vec4<f32>(in.position, 1.0);
    let cascade = shadow_cascade_index.x;
    if (cascade == 1u) {
        out.pos = frame.light_space_1 * world;
    } else if (cascade == 2u) {
        out.pos = frame.light_space_2 * world;
    } else {
        out.pos = frame.light_space * world;
    }
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_shadow_masked(in: VsShadowMaskedOut) {
    // Same KHR_texture_transform the forward pass applies to this slot.
    let uv = vec2<f32>(
        prim.base_uv_row0.x * in.uv.x + prim.base_uv_row0.y * in.uv.y + prim.base_uv_row0.z,
        prim.base_uv_row1.x * in.uv.x + prim.base_uv_row1.y * in.uv.y + prim.base_uv_row1.z,
    );
    // emissive_factor.w carries the MASK cutoff (0 = never discard).
    let alpha = prim.base_color.a
        * textureSampleLevel(base_color_tex, base_color_sampler, uv, 0.0).a;
    if (alpha < prim.emissive_factor.w) {
        discard;
    }
}

fn shadow_factor(view_depth: f32, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    // Cascade selection by view distance.
    var cascade = 0;
    if (view_depth > frame.cascade_splits.x) {
        cascade = 1;
    }
    if (view_depth > frame.cascade_splits.y) {
        cascade = 2;
    }
    let count = i32(frame.cascade_splits.z);
    if (cascade > count - 1) {
        cascade = count - 1;
    }

    var light_space_pos: vec4<f32>;
    if (cascade == 1) {
        light_space_pos = frame.light_space_1 * vec4<f32>(world_pos, 1.0);
    } else if (cascade == 2) {
        light_space_pos = frame.light_space_2 * vec4<f32>(world_pos, 1.0);
    } else {
        light_space_pos = frame.light_space * vec4<f32>(world_pos, 1.0);
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
        let l = normalize(frame.light_dir_ambient.xyz);
        let cos_sun = max(dot(dir, l), 0.0);
        let sun = pow(cos_sun, 1200.0) * 24.0 + pow(cos_sun, 48.0) * 0.5;
        color += vec3<f32>(1.0, 0.95, 0.85) * sun
            * (frame.light_color_intensity.w * 0.4);
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
    @builtin(position) frag_coord: vec4<f32>,
) -> vec3<f32> {
    var total = vec3<f32>(0.0);

    // Determine tile for this fragment and read per-tile light list.
    let tile_x = u32(frag_coord.x) / 16u;
    let tile_y = u32(frag_coord.y) / 16u;
    // tile_counts (width, height) are in frame.cascade_splits.zw.
    // Fallback: use all lights when tile grid is unavailable.
    let tile_w = u32(frame.cascade_splits.z);
    let tile_h = u32(frame.cascade_splits.w);
    var tile_count = 0i32;
    var tile_offset = 0u;

    // Clamp tile index to valid range.
    let tx = min(tile_x, max(tile_w, 1u) - 1u);
    let ty = min(tile_y, max(tile_h, 1u) - 1u);
    let tile_index = ty * tile_w + tx;

    let grid_entry = tile_light_grid[tile_index];
    let count = i32(grid_entry.x);
    tile_offset = grid_entry.y;

    let total_lights = i32(frame.camera_position.w);
    // If tile grid is empty or invalid, iterate all lights as fallback.
    if (count <= 0 || tile_w == 0u) {
        // Full iteration fallback uses the global count.
        tile_count = total_lights;
    } else {
        tile_count = count;
    }

    for (var j = 0u; j < u32(abs(tile_count)); j = j + 1u) {
        let light_idx = select(j, tile_light_indices[tile_offset + j], count > 0);
        let a = punctual_lights[light_idx * 4u];
        let b = punctual_lights[light_idx * 4u + 1u];
        let cvec = punctual_lights[light_idx * 4u + 2u];
        let dvec = punctual_lights[light_idx * 4u + 3u];
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
    // Per-slot UV set: material_flags.y is a bitmask (bit 0 base .. 4 occlusion);
    // a set bit selects TEXCOORD_1 (baked AO on UV1 is the standard export).
    let uv_mask = u32(prim.material_flags.y);
    let base_in = select(in.uv, in.uv1, (uv_mask & 1u) != 0u);
    let mr_in = select(in.uv, in.uv1, (uv_mask & 2u) != 0u);
    let normal_in = select(in.uv, in.uv1, (uv_mask & 4u) != 0u);
    let emissive_in = select(in.uv, in.uv1, (uv_mask & 8u) != 0u);
    let occlusion_in = select(in.uv, in.uv1, (uv_mask & 16u) != 0u);

    // All implicit-derivative samples up front, in uniform control flow.
    // KHR_texture_transform applies to the base slot's chosen set.
    let base_uv = vec2<f32>(
        prim.base_uv_row0.x * base_in.x + prim.base_uv_row0.y * base_in.y
            + prim.base_uv_row0.z,
        prim.base_uv_row1.x * base_in.x + prim.base_uv_row1.y * base_in.y
            + prim.base_uv_row1.z,
    );
    let base_sample = textureSample(base_color_tex, base_color_sampler, base_uv);
    let mr_sample = textureSample(metal_rough_tex, metal_rough_sampler, mr_in);
    let normal_sample = textureSample(normal_tex, normal_sampler, normal_in);
    let emissive_sample = textureSample(emissive_tex, emissive_sampler, emissive_in);
    let occlusion_sample = textureSample(occlusion_tex, occlusion_sampler, occlusion_in);

    // glTF: COLOR_0 multiplies the base color factor and texture.
    let albedo = prim.base_color * base_sample * in.vertex_color;
    // MASK alpha mode: emissive_factor.w carries the cutoff (0 = keep all).
    if (albedo.a < prim.emissive_factor.w) {
        discard;
    }
    // KHR_materials_unlit: emit the base color and stop. The spec defines the
    // result as exactly base_color, with no lighting, IBL, shadowing or
    // emissive contribution - which is the whole point of the extension, so
    // this returns BEFORE any of that is computed rather than multiplying it
    // out afterwards. The alpha cutoff above still applies.
    if (prim.material_flags.x > 0.5) {
        return albedo;
    }
    // glTF: metallic in B, roughness in G.
    let metallic = clamp(prim.material_factors.x * mr_sample.b, 0.0, 1.0);
    let roughness = clamp(prim.material_factors.y * mr_sample.g, 0.045, 1.0);
    let occlusion = mix(1.0, occlusion_sample.r, prim.material_factors.z);
    let emissive = prim.emissive_factor.rgb * emissive_sample.rgb;

    // TBN normal mapping (glTF convention: +Z out of the surface).
    let n_geom = normalize(in.world_normal);
    let t = normalize(in.world_tangent.xyz - n_geom * dot(n_geom, in.world_tangent.xyz));
    let b = cross(n_geom, t) * in.world_tangent.w;
    var n_ts = normal_sample.xyz * 2.0 - 1.0;
    n_ts = vec3<f32>(n_ts.xy * prim.material_factors.w, n_ts.z);
    let n = normalize(mat3x3<f32>(t, b, n_geom) * n_ts);

    let l = normalize(frame.light_dir_ambient.xyz);
    let v = normalize(frame.camera_position.xyz - in.world_position);
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
    let radiance = frame.light_color_intensity.rgb * frame.light_color_intensity.w;

    // IBL. Two paths: a prefiltered environment when one is bound, and the
    // analytic sky/ground approximation otherwise.
    //
    // Both are evaluated every fragment and chosen between with `select`, not
    // `mix`: `mix(a, b, 0.0)` is a * 1 + b * 0, which is only bit-identical to
    // `a` if the compiler is generous, while `select` picks one operand
    // untouched. That matters because the no-environment path must render
    // exactly what it rendered before this feature existed - every golden test
    // in the suite depends on it.
    //
    // The cost of evaluating both is three texture fetches from 1x1 textures
    // on the fallback path. The alternative, branching, would either need
    // uniform control flow around the samples (they use explicit LODs, so it
    // is legal, but Chrome's validator has historically been strict about
    // cube samples in divergent flow) or a second pipeline.
    let ibl_strength = frame.light_dir_ambient.w;
    let k_s_ibl = fresnel_schlick(n_dot_v, f0);
    let reflected = reflect(-v, n);

    let env_enabled = ibl_params.enabled_maxmip_intensity.x > 0.5;
    let env_intensity = ibl_params.enabled_maxmip_intensity.z;

    // Prefiltered mip is roughness * max_mip, the inverse of the bake's
    // roughness = mip / (mip_count - 1) (see render::ibl).
    let irradiance_env =
        textureSampleLevel(irradiance_map, ibl_sampler, n, 0.0).rgb * env_intensity;
    let prefiltered = textureSampleLevel(
        prefiltered_map,
        ibl_sampler,
        reflected,
        roughness * ibl_params.enabled_maxmip_intensity.y,
    ).rgb * env_intensity;
    // u = N.V, v = roughness; r is the scale on F0, g the bias.
    let env_brdf =
        textureSampleLevel(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;

    let irradiance_analytic = hemisphere_irradiance(n);
    let irradiance = select(irradiance_analytic, irradiance_env, env_enabled);
    let diffuse_ibl = (vec3<f32>(1.0) - k_s_ibl) * (1.0 - metallic) * albedo.rgb * irradiance;

    let env_analytic = mix(
        sky_radiance(reflected, true),
        irradiance_analytic,
        roughness * roughness,
    );
    let specular_analytic = env_analytic * env_brdf_approx(f0, roughness, n_dot_v);
    let specular_split_sum = prefiltered * (f0 * env_brdf.x + vec3<f32>(env_brdf.y));
    let specular_ibl = select(specular_analytic, specular_split_sum, env_enabled);
    let ambient = ibl_strength * occlusion * (diffuse_ibl + specular_ibl);

    let punctual =
        punctual_lighting(in.world_position, n, v, albedo.rgb, metallic, roughness, f0, in.position);

    let color =
        (diffuse + specular) * radiance * n_dot_l * shadow + punctual + ambient + emissive;
    return vec4<f32>(color, albedo.a);
}
