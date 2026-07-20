// Split-sum image-based lighting precompute: equirectangular HDR -> cubemap,
// cosine-convolved irradiance, GGX-prefiltered specular, and the environment
// BRDF lookup table.
//
// Every entry point is a fragment stage over a fullscreen triangle writing one
// cube face (a single array layer of the cube texture) or the 2D LUT. Compute
// with storage textures was the obvious alternative and was rejected: WebGPU
// core has no `rg16float` storage format, so the BRDF LUT would have needed a
// second mechanism anyway, and `rgba16float` storage is write-only - the
// downsample chain reads its own previous mip. Render passes do all four with
// one pipeline shape.
//
// Sampling of the source *equirect* uses textureLoad plus a hand-written
// bilinear filter rather than a sampler. The source is Rgba32Float and
// float32-filterable is an optional WebGPU feature the context does not
// request (see context.rs); loading and lerping keeps the web target intact
// without a half-float conversion on the CPU side.

const PI: f32 = 3.14159265359;

struct Params {
    // x: cube face index, y: roughness, z: sample count, w: source mip level
    face_roughness_samples_mip: vec4<f32>,
    // x: source cube resolution in texels (for the prefilter mip heuristic)
    source_resolution: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src_equirect: texture_2d<f32>;
@group(0) @binding(2) var src_cube: texture_cube<f32>;
@group(0) @binding(3) var src_sampler: sampler;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

/// Fullscreen triangle; `uv` is 0..1 across the target with y running down,
/// matching the render target's top-left origin.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var out: VsOut;
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

/// World direction of a point on cube face `face` at `uv`.
///
/// The face basis is the D3D/Vulkan/WebGPU cube convention (+X, -X, +Y, -Y,
/// +Z, -Z in layer order). Getting a face's handedness wrong here mirrors that
/// face only, which is invisible on a smooth environment and glaring on a
/// structured one - hence the seam test in tests/ibl.rs.
fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let a = 2.0 * uv.x - 1.0;
    let b = 1.0 - 2.0 * uv.y;
    var dir: vec3<f32>;
    switch face {
        case 0u: { dir = vec3<f32>(1.0, b, -a); }
        case 1u: { dir = vec3<f32>(-1.0, b, a); }
        case 2u: { dir = vec3<f32>(a, 1.0, -b); }
        case 3u: { dir = vec3<f32>(a, -1.0, b); }
        case 4u: { dir = vec3<f32>(a, b, 1.0); }
        default: { dir = vec3<f32>(-a, b, -1.0); }
    }
    return normalize(dir);
}

/// Equirectangular UV for a world direction: u is longitude, v is latitude
/// with +Y at v = 0.
fn equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let longitude = atan2(dir.z, dir.x);
    let latitude = asin(clamp(dir.y, -1.0, 1.0));
    return vec2<f32>(longitude / (2.0 * PI) + 0.5, 0.5 - latitude / PI);
}

/// Bilinear fetch from the equirect source.
///
/// Longitude wraps and latitude clamps, which is what an equirect map means
/// geometrically. A ClampToEdge sampler on both axes would darken a seam one
/// texel wide down the -X meridian of every derived map.
fn sample_equirect(dir: vec3<f32>) -> vec3<f32> {
    let dims = vec2<i32>(textureDimensions(src_equirect, 0));
    let texel = equirect_uv(dir) * vec2<f32>(dims) - 0.5;
    let base = floor(texel);
    let frac = texel - base;

    let x0 = i32(base.x);
    let y0 = i32(base.y);
    let xa = ((x0 % dims.x) + dims.x) % dims.x;
    let xb = (((x0 + 1) % dims.x) + dims.x) % dims.x;
    let ya = clamp(y0, 0, dims.y - 1);
    let yb = clamp(y0 + 1, 0, dims.y - 1);

    let c00 = textureLoad(src_equirect, vec2<i32>(xa, ya), 0).rgb;
    let c10 = textureLoad(src_equirect, vec2<i32>(xb, ya), 0).rgb;
    let c01 = textureLoad(src_equirect, vec2<i32>(xa, yb), 0).rgb;
    let c11 = textureLoad(src_equirect, vec2<i32>(xb, yb), 0).rgb;
    return mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);
}

@fragment
fn fs_equirect_to_cube(in: VsOut) -> @location(0) vec4<f32> {
    let dir = cube_direction(u32(params.face_roughness_samples_mip.x), in.uv);
    return vec4<f32>(sample_equirect(dir), 1.0);
}

/// One mip of the environment cube from the one above it.
///
/// A target texel at mip m sits exactly on the corner shared by its four
/// parents at mip m-1, so a single bilinear tap IS their box average - no
/// explicit 2x2 gather needed. The cube projection makes that only
/// approximately true near face corners, which is well inside the error a
/// downsample chain is allowed.
@fragment
fn fs_downsample_cube(in: VsOut) -> @location(0) vec4<f32> {
    let dir = cube_direction(u32(params.face_roughness_samples_mip.x), in.uv);
    return textureSampleLevel(src_cube, src_sampler, dir, params.face_roughness_samples_mip.w);
}

/// Cosine-weighted convolution of the environment.
///
/// Midpoint rule over (phi, theta) rather than the usual `for (phi = 0; phi <
/// 2PI; phi += delta)` accumulation: with a fixed delta the loop bound does
/// not divide the interval, and the left-endpoint bias makes a constant
/// environment convolve to ~0.997 of itself. That 0.3% is small enough to
/// eyeball as correct and large enough to hide a real weighting bug, so the
/// quadrature is exact-count and centred instead. Measured: a constant
/// environment now round-trips to within 0.03%.
///
/// The stored value is E / PI - the quantity that multiplies albedo directly,
/// which is what every real-time IBL implementation puts in this map. So a
/// constant environment of radiance L stores L, not PI * L.
const IRRADIANCE_PHI_STEPS: u32 = 128u;
const IRRADIANCE_THETA_STEPS: u32 = 64u;

@fragment
fn fs_irradiance(in: VsOut) -> @location(0) vec4<f32> {
    let normal = cube_direction(u32(params.face_roughness_samples_mip.x), in.uv);

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    var irradiance = vec3<f32>(0.0);
    for (var i = 0u; i < IRRADIANCE_PHI_STEPS; i = i + 1u) {
        let phi = 2.0 * PI * (f32(i) + 0.5) / f32(IRRADIANCE_PHI_STEPS);
        for (var j = 0u; j < IRRADIANCE_THETA_STEPS; j = j + 1u) {
            let theta = 0.5 * PI * (f32(j) + 0.5) / f32(IRRADIANCE_THETA_STEPS);
            let sin_theta = sin(theta);
            let cos_theta = cos(theta);
            let local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
            let world = local.x * tangent + local.y * bitangent + local.z * normal;
            // cos(theta) is the Lambert term, sin(theta) the solid angle of
            // the (phi, theta) parameterisation. Dropping the sin is THE
            // classic bug here: it over-weights the pole and the result stops
            // being the environment's average even for a constant one.
            irradiance += textureSampleLevel(src_cube, src_sampler, world, 0.0).rgb
                * cos_theta * sin_theta;
        }
    }
    let samples = f32(IRRADIANCE_PHI_STEPS * IRRADIANCE_THETA_STEPS);
    return vec4<f32>(PI * irradiance / samples, 1.0);
}

/// Van der Corput radical inverse, base 2.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(count), radical_inverse_vdc(i));
}

/// GGX/Trowbridge-Reitz importance sample around `normal`.
fn importance_sample_ggx(xi: vec2<f32>, normal: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    // Both clamps are load-bearing, not defensive noise. As roughness -> 0,
    // a*a underflows against 1.0 and the ratio becomes (1 - xi.y) / (1 - xi.y),
    // which rounds to just above 1 for about half the sample sequence; the
    // unclamped sqrt(1 - cos^2) is then sqrt(-epsilon) = NaN, `n_dot_l <= 0`
    // is false for NaN in some backends and true in others, and the sample is
    // silently dropped. Measured before the clamp: the mirror row of the BRDF
    // LUT summed to 0.563 instead of 1.0 - a 44% energy loss that looks
    // exactly like a plausible-but-dark material.
    let cos_theta = sqrt(clamp((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y), 0.0, 1.0));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let half_local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.z) < 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(half_local.x * tangent + half_local.y * bitangent + half_local.z * normal);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-6);
}

/// Prefiltered specular radiance for one roughness (one mip of the cube).
///
/// The split-sum approximation's first half. N = V = R is the standard
/// simplification: it drops the stretched grazing highlight, which is the
/// price of storing the result in a cubemap indexed only by reflection vector.
@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let normal = cube_direction(u32(params.face_roughness_samples_mip.x), in.uv);
    let roughness = params.face_roughness_samples_mip.y;
    let sample_count = u32(params.face_roughness_samples_mip.z);
    let resolution = params.source_resolution.x;

    // Mip 0 is the environment itself. Importance sampling at roughness 0
    // degenerates to sample_count identical taps, so this is not an
    // optimisation but the exact answer, cheaper.
    if (roughness <= 0.0) {
        return textureSampleLevel(src_cube, src_sampler, normal, 0.0);
    }

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley(i, sample_count);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        // V = N, so L is the reflection of N about H.
        let light = normalize(2.0 * dot(normal, half_vector) * half_vector - normal);
        let n_dot_l = dot(normal, light);
        if (n_dot_l <= 0.0) {
            continue;
        }

        // Sample the environment at a mip whose texel solid angle matches the
        // sample's, per Krivanek/Colbert. Without this a high-contrast
        // environment (a sun disc, a bright window) aliases into a spray of
        // fireflies that no amount of samples removes.
        // V = N here, so H.V is H.N and the usual D * NdotH / (4 * HdotV)
        // collapses to D / 4.
        let n_dot_h = max(dot(normal, half_vector), 0.0);
        let pdf = distribution_ggx(n_dot_h, roughness) * 0.25 + 1e-4;
        let texel_solid_angle = 4.0 * PI / (6.0 * resolution * resolution);
        let sample_solid_angle = 1.0 / (f32(sample_count) * pdf);
        let mip = max(0.5 * log2(sample_solid_angle / texel_solid_angle), 0.0);

        color += textureSampleLevel(src_cube, src_sampler, light, mip).rgb * n_dot_l;
        total_weight += n_dot_l;
    }
    return vec4<f32>(color / max(total_weight, 1e-4), 1.0);
}

/// Smith geometry term, IBL k (r*r/2), not the direct-lighting k.
fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g1 * g2;
}

/// The split-sum environment BRDF: scale and bias on F0.
///
/// u = N.V, v = roughness, both at texel centres - so N.V never reaches 0,
/// where the term is singular.
const BRDF_SAMPLE_COUNT: u32 = 1024u;

@fragment
fn fs_brdf_lut(in: VsOut) -> @location(0) vec2<f32> {
    let n_dot_v = in.uv.x;
    let roughness = in.uv.y;

    let view = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let normal = vec3<f32>(0.0, 0.0, 1.0);

    var scale = 0.0;
    var bias = 0.0;
    for (var i = 0u; i < BRDF_SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, BRDF_SAMPLE_COUNT);
        let half_vector = importance_sample_ggx(xi, normal, roughness);
        let light = normalize(2.0 * dot(view, half_vector) * half_vector - view);

        let n_dot_l = max(light.z, 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }
        let n_dot_h = max(half_vector.z, 0.0);
        let v_dot_h = max(dot(view, half_vector), 0.0);

        let g = geometry_smith_ibl(max(n_dot_v, 1e-4), n_dot_l, roughness);
        let g_vis = (g * v_dot_h) / max(n_dot_h * max(n_dot_v, 1e-4), 1e-6);
        // Fresnel factored into (1 - Fc) * F0 + Fc, so the integral splits
        // into a scale on F0 and an F0-independent bias.
        let fc = pow(1.0 - v_dot_h, 5.0);
        scale += (1.0 - fc) * g_vis;
        bias += fc * g_vis;
    }
    return vec2<f32>(scale, bias) / f32(BRDF_SAMPLE_COUNT);
}
