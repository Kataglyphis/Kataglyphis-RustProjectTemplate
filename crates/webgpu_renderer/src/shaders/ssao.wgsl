// Depth-only SSAO: view-space position reconstruction from the depth
// buffer (textureLoad, no samplers, no derivatives - uniformity-safe on
// web), normals from depth neighbors, fixed hemisphere kernel, plus a 3x3
// box blur entry point. Runs at half resolution.

struct SsaoUniforms {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    // x: radius (view units), y: bias, z: intensity, w: unused
    params: vec4<f32>,
};

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> u: SsaoUniforms;

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

fn load_depth(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let coords = clamp(
        vec2<i32>(uv * dims),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    );
    return textureLoad(depth_tex, coords, 0);
}

fn view_pos_at(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let v = u.inv_proj * ndc;
    return v.xyz / v.w;
}

const KERNEL_SIZE: i32 = 12;
// Hemisphere (z > 0) sample directions, roughly cosine-distributed and
// scaled towards the center.
const KERNEL: array<vec3<f32>, 12> = array<vec3<f32>, 12>(
    vec3<f32>(0.204, 0.104, 0.301),
    vec3<f32>(-0.322, 0.170, 0.222),
    vec3<f32>(0.109, -0.372, 0.284),
    vec3<f32>(-0.114, -0.184, 0.451),
    vec3<f32>(0.451, 0.322, 0.339),
    vec3<f32>(-0.532, -0.281, 0.240),
    vec3<f32>(0.320, -0.564, 0.353),
    vec3<f32>(-0.310, 0.610, 0.346),
    vec3<f32>(0.671, -0.166, 0.474),
    vec3<f32>(-0.144, -0.702, 0.482),
    vec3<f32>(-0.658, 0.397, 0.514),
    vec3<f32>(0.442, 0.679, 0.531),
);

@fragment
fn fs_ssao(in: VsOut) -> @location(0) vec4<f32> {
    let depth = load_depth(in.uv);
    if (depth >= 1.0) {
        // Sky: fully unoccluded.
        return vec4<f32>(1.0);
    }
    let p = view_pos_at(in.uv, depth);

    let dims = vec2<f32>(textureDimensions(depth_tex));
    let du = vec2<f32>(1.0 / dims.x, 0.0);
    let dv = vec2<f32>(0.0, 1.0 / dims.y);
    let px = view_pos_at(in.uv + du, load_depth(in.uv + du));
    let py = view_pos_at(in.uv + dv, load_depth(in.uv + dv));
    let n = normalize(cross(px - p, py - p));

    // Orthonormal basis around the normal.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.98) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(up, n));
    let b = cross(n, t);

    let radius = u.params.x;
    let bias = u.params.y;
    var occlusion = 0.0;
    for (var i = 0; i < KERNEL_SIZE; i = i + 1) {
        let k = KERNEL[i];
        let dir = t * k.x + b * k.y + n * k.z;
        let sample_pos = p + dir * radius;

        let clip = u.proj * vec4<f32>(sample_pos, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let sample_uv = vec2<f32>(
            (clip.x / clip.w) * 0.5 + 0.5,
            0.5 - (clip.y / clip.w) * 0.5,
        );
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }
        let scene_z = view_pos_at(sample_uv, load_depth(sample_uv)).z;
        // View space looks down -Z: nearer surfaces have GREATER z.
        let range_check = smoothstep(0.0, 1.0, radius / max(abs(p.z - scene_z), 1e-4));
        if (scene_z >= sample_pos.z + bias) {
            occlusion = occlusion + range_check;
        }
    }
    let ao = 1.0 - u.params.z * (occlusion / f32(KERNEL_SIZE));
    return vec4<f32>(clamp(ao, 0.0, 1.0));
}

// 3x3 box blur over the raw AO (bound as depth_tex slot? no - separate
// entry using a regular texture). The blur pass binds the raw AO texture
// at binding 2 to keep one bind group layout.
@group(0) @binding(2) var ao_tex: texture_2d<f32>;

@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(ao_tex));
    let center = vec2<i32>(in.uv * dims);
    let max_coord = vec2<i32>(dims) - vec2<i32>(1);
    var total = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = clamp(center + vec2<i32>(x, y), vec2<i32>(0), max_coord);
            total = total + textureLoad(ao_tex, c, 0).r;
        }
    }
    return vec4<f32>(total / 9.0);
}
