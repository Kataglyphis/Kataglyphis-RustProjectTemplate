// Luminance histogram over the HDR target, for auto-exposure.
//
// The binning here MUST match render::auto_exposure::histogram_bin on the CPU
// side. That function is unit-tested (monotonicity, clamping, the reserved
// black bin, round-tripping through bin_luminance) and a headless test
// compares this shader's output against it - so the duplication is checked
// rather than hoped at.
//
// Bin 0 is reserved for "effectively black". Without it, near-zero luminance
// dominates the log-space average in any scene with background: log2 of a
// tiny number is a large negative that drags the mean down and blows the
// exposure up.

const HISTOGRAM_BINS: u32 = 64u;
const MIN_LOG_LUMINANCE: f32 = -10.0;
const MAX_LOG_LUMINANCE: f32 = 4.0;
const BLACK_THRESHOLD: f32 = 1e-6;

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 64>;

// Rec. 709 luma, matching the CPU-side luminance used by the golden tests.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn histogram_bin(lum: f32) -> u32 {
    // `!(lum > threshold)` rather than `lum <= threshold` so NaN lands in bin
    // 0 instead of indexing out of range.
    if (!(lum > BLACK_THRESHOLD)) {
        return 0u;
    }
    let log_luminance = log2(lum);
    let normalized = clamp(
        (log_luminance - MIN_LOG_LUMINANCE) / (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE),
        0.0,
        1.0,
    );
    let span = f32(HISTOGRAM_BINS - 2u);
    return 1u + min(u32(normalized * span), HISTOGRAM_BINS - 2u);
}

// One thread per pixel. 16x16 keeps a workgroup within the 256-invocation
// limit that the WebGPU baseline guarantees.
@compute @workgroup_size(16, 16, 1)
fn cs_build_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(hdr_tex);
    // Dispatches are rounded up to whole workgroups, so the tail threads fall
    // outside the image and must not contribute - otherwise the clamped edge
    // texel is counted repeatedly and biases the exposure.
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let color = textureLoad(hdr_tex, vec2<i32>(i32(id.x), i32(id.y)), 0).rgb;
    let bin = histogram_bin(luminance(color));
    atomicAdd(&histogram[bin], 1u);
}

// Zeroes the histogram. A separate entry point rather than a queue write so
// the reset stays on the GPU timeline and cannot race the build pass.
@compute @workgroup_size(64, 1, 1)
fn cs_clear_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < HISTOGRAM_BINS) {
        atomicStore(&histogram[id.x], 0u);
    }
}
