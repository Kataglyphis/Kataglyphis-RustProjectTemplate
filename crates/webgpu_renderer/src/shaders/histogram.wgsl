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

// ---------------------------------------------------------------------------
// Reduction: histogram -> adapted exposure, entirely on the GPU.
//
// Deliberately never reads back to the CPU. A per-frame readback would
// serialise the pipeline behind a map_async round trip; the exposure the
// tonemap needs can just stay in a buffer the tonemap reads.
//
// Mirrors render::auto_exposure::{average_luminance, exposure_ev_for_luminance,
// adapt_exposure_ev}, all of which are unit-tested on the CPU side.

const EXPOSURE_KEY: f32 = 0.18;

struct ExposureParams {
    // x: seconds since the previous frame
    // y: adaptation rate constant (0 disables adaptation)
    // z: 1 when auto-exposure is enabled, else 0
    // w: manual exposure EV, used when auto is off
    values: vec4<f32>,
};

// The reduction reads the SAME storage binding the build pass writes, via
// atomicLoad, rather than aliasing it to a second read-only binding. wgpu
// rejects that: STORAGE_READ_WRITE is an exclusive usage and cannot coexist
// with STORAGE_READ_ONLY for one buffer inside a dispatch scope.
@group(0) @binding(2) var<storage, read_write> exposure_state: array<f32, 2>;
@group(0) @binding(3) var<uniform> exposure_params: ExposureParams;

// Inverse of histogram_bin, evaluated at the bin centre.
fn bin_luminance(bin: u32) -> f32 {
    if (bin == 0u) {
        return 0.0;
    }
    let index = f32(bin - 1u) + 0.5;
    let normalized = index / f32(HISTOGRAM_BINS - 2u);
    return exp2(MIN_LOG_LUMINANCE + normalized * (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE));
}

// One invocation: 64 bins is far too little work to justify a parallel
// reduction, and a single thread removes any question of shared-memory
// barriers being right.
@compute @workgroup_size(1, 1, 1)
fn cs_reduce_exposure() {
    let dt = exposure_params.values.x;
    let speed = exposure_params.values.y;
    let auto_enabled = exposure_params.values.z;
    let manual_ev = exposure_params.values.w;

    if (auto_enabled < 0.5) {
        // Manual mode still writes the buffer, so the tonemap has one source
        // of truth and switching modes cannot leave a stale auto value behind.
        exposure_state[0] = manual_ev;
        exposure_state[1] = manual_ev;
        return;
    }

    // Geometric mean over the non-black bins.
    var weighted_log_sum = 0.0;
    var counted = 0.0;
    for (var bin = 1u; bin < HISTOGRAM_BINS; bin = bin + 1u) {
        let count = f32(atomicLoad(&histogram[bin]));
        if (count > 0.0) {
            weighted_log_sum = weighted_log_sum + log2(bin_luminance(bin)) * count;
            counted = counted + count;
        }
    }

    let current_ev = exposure_state[0];

    if (counted <= 0.0) {
        // Nothing but black: hold the previous exposure rather than adapt to
        // a meaningless number. Deriving one here would drive exposure to
        // infinity on the first frame of a scene that has not loaded yet.
        exposure_state[1] = current_ev;
        return;
    }

    let average_luminance = exp2(weighted_log_sum / counted);
    let target_ev = log2(EXPOSURE_KEY / average_luminance);

    // Exponential, not a plain lerp: a lerp converges twice as fast at 120 Hz
    // as at 60, so a speed tuned on one machine is wrong on every other.
    var adapted = target_ev;
    if (dt > 0.0 && speed > 0.0) {
        let blend = clamp(1.0 - exp(-dt * speed), 0.0, 1.0);
        adapted = current_ev + (target_ev - current_ev) * blend;
    }

    exposure_state[0] = adapted;
    exposure_state[1] = target_ev;
}
