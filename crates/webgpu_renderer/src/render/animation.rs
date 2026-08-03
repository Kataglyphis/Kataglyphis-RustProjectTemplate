//! Pure keyframe sampling over glTF animation tracks.
//!
//! Step / linear / cubic-spline interpolation for translation, scale,
//! rotation and morph-weight channels. No GPU types, no `ForwardRenderer` —
//! callers own retargeting the sampled values onto nodes/primitives.

use glam::{Quat, Vec3, Vec4};

use crate::scene::Interpolation;

/// Returns `(i0, i1, frac, td)`: the bracketing keyframe indices, the lerp
/// fraction in `[0,1]` (clamped away from a zero-length span), and the true
/// segment duration `times[i1] - times[i0]` (0.0 when `i0 == i1`). `td` is
/// the glTF CUBICSPLINE tangent scale and must be the *unclamped* span —
/// callers must not reuse `frac`'s clamped span for it.
pub(crate) fn keyframe_lerp_indices(times: &[f32], t: f32) -> (usize, usize, f32, f32) {
    if times.is_empty() {
        return (0, 0, 0.0, 0.0);
    }
    if t <= times[0] {
        return (0, 0, 0.0, 0.0);
    }
    if t >= *times.last().unwrap() {
        let last = times.len() - 1;
        return (last, last, 0.0, 0.0);
    }
    // `times` is sorted (glTF requires it), so binary-search the first index
    // whose time is >= t; the bracketing segment is the one just before it.
    let i = times.partition_point(|&x| x < t) - 1;
    let span = times[i + 1] - times[i];
    (i, i + 1, (t - times[i]) / span.max(1e-6), span)
}

/// glTF CUBICSPLINE Hermite basis weights for (value0, out_tangent0, value1,
/// in_tangent1) at local time `t` in [0,1] over a segment of duration `td`.
/// The tangent weights are scaled by `td` per the glTF spec.
pub(crate) fn cubic_spline_weights(t: f32, td: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        td * (t3 - 2.0 * t2 + t),
        -2.0 * t3 + 3.0 * t2,
        td * (t3 - t2),
    )
}

/// Sample a Vec3 channel (translation/scale) between keyframes `i0` and `i1` at
/// fraction `frac`, honoring the interpolation mode. `dt` is the segment's time
/// span (used by CubicSpline). For CubicSpline the array is 3x `times`
/// (in-tangent, value, out-tangent per keyframe).
pub(crate) fn sample_vec3(
    values: &[Vec3],
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Option<Vec3> {
    match interp {
        Interpolation::Step => values.get(i0).copied(),
        Interpolation::Linear => Some(values.get(i0)?.lerp(*values.get(i1)?, frac)),
        Interpolation::CubicSpline => {
            let p0 = *values.get(3 * i0 + 1)?;
            let m0 = *values.get(3 * i0 + 2)?;
            let m1 = *values.get(3 * i1)?;
            let p1 = *values.get(3 * i1 + 1)?;
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            Some(p0 * w0 + m0 * w1 + p1 * w2 + m1 * w3)
        }
    }
}

/// Sample a rotation channel. CubicSpline interpolates the quaternion components
/// with the Hermite basis and renormalizes (the glTF-spec approximation);
/// Linear uses slerp; Step holds the keyframe.
pub(crate) fn sample_quat(
    values: &[Quat],
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Option<Quat> {
    match interp {
        Interpolation::Step => values.get(i0).copied(),
        Interpolation::Linear => Some(values.get(i0)?.slerp(*values.get(i1)?, frac)),
        Interpolation::CubicSpline => {
            let p0 = *values.get(3 * i0 + 1)?;
            let m0 = *values.get(3 * i0 + 2)?;
            let m1 = *values.get(3 * i1)?;
            let p1 = *values.get(3 * i1 + 1)?;
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            let v4 = |q: Quat| Vec4::new(q.x, q.y, q.z, q.w);
            let v = v4(p0) * w0 + v4(m0) * w1 + v4(p1) * w2 + v4(m1) * w3;
            Some(Quat::from_vec4(v.normalize()))
        }
    }
}

/// Sample a morph-weights channel: `n` weights per keyframe, returned as a Vec
/// of length `n`. Under CubicSpline the channel stores `3 * n` per keyframe as
/// three contiguous `n`-blocks (in-tangents, values, out-tangents), each target
/// Hermite-interpolated with the same basis the vector/quaternion paths use. On
/// any out-of-range index the weight falls back to 0 rather than panicking.
pub(crate) fn sample_morph_weights(
    values: &[f32],
    n: usize,
    interp: Interpolation,
    i0: usize,
    i1: usize,
    frac: f32,
    dt: f32,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let at = |idx: usize| values.get(idx).copied().unwrap_or(0.0);
    match interp {
        Interpolation::Step => (0..n).map(|k| at(i0 * n + k)).collect(),
        Interpolation::Linear => (0..n)
            .map(|k| {
                let a = at(i0 * n + k);
                let b = at(i1 * n + k);
                a + (b - a) * frac
            })
            .collect(),
        Interpolation::CubicSpline => {
            let (w0, w1, w2, w3) = cubic_spline_weights(frac, dt);
            (0..n)
                .map(|k| {
                    let p0 = at(i0 * 3 * n + n + k); // value at start
                    let m0 = at(i0 * 3 * n + 2 * n + k); // out-tangent at start
                    let m1 = at(i1 * 3 * n + k); // in-tangent at end
                    let p1 = at(i1 * 3 * n + n + k); // value at end
                    p0 * w0 + m0 * w1 + p1 * w2 + m1 * w3
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_spline_weights_collapse_to_the_keyframe_at_the_ends() {
        // At t=0 only the value0 weight is 1; at t=1 only the value1 weight is 1.
        let (a, b, c, d) = cubic_spline_weights(0.0, 3.0);
        assert!((a - 1.0).abs() < 1e-6 && b.abs() < 1e-6 && c.abs() < 1e-6 && d.abs() < 1e-6);
        let (a, b, c, d) = cubic_spline_weights(1.0, 3.0);
        assert!(a.abs() < 1e-6 && b.abs() < 1e-6 && (c - 1.0).abs() < 1e-6 && d.abs() < 1e-6);
    }

    #[test]
    fn cubic_spline_vec3_hits_keyframe_values_at_segment_ends() {
        // 2 keyframes, cubic layout: [in0, v0, out0, in1, v1, out1]. The tangents
        // are non-zero to prove the ends ignore them.
        let v0 = Vec3::new(1.0, 2.0, 3.0);
        let v1 = Vec3::new(4.0, 5.0, 6.0);
        let vals = vec![Vec3::ONE, v0, Vec3::splat(9.0), Vec3::splat(-7.0), v1, Vec3::ONE];
        let start = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 0.0, 2.0).unwrap();
        let end = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 1.0, 2.0).unwrap();
        assert!((start - v0).length() < 1e-5, "t=0 should be v0, got {start:?}");
        assert!((end - v1).length() < 1e-5, "t=1 should be v1, got {end:?}");
    }

    #[test]
    fn cubic_spline_vec3_zero_tangents_reduce_to_smoothstep_midpoint() {
        // With zero tangents the Hermite basis is h00·v0 + h01·v1; at t=0.5 both
        // are 0.5, so the midpoint is the linear midpoint.
        let vals = vec![Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ONE, Vec3::ZERO];
        let mid = sample_vec3(&vals, Interpolation::CubicSpline, 0, 1, 0.5, 1.0).unwrap();
        assert!(
            (mid - Vec3::splat(0.5)).length() < 1e-5,
            "zero-tangent cubic midpoint should be 0.5, got {mid:?}"
        );
    }

    #[test]
    fn step_holds_and_linear_interpolates_vec3() {
        let vals = vec![Vec3::ZERO, Vec3::splat(2.0)];
        assert_eq!(
            sample_vec3(&vals, Interpolation::Step, 0, 1, 0.9, 1.0).unwrap(),
            Vec3::ZERO,
            "Step must hold the left keyframe"
        );
        let lin = sample_vec3(&vals, Interpolation::Linear, 0, 1, 0.5, 1.0).unwrap();
        assert!((lin - Vec3::splat(1.0)).length() < 1e-6);
    }

    #[test]
    fn cubic_spline_quat_ends_are_the_normalized_keyframes() {
        let q0 = Quat::from_rotation_y(0.3);
        let q1 = Quat::from_rotation_y(1.1);
        let vals = vec![Quat::IDENTITY, q0, Quat::IDENTITY, Quat::IDENTITY, q1, Quat::IDENTITY];
        let start = sample_quat(&vals, Interpolation::CubicSpline, 0, 1, 0.0, 1.0).unwrap();
        let end = sample_quat(&vals, Interpolation::CubicSpline, 0, 1, 1.0, 1.0).unwrap();
        // dot ~= 1 means (near-)identical orientation (sign-agnostic).
        assert!(start.dot(q0).abs() > 0.999, "t=0 should match q0");
        assert!(end.dot(q1).abs() > 0.999, "t=1 should match q1");
        assert!((start.length() - 1.0).abs() < 1e-5, "result must be a unit quaternion");
    }

    #[test]
    fn morph_weights_linear_and_step_stride_two_targets() {
        // 2 targets, 2 keyframes, flattened [k0t0, k0t1, k1t0, k1t1].
        let vals = vec![0.0, 1.0, 1.0, 3.0];
        let lin = sample_morph_weights(&vals, 2, Interpolation::Linear, 0, 1, 0.5, 1.0);
        assert_eq!(lin, vec![0.5, 2.0], "each target lerps independently");
        let step = sample_morph_weights(&vals, 2, Interpolation::Step, 0, 1, 0.9, 1.0);
        assert_eq!(step, vec![0.0, 1.0], "Step holds the left keyframe per target");
    }

    #[test]
    fn morph_weights_cubic_hits_keyframe_values_at_ends() {
        // 1 target, 2 keyframes, cubic layout is [in, value, out] per keyframe:
        // [in0, v0, out0, in1, v1, out1] with non-zero tangents to prove the
        // ends ignore them.
        let vals = vec![9.0, 0.2, -4.0, 7.0, 0.8, -3.0];
        let start = sample_morph_weights(&vals, 1, Interpolation::CubicSpline, 0, 1, 0.0, 1.0);
        let end = sample_morph_weights(&vals, 1, Interpolation::CubicSpline, 0, 1, 1.0, 1.0);
        assert!((start[0] - 0.2).abs() < 1e-5, "t=0 is value0, got {}", start[0]);
        assert!((end[0] - 0.8).abs() < 1e-5, "t=1 is value1, got {}", end[0]);
    }

    #[test]
    fn keyframe_lookup_matches_a_linear_scan_over_many_times() {
        // Irregularly spaced sorted times.
        let times: Vec<f32> = (0..64)
            .map(|i| i as f32 * 0.37 + (i as f32 * 1.7).sin() * 0.1)
            .scan(0.0, |acc, x| {
                *acc += x.max(0.01);
                Some(*acc)
            })
            .collect();

        fn linear_scan(times: &[f32], t: f32) -> (usize, usize, f32) {
            if times.is_empty() {
                return (0, 0, 0.0);
            }
            if t <= times[0] {
                return (0, 0, 0.0);
            }
            if t >= *times.last().unwrap() {
                let last = times.len() - 1;
                return (last, last, 0.0);
            }
            let mut i = 0;
            while i + 1 < times.len() && times[i + 1] < t {
                i += 1;
            }
            let span = (times[i + 1] - times[i]).max(1e-6);
            (i, i + 1, (t - times[i]) / span)
        }

        let mut probes: Vec<f32> = vec![times[0] - 1.0, *times.last().unwrap() + 1.0];
        for &kt in &times {
            probes.push(kt);
        }
        for w in times.windows(2) {
            probes.push((w[0] + w[1]) * 0.5);
        }

        for t in probes {
            let (i0, i1, frac, _dt) = keyframe_lerp_indices(&times, t);
            let (ei0, ei1, efrac) = linear_scan(&times, t);
            assert_eq!((i0, i1), (ei0, ei1), "index mismatch at t={t}");
            assert!(
                (frac - efrac).abs() < 1e-6,
                "frac mismatch at t={t}: got {frac}, expected {efrac}"
            );
        }
    }

    #[test]
    fn keyframe_lookup_returns_the_segment_duration() {
        let times = [0.0, 1.0, 3.0, 6.0];
        let (i0, i1, _frac, dt) = keyframe_lerp_indices(&times, 2.0);
        assert_eq!((i0, i1), (1, 2));
        assert!((dt - (times[2] - times[1])).abs() < 1e-6);

        let (i0, i1, _frac, dt) = keyframe_lerp_indices(&times, -1.0);
        assert_eq!((i0, i1), (0, 0));
        assert_eq!(dt, 0.0);

        let (i0, i1, _frac, dt) = keyframe_lerp_indices(&times, 10.0);
        assert_eq!((i0, i1), (3, 3));
        assert_eq!(dt, 0.0);
    }

    #[test]
    fn morph_weights_out_of_range_falls_back_to_zero() {
        // A malformed/short channel must not panic; missing samples read 0.
        let vals = vec![0.5];
        let w = sample_morph_weights(&vals, 3, Interpolation::Linear, 0, 1, 0.5, 1.0);
        assert_eq!(w.len(), 3);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert!(sample_morph_weights(&vals, 0, Interpolation::Linear, 0, 1, 0.5, 1.0).is_empty());
    }
}
