//! Auto-exposure maths: luminance histogram binning, average extraction, and
//! temporal adaptation.
//!
//! Deliberately free functions over plain numbers, with no wgpu types. The GPU
//! side of auto-exposure is a compute shader writing a histogram and a second
//! pass reducing it, and neither is testable without a device - but the parts
//! that are actually easy to get wrong are all here:
//!
//! - log-space binning, where an off-by-one or a wrong base silently shifts
//!   every measurement;
//! - the empty/black-scene case, where a naive average is 0 and the derived
//!   exposure is +infinity;
//! - adaptation direction and framerate independence, where "it converges"
//!   and "it converges at the same speed on a 30 Hz and a 300 Hz machine" are
//!   different properties and only one of them is obvious.

/// Number of histogram bins. 64 is enough resolution for exposure decisions
/// (each bin spans ~0.2 EV over the default range) while staying a single
/// workgroup's worth of shared memory on the GPU side.
pub const HISTOGRAM_BINS: usize = 64;

/// Luminance range the histogram covers, in log2 units. Values outside are
/// clamped into the end bins rather than dropped: a scene brighter than the
/// range should read as "very bright", not as "no data".
pub const MIN_LOG_LUMINANCE: f32 = -10.0;
pub const MAX_LOG_LUMINANCE: f32 = 4.0;

/// Middle grey. The exposure that maps average scene luminance onto this is
/// what "correctly exposed" means here.
pub const EXPOSURE_KEY: f32 = 0.18;

/// Which histogram bin a linear luminance falls into.
///
/// Bin 0 is reserved for "effectively black". Without that, near-zero
/// luminance dominates the log-space average in any scene with background -
/// log2 of a tiny number is a large negative that drags the mean down and
/// blows the exposure up.
pub fn histogram_bin(luminance: f32) -> usize {
    if luminance.is_nan() || luminance <= 1e-6 {
        return 0;
    }
    let log_luminance = luminance.log2();
    let normalized = ((log_luminance - MIN_LOG_LUMINANCE)
        / (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE))
        .clamp(0.0, 1.0);
    // Bins 1..HISTOGRAM_BINS-1 carry the actual range; scale into that span.
    1 + ((normalized * (HISTOGRAM_BINS - 2) as f32) as usize).min(HISTOGRAM_BINS - 2)
}

/// Representative linear luminance for a bin, i.e. the inverse of
/// [`histogram_bin`] evaluated at the bin's centre.
pub fn bin_luminance(bin: usize) -> f32 {
    if bin == 0 {
        return 0.0;
    }
    let index = (bin - 1) as f32 + 0.5;
    let normalized = index / (HISTOGRAM_BINS - 2) as f32;
    let log_luminance = MIN_LOG_LUMINANCE + normalized * (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE);
    log_luminance.exp2()
}

/// Geometric mean luminance of a histogram, ignoring the black bin.
///
/// Returns `None` when nothing was sampled or every sample was black, so the
/// caller can hold the previous exposure rather than adapt to a meaningless
/// number. Returning 0.0 here would push exposure to infinity on the first
/// frame of a scene that has not finished loading.
pub fn average_luminance(histogram: &[u32]) -> Option<f32> {
    let mut weighted_log_sum = 0.0f64;
    let mut counted = 0u64;

    for (bin, &count) in histogram.iter().enumerate().skip(1) {
        if count == 0 {
            continue;
        }
        let luminance = bin_luminance(bin);
        if luminance <= 0.0 {
            continue;
        }
        weighted_log_sum += f64::from(luminance.log2()) * f64::from(count);
        counted += u64::from(count);
    }

    if counted == 0 {
        return None;
    }
    Some(((weighted_log_sum / counted as f64) as f32).exp2())
}

/// Exposure multiplier that maps `average_luminance` onto middle grey.
pub fn exposure_for_luminance(average_luminance: f32) -> f32 {
    // Guard the divide: a caller that ignored average_luminance's None and
    // passed 0 would otherwise get infinity, and infinity times any colour is
    // a NaN frame.
    if average_luminance.is_nan() || average_luminance <= 1e-6 {
        return 1.0;
    }
    EXPOSURE_KEY / average_luminance
}

/// Same, expressed in EV so it can be compared against the manual slider.
pub fn exposure_ev_for_luminance(average_luminance: f32) -> f32 {
    exposure_for_luminance(average_luminance).log2()
}

/// Moves `current_ev` toward `target_ev` at a rate independent of framerate.
///
/// `speed` is the exponential rate constant: higher adapts faster. The
/// framerate independence is the point - a naive `current + (target -
/// current) * speed` converges roughly twice as fast at 120 Hz as at 60 Hz,
/// so a value tuned on one machine is wrong on every other.
pub fn adapt_exposure_ev(
    current_ev: f32,
    target_ev: f32,
    delta_time_seconds: f32,
    speed: f32,
) -> f32 {
    if !current_ev.is_finite() {
        return target_ev;
    }
    if delta_time_seconds <= 0.0 || speed <= 0.0 {
        return current_ev;
    }
    let blend = 1.0 - (-delta_time_seconds * speed).exp();
    current_ev + (target_ev - current_ev) * blend.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_and_near_black_land_in_the_reserved_bin() {
        assert_eq!(histogram_bin(0.0), 0);
        assert_eq!(histogram_bin(1e-9), 0);
        assert_eq!(
            histogram_bin(-1.0),
            0,
            "negative luminance is not a value, not a dark value"
        );
        assert_eq!(histogram_bin(f32::NAN), 0, "NaN must not index a bin");
    }

    #[test]
    fn brighter_luminance_never_lands_in_a_lower_bin() {
        let mut previous = histogram_bin(1e-5);
        for step in 0..200 {
            let luminance = 1e-4 * 1.15f32.powi(step);
            let bin = histogram_bin(luminance);
            assert!(
                bin >= previous,
                "bin went down from {previous} to {bin} at luminance {luminance}"
            );
            assert!(bin < HISTOGRAM_BINS, "bin {bin} is out of range");
            previous = bin;
        }
    }

    #[test]
    fn out_of_range_luminance_clamps_instead_of_wrapping() {
        // Above the range must read as the brightest bin, not wrap to a dark
        // one - an overexposed scene that reports "dark" would drive exposure
        // the wrong way and stay overexposed.
        assert_eq!(histogram_bin(1e9), HISTOGRAM_BINS - 1);
        // Below the range but not black: the darkest non-black bin.
        assert_eq!(histogram_bin(1e-5), 1);
    }

    #[test]
    fn bin_luminance_round_trips_through_histogram_bin() {
        for bin in 1..HISTOGRAM_BINS - 1 {
            let luminance = bin_luminance(bin);
            assert_eq!(
                histogram_bin(luminance),
                bin,
                "bin {bin} -> luminance {luminance} -> bin {}",
                histogram_bin(luminance)
            );
        }
    }

    #[test]
    fn empty_or_all_black_histograms_report_no_measurement() {
        assert_eq!(average_luminance(&[0u32; HISTOGRAM_BINS]), None);

        let mut only_black = [0u32; HISTOGRAM_BINS];
        only_black[0] = 10_000;
        assert_eq!(
            average_luminance(&only_black),
            None,
            "a fully black frame must report no measurement, not an average of zero"
        );
    }

    #[test]
    fn average_of_a_single_populated_bin_is_that_bin() {
        let mut histogram = [0u32; HISTOGRAM_BINS];
        let bin = 30;
        histogram[bin] = 500;

        let average = average_luminance(&histogram).expect("a populated histogram has an average");
        let expected = bin_luminance(bin);
        assert!(
            (average - expected).abs() / expected < 0.01,
            "average {average} should be bin {bin}'s luminance {expected}"
        );
    }

    #[test]
    fn average_is_geometric_so_one_bright_pixel_does_not_dominate() {
        // A dim scene with a single blown-out highlight. An arithmetic mean
        // would be dragged up by the highlight and underexpose everything
        // else; the geometric mean is what makes exposure track the bulk of
        // the image.
        let dim_bin = 20;
        let bright_bin = HISTOGRAM_BINS - 2;
        let mut histogram = [0u32; HISTOGRAM_BINS];
        histogram[dim_bin] = 10_000;
        histogram[bright_bin] = 1;

        let average = average_luminance(&histogram).unwrap();
        let dim = bin_luminance(dim_bin);
        assert!(
            average < dim * 2.0,
            "one bright sample moved the average from {dim} to {average}"
        );
    }

    #[test]
    fn exposure_maps_average_luminance_onto_middle_grey() {
        for &luminance in &[0.02f32, 0.18, 1.0, 40.0] {
            let exposure = exposure_for_luminance(luminance);
            let exposed = luminance * exposure;
            assert!(
                (exposed - EXPOSURE_KEY).abs() < 1e-4,
                "luminance {luminance} exposed to {exposed}, expected {EXPOSURE_KEY}"
            );
        }
    }

    #[test]
    fn a_dark_scene_brightens_and_a_bright_scene_darkens() {
        // Direction, stated as a property rather than a magic number.
        assert!(
            exposure_ev_for_luminance(0.01) > 0.0,
            "a dark scene must expose up"
        );
        assert!(
            exposure_ev_for_luminance(10.0) < 0.0,
            "a bright scene must expose down"
        );
        assert!(
            exposure_ev_for_luminance(EXPOSURE_KEY).abs() < 1e-4,
            "a scene already at middle grey needs no correction"
        );
    }

    #[test]
    fn zero_luminance_yields_a_finite_exposure() {
        // The guard that stops a not-yet-loaded scene producing a NaN frame.
        assert!(exposure_for_luminance(0.0).is_finite());
        assert!(exposure_ev_for_luminance(0.0).is_finite());
    }

    #[test]
    fn adaptation_converges_toward_the_target() {
        let mut ev = 0.0f32;
        let target = 3.0f32;
        for _ in 0..600 {
            ev = adapt_exposure_ev(ev, target, 1.0 / 60.0, 3.0);
        }
        assert!((ev - target).abs() < 0.01, "did not converge: {ev}");
    }

    #[test]
    fn adaptation_is_framerate_independent() {
        // The property a naive lerp fails. Same wall-clock time, different
        // step sizes, must land in the same place.
        let target = 4.0f32;
        let seconds = 0.5f32;

        // A wide step-count spread on purpose. With 30 vs 300 steps a naive
        // lerp lands within 0.03 of the correct answer, which is too close to
        // float noise to assert on; 8 vs 2000 makes the two formulations
        // disagree by more than an EV.
        let mut slow = 0.0f32;
        for _ in 0..8 {
            slow = adapt_exposure_ev(slow, target, seconds / 8.0, 2.5);
        }

        let mut fast = 0.0f32;
        for _ in 0..2000 {
            fast = adapt_exposure_ev(fast, target, seconds / 2000.0, 2.5);
        }

        assert!(
            (slow - fast).abs() < 0.02,
            "8 steps gave {slow}, 2000 steps gave {fast} over the same {seconds}s"
        );
    }

    #[test]
    fn adaptation_never_overshoots_or_runs_backwards() {
        // A large dt must not push past the target and oscillate.
        let overshoot = adapt_exposure_ev(0.0, 5.0, 1000.0, 10.0);
        assert!(
            (0.0..=5.0 + 1e-4).contains(&overshoot),
            "overshot to {overshoot}"
        );

        // Downward adaptation is the same property in the other direction.
        let downward = adapt_exposure_ev(5.0, 0.0, 1000.0, 10.0);
        assert!((-1e-4..=5.0).contains(&downward), "undershot to {downward}");
    }

    #[test]
    fn a_stalled_frame_or_disabled_adaptation_holds_the_current_value() {
        assert_eq!(
            adapt_exposure_ev(1.5, 4.0, 0.0, 3.0),
            1.5,
            "dt 0 must not move exposure"
        );
        assert_eq!(
            adapt_exposure_ev(1.5, 4.0, 0.016, 0.0),
            1.5,
            "speed 0 disables adaptation"
        );
    }

    #[test]
    fn a_non_finite_current_value_recovers_instead_of_propagating() {
        // If exposure ever becomes NaN the frame is lost; adaptation must be
        // able to climb out rather than staying NaN forever.
        assert_eq!(adapt_exposure_ev(f32::NAN, 2.0, 0.016, 3.0), 2.0);
        assert_eq!(adapt_exposure_ev(f32::INFINITY, 2.0, 0.016, 3.0), 2.0);
    }
}
