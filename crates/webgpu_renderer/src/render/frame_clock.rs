//! Wall-clock frame pacing for time-dependent effects (currently
//! [`crate::render::forward::ForwardRenderer::frame_delta_seconds`], which
//! drives auto-exposure adaptation).
//!
//! The clamping and "no previous tick yet" logic is the part worth testing;
//! the timestamp itself is not, so [`FrameClock::tick_at`] takes an injected
//! seconds-since-some-epoch value and never touches a real clock. Callers
//! (the native `viewer` example, the wasm demo) supply that value from
//! whatever monotonic source their platform has - `Instant` natively,
//! `Performance::now()` on wasm, where `std::time::Instant` panics.

/// Upper bound on a reported delta. A suspended tab or a debugger pause must
/// not slam whatever adapts on this value (auto-exposure) with a
/// multi-second step.
pub const MAX_DELTA_SECONDS: f32 = 0.25;

/// Reported before the first real tick, matching the nominal rate
/// `frame_delta_seconds` defaulted to before any caller wrote it.
pub const NOMINAL_DELTA_SECONDS: f32 = 1.0 / 60.0;

/// Turns a monotonic seconds-since-epoch reading into a clamped
/// frame-to-frame delta.
#[derive(Default)]
pub struct FrameClock {
    last_seconds: Option<f64>,
}

impl FrameClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// `now_seconds` must come from a monotonic source (e.g. an `Instant`
    /// fixed at startup, elapsed as `as_secs_f64()`, or
    /// `Performance::now() / 1000.0`). Not required to be strictly
    /// increasing: a backwards or repeated reading clamps to `0.0` rather
    /// than producing a negative delta.
    pub fn tick_at(&mut self, now_seconds: f64) -> f32 {
        let delta = match self.last_seconds {
            None => NOMINAL_DELTA_SECONDS,
            Some(last) => ((now_seconds - last) as f32).clamp(0.0, MAX_DELTA_SECONDS),
        };
        self.last_seconds = Some(now_seconds);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_tick_is_nominal() {
        let mut clock = FrameClock::new();
        assert_eq!(clock.tick_at(1_234.0), NOMINAL_DELTA_SECONDS);
    }

    #[test]
    fn consecutive_ticks_match_the_injected_gap() {
        let mut clock = FrameClock::new();
        clock.tick_at(10.0);
        assert!((clock.tick_at(10.016) - 0.016).abs() < 1e-6);
        assert!((clock.tick_at(10.1) - (10.1 - 10.016)).abs() < 1e-6);
    }

    #[test]
    fn a_five_second_gap_clamps_instead_of_slamming_the_caller() {
        let mut clock = FrameClock::new();
        clock.tick_at(0.0);
        assert_eq!(clock.tick_at(5.0), MAX_DELTA_SECONDS);
    }

    #[test]
    fn a_backwards_or_repeated_reading_clamps_to_zero_not_negative() {
        // A coarse or glitching timer must not hand adapt_exposure_ev a
        // negative delta.
        let mut clock = FrameClock::new();
        clock.tick_at(10.0);
        assert_eq!(clock.tick_at(9.5), 0.0);
        assert_eq!(clock.tick_at(9.5), 0.0);
    }
}
