/// Minimal deterministic LCG (linear congruential generator) PRNG.
///
/// Used across burn demos for reproducible random data generation without
/// pulling in a full `rand` dependency. **Not** cryptographically secure.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return a raw `u32`.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32)
    }

    /// Approximate standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
