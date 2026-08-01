//! Punctual-light packing: the host side of the storage buffer
//! `forward.slang` reads as `punctualLights`.
//!
//! Each light occupies four consecutive `vec4` rows, `[lightIdx * 4 .. + 4]`:
//!
//! - row 0 (`a`): `position.xyz`, `kind` (`1.0` point, `2.0` spot, `3.0`
//!   directional) in `.w`
//! - row 1 (`b`): `color * intensity` in `.rgb`, `range` in `.w`
//! - row 2 (`cvec`): `direction.xyz`, `cos_inner` in `.w`
//! - row 3 (`dvec`): `cos_outer` in `.x`, unused `.yzw`
//!
//! `forward.slang`'s `for` loop over `punctualLights` (around line 317-345)
//! is the decoder: it reads `kind` from row `a.w` (`> 2.5` directional,
//! `> 1.5` spot) and spot cone attenuation as
//! `smoothstep(dvec.x, cvec.w, cosAngle)`, i.e. outer cosine from row 3's
//! `.x` and inner cosine from row 2's `.w`. Changing this layout means
//! changing both sides together.

use crate::scene::{CpuLight, CpuLightKind};

pub const MAX_PUNCTUAL_LIGHTS: usize = 256;

pub(crate) fn pack_punctual_lights(
    lights: &[CpuLight],
) -> ([[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4], u32) {
    let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
    let count = lights.len().min(MAX_PUNCTUAL_LIGHTS);
    for (i, light) in lights.iter().take(count).enumerate() {
        let (kind, cos_inner, cos_outer) = match light.kind {
            CpuLightKind::Point => (1.0, 0.0, 0.0),
            CpuLightKind::Spot {
                cos_inner,
                cos_outer,
            } => (2.0, cos_inner, cos_outer),
            CpuLightKind::Directional => (3.0, 0.0, 0.0),
        };
        let base = i * 4;
        packed[base] = [
            light.position[0],
            light.position[1],
            light.position[2],
            kind,
        ];
        packed[base + 1] = [
            light.color[0] * light.intensity,
            light.color[1] * light.intensity,
            light.color[2] * light.intensity,
            light.range,
        ];
        packed[base + 2] = [
            light.direction[0],
            light.direction[1],
            light.direction[2],
            cos_inner,
        ];
        packed[base + 3] = [cos_outer, 0.0, 0.0, 0.0];
    }
    (packed, count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_light() -> CpuLight {
        CpuLight {
            kind: CpuLightKind::Point,
            color: [1.0, 0.5, 0.25],
            intensity: 2.0,
            range: 10.0,
            position: [1.0, 2.0, 3.0],
            direction: [0.0, -1.0, 0.0],
        }
    }

    fn spot_light(cos_inner: f32, cos_outer: f32) -> CpuLight {
        CpuLight {
            kind: CpuLightKind::Spot {
                cos_inner,
                cos_outer,
            },
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 5.0,
            position: [0.0, 1.0, 0.0],
            direction: [0.0, -1.0, 0.0],
        }
    }

    fn directional_light() -> CpuLight {
        CpuLight {
            kind: CpuLightKind::Directional,
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            range: 0.0,
            position: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn point_spot_and_directional_pack_their_kind_discriminant() {
        let lights = [point_light(), spot_light(0.9, 0.8), directional_light()];
        let (packed, count) = pack_punctual_lights(&lights);
        assert_eq!(count, 3);

        // forward.slang: `kind > 2.5` is directional, else `kind > 1.5` is spot,
        // else point. The packed values must land on the correct side of both
        // thresholds, not merely equal 1.0/2.0/3.0.
        let point_kind = packed[0 * 4][3];
        let spot_kind = packed[1 * 4][3];
        let directional_kind = packed[2 * 4][3];

        assert_eq!(point_kind, 1.0);
        assert!(point_kind <= 1.5, "point must not be read as spot");

        assert_eq!(spot_kind, 2.0);
        assert!(spot_kind > 1.5, "spot must be read as spot, not point");
        assert!(spot_kind <= 2.5, "spot must not be read as directional");

        assert_eq!(directional_kind, 3.0);
        assert!(
            directional_kind > 2.5,
            "directional must be read as directional"
        );
    }

    #[test]
    fn intensity_is_premultiplied_into_the_colour_row() {
        let lights = [point_light()];
        let (packed, _) = pack_punctual_lights(&lights);
        let row1 = packed[1];
        assert_eq!(row1[0], 1.0 * 2.0);
        assert_eq!(row1[1], 0.5 * 2.0);
        assert_eq!(row1[2], 0.25 * 2.0);
        assert_eq!(row1[3], 10.0, "range lives in row 1's .w");
    }

    #[test]
    fn lights_beyond_the_cap_are_dropped_not_wrapped() {
        let lights: Vec<CpuLight> = (0..MAX_PUNCTUAL_LIGHTS + 5)
            .map(|i| CpuLight {
                position: [i as f32, 0.0, 0.0],
                ..point_light()
            })
            .collect();
        let (packed, count) = pack_punctual_lights(&lights);
        assert_eq!(count as usize, MAX_PUNCTUAL_LIGHTS);

        let last_base = (MAX_PUNCTUAL_LIGHTS - 1) * 4;
        assert_eq!(
            packed[last_base][0],
            (MAX_PUNCTUAL_LIGHTS - 1) as f32,
            "last packed record must be light index MAX_PUNCTUAL_LIGHTS - 1, not a wraparound over the front"
        );

        // Nothing beyond the cap was ever written.
        assert_eq!(packed[MAX_PUNCTUAL_LIGHTS * 4 - 4][3], 1.0);
    }

    #[test]
    fn spot_cone_angles_land_in_the_slots_smoothstep_reads() {
        let lights = [spot_light(0.9, 0.7)];
        let (packed, _) = pack_punctual_lights(&lights);
        // forward.slang: `smoothstep(dvec.x, cvec.w, cosAngle)` — dvec is row
        // base+3, cvec is row base+2.
        let cvec = packed[2];
        let dvec = packed[3];
        assert_eq!(cvec[3], 0.9, "cos_inner must be in packed[base+2][3]");
        assert_eq!(dvec[0], 0.7, "cos_outer must be in packed[base+3][0]");
    }
}
