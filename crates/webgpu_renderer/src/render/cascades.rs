//! Pure-CPU shadow-cascade fitting, split out of `forward.rs`.
//!
//! The shader picks a cascade by *eye distance*
//! (`forward.slang:151`, `o.viewDepth = distance(worldPos, camera_position)`,
//! interpolated from per-vertex values - see below) and compares it against
//! `frame.cascade_splits.xy` (`forward.slang:198-211`). Those splits
//! therefore have to live in eye-distance units, and each cascade's
//! light-space box has to cover whatever the shader can route to it. The
//! previous `update_cascades` sized ITS boxes, and picked its splits, from
//! the scene's own radius alone - a quantity that does not move with the
//! camera. Zoom the camera out and every fragment's real eye distance
//! outgrows both (fixed) splits, so cascade selection always lands on the
//! last cascade and the two near ones are never sampled.
//!
//! `fit_cascades` keeps the box *shapes* unchanged (a near box hugging the
//! camera focus, a mid box around the orbit target, a far box around the
//! whole scene - `light_matrix_for` is untouched), but derives the near/mid
//! *radii* - and therefore the splits, which are `2 * radius` - from the
//! camera's actual distance to the scene, floored at the scene's own radius.
//! The floor matters: `viewDepth` is only ever linearly interpolated from
//! per-vertex distances (not recomputed per-fragment), which is a poor
//! approximation on a large, coarsely-triangulated receiver - measured
//! against `tests/assets/cube_on_plane.gltf`'s single-quad ground plane, the
//! interpolated value can be off by several units. The old scene-radius
//! sizing happened to be generous enough to absorb that error for every
//! camera position the existing GPU golden tests exercise (all of which sit
//! within the scene's own radius); an earlier version of this module that
//! additionally re-derived box *position/size* from the exact camera frustum
//! shrank the boxes enough to expose it and regressed
//! `shadow_darkens_plane_under_cube` to zero shadowed pixels. Flooring
//! `dist_to_center` at `scene_radius` reproduces the old (already-correct)
//! sizing exactly whenever the camera sits at or inside the scene's radius,
//! and only grows the boxes once the camera moves farther out than that -
//! which is exactly the situation the bug report describes.

use glam::{Mat4, Vec3};

use crate::render::forward::CASCADE_COUNT;
use crate::scene::camera::OrbitCamera;

pub(crate) struct CascadeFit {
    pub splits: [f32; 2],
    pub matrices: [Mat4; CASCADE_COUNT],
}

/// Fits one orthographic light matrix per cascade: cascade 0 hugs the camera
/// focus (crisp near shadows), cascade 1 sits around the orbit target,
/// cascade 2 covers the whole scene. `light_dir` is a parameter rather than
/// read off a renderer so this is callable from a plain unit test, without a
/// GPU or a `ForwardRenderer`.
pub(crate) fn fit_cascades(camera: &OrbitCamera, scene_min: Vec3, scene_max: Vec3, light_dir: Vec3) -> CascadeFit {
    let scene_center = (scene_min + scene_max) * 0.5;
    let mut scene_radius = ((scene_max - scene_min).length() * 0.5).max(1e-3);
    if !scene_radius.is_finite() {
        scene_radius = 1.0;
    }

    // See the module doc comment for why this is floored at scene_radius
    // rather than used raw.
    let dist_to_center = (scene_center - camera.eye()).length().max(scene_radius);

    let near_radius = (dist_to_center * 0.35).max(0.5);
    let mid_radius = (dist_to_center * 0.7).max(1.0);
    let splits = [near_radius * 2.0, mid_radius * 2.0];

    let focus_near = camera.target.lerp(camera.eye(), 0.15);
    let cascades = [(focus_near, near_radius), (camera.target, mid_radius), (scene_center, scene_radius)];

    let mut matrices = [Mat4::IDENTITY; CASCADE_COUNT];
    for (i, (center, radius)) in cascades.into_iter().enumerate() {
        matrices[i] = light_matrix_for(center, radius, light_dir);
    }

    CascadeFit { splits, matrices }
}

/// One cascade's orthographic light matrix, fitted to a world-space sphere
/// (`center`, `radius`). Pull the light eye back along `-light_dir` far
/// enough that the sphere sits entirely in front of it, then bound a square
/// ortho box around it.
fn light_matrix_for(center: Vec3, radius: f32, light_dir: Vec3) -> Mat4 {
    let light_dir = light_dir.normalize_or_zero();
    let light_dir = if light_dir == Vec3::ZERO { Vec3::Y } else { light_dir };
    let up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    // Pull the eye far back so casters outside the cascade box still fit in
    // the depth range.
    let eye = center + light_dir * (radius * 4.0);
    let view = Mat4::look_at_rh(eye, center, up);
    let projection = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 8.0);
    projection * view
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zoomed_out_camera() -> OrbitCamera {
        OrbitCamera {
            radius: 200.0,
            ..OrbitCamera::default()
        }
    }

    #[test]
    fn splits_track_the_camera_instead_of_staying_scene_fixed() {
        // THE bug this module fixes: splits derived purely from scene radius
        // do not move when the camera does, so a camera far from the scene
        // eventually has every fragment's eye distance outgrow both splits
        // and cascade selection gets stuck on the last cascade. A correct
        // fit pushes the splits out roughly in step with eye distance.
        let scene_min = Vec3::splat(-1.0);
        let scene_max = Vec3::splat(1.0);
        let light_dir = Vec3::Y;

        let near_fit = fit_cascades(&OrbitCamera::default(), scene_min, scene_max, light_dir);
        let far_fit = fit_cascades(&zoomed_out_camera(), scene_min, scene_max, light_dir);

        assert!(
            far_fit.splits[0] > near_fit.splits[0] * 10.0,
            "far camera's first split ({}) must grow with eye distance, not stay near the close camera's ({})",
            far_fit.splits[0],
            near_fit.splits[0]
        );
        assert!(
            far_fit.splits[1] > far_fit.splits[0],
            "splits must stay ordered: {:?}",
            far_fit.splits
        );
    }

    #[test]
    fn a_camera_inside_the_scene_radius_matches_the_original_scene_radius_derived_split() {
        // Regression pin for `shadow_darkens_plane_under_cube` /
        // `first_frame_uses_the_correct_cascade_and_tile_counts`: those GPU
        // golden tests use a camera whose distance to the scene center is
        // LESS than the scene's own radius, which must fall back to exactly
        // the original (already-correct, already shadow-map-verified)
        // scene-radius-derived sizing.
        let scene_min = Vec3::new(-4.0, 0.0, -4.0);
        let scene_max = Vec3::new(4.0, 1.4, 4.0);
        let camera = OrbitCamera {
            radius: 6.0,
            pitch_deg: 55.0,
            ..OrbitCamera::default()
        };
        let scene_center = (scene_min + scene_max) * 0.5;
        let scene_radius = (scene_max - scene_min).length() * 0.5;
        assert!(
            (scene_center - camera.eye()).length() < scene_radius,
            "precondition: this camera must sit inside the scene's own radius"
        );

        let fit = fit_cascades(&camera, scene_min, scene_max, Vec3::new(-1.0, 0.7, -0.3));

        let expected_near = (scene_radius * 0.35).max(0.5);
        let expected_mid = (scene_radius * 0.7).max(1.0);
        assert!(
            (fit.splits[0] - expected_near * 2.0).abs() < 1e-3,
            "got {}, expected {}",
            fit.splits[0],
            expected_near * 2.0
        );
        assert!(
            (fit.splits[1] - expected_mid * 2.0).abs() < 1e-3,
            "got {}, expected {}",
            fit.splits[1],
            expected_mid * 2.0
        );
    }

    #[test]
    fn cascade_matrices_stay_finite_and_non_degenerate() {
        let scene_min = Vec3::splat(-1.0);
        let scene_max = Vec3::splat(1.0);
        for camera in [OrbitCamera::default(), zoomed_out_camera()] {
            let fit = fit_cascades(&camera, scene_min, scene_max, Vec3::new(-0.55, -1.0, -0.35));
            for (i, m) in fit.matrices.iter().enumerate() {
                assert!(m.is_finite(), "cascade {i} matrix is not finite: {m:?}");
                assert!(m.determinant().abs() > 1e-12, "cascade {i} matrix collapsed");
            }
        }
    }
}
