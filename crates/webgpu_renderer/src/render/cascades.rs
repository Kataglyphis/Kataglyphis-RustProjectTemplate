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
//! whole scene), but derives the near/mid *radii* - and therefore the
//! splits, which are `2 * radius` - from the camera's actual distance to the
//! scene, floored at the scene's own radius and then QUANTIZED to powers of
//! two of that radius. The floor matters: `viewDepth` is only ever linearly
//! interpolated from per-vertex distances (not recomputed per-fragment),
//! which is a poor approximation on a large, coarsely-triangulated receiver -
//! measured against `tests/assets/cube_on_plane.gltf`'s single-quad ground
//! plane, the interpolated value can be off by several units. The old
//! scene-radius sizing happened to be generous enough to absorb that error
//! for every camera position the existing GPU golden tests exercise (all of
//! which sit within the scene's own radius); an earlier version of this
//! module that additionally re-derived box *position/size* from the exact
//! camera frustum shrank the boxes enough to expose it and regressed
//! `shadow_darkens_plane_under_cube` to zero shadowed pixels. Flooring
//! `dist_to_center` at `scene_radius` reproduces the old (already-correct)
//! sizing exactly whenever the camera sits at or inside the scene's radius,
//! and only grows the boxes once the camera moves farther out than that -
//! which is exactly the situation the bug report describes. The quantization
//! matters separately: sizing the near/mid radii from the RAW (continuous)
//! distance re-scales the texel grid on every dolly of the camera, which
//! defeats the whole-texel snap described below (ingredient 3) for exactly
//! the motion that snap exists to fix - a static shadow edge would still
//! crawl, just from the box's own size changing instead of its center.
//! Snapping distance to the nearest power-of-two multiple of `scene_radius`
//! (rounding away from the origin, so the floor case above stays bit-for-bit
//! exact) makes the box size a step function of camera distance instead: it
//! only changes at discrete zoom thresholds, where a one-frame reprojection
//! is an acceptable, rare cost, and is otherwise perfectly stable under
//! dollying.
//!
//! Each cascade's box is then STABILIZED against camera motion, mirroring
//! `CascadedShadowMapMath.cpp:154-201`. Three ingredients, each necessary:
//!
//! 1. A WORLD-FIXED light basis (pure rotation about the origin, built from
//!    `light_dir` alone). A basis anchored at a moving slice center absorbs
//!    camera translation continuously, and no snap applied afterwards can
//!    undo that.
//! 2. A box sized from each cascade's `radius` alone - a function of the
//!    slice geometry (see above), not of instantaneous camera position - so
//!    its texel footprint never changes as the camera turns, nor (thanks to
//!    the quantization above) as the camera dollies, except at the discrete
//!    zoom steps where a one-time reprojection is expected.
//! 3. The box center snapped to whole texels in that fixed basis, so the box
//!    only ever moves in texel increments and a static shadow edge always
//!    lands on the same texels. The box is padded by one texel of the box
//!    that is actually PROJECTED, not of the raw radius: `texel_world` must
//!    equal `2 * half_extent / shadow_map_size`, or the grid the center
//!    snaps to and the grid the box projects disagree and a "whole texel"
//!    snap drifts by a fraction of a texel every step. Solving
//!    `half_extent = radius + 2 * half_extent / shadow_map_size` gives
//!    `half_extent = radius * shadow_map_size / (shadow_map_size - 2)`.
//!
//! Depth (near/far) is deliberately NOT tightened to the sphere the way the
//! C++ version tightens to its exact frustum corners: per-cascade caster
//! culling is disabled here (every primitive is submitted to every cascade's
//! shadow pass, see `shadow_caster_bundle_is_cached_across_frames`), so a
//! caster can legitimately sit outside a given cascade's sphere and still
//! need to survive that cascade's depth range. A tight `±radius` bound
//! measurably clipped such casters (`caster_culling_engages_and_shadows_survive`
//! went from a real shadow to none), so the depth window stays proportional
//! to `radius` but generous - the same scale as the legacy eye-pulled-back
//! `radius * 4` / `far = radius * 8` window it replaces. Near may legitimately
//! come out negative (the basis is anchored at the origin, not behind the
//! scene); the orthographic constructor accepts that.

// glam 0.33 moved the camera constructors off `Mat4` and split them by clip-
// space convention. `directx` is glam's name for NDC Z in [0,1] with Y up —
// which is also wgpu's and Metal's — and it reproduces the old
// `Mat4::perspective_rh`/`orthographic_rh` bit for bit (verified 2026-08-07).
use glam::camera::rh::proj::directx as clip;
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec3};

use crate::render::forward::{CASCADE_COUNT, SHADOW_MAP_SIZE};
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
pub(crate) fn fit_cascades(
    camera: &OrbitCamera,
    scene_min: Vec3,
    scene_max: Vec3,
    light_dir: Vec3,
) -> CascadeFit {
    let scene_center = (scene_min + scene_max) * 0.5;
    let mut scene_radius = ((scene_max - scene_min).length() * 0.5).max(1e-3);
    if !scene_radius.is_finite() {
        scene_radius = 1.0;
    }

    // See the module doc comment for why this is floored at scene_radius
    // rather than used raw.
    let mut raw_dist = (scene_center - camera.eye()).length().max(scene_radius);
    if !raw_dist.is_finite() {
        raw_dist = scene_radius;
    }
    // Sizing the cascade from a CONTINUOUS camera distance re-scales the
    // texel grid every frame, which defeats the whole-texel snap in
    // stabilized_light_matrix_for. Quantize to powers of two of the scene's
    // own radius: the box then changes size only at discrete zoom steps
    // (where a one-frame reprojection is invisible) and is otherwise fixed.
    let mut steps = (raw_dist / scene_radius).log2().ceil().max(0.0);
    if !steps.is_finite() {
        steps = 0.0;
    }
    let dist_to_center = scene_radius * steps.exp2();

    let near_radius = (dist_to_center * 0.35).max(0.5);
    let mid_radius = (dist_to_center * 0.7).max(1.0);
    let splits = [near_radius * 2.0, mid_radius * 2.0];

    let focus_near = camera.target.lerp(camera.eye(), 0.15);
    let cascades = [
        (focus_near, near_radius),
        (camera.target, mid_radius),
        (scene_center, scene_radius),
    ];

    let light_dir = light_dir.normalize_or_zero();
    let light_dir = if light_dir == Vec3::ZERO {
        Vec3::Y
    } else {
        light_dir
    };
    let up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    // World-fixed: a pure rotation about the origin, independent of any
    // cascade's center. See the module doc comment, ingredient 1. Eye is
    // `light_dir` (not `-light_dir`, unlike the C++ reference): this crate's
    // `light_dir` already points FROM the surface TOWARD the light (the
    // legacy `light_matrix_for` placed its eye at `center + light_dir * d`,
    // i.e. on the light's side), whereas the C++ `lightDir` is the ray's
    // travel direction, the opposite convention.
    let light_basis = look_at_mat4(light_dir, Vec3::ZERO, up);

    let mut matrices = [Mat4::IDENTITY; CASCADE_COUNT];
    for (i, (center, radius)) in cascades.into_iter().enumerate() {
        matrices[i] = stabilized_light_matrix_for(light_basis, center, radius, SHADOW_MAP_SIZE);
    }

    CascadeFit { splits, matrices }
}

/// One cascade's orthographic light matrix, fitted to a world-space sphere
/// (`center`, `radius`) and stabilized in the given world-fixed `light_basis`
/// so it only ever moves in whole-texel increments. See the module doc
/// comment for why each step is necessary.
fn stabilized_light_matrix_for(
    light_basis: Mat4,
    center: Vec3,
    radius: f32,
    shadow_map_size: u32,
) -> Mat4 {
    // shadow_map_size <= 2 has no consistent solution (the pad would
    // swallow the whole box); fall back to the unsnapped tight fit.
    let (half_extent, texel_world) = if shadow_map_size > 2 {
        let half_extent = radius * shadow_map_size as f32 / (shadow_map_size as f32 - 2.0);
        let texel_world = (2.0 * half_extent) / shadow_map_size as f32;
        (half_extent, texel_world)
    } else {
        (radius, 0.0)
    };
    let center_ls = light_basis.transform_point3(center);
    let (snapped_x, snapped_y) = if texel_world > 0.0 {
        (
            (center_ls.x / texel_world).floor() * texel_world,
            (center_ls.y / texel_world).floor() * texel_world,
        )
    } else {
        (center_ls.x, center_ls.y)
    };

    // Depth range: unlike the C++ reference this cascade has no exact frustum
    // corners to bound, only a sphere approximation whose caster may sit
    // anywhere in the wider scene (per-cascade caster culling is disabled -
    // see `shadow_caster_bundle_is_cached_across_frames` - so every primitive
    // is submitted to every cascade's pass and relies on this range not
    // clipping it). Pad generously in proportion to `radius`, matching the
    // legacy eye-pulled-back-by-`radius * 4` / `far = radius * 8` window this
    // replaces (that window was symmetric around `center`, roughly `radius *
    // 4` either side) rather than tightening to just the sphere itself, which
    // measurably clips real casters - see `caster_culling_engages_and_shadows_survive`.
    let depth_pad = radius * 4.0;
    let near = -center_ls.z - depth_pad;
    let mut far = -center_ls.z + depth_pad;
    if far <= near {
        far = near + 1.0;
    }

    let projection = clip::orthographic(
        snapped_x - half_extent,
        snapped_x + half_extent,
        snapped_y - half_extent,
        snapped_y + half_extent,
        near,
        far,
    );
    projection * light_basis
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
                assert!(
                    m.determinant().abs() > 1e-12,
                    "cascade {i} matrix collapsed"
                );
            }
        }
    }

    /// Texel-space coordinate of a world point under one cascade's matrix:
    /// NDC scaled so one shadow-map texel is exactly 1.0. Whole-texel motion
    /// of the box shows up here as integer deltas. Mirrors
    /// `cascadedShadowMapSuite.cpp`'s `texel_space` helper.
    fn texel_space(view_proj: Mat4, world: Vec3) -> (f32, f32) {
        let clip = view_proj * world.extend(1.0);
        (
            (clip.x / clip.w) * (SHADOW_MAP_SIZE as f32 / 2.0),
            (clip.y / clip.w) * (SHADOW_MAP_SIZE as f32 / 2.0),
        )
    }

    #[test]
    fn cascades_shift_by_whole_texels_under_camera_motion() {
        // The shimmer bug: refitting the ortho box every frame moves every
        // shadow edge by whatever sub-texel amount the camera moved, and
        // edges crawl. Stabilized cascades may only move the box in WHOLE
        // texel increments, so a static world point's projected texel
        // position may only shift by a (near-)integer amount.
        let scene_min = Vec3::splat(-1.0);
        let scene_max = Vec3::splat(1.0);
        let light_dir = Vec3::new(-0.4, -1.0, -0.2);
        let probe = Vec3::new(0.2, 0.3, 0.1);

        let camera_a = OrbitCamera::default();
        // Sub-texel: the near cascade's texel size here is ~1.4e-3 world
        // units (radius ~1.4, SHADOW_MAP_SIZE 2048); this offset is well
        // under one texel.
        let camera_b = OrbitCamera {
            target: camera_a.target + Vec3::new(4e-4, 0.0, 0.0),
            ..camera_a.clone()
        };

        let fit_a = fit_cascades(&camera_a, scene_min, scene_max, light_dir);
        let fit_b = fit_cascades(&camera_b, scene_min, scene_max, light_dir);

        let (ax, ay) = texel_space(fit_a.matrices[0], probe);
        let (bx, by) = texel_space(fit_b.matrices[0], probe);
        let (dx, dy) = (ax - bx, ay - by);
        assert!(
            (dx - dx.round()).abs() < 5e-2,
            "x moved by a fractional texel: {dx}"
        );
        assert!(
            (dy - dy.round()).abs() < 5e-2,
            "y moved by a fractional texel: {dy}"
        );
    }

    #[test]
    fn cascades_stay_texel_aligned_over_long_camera_travel() {
        // The off-by-2/shadow_map_size bug: texel_world was computed from the
        // UNPADDED radius, but the box that is actually projected is
        // 2*half_extent wide (half_extent = radius + texel_world), so the
        // true texel size is texel_world * (1 + 2/shadow_map_size), not
        // texel_world. Snapping the center by k grid steps then moves a fixed
        // world point by only k/(1+2/shadow_map_size) texels - short of an
        // integer by ~2k/shadow_map_size texels. That error is invisible for
        // the sub-texel offset used above (k is 0 or 1), so move the box
        // center by well over one texel instead (the near cascade's texel
        // here is ~1.4e-3 world units, so 0.5 is hundreds of grid steps),
        // which accumulates the drift past the 5e-2 tolerance unless the fix
        // (deriving texel_world from the padded half_extent) is in place.
        //
        // Unlike `cascades_shift_by_whole_texels_under_camera_motion`, this
        // calls `stabilized_light_matrix_for` directly with a FIXED radius
        // rather than routing the travel through `fit_cascades`: `near_radius`
        // there scales with camera-to-scene distance (see the module doc
        // comment), a separate and correct behavior that would itself change
        // the box size between the two samples and swamp the texel-alignment
        // signal this test targets.
        let light_dir = Vec3::new(-0.4, -1.0, -0.2).normalize();
        let up = if light_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let light_basis = look_at_mat4(light_dir, Vec3::ZERO, up);
        let radius = 1.4_f32;
        let probe = Vec3::new(0.2, 0.3, 0.1);

        let center_a = Vec3::ZERO;
        let center_b = center_a + Vec3::new(0.5, 0.0, 0.0);

        let matrix_a = stabilized_light_matrix_for(light_basis, center_a, radius, SHADOW_MAP_SIZE);
        let matrix_b = stabilized_light_matrix_for(light_basis, center_b, radius, SHADOW_MAP_SIZE);

        let (ax, ay) = texel_space(matrix_a, probe);
        let (bx, by) = texel_space(matrix_b, probe);
        let (dx, dy) = (ax - bx, ay - by);
        assert!(
            (dx - dx.round()).abs() < 5e-2,
            "x drifted off whole-texel alignment: {dx}"
        );
        assert!(
            (dy - dy.round()).abs() < 5e-2,
            "y drifted off whole-texel alignment: {dy}"
        );
    }

    /// Mirrors `forward.slang:198-227`'s cascade selection + projection, up
    /// through the `uv`/`proj.z` range check at `:212` (the point where the
    /// shader currently gives up and returns "fully lit"). Returns whether
    /// `world` lands inside cascade `c`'s box.
    fn shader_covers(view_proj: Mat4, world: Vec3) -> bool {
        let clip = view_proj * world.extend(1.0);
        if clip.w.abs() < 1e-6 {
            return false;
        }
        let proj = clip.truncate() / clip.w;
        let u = proj.x * 0.5 + 0.5;
        let v = 0.5 - proj.y * 0.5;
        (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) && (0.0..=1.0).contains(&proj.z)
    }

    /// `forward.slang:198-203`'s cascade selection rule, in isolation.
    fn select_cascade(eye_distance: f32, splits: [f32; 2]) -> usize {
        let mut cascade = 0usize;
        if eye_distance > splits[0] {
            cascade = 1;
        }
        if eye_distance > splits[1] {
            cascade = 2;
        }
        cascade.min(CASCADE_COUNT - 1)
    }

    #[test]
    fn every_selectable_eye_distance_lands_inside_the_cascade_it_selects() {
        // The near-band hole: a fragment closer to the camera than roughly
        // half the near split is routed to cascade 0 by `select_cascade`,
        // but cascade 0's box hugs the camera FOCUS point
        // (`camera.target.lerp(camera.eye(), 0.15)`), not the camera itself,
        // so a point that close to the eye can fall outside it. The fix
        // (`forward.slang`'s `shadow_visibility`) retries the next coarser
        // cascade before giving up, so the invariant this test pins is
        // UNION coverage: some cascade at or above the selected index must
        // contain the fragment, not necessarily the selected one alone.
        let scene_min = Vec3::new(-4.0, 0.0, -4.0);
        let scene_max = Vec3::new(4.0, 1.4, 4.0);
        let light_dir = Vec3::new(-1.0, -0.3, -1.0);
        let camera = OrbitCamera {
            radius: 6.0,
            pitch_deg: 5.0,
            ..OrbitCamera::default()
        };

        let fit = fit_cascades(&camera, scene_min, scene_max, light_dir);
        let eye = camera.eye();
        let view_dir = (camera.target - eye).normalize();

        let lo = 0.05 * fit.splits[0];
        let hi = 1.5 * fit.splits[1];
        const STEPS: u32 = 64;

        let mut uncovered = Vec::new();
        for i in 0..=STEPS {
            let t = i as f32 / STEPS as f32;
            let eye_distance = lo + t * (hi - lo);
            let world = eye + view_dir * eye_distance;
            let selected = select_cascade(eye_distance, fit.splits);

            let covered_by_fallback =
                (selected..CASCADE_COUNT).any(|c| shader_covers(fit.matrices[c], world));
            if !covered_by_fallback {
                uncovered.push((eye_distance, selected));
            }
        }

        assert!(
            uncovered.is_empty(),
            "no cascade from the selected one through {} covers the fragment for {} of {} \
             sampled eye distances (eye_distance, selected cascade): {:?}",
            CASCADE_COUNT - 1,
            uncovered.len(),
            STEPS + 1,
            uncovered
        );
    }

    #[test]
    fn cascade_box_size_is_invariant_under_camera_motion() {
        // The other half of the shimmer: if the box resized as the camera
        // turned, the world-size of a texel would breathe. As long as the
        // camera stays within the scene's own radius (the floor in
        // `fit_cascades`), the box size must be bit-for-bit stable across
        // rotation of the camera.
        let scene_min = Vec3::splat(-50.0);
        let scene_max = Vec3::splat(50.0);
        let light_dir = Vec3::new(-0.3, -1.0, 0.2);

        let camera_a = OrbitCamera {
            radius: 5.0,
            yaw_deg: 10.0,
            pitch_deg: 15.0,
            ..OrbitCamera::default()
        };
        let camera_b = OrbitCamera {
            radius: 5.0,
            yaw_deg: 200.0,
            pitch_deg: 60.0,
            ..OrbitCamera::default()
        };

        let fit_a = fit_cascades(&camera_a, scene_min, scene_max, light_dir);
        let fit_b = fit_cascades(&camera_b, scene_min, scene_max, light_dir);

        // viewProj = ortho * light_basis; row norms of the upper 3x3 recover
        // the ortho scales (1 / half_extent) since light_basis is a pure
        // rotation.
        let row_norm =
            |m: &Mat4, row: usize| Vec3::new(m.x_axis[row], m.y_axis[row], m.z_axis[row]).length();

        for i in 0..CASCADE_COUNT {
            for row in 0..2 {
                let a = row_norm(&fit_a.matrices[i], row);
                let b = row_norm(&fit_b.matrices[i], row);
                assert!(
                    (a - b).abs() < 1e-4,
                    "cascade {i} row {row} scale breathes with camera motion: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn cascade_box_size_is_invariant_under_small_camera_dolly() {
        // The dolly half of the shimmer bug: `near_radius`/`mid_radius` used
        // to scale continuously with `dist_to_center`, so moving the camera
        // one millimetre toward or away from the scene re-scaled the texel
        // grid every frame - defeating the whole-texel snap in
        // `stabilized_light_matrix_for` for exactly the motion that snap
        // exists to fix. Both cameras sit well outside `scene_radius` (so
        // the floor does not bind) and close enough together that they must
        // land on the same quantized zoom step.
        let scene_min = Vec3::splat(-1.0);
        let scene_max = Vec3::splat(1.0);
        let light_dir = Vec3::new(-0.3, -1.0, 0.2);

        let camera_a = OrbitCamera {
            radius: 50.0,
            ..OrbitCamera::default()
        };
        let camera_b = OrbitCamera {
            radius: 50.3,
            ..OrbitCamera::default()
        };

        let fit_a = fit_cascades(&camera_a, scene_min, scene_max, light_dir);
        let fit_b = fit_cascades(&camera_b, scene_min, scene_max, light_dir);

        let row_norm =
            |m: &Mat4, row: usize| Vec3::new(m.x_axis[row], m.y_axis[row], m.z_axis[row]).length();

        for i in 0..CASCADE_COUNT {
            for row in 0..2 {
                let a = row_norm(&fit_a.matrices[i], row);
                let b = row_norm(&fit_b.matrices[i], row);
                assert!(
                    (a - b).abs() < 1e-4,
                    "cascade {i} row {row} scale breathes with a small camera dolly: {a} vs {b}"
                );
            }
        }
    }
}
