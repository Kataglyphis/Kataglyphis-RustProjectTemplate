//! Pure-CPU bounds and frustum-culling geometry, split out of `forward.rs`.
//!
//! > **World bounds must cover every pose the geometry can actually reach —
//! > not the pose it was authored in.**
//!
//! See `docs/renderer-bounds-invariant.md` (repo root) for the full rule, why
//! it keeps recurring, and every consumer that reads bounds.

use glam::{Mat4, Vec3};

use crate::render::forward::MAX_JOINTS;
use crate::scene::{CpuScene, CpuSkin};

/// View-frustum from a wgpu-convention (0..1 depth) view-projection matrix
/// (Gribb-Hartmann plane extraction; plane normals point inward).
pub(crate) struct Frustum {
    planes: [glam::Vec4; 6],
}

impl Frustum {
    pub(crate) fn from_view_proj(m: &Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);
        Self {
            planes: [
                r3 + r0, // left
                r3 - r0, // right
                r3 + r1, // bottom
                r3 - r1, // top
                r2,      // near (z >= 0 in 0..1 depth)
                r3 - r2, // far
            ],
        }
    }

    /// Positive-vertex test: the AABB is outside when its most favorable
    /// corner is behind any plane.
    pub(crate) fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.test_planes(min, max, &self.planes)
    }

    /// Visibility test for SHADOW CASTERS: identical, except the near plane
    /// is ignored - a correctness requirement, not an optimisation. A
    /// cascade's ortho box is fitted to the camera slice it covers; geometry
    /// between the light and that box lies outside the near plane but still
    /// casts into it along the box's own depth axis. Culling it produces the
    /// missing-shadow-from-tall-geometry bug the C++ engine documents in
    /// scene/Frustum.ixx. Side and far planes are safe: a caster outside them
    /// in the light's XY casts its shadow outside the map too.
    ///
    /// Only the tests below exercise it today — the cascade pass still culls
    /// casters with the near-plane-inclusive [`Self::intersects_aabb`]. Kept
    /// (and tested) so wiring it up is a one-line change, not a rewrite.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn intersects_aabb_as_caster(&self, min: Vec3, max: Vec3) -> bool {
        let [left, right, bottom, top, _near, far] = &self.planes;
        self.test_planes(min, max, &[*left, *right, *bottom, *top, *far])
    }

    fn test_planes(&self, min: Vec3, max: Vec3, planes: &[glam::Vec4]) -> bool {
        for plane in planes {
            let p = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );
            if plane.truncate().dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Inverse-transpose of a model matrix, guarded against singular input.
///
/// A zero-scale node is how Blender (and plenty of exporters) hide an object, so
/// singular model matrices arrive in ordinary files. `Mat4::inverse` on one
/// yields inf/NaN, which poisons the normal matrix and shades that primitive as
/// garbage. Fall back to identity: the geometry is degenerate anyway, and a
/// finite wrong normal is strictly better than a NaN one.
pub(crate) fn normal_matrix_of(model: Mat4) -> Mat4 {
    let inv = model.inverse();
    if inv.is_finite() {
        inv.transpose()
    } else {
        Mat4::IDENTITY
    }
}

/// True when `p` lies inside the AABB expanded by the SAME margin the occlusion
/// proxy box uses (`occlusion_bbox.wgsl`: 2% of the half-extent plus 1 cm), so
/// this CPU test agrees with the box actually rasterised rather than a slightly
/// different one. Thin wrapper over [`crate::render::occlusion::aabb_contains`],
/// which `OcclusionQueries::record` also uses to force-visible a primitive the
/// camera sits inside - kept here too since the GPU-culling path (no such
/// baked-in forcing) still needs this guard at the draw-skip decision below.
pub(crate) fn aabb_contains_point(min: Vec3, max: Vec3, p: Vec3) -> bool {
    crate::render::occlusion::aabb_contains(
        min,
        max,
        p,
        crate::render::occlusion::CONTAINMENT_MARGIN,
    )
}

/// Bounds covering every instance of `pre`.
///
/// See `docs/renderer-bounds-invariant.md` for the rule these helpers exist to
/// uphold, the full list of consumers that read bounds, and why each over-cover
/// argument is a proof rather than a fudge factor. Eight bugs in this renderer
/// were the same bug; that document is the checklist for not writing a ninth.
///
/// The shader builds its world position as `instance_matrix * skin_matrix * v`,
/// so instance transforms apply ON TOP of the posed box. With bounds left at the
/// un-instanced position the frustum test culls the whole primitive - every
/// instance with it - as soon as the base position leaves the view, even while
/// the instances are on screen. Empty means the default single identity
/// instance, for which the bounds are unchanged.
pub(crate) fn instanced_bounds(pre: (Vec3, Vec3), instances: &[Mat4]) -> (Vec3, Vec3) {
    if instances.is_empty() {
        return pre;
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for m in instances {
        let (lo, hi) = transform_aabb(*m, pre.0, pre.1);
        min = min.min(lo);
        max = max.max(hi);
    }
    (min, max)
}

/// Widen world bounds so they cover every pose the skin can put this geometry
/// in.
///
/// A skinned vertex ignores the node/model matrix entirely (`skin_matrix` in
/// forward.wgsl returns the joint blend and only falls back to `uniforms.model`
/// when the weights are zero), so node-derived bounds describe the bind pose,
/// not what is drawn. Skin weights are normalized, which makes the skinned
/// position a CONVEX COMBINATION of `J_i * v`; it therefore lies inside the
/// union of the per-joint boxes, so unioning them is conservative and never
/// culls visible geometry. The caller's node-derived box is kept in the union
/// because a skinned primitive may still contain zero-weight vertices.
pub(crate) fn widen_bounds_for_skin(
    bounds: (Vec3, Vec3),
    local_min: Vec3,
    local_max: Vec3,
    skin: &CpuSkin,
    world: &[Mat4],
) -> (Vec3, Vec3) {
    let (mut min, mut max) = bounds;
    for (i, &joint_node) in skin.joints.iter().take(MAX_JOINTS).enumerate() {
        let jw = world.get(joint_node).copied().unwrap_or(Mat4::IDENTITY);
        let ib = skin
            .inverse_bind_matrices
            .get(i)
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        let (lo, hi) = transform_aabb(jw * ib, local_min, local_max);
        min = min.min(lo);
        max = max.max(hi);
    }
    (min, max)
}

pub(crate) fn primitive_world_aabb(prim: &crate::scene::CpuPrimitive) -> (Vec3, Vec3) {
    // Morphed primitives go through the pose-covering local bounds and are then
    // transformed as a box, which stays conservative under rotation. Everything
    // else keeps the exact per-vertex transform - a tighter box, and the path
    // every non-morph primitive used before morph targets existed.
    if !prim.morph_targets.is_empty() {
        let (lo, hi) = primitive_local_aabb(prim);
        return transform_aabb(prim.transform, lo, hi);
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &prim.vertices {
        let world = prim
            .transform
            .transform_point3(Vec3::from_array(vertex.position));
        // Same reasoning as primitive_local_aabb: never let a non-finite vertex
        // (or a non-finite transform) reach the cascade fitting.
        if !world.is_finite() {
            continue;
        }
        min = min.min(world);
        max = max.max(world);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

/// Local-space bounds that cover every pose the primitive can reach.
///
/// Morph targets move vertices away from the neutral pose, so bounds taken from
/// `vertices` alone would be too small and frustum culling would drop a morphed
/// primitive while it is still on screen. For weights in [0,1] the reachable
/// extremes of a vertex are its position plus the sum of the negative deltas
/// (lower corner) and plus the sum of the positive deltas (upper corner), so
/// this is exact over that weight range and conservative outside it. Primitives
/// without morph targets are unaffected.
pub(crate) fn primitive_local_aabb(prim: &crate::scene::CpuPrimitive) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (i, vertex) in prim.vertices.iter().enumerate() {
        let p = Vec3::from_array(vertex.position);
        // Skip non-finite positions instead of letting them propagate. A single
        // NaN/inf vertex otherwise NaNs these bounds, then the scene bounds, then
        // the cascade radius - and ALL THREE cascade matrices become NaN, so
        // shadows break for every object in the scene, not just the bad mesh.
        // (`Frustum::test_planes` also treats NaN as visible, so the bad
        // primitive would additionally never cull.)
        if !p.is_finite() {
            continue;
        }
        let (mut lo, mut hi) = (p, p);
        for target in &prim.morph_targets {
            if let Some(d) = target.position_deltas.get(i) {
                lo += d.min(Vec3::ZERO);
                hi += d.max(Vec3::ZERO);
            }
        }
        min = min.min(lo);
        max = max.max(hi);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

pub(crate) fn transform_aabb(m: Mat4, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let mut out_min = Vec3::splat(f32::INFINITY);
    let mut out_max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let p = m.transform_point3(corner);
        out_min = out_min.min(p);
        out_max = out_max.max(p);
    }
    (out_min, out_max)
}

pub(crate) fn compute_world_bounds(scene: &CpuScene) -> Option<(Vec3, Vec3)> {
    // Union the per-primitive world AABBs rather than raw vertices: that helper
    // covers the morphed pose, so a morphing scene cannot report bounds smaller
    // than the geometry it actually draws (these bounds fit the shadow
    // cascades). For primitives without morph targets the helper still does the
    // exact per-vertex transform, so this is identical to the previous result.
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for prim in &scene.primitives {
        if prim.vertices.is_empty() {
            continue;
        }
        let (lo, hi) = primitive_world_aabb(prim);
        bounds = Some(match bounds {
            None => (lo, hi),
            Some((min, max)) => (min.min(lo), max.max(hi)),
        });
    }
    bounds
}

#[cfg(test)]
mod tests {
    // glam 0.33 moved the camera constructors off `Mat4` and split them by
    // clip-space convention. `directx` is glam's name for NDC Z in [0,1] with
    // Y up — which is also wgpu's and Metal's — and it reproduces the old
    // `Mat4::perspective_rh`/`orthographic_rh` bit for bit (verified against
    // glam 0.33.3 on 2026-08-07).
    use glam::camera::rh::proj::directx as clip;
    use glam::camera::rh::view::look_at_mat4;

    #[test]
    fn caster_test_ignores_the_near_plane_and_nothing_else() {
        // Ortho box [-1,1]^3: something BEHIND the near plane (z just above
        // 1 in view space, i.e. between the light and the box) must survive
        // the caster test - its shadow still falls into the box - while the
        // full test rejects it. Anything outside a SIDE plane must fail both.
        let light = clip::orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
            * look_at_mat4(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, Vec3::Y);

        let behind_near = (Vec3::new(-0.1, -0.1, 1.4), Vec3::new(0.1, 0.1, 1.6));
        let frustum = Frustum::from_view_proj(&light);
        assert!(
            !frustum.intersects_aabb(behind_near.0, behind_near.1),
            "the full test must reject a box behind the near plane"
        );
        assert!(
            frustum.intersects_aabb_as_caster(behind_near.0, behind_near.1),
            "a caster between the light and the box still shadows into it"
        );

        let beside = (Vec3::new(5.0, -0.1, 0.0), Vec3::new(5.2, 0.1, 0.2));
        assert!(
            !frustum.intersects_aabb_as_caster(beside.0, beside.1),
            "outside a side plane, the shadow lands outside the map too"
        );

        let inside = (Vec3::splat(-0.2), Vec3::splat(0.2));
        assert!(frustum.intersects_aabb_as_caster(inside.0, inside.1));
    }
    use super::*;
    use crate::scene::camera::OrbitCamera;

    #[test]
    fn a_singular_model_matrix_yields_a_finite_normal_matrix() {
        // Zero scale on an axis is how Blender hides an object, so singular
        // matrices arrive in ordinary files. inverse() is inf/NaN there, and a
        // NaN normal matrix shades the primitive as garbage.
        let squashed = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
        assert!(
            !squashed.inverse().is_finite(),
            "precondition: inverse is not finite"
        );
        assert!(
            normal_matrix_of(squashed).is_finite(),
            "guard must return something finite"
        );
        // A well-formed matrix must still get the real inverse-transpose.
        let ok = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        let expected = ok.inverse().transpose();
        assert!((normal_matrix_of(ok) - expected).abs_diff_eq(Mat4::ZERO, 1e-6));
    }

    #[test]
    fn a_non_finite_vertex_cannot_poison_the_bounds() {
        // One bad vertex used to NaN these bounds, then the scene bounds, then
        // the cascade radius - breaking shadows for EVERY object in the scene.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube.gltf");
        let scene = crate::load_gltf(path).expect("cube.gltf must load");
        let mut prim = scene.primitives[0].clone();
        prim.vertices[0].position = [f32::NAN, 1.0, f32::INFINITY];

        let (min, max) = primitive_local_aabb(&prim);
        assert!(
            min.is_finite() && max.is_finite(),
            "local bounds must stay finite, got {min:?}..{max:?}"
        );
        let (wmin, wmax) = primitive_world_aabb(&prim);
        assert!(
            wmin.is_finite() && wmax.is_finite(),
            "world bounds must stay finite, got {wmin:?}..{wmax:?}"
        );
        // And the surviving vertices must still define a real box.
        assert!(
            max.x > min.x,
            "the good vertices must still bound something"
        );
    }

    #[test]
    fn aabb_contains_point_matches_the_occlusion_proxy_margin() {
        // The guard that keeps a camera-containing primitive drawn must agree
        // with the box occlusion_bbox.wgsl actually rasterises, or it protects a
        // slightly different volume than the one that misbehaves. That shader
        // expands by `half * 0.02 + 0.01`.
        let (min, max) = (Vec3::splat(-0.5), Vec3::splat(0.5));
        let margin = 0.5 * 0.02 + 0.01; // half-extent 0.5

        assert!(
            aabb_contains_point(min, max, Vec3::ZERO),
            "centre is inside"
        );
        // Just inside the expanded face.
        let inside = 0.5 + margin - 1e-4;
        assert!(aabb_contains_point(min, max, Vec3::new(inside, 0.0, 0.0)));
        // Just outside it.
        let outside = 0.5 + margin + 1e-3;
        assert!(!aabb_contains_point(min, max, Vec3::new(outside, 0.0, 0.0)));
        // Must hold on every axis, not just x.
        assert!(!aabb_contains_point(min, max, Vec3::new(0.0, 0.0, outside)));
    }

    #[test]
    fn morph_targets_expand_the_culling_bounds() {
        // cube_morph.gltf is the unit cube (positions in [-0.5, 0.5]) plus one
        // target that lifts every vertex +1.0 in Y. Bounds taken from the
        // neutral pose alone would stop at y = 0.5, so a fully-weighted morph
        // would sit outside its own AABB and the frustum test could cull it
        // while it is plainly on screen.
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/cube_morph.gltf");
        let scene = crate::load_gltf(path).expect("cube_morph.gltf must load");
        let prim = &scene.primitives[0];
        assert_eq!(prim.morph_targets.len(), 1, "fixture must carry a target");

        let (min, max) = primitive_local_aabb(prim);
        assert!(
            (max.y - 1.5).abs() < 1e-5,
            "bounds must reach the fully-morphed pose (0.5 + 1.0), got max.y={}",
            max.y
        );
        // The neutral pose stays inside: the +Y target only has positive deltas,
        // so the lower bound must NOT be dragged upward.
        assert!(
            (min.y + 0.5).abs() < 1e-5,
            "unmorphed extent must be preserved, got min.y={}",
            min.y
        );
        // X/Z are untouched by the target and must stay exactly the cube's.
        assert!((max.x - 0.5).abs() < 1e-5 && (min.x + 0.5).abs() < 1e-5);
    }

    #[test]
    fn frustum_culls_out_of_view_aabbs() {
        let camera = OrbitCamera::default();
        let frustum = Frustum::from_view_proj(&camera.view_projection(1.0));

        // Cube at the orbit target is visible.
        assert!(frustum.intersects_aabb(Vec3::splat(-0.5), Vec3::splat(0.5)));
        // A cube far off to the side is culled.
        assert!(
            !frustum.intersects_aabb(Vec3::new(1000.0, -0.5, -0.5), Vec3::new(1001.0, 0.5, 0.5))
        );
        // Behind the camera is culled.
        let eye = camera.eye();
        let behind = eye + (eye - Vec3::ZERO).normalize() * 10.0;
        assert!(!frustum.intersects_aabb(behind - Vec3::splat(0.4), behind + Vec3::splat(0.4)));
    }
}
