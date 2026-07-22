//! Mesh simplification and runtime LOD selection.
//!
//! The simplifier is vertex-clustering (grid quantization): vertices in the
//! same grid cell are merged to their centroid and degenerate triangles
//! dropped. It is not quadric-error decimation — it is O(n), dependency-free,
//! deterministic, and good enough to cut triangle counts on dense
//! photogrammetry meshes where the LOD is only ever seen at distance.
//! Swapping in meshoptimizer or a QEM pass later only replaces
//! `simplify_primitive`.
//!
//! QEM would place merged vertices to minimise distance to the original
//! SURFACE rather than to the original vertices, which preserves silhouettes
//! and creases that clustering rounds off. That is the upgrade this module is
//! shaped for and has not had.

use std::collections::HashMap;

use glam::Vec3;

use crate::scene::{CpuPrimitive, Vertex};

/// One level of detail: a simplified primitive plus the distance beyond
/// which it should be used.
#[derive(Clone, Debug)]
pub struct Lod {
    pub primitive: CpuPrimitive,
    /// Switch to this level when the camera is farther than this (world units).
    pub min_distance: f32,
}

/// Simplifies a primitive by clustering vertices onto a grid.
///
/// `cell_ratio` is the grid cell size as a fraction of the mesh's bounding
/// box diagonal (e.g. 0.02 = 2%). Larger values simplify harder.
pub fn simplify_primitive(prim: &CpuPrimitive, cell_ratio: f32) -> CpuPrimitive {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in &prim.vertices {
        let p = Vec3::from_array(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    let diagonal = (max - min).length();
    if !diagonal.is_finite() || diagonal <= 0.0 {
        return prim.clone();
    }
    let cell = (diagonal * cell_ratio).max(1e-6);

    // Map each grid cell to a merged vertex.
    //
    // Position is the CENTROID of the vertices that fell in the cell, not the
    // first one seen. First-wins snaps every vertex to whichever happened to
    // be visited first, so the simplified surface jitters toward an arbitrary
    // corner of each cell and the error depends on vertex ORDER - reordering
    // an unchanged mesh changed the result. The centroid minimises squared
    // distance to the merged vertices and is order-independent.
    //
    // Normals are averaged for the same reason: a merged vertex that kept one
    // input normal shades as a facet.
    let mut cell_to_index: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut normal_sums: Vec<Vec3> = Vec::new();
    let mut position_sums: Vec<Vec3> = Vec::new();
    let mut merge_counts: Vec<f32> = Vec::new();
    let mut remap: Vec<u32> = Vec::with_capacity(prim.vertices.len());

    for v in &prim.vertices {
        let p = Vec3::from_array(v.position);
        let key = (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        );
        let index = *cell_to_index.entry(key).or_insert_with(|| {
            vertices.push(*v);
            normal_sums.push(Vec3::ZERO);
            position_sums.push(Vec3::ZERO);
            merge_counts.push(0.0);
            (vertices.len() - 1) as u32
        });
        normal_sums[index as usize] += Vec3::from_array(v.normal);
        position_sums[index as usize] += p;
        merge_counts[index as usize] += 1.0;
        remap.push(index);
    }

    for (index, vertex) in vertices.iter_mut().enumerate() {
        let n = normal_sums[index].normalize_or_zero();
        if n != Vec3::ZERO {
            vertex.normal = n.to_array();
        }
        let count = merge_counts[index];
        if count > 0.0 {
            vertex.position = (position_sums[index] / count).to_array();
        }
    }

    // Rebuild indices, dropping triangles that collapsed to a line/point.
    let mut indices = Vec::with_capacity(prim.indices.len());
    for tri in prim.indices.chunks_exact(3) {
        let (a, b, c) = (
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        );
        if a != b && b != c && a != c {
            indices.extend_from_slice(&[a, b, c]);
        }
    }

    CpuPrimitive {
        vertices,
        indices,
        transform: prim.transform,
        node_index: prim.node_index,
        skin_index: prim.skin_index,
        material: prim.material.clone(),
        // Simplification changes the vertex count, so per-vertex morph deltas
        // can't be carried; simplified LODs render unmorphed.
        morph_targets: Vec::new(),
        morph_weights: Vec::new(),
    }
}

/// Builds an LOD chain for a primitive. `switch_distances` gives the
/// camera distance at which each successive level takes over; each level
/// simplifies twice as aggressively as the previous one.
pub fn build_lod_chain(prim: &CpuPrimitive, switch_distances: &[f32]) -> Vec<Lod> {
    build_lod_chain_with(prim, switch_distances, Simplifier::VertexClustering)
}

/// Which simplifier a chain is built with.
///
/// The two take different ratio meanings, which is why this is an enum rather
/// than a function pointer: clustering's ratio is a GRID CELL SIZE (bigger
/// cell, fewer vertices) while QEM's is a TRIANGLE BUDGET (smaller fraction,
/// fewer triangles). They move in opposite directions and are not
/// interchangeable numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simplifier {
    /// Grid quantization. O(n), and cannot represent a feature smaller than
    /// its cell - see the spike measurement in `qem.rs`.
    VertexClustering,
    /// Quadric error metrics. Keeps silhouettes and creases at the same
    /// triangle budget; costs a heap and a pass over the one-ring per collapse.
    Quadric,
}

/// Builds an LOD chain with an explicit simplifier.
///
/// Each level is twice as aggressive as the one before it, under whichever
/// ratio convention the simplifier uses.
pub fn build_lod_chain_with(
    prim: &CpuPrimitive,
    switch_distances: &[f32],
    simplifier: Simplifier,
) -> Vec<Lod> {
    let mut chain = Vec::with_capacity(switch_distances.len());
    // Clustering grows its cell; QEM shrinks its budget. Same "twice as
    // aggressive per level", opposite directions.
    let mut cluster_ratio = 0.02;
    let mut keep_fraction = 0.5;
    for &distance in switch_distances {
        let primitive = match simplifier {
            Simplifier::VertexClustering => simplify_primitive(prim, cluster_ratio),
            Simplifier::Quadric => crate::scene::qem::simplify_primitive_qem(prim, keep_fraction),
        };
        chain.push(Lod {
            primitive,
            min_distance: distance,
        });
        cluster_ratio *= 2.0;
        keep_fraction *= 0.5;
    }
    chain
}

/// Picks the LOD index for a camera distance: the last level whose
/// `min_distance` the camera has passed, or `None` for the full-detail mesh.
pub fn select_lod(chain: &[Lod], distance: f32) -> Option<usize> {
    select_lod_by_distance_iter(chain.iter().map(|lod| lod.min_distance), distance)
}

/// The same rule as `select_lod`, over the switch distances alone.
///
/// The renderer keeps only GPU buffers per level - the CPU `Lod` chain is
/// dropped once uploaded - so it has no `&[Lod]` to hand `select_lod`. Rather
/// than let the render path grow a second, silently divergent rule, both
/// forms funnel through `select_lod_by_distance_iter`.
pub fn select_lod_by_distance(min_distances: &[f32], distance: f32) -> Option<usize> {
    select_lod_by_distance_iter(min_distances.iter().copied(), distance)
}

/// Last level whose switch distance the camera has passed. Levels are
/// expected in ascending distance order, which is how `build_lod_chain_with`
/// emits them; scanning to the end rather than stopping at the first miss
/// means an out-of-order list degrades to "the farthest match" instead of
/// silently picking level 0.
fn select_lod_by_distance_iter(
    min_distances: impl Iterator<Item = f32>,
    distance: f32,
) -> Option<usize> {
    let mut chosen = None;
    for (i, min_distance) in min_distances.enumerate() {
        if distance >= min_distance {
            chosen = Some(i);
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::CpuMaterial;
    use glam::Mat4;

    /// A dense grid mesh in the XY plane.
    fn grid_primitive(n: usize) -> CpuPrimitive {
        let mut vertices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                vertices.push(Vertex {
                    position: [x as f32 / n as f32, y as f32 / n as f32, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    joints: [0.0; 4],
                    weights: [0.0; 4],
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv1: [0.0, 0.0],
                });
            }
        }
        let mut indices = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (y * n + x) as u32;
                let right = i + 1;
                let down = i + n as u32;
                let diag = down + 1;
                indices.extend_from_slice(&[i, right, diag, i, diag, down]);
            }
        }
        CpuPrimitive {
            vertices,
            indices,
            transform: Mat4::IDENTITY,
            node_index: None,
            skin_index: None,
            material: CpuMaterial::default(),
            morph_targets: Vec::new(),
            morph_weights: Vec::new(),
        }
    }

    #[test]
    fn simplification_reduces_geometry() {
        let full = grid_primitive(32);
        let simplified = simplify_primitive(&full, 0.08);

        assert!(
            simplified.vertices.len() < full.vertices.len() / 2,
            "expected a big vertex reduction: {} -> {}",
            full.vertices.len(),
            simplified.vertices.len()
        );
        assert!(!simplified.indices.is_empty(), "mesh collapsed entirely");
        assert_eq!(
            simplified.indices.len() % 3,
            0,
            "indices must stay triangles"
        );
        // Every index must address a surviving vertex.
        let max_index = *simplified.indices.iter().max().unwrap() as usize;
        assert!(max_index < simplified.vertices.len());
    }

    #[test]
    fn harder_ratio_simplifies_more() {
        let full = grid_primitive(32);
        let light = simplify_primitive(&full, 0.04);
        let heavy = simplify_primitive(&full, 0.16);
        assert!(heavy.vertices.len() < light.vertices.len());
    }

    #[test]
    fn lod_selection_follows_distance() {
        let full = grid_primitive(16);
        let chain = build_lod_chain(&full, &[10.0, 50.0]);
        assert_eq!(chain.len(), 2);
        assert!(chain[1].primitive.vertices.len() <= chain[0].primitive.vertices.len());

        assert_eq!(select_lod(&chain, 1.0), None); // near: full detail
        assert_eq!(select_lod(&chain, 20.0), Some(0));
        assert_eq!(select_lod(&chain, 100.0), Some(1));
    }

    #[test]
    fn degenerate_input_is_returned_unchanged() {
        let mut prim = grid_primitive(4);
        // Collapse every vertex to one point: zero diagonal.
        for v in &mut prim.vertices {
            v.position = [0.0, 0.0, 0.0];
        }
        let simplified = simplify_primitive(&prim, 0.05);
        assert_eq!(simplified.vertices.len(), prim.vertices.len());
    }

    /// Mean distance from each ORIGINAL vertex to the merged vertex it was
    /// mapped onto - i.e. how far the simplifier moved the surface.
    fn mean_displacement(original: &CpuPrimitive, cell_ratio: f32) -> f32 {
        let simplified = simplify_primitive(original, cell_ratio);

        // Re-derive the mapping the simplifier used: nearest merged vertex.
        // Cheap enough for a test-sized grid and independent of the internal
        // remap table, so this measures the RESULT rather than the bookkeeping.
        let mut total = 0.0f32;
        for v in &original.vertices {
            let p = Vec3::from_array(v.position);
            let nearest = simplified
                .vertices
                .iter()
                .map(|s| (Vec3::from_array(s.position) - p).length())
                .fold(f32::INFINITY, f32::min);
            total += nearest;
        }
        total / original.vertices.len() as f32
    }

    #[test]
    fn merged_vertices_sit_at_the_cell_centroid() {
        // Four vertices in one cell, at known offsets. The merged position
        // must be their average, not whichever was visited first.
        let mut prim = grid_primitive(2);
        prim.vertices[0].position = [0.0, 0.0, 0.0];
        prim.vertices[1].position = [1.0, 0.0, 0.0];
        prim.vertices[2].position = [0.0, 1.0, 0.0];
        prim.vertices[3].position = [1.0, 1.0, 0.0];

        // A cell ratio large enough to swallow the whole mesh.
        let simplified = simplify_primitive(&prim, 10.0);
        assert_eq!(
            simplified.vertices.len(),
            1,
            "the whole mesh should collapse to one vertex"
        );

        let merged = Vec3::from_array(simplified.vertices[0].position);
        let expected = Vec3::new(0.5, 0.5, 0.0);
        assert!(
            (merged - expected).length() < 1e-5,
            "merged vertex at {merged:?}, expected the centroid {expected:?}"
        );
    }

    #[test]
    fn simplification_is_independent_of_vertex_order() {
        // The property first-wins merging did NOT have: reversing the vertex
        // list changed which position each cell kept, so an unchanged mesh
        // simplified differently depending on how it was authored.
        //
        // The jitter is load-bearing. A regular grid is SYMMETRIC under
        // reversal - first-wins picks each cell's min corner forwards and its
        // max corner backwards, and for a symmetric grid those two sets are
        // identical, so the test passed against the very bug it exists for.
        // Verified: with first-wins restored, this fails only with the jitter.
        let mut original = grid_primitive(16);
        for (i, v) in original.vertices.iter_mut().enumerate() {
            let n = i as f32;
            v.position[0] += (n * 0.37).fract() * 0.01;
            v.position[1] += (n * 0.71).fract() * 0.01;
            v.position[2] += (n * 0.13).fract() * 0.01;
        }

        let mut reordered = original.clone();
        reordered.vertices.reverse();
        let last = (original.vertices.len() - 1) as u32;
        for index in &mut reordered.indices {
            *index = last - *index;
        }

        let a = simplify_primitive(&original, 0.1);
        let b = simplify_primitive(&reordered, 0.1);

        assert_eq!(
            a.vertices.len(),
            b.vertices.len(),
            "vertex order changed the simplified vertex count"
        );

        // Same set of positions, order aside - compared with a tolerance, not
        // for bit equality.
        //
        // Centroid merging is order-independent in exact arithmetic but not in
        // f32: summing the same positions in a different sequence rounds
        // differently in the last digits. An exact comparison (quantise, sort,
        // assert_eq) fails on that alone, which is what a first attempt here
        // did. The property worth asserting is that the surface is the same,
        // and 1e-4 is far tighter than the cell size while leaving room for
        // summation order.
        let key = |v: &Vertex| {
            let p = v.position;
            (p[0].to_bits(), p[1].to_bits(), p[2].to_bits())
        };
        let mut sorted_a: Vec<&Vertex> = a.vertices.iter().collect();
        let mut sorted_b: Vec<&Vertex> = b.vertices.iter().collect();
        sorted_a.sort_by_key(|v| key(v));
        sorted_b.sort_by_key(|v| key(v));

        for (va, vb) in sorted_a.iter().zip(&sorted_b) {
            let pa = Vec3::from_array(va.position);
            let pb = Vec3::from_array(vb.position);
            assert!(
                (pa - pb).length() < 1e-4,
                "vertex order changed a merged position: {pa:?} vs {pb:?}"
            );
        }
    }

    #[test]
    fn displacement_stays_within_the_cell_size() {
        // The bound clustering can actually promise: no vertex moves further
        // than the cell it was merged within. A regression that placed merged
        // vertices outside their cell would pass the reduction tests and
        // quietly warp the mesh.
        let original = grid_primitive(24);
        let cell_ratio = 0.1f32;

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &original.vertices {
            let p = Vec3::from_array(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let cell = (max - min).length() * cell_ratio;

        let mean = mean_displacement(&original, cell_ratio);
        assert!(
            mean < cell,
            "mean displacement {mean} exceeds one cell ({cell}); merged vertices are leaving their cells"
        );
        assert!(
            mean > 0.0,
            "nothing moved at all - the mesh was not simplified"
        );
    }
}
