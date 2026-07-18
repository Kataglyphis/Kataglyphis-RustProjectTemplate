//! Mesh simplification and runtime LOD selection.
//!
//! The simplifier is vertex-clustering (grid quantization): vertices are
//! snapped to a grid, merged, and degenerate triangles dropped. It is not
//! quadric-error decimation — it is O(n), dependency-free, deterministic,
//! and good enough to cut triangle counts on dense photogrammetry meshes
//! where the LOD is only ever seen at distance. Swapping in meshoptimizer
//! later only replaces `simplify_primitive`.

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

    // Map each grid cell to a merged vertex (first-wins position, averaged
    // normal so shading stays smooth).
    let mut cell_to_index: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut normal_sums: Vec<Vec3> = Vec::new();
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
            (vertices.len() - 1) as u32
        });
        normal_sums[index as usize] += Vec3::from_array(v.normal);
        remap.push(index);
    }

    for (vertex, sum) in vertices.iter_mut().zip(&normal_sums) {
        let n = sum.normalize_or_zero();
        if n != Vec3::ZERO {
            vertex.normal = n.to_array();
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
    }
}

/// Builds an LOD chain for a primitive. `switch_distances` gives the
/// camera distance at which each successive level takes over; each level
/// simplifies twice as aggressively as the previous one.
pub fn build_lod_chain(prim: &CpuPrimitive, switch_distances: &[f32]) -> Vec<Lod> {
    let mut chain = Vec::with_capacity(switch_distances.len());
    let mut ratio = 0.02;
    for &distance in switch_distances {
        chain.push(Lod {
            primitive: simplify_primitive(prim, ratio),
            min_distance: distance,
        });
        ratio *= 2.0;
    }
    chain
}

/// Picks the LOD index for a camera distance: the last level whose
/// `min_distance` the camera has passed, or `None` for the full-detail mesh.
pub fn select_lod(chain: &[Lod], distance: f32) -> Option<usize> {
    let mut chosen = None;
    for (i, lod) in chain.iter().enumerate() {
        if distance >= lod.min_distance {
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
        assert_eq!(simplified.indices.len() % 3, 0, "indices must stay triangles");
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
}
