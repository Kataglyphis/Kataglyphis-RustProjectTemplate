//! Quadric error decimation, measured against the clustering simplifier it
//! is meant to beat.
//!
//! The properties worth pinning are geometric, not structural: does the
//! output still describe the same SURFACE, and does it keep the features
//! clustering loses. So the assertions here go through a point-to-surface
//! distance rather than through the simplifier's internal bookkeeping - a
//! rewrite of the collapse machinery should not need this file edited.

use glam::{Mat4, Vec3};
use kataglyphis_webgpu_renderer::scene::lod::simplify_primitive;
use kataglyphis_webgpu_renderer::scene::qem::simplify_primitive_qem;
use kataglyphis_webgpu_renderer::scene::{CpuMaterial, CpuPrimitive, Vertex};

fn vertex(position: [f32; 3], normal: [f32; 3]) -> Vertex {
    Vertex {
        position,
        normal,
        uv: [position[0], position[1]],
        tangent: [1.0, 0.0, 0.0, 1.0],
        joints: [0.0; 4],
        weights: [0.0; 4],
    }
}

fn primitive(vertices: Vec<Vertex>, indices: Vec<u32>) -> CpuPrimitive {
    CpuPrimitive {
        vertices,
        indices,
        transform: Mat4::IDENTITY,
        node_index: None,
        skin_index: None,
        material: CpuMaterial::default(),
    }
}

/// A flat `n` x `n` triangulated grid over the unit square in Z = 0.
fn plane_grid(n: usize) -> CpuPrimitive {
    let mut vertices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            vertices.push(vertex(
                [x as f32 / (n - 1) as f32, y as f32 / (n - 1) as f32, 0.0],
                [0.0, 0.0, 1.0],
            ));
        }
    }
    let mut indices = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let i = (y * n + x) as u32;
            indices.extend_from_slice(&[i, i + 1, i + n as u32 + 1]);
            indices.extend_from_slice(&[i, i + n as u32 + 1, i + n as u32]);
        }
    }
    primitive(vertices, indices)
}

/// A bumpy grid: the plane displaced by a smooth function, so collapses have
/// real (non-zero) costs to rank.
fn bumpy_grid(n: usize) -> CpuPrimitive {
    let mut prim = plane_grid(n);
    for v in &mut prim.vertices {
        let (x, y) = (v.position[0], v.position[1]);
        v.position[2] = 0.25 * (x * 9.0).sin() * (y * 7.0).cos();
    }
    prim
}

/// A flat square base with one vertex pulled up into a tall thin spike.
///
/// The spike is a single vertex of an otherwise planar grid, which is
/// exactly the configuration centroid clustering cannot represent: the tip
/// shares a cell with its flat neighbours and gets averaged down toward
/// them, while QEM sees an enormous quadric there and refuses to remove it.
fn spiked_grid(n: usize, height: f32) -> CpuPrimitive {
    let mut prim = plane_grid(n);
    let tip = (n / 2) * n + n / 2;
    prim.vertices[tip].position[2] = height;
    prim
}

fn triangle_count(prim: &CpuPrimitive) -> usize {
    prim.indices.len() / 3
}

fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Ericson's closest-point-on-triangle, region by region.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// Largest distance from any original vertex to the simplified surface.
fn max_deviation(original: &CpuPrimitive, simplified: &CpuPrimitive) -> f32 {
    if simplified.indices.is_empty() {
        return f32::INFINITY;
    }
    let mut worst = 0.0f32;
    for v in &original.vertices {
        let p = Vec3::from_array(v.position);
        let mut nearest = f32::INFINITY;
        for tri in simplified.indices.chunks_exact(3) {
            let a = Vec3::from_array(simplified.vertices[tri[0] as usize].position);
            let b = Vec3::from_array(simplified.vertices[tri[1] as usize].position);
            let c = Vec3::from_array(simplified.vertices[tri[2] as usize].position);
            nearest = nearest.min(point_triangle_distance(p, a, b, c));
        }
        worst = worst.max(nearest);
    }
    worst
}

/// Highest Z reached by the rendered SURFACE - how much of the spike survived.
///
/// Referenced vertices only. Clustering leaves the merged-away vertices in
/// its buffer, so a naive max over `vertices` reports the full spike height
/// from a vertex that no triangle uses and nothing draws.
fn peak_height(prim: &CpuPrimitive) -> f32 {
    prim.indices.iter().fold(f32::NEG_INFINITY, |acc, &i| {
        acc.max(prim.vertices[i as usize].position[2])
    })
}

fn assert_well_formed(prim: &CpuPrimitive) {
    assert_eq!(
        prim.indices.len() % 3,
        0,
        "output must stay a triangle list"
    );
    for tri in prim.indices.chunks_exact(3) {
        for &i in tri {
            assert!(
                (i as usize) < prim.vertices.len(),
                "index {i} out of range ({} vertices)",
                prim.vertices.len()
            );
        }
        assert!(
            tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2],
            "degenerate triangle {tri:?} survived"
        );
    }
    // Every emitted vertex must be referenced; a compaction bug that leaves
    // orphans would otherwise pass silently and bloat the buffer.
    let mut used = vec![false; prim.vertices.len()];
    for &i in &prim.indices {
        used[i as usize] = true;
    }
    assert!(
        used.iter().all(|&u| u),
        "unreferenced vertices in the output"
    );
}

#[test]
fn reaches_roughly_the_requested_triangle_count() {
    let full = bumpy_grid(24);
    let before = triangle_count(&full);
    for ratio in [0.5f32, 0.25, 0.1] {
        let out = simplify_primitive_qem(&full, ratio);
        assert_well_formed(&out);
        let after = triangle_count(&out);
        let achieved = after as f32 / before as f32;
        assert!(
            achieved <= ratio * 1.25 + 0.01,
            "ratio {ratio}: asked to keep {ratio}, kept {achieved} ({after}/{before})"
        );
        assert!(after > 0, "ratio {ratio}: mesh collapsed entirely");
    }
}

#[test]
fn harder_decimation_costs_more_error() {
    let full = bumpy_grid(24);
    let mut previous = -1.0f32;
    for ratio in [0.8f32, 0.5, 0.25, 0.1] {
        let deviation = max_deviation(&full, &simplify_primitive_qem(&full, ratio));
        assert!(
            deviation >= previous - 1e-6,
            "error fell when decimating harder: ratio {ratio} gave {deviation}, previous {previous}"
        );
        previous = deviation;
    }
}

#[test]
fn a_plane_decimates_to_near_minimal_with_no_error() {
    // The QEM signature result. Every interior collapse on a co-planar mesh
    // has exactly zero cost, so the whole grid should fold down to the few
    // triangles that span the square - and the surface must not move at all.
    // Clustering cannot do this: it is bounded by its cell grid regardless
    // of how flat the input is.
    //
    // The ratio is small but not zero: decimation is driven by a triangle
    // budget, not by an error threshold, so a target of zero really does
    // consume the mesh - boundary quadrics make the last collapses
    // expensive, not illegal.
    let full = plane_grid(16);
    let out = simplify_primitive_qem(&full, 0.01);
    assert_well_formed(&out);

    // Measured: 450 triangles collapse to 4, max deviation 5.06e-6 - which
    // is f32 round-off on unit-scale coordinates, not surface movement.
    // 1e-4 leaves margin over that while staying four orders below the
    // mesh's own size.
    assert!(
        triangle_count(&out) <= 6,
        "a flat grid should collapse to a handful of triangles, got {}",
        triangle_count(&out)
    );
    let deviation = max_deviation(&full, &out);
    assert!(
        deviation < 1e-4,
        "a co-planar mesh must decimate without moving the surface, deviation {deviation}"
    );
}

#[test]
fn a_sharp_spike_survives_qem_where_clustering_rounds_it_off() {
    // The headline comparison. Both simplifiers are held to the same
    // triangle budget: clustering runs first and QEM is then asked for no
    // more triangles than clustering produced, so QEM cannot win by simply
    // keeping more geometry.
    const HEIGHT: f32 = 2.0;
    let full = spiked_grid(17, HEIGHT);

    let clustered = simplify_primitive(&full, 0.12);
    let budget = triangle_count(&clustered) as f32 / triangle_count(&full) as f32;
    let qem = simplify_primitive_qem(&full, budget);
    assert_well_formed(&qem);
    assert!(
        triangle_count(&qem) <= triangle_count(&clustered),
        "QEM was given a larger budget than clustering: {} vs {}",
        triangle_count(&qem),
        triangle_count(&clustered)
    );

    let qem_peak = peak_height(&qem);
    let clustered_peak = peak_height(&clustered);
    let qem_deviation = max_deviation(&full, &qem);
    let clustered_deviation = max_deviation(&full, &clustered);

    // Measured on this mesh (17x17 grid, 512 triangles, both cut to 18):
    //   QEM        surface peak 2.000000, max deviation 6.66e-8
    //   clustering surface peak 0.000000, max deviation 2.0000
    // Clustering does not merely shorten the spike, it loses it entirely -
    // the tip lands alone in its own grid cell, survives as a vertex, and
    // then every triangle that referenced it is dropped as degenerate, so
    // the drawn surface is flat and the tip is a full 2.0 away from it.
    // QEM instead places the merged vertex exactly at the apex, because the
    // cone of steep face planes there makes the 3x3 well-conditioned and its
    // solution is the apex itself.
    //
    // The thresholds are those measurements with margin, not predictions.
    // The gap is large enough that they do not need to be tight; a
    // regression that half-loses the spike still trips them.
    assert!(
        qem_peak > HEIGHT * 0.95,
        "QEM lost the spike: surface peak {qem_peak} of {HEIGHT}"
    );
    assert!(
        clustered_peak < HEIGHT * 0.5,
        "clustering unexpectedly kept the spike ({clustered_peak}); \
         the comparison no longer measures anything"
    );
    assert!(
        qem_deviation < clustered_deviation * 0.01,
        "QEM deviation {qem_deviation} is not meaningfully better than \
         clustering's {clustered_deviation}"
    );
}

#[test]
fn output_is_bit_identical_across_runs() {
    // f32/f64 costs in a heap are the classic source of nondeterminism here:
    // equal-cost edges pop in whatever order the heap happens to store them
    // unless the comparator breaks ties on vertex index. A flat region of
    // the bumpy grid produces plenty of exact ties to expose that.
    let full = bumpy_grid(20);
    let a = simplify_primitive_qem(&full, 0.3);
    let b = simplify_primitive_qem(&full, 0.3);

    assert_eq!(a.indices, b.indices, "index buffer differed between runs");
    assert_eq!(a.vertices.len(), b.vertices.len());
    for (va, vb) in a.vertices.iter().zip(&b.vertices) {
        assert_eq!(
            va.position.map(f32::to_bits),
            vb.position.map(f32::to_bits),
            "vertex position differed between runs"
        );
        assert_eq!(va.normal.map(f32::to_bits), vb.normal.map(f32::to_bits));
        assert_eq!(va.uv.map(f32::to_bits), vb.uv.map(f32::to_bits));
    }
}

#[test]
fn attributes_are_carried_through_collapses() {
    // Not just positions: a collapse that dropped UVs or left normals
    // unnormalised would pass every geometric assertion above and then shade
    // and texture wrongly.
    let mut full = bumpy_grid(16);
    for v in &mut full.vertices {
        let n = Vec3::new(v.position[0], v.position[1], 1.0).normalize();
        v.normal = n.to_array();
        v.uv = [v.position[0], v.position[1]];
    }
    let out = simplify_primitive_qem(&full, 0.25);
    for v in &out.vertices {
        let length = Vec3::from_array(v.normal).length();
        assert!(
            (length - 1.0).abs() < 1e-4,
            "normal not renormalised after collapse: length {length}"
        );
        assert!(
            v.uv[0].is_finite() && v.uv[1].is_finite(),
            "uv went non-finite: {:?}",
            v.uv
        );
        assert!((-0.01..=1.01).contains(&v.uv[0]) && (-0.01..=1.01).contains(&v.uv[1]));
    }
}

#[test]
fn degenerate_inputs_do_not_panic() {
    let empty = primitive(Vec::new(), Vec::new());
    assert!(simplify_primitive_qem(&empty, 0.5).indices.is_empty());

    let no_indices = primitive(vec![vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])], Vec::new());
    assert!(simplify_primitive_qem(&no_indices, 0.5).indices.is_empty());

    let single = primitive(
        vec![
            vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ],
        vec![0, 1, 2],
    );
    for ratio in [0.0f32, 0.5, 1.0, 2.0, -3.0, f32::NAN, f32::INFINITY] {
        assert_well_formed(&simplify_primitive_qem(&single, ratio));
    }

    let identical = primitive(
        vec![vertex([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]); 6],
        vec![0, 1, 2, 3, 4, 5],
    );
    assert_well_formed(&simplify_primitive_qem(&identical, 0.5));

    // Zero-area faces mixed into a real mesh: the plane is undefined there
    // and must be skipped rather than producing a NaN quadric.
    let mut with_slivers = bumpy_grid(8);
    with_slivers.indices.extend_from_slice(&[0, 0, 1, 2, 2, 2]);
    assert_well_formed(&simplify_primitive_qem(&with_slivers, 0.5));
}
