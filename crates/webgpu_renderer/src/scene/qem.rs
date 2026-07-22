//! Garland-Heckbert quadric error metric decimation.
//!
//! The upgrade `lod::simplify_primitive` documents as missing. Clustering
//! merges vertices to a cell centroid, so it minimises distance to the
//! original VERTICES and rounds off anything smaller than a cell. QEM places
//! the merged vertex where the summed squared distance to the original
//! PLANES is smallest, so a crease or a spike survives at ratios where
//! clustering has already flattened it.
//!
//! Clustering stays: it is O(n) and needs no adjacency, which is still the
//! right trade for distant photogrammetry LODs. This is the O(n log n)
//! quality path, chosen per primitive by the caller.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use glam::{DMat3, DVec3, Vec3};

use crate::scene::{CpuPrimitive, Vertex};

/// Weight applied to the plane constraints synthesised along open boundaries.
///
/// Boundary edges have only one incident face, so nothing in the ordinary
/// face quadric resists pulling them inward - an open mesh erodes from its
/// rim. Garland's fix is a virtual plane through the edge perpendicular to
/// its face; the weight decides how much stronger than a real face it is.
/// 100 was enough to pin the border of the flat grid in the tests while
/// still letting boundary vertices slide ALONG the rim (which costs nothing,
/// because that motion stays inside the constraint plane).
const BOUNDARY_WEIGHT: f64 = 100.0;

/// Relative determinant below which the 3x3 solve is treated as singular.
///
/// Scaled by the matrix norm cubed so the test is on conditioning, not on
/// the mesh's units - an absolute epsilon rejects everything on a
/// millimetre-scale mesh and nothing on a kilometre-scale one.
const SINGULAR_EPSILON: f64 = 1e-10;

/// Symmetric 4x4 error quadric, upper triangle only.
///
/// Order: a2 ab ac ad b2 bc bd c2 cd d2, for the plane (a, b, c, d).
/// f64 rather than f32 because these are sums of squares of coordinates:
/// accumulating a few hundred face quadrics in f32 on a mesh that is not
/// centred on the origin loses the small off-diagonal terms the 3x3 solve
/// depends on, and the solve then reports "singular" on curved regions.
#[derive(Clone, Copy, Default, Debug)]
struct Quadric {
    m: [f64; 10],
}

impl Quadric {
    /// The fundamental quadric K_p = p * p^T for a plane, times `weight`.
    fn from_plane(n: DVec3, d: f64, weight: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ]
            .map(|v| v * weight),
        }
    }

    fn add(&mut self, other: &Quadric) {
        for (dst, src) in self.m.iter_mut().zip(other.m.iter()) {
            *dst += *src;
        }
    }

    /// v^T Q v for the homogeneous point (v, 1).
    fn error(&self, v: DVec3) -> f64 {
        let [a2, ab, ac, ad, b2, bc, bd, c2, cd, d2] = self.m;
        a2 * v.x * v.x
            + 2.0 * ab * v.x * v.y
            + 2.0 * ac * v.x * v.z
            + 2.0 * ad * v.x
            + b2 * v.y * v.y
            + 2.0 * bc * v.y * v.z
            + 2.0 * bd * v.y
            + c2 * v.z * v.z
            + 2.0 * cd * v.z
            + d2
    }

    /// The position minimising `error`, or `None` when the derived 3x3 is
    /// too ill-conditioned to trust - which is the COMMON case, not the rare
    /// one: every planar region gives a rank-1 system, and every straight
    /// crease a rank-2 one.
    fn optimal_position(&self) -> Option<DVec3> {
        let [a2, ab, ac, ad, b2, bc, bd, c2, cd, _] = self.m;
        let a = DMat3::from_cols(
            DVec3::new(a2, ab, ac),
            DVec3::new(ab, b2, bc),
            DVec3::new(ac, bc, c2),
        );
        let norm = [a2, ab, ac, b2, bc, c2]
            .iter()
            .fold(0.0f64, |acc, v| acc.max(v.abs()));
        if norm <= 0.0 {
            return None;
        }
        let det = a.determinant();
        if det.abs() < SINGULAR_EPSILON * norm * norm * norm {
            return None;
        }
        let solution = a.inverse() * DVec3::new(-ad, -bd, -cd);
        solution.is_finite().then_some(solution)
    }
}

/// A pending collapse. Costs are pushed rather than decrease-keyed, so an
/// entry is stale once either endpoint has been collapsed into since it was
/// queued; `versions` detects that on pop.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// Cost as an order-preserving integer key. Ordering f64 directly is not
    /// a total order, and `partial_cmp().unwrap()` in a heap comparator is a
    /// panic waiting for the first NaN quadric on a degenerate face.
    key: u64,
    v0: u32,
    v1: u32,
    version0: u32,
    version1: u32,
    target: DVec3,
}

impl Candidate {
    fn order(&self) -> (u64, u32, u32) {
        (self.key, self.v0, self.v1)
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.order() == other.order()
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    /// Reversed: `BinaryHeap` is a max-heap and we want the cheapest
    /// collapse. Ties break on vertex index so the sequence of collapses is
    /// fixed even when a mesh has many equal-cost edges - a flat grid has
    /// thousands of exactly-zero-cost edges, and without the tiebreak the
    /// output depends on heap internals.
    fn cmp(&self, other: &Self) -> Ordering {
        other.order().cmp(&self.order())
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Maps a non-negative f64 onto a u64 that sorts the same way.
fn cost_key(cost: f64) -> u64 {
    if cost.is_nan() {
        return u64::MAX;
    }
    cost.max(0.0).to_bits()
}

/// Simplifies a primitive with quadric error metric decimation.
///
/// `target_ratio` is the fraction of TRIANGLES to keep (0.25 = a quarter).
/// Values >= 1.0 return the input unchanged. The result may have more
/// triangles than requested when no further collapse is legal - boundary
/// constraints and the normal-flip rejection both stop decimation early, by
/// design.
pub fn simplify_primitive_qem(prim: &CpuPrimitive, target_ratio: f32) -> CpuPrimitive {
    let face_count = prim.indices.len() / 3;
    if face_count == 0 || prim.vertices.is_empty() || target_ratio >= 1.0 {
        return prim.clone();
    }

    let Some(mesh) = WeldedMesh::build(prim) else {
        return prim.clone();
    };
    let mut mesh = mesh;

    let target_faces = if target_ratio <= 0.0 {
        0
    } else {
        (face_count as f64 * target_ratio as f64).round() as usize
    };
    mesh.decimate(target_faces);
    mesh.into_primitive(prim)
}

/// The mesh in the form decimation needs: positionally welded vertices, face
/// adjacency, and one quadric per vertex.
struct WeldedMesh {
    positions: Vec<DVec3>,
    /// Attributes for each welded vertex, blended as collapses proceed.
    attributes: Vec<Vertex>,
    quadrics: Vec<Quadric>,
    alive: Vec<bool>,
    versions: Vec<u32>,
    /// Face indices touching each vertex. May contain removed faces and
    /// duplicates; consumers filter.
    incident: Vec<Vec<u32>>,
    faces: Vec<[u32; 3]>,
    face_alive: Vec<bool>,
    live_faces: usize,
}

impl WeldedMesh {
    fn build(prim: &CpuPrimitive) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &prim.vertices {
            let p = Vec3::from_array(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let diagonal = (max - min).length();
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return None;
        }

        // Weld first. A glTF index buffer routinely stores the same corner
        // several times (split UVs, split normals), and every one of those
        // duplicates makes its edges look like boundary edges: two faces meet
        // there geometrically but reference different indices. Without
        // welding, BOUNDARY_WEIGHT would pin the entire interior of a
        // hard-edged mesh and nothing would decimate.
        let epsilon = (diagonal as f64) * 1e-6;
        let mut key_to_welded: HashMap<[i64; 3], u32> = HashMap::new();
        let mut remap = Vec::with_capacity(prim.vertices.len());
        let mut positions: Vec<DVec3> = Vec::new();
        let mut attributes: Vec<Vertex> = Vec::new();

        for v in &prim.vertices {
            let p = DVec3::new(
                v.position[0] as f64,
                v.position[1] as f64,
                v.position[2] as f64,
            );
            let key = [
                (p.x / epsilon).round() as i64,
                (p.y / epsilon).round() as i64,
                (p.z / epsilon).round() as i64,
            ];
            let index = *key_to_welded.entry(key).or_insert_with(|| {
                positions.push(p);
                attributes.push(*v);
                (positions.len() - 1) as u32
            });
            remap.push(index);
        }

        let mut faces: Vec<[u32; 3]> = Vec::with_capacity(prim.indices.len() / 3);
        for tri in prim.indices.chunks_exact(3) {
            let (a, b, c) = (
                remap[tri[0] as usize],
                remap[tri[1] as usize],
                remap[tri[2] as usize],
            );
            if a != b && b != c && a != c {
                faces.push([a, b, c]);
            }
        }
        if faces.is_empty() {
            return None;
        }

        let vertex_count = positions.len();
        let mut incident = vec![Vec::new(); vertex_count];
        for (face_index, face) in faces.iter().enumerate() {
            for &v in face {
                incident[v as usize].push(face_index as u32);
            }
        }

        let mut mesh = Self {
            positions,
            attributes,
            quadrics: vec![Quadric::default(); vertex_count],
            alive: vec![true; vertex_count],
            versions: vec![0; vertex_count],
            incident,
            live_faces: faces.len(),
            face_alive: vec![true; faces.len()],
            faces,
        };
        mesh.accumulate_quadrics();
        Some(mesh)
    }

    /// Face quadrics (area-weighted) plus boundary constraint quadrics.
    ///
    /// Area weighting is Garland's: an unweighted sum lets a fan of slivers
    /// outvote one large face that actually describes the surface.
    fn accumulate_quadrics(&mut self) {
        for face_index in 0..self.faces.len() {
            let face = self.faces[face_index];
            let Some((normal, area)) = self.face_plane(face) else {
                continue;
            };
            let d = -normal.dot(self.positions[face[0] as usize]);
            let q = Quadric::from_plane(normal, d, area);
            for &v in &face {
                self.quadrics[v as usize].add(&q);
            }
        }

        // Count faces per undirected edge to find the open rim.
        let mut edge_faces: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for (face_index, face) in self.faces.iter().enumerate() {
            for i in 0..3 {
                let key = edge_key(face[i], face[(i + 1) % 3]);
                let entry = edge_faces.entry(key).or_insert((0, face_index as u32));
                entry.0 += 1;
            }
        }

        // Sorted so the accumulation order (and thus f64 rounding) does not
        // depend on HashMap iteration order.
        let mut boundary: Vec<((u32, u32), u32)> = edge_faces
            .iter()
            .filter(|(_, (count, _))| *count == 1)
            .map(|(k, (_, face))| (*k, *face))
            .collect();
        boundary.sort_unstable();

        for ((a, b), face_index) in boundary {
            let face = self.faces[face_index as usize];
            let Some((normal, _)) = self.face_plane(face) else {
                continue;
            };
            let pa = self.positions[a as usize];
            let edge = self.positions[b as usize] - pa;
            let length_sq = edge.length_squared();
            let plane_normal = edge.cross(normal);
            if plane_normal.length_squared() <= 0.0 || length_sq <= 0.0 {
                continue;
            }
            let plane_normal = plane_normal.normalize();
            let d = -plane_normal.dot(pa);
            let q = Quadric::from_plane(plane_normal, d, BOUNDARY_WEIGHT * length_sq);
            self.quadrics[a as usize].add(&q);
            self.quadrics[b as usize].add(&q);
        }
    }

    /// Unit normal and area of a face, or `None` if it is degenerate.
    fn face_plane(&self, face: [u32; 3]) -> Option<(DVec3, f64)> {
        let p0 = self.positions[face[0] as usize];
        let cross =
            (self.positions[face[1] as usize] - p0).cross(self.positions[face[2] as usize] - p0);
        let length = cross.length();
        if length <= 0.0 || !length.is_finite() {
            return None;
        }
        Some((cross / length, 0.5 * length))
    }

    /// Face normal with `replaced` moved to `position` (unnormalised).
    fn face_normal_with(&self, face: [u32; 3], replaced: u32, position: DVec3) -> DVec3 {
        let at = |v: u32| {
            if v == replaced {
                position
            } else {
                self.positions[v as usize]
            }
        };
        (at(face[1]) - at(face[0])).cross(at(face[2]) - at(face[0]))
    }

    fn unique_edges(&self) -> Vec<(u32, u32)> {
        let mut edges: Vec<(u32, u32)> = self
            .faces
            .iter()
            .flat_map(|f| {
                [
                    edge_key(f[0], f[1]),
                    edge_key(f[1], f[2]),
                    edge_key(f[2], f[0]),
                ]
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// Collapse target and cost for an edge.
    ///
    /// The fallback is not an edge case: on any flat or straight-crease
    /// region the 3x3 is singular, and returning "no collapse" there would
    /// leave exactly the regions that are cheapest to simplify untouched.
    fn evaluate(&self, v0: u32, v1: u32) -> (DVec3, f64) {
        let mut q = self.quadrics[v0 as usize];
        q.add(&self.quadrics[v1 as usize]);
        let p0 = self.positions[v0 as usize];
        let p1 = self.positions[v1 as usize];

        if let Some(optimal) = q.optimal_position() {
            let cost = q.error(optimal);
            // A negative error is round-off on an exactly-zero quadric, not a
            // better-than-perfect placement.
            if cost.is_finite() && cost >= -1e-9 {
                return (optimal, cost.max(0.0));
            }
        }

        let mut best = (p0, q.error(p0));
        for candidate in [p1, 0.5 * (p0 + p1)] {
            let cost = q.error(candidate);
            if cost < best.1 {
                best = (candidate, cost);
            }
        }
        (
            best.0,
            if best.1.is_finite() {
                best.1.max(0.0)
            } else {
                f64::MAX
            },
        )
    }

    fn push(&self, heap: &mut BinaryHeap<Candidate>, v0: u32, v1: u32) {
        let (a, b) = edge_key(v0, v1);
        if a == b || !self.alive[a as usize] || !self.alive[b as usize] {
            return;
        }
        let (target, cost) = self.evaluate(a, b);
        heap.push(Candidate {
            key: cost_key(cost),
            v0: a,
            v1: b,
            version0: self.versions[a as usize],
            version1: self.versions[b as usize],
            target,
        });
    }

    fn decimate(&mut self, target_faces: usize) {
        let mut heap = BinaryHeap::new();
        for (a, b) in self.unique_edges() {
            self.push(&mut heap, a, b);
        }

        while self.live_faces > target_faces {
            let Some(candidate) = heap.pop() else { break };
            let (v0, v1) = (candidate.v0 as usize, candidate.v1 as usize);
            if !self.alive[v0]
                || !self.alive[v1]
                || self.versions[v0] != candidate.version0
                || self.versions[v1] != candidate.version1
            {
                continue;
            }
            if !self.collapse(candidate.v0, candidate.v1, candidate.target) {
                continue;
            }
            let neighbours = self.neighbours(candidate.v0);
            for n in neighbours {
                self.push(&mut heap, candidate.v0, n);
            }
        }
    }

    /// Live faces touching `v`, deduplicated.
    fn live_incident(&self, v: u32) -> Vec<u32> {
        let mut faces: Vec<u32> = self.incident[v as usize]
            .iter()
            .copied()
            .filter(|&f| self.face_alive[f as usize])
            .collect();
        faces.sort_unstable();
        faces.dedup();
        faces
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut result: Vec<u32> = self
            .live_incident(v)
            .into_iter()
            .flat_map(|f| self.faces[f as usize])
            .filter(|&n| n != v)
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Collapses `v1` into `v0` at `target`. Returns false (changing
    /// nothing) if the collapse would fold the surface.
    fn collapse(&mut self, v0: u32, v1: u32, target: DVec3) -> bool {
        let faces0 = self.live_incident(v0);
        let faces1 = self.live_incident(v1);

        // Faces containing both endpoints vanish; every other incident face
        // is reshaped and must keep its orientation. Skipping this check is
        // what makes a naive QEM implementation produce visibly folded
        // geometry - the metric itself is happy to put a vertex on the far
        // side of its own one-ring.
        for (&face_index, moved) in faces0
            .iter()
            .map(|f| (f, v0))
            .chain(faces1.iter().map(|f| (f, v1)))
        {
            let face = self.faces[face_index as usize];
            if face.contains(&v0) && face.contains(&v1) {
                continue;
            }
            let Some((before, _)) = self.face_plane(face) else {
                continue;
            };
            let after = self.face_normal_with(face, moved, target);
            if after.length_squared() <= 0.0 || before.dot(after) <= 0.0 {
                return false;
            }
        }

        let p0 = self.positions[v0 as usize];
        let p1 = self.positions[v1 as usize];
        self.attributes[v0 as usize] = blend(
            &self.attributes[v0 as usize],
            &self.attributes[v1 as usize],
            interpolation_parameter(p0, p1, target),
        );
        self.positions[v0 as usize] = target;
        self.attributes[v0 as usize].position = target.as_vec3().to_array();

        let mut removed = 0usize;
        for &face_index in faces1.iter() {
            let face = &mut self.faces[face_index as usize];
            if face.contains(&v0) {
                self.face_alive[face_index as usize] = false;
                removed += 1;
                continue;
            }
            for slot in face.iter_mut() {
                if *slot == v1 {
                    *slot = v0;
                }
            }
            self.incident[v0 as usize].push(face_index);
        }
        self.live_faces -= removed;

        self.alive[v1 as usize] = false;
        let q1 = self.quadrics[v1 as usize];
        self.quadrics[v0 as usize].add(&q1);
        self.versions[v0 as usize] += 1;
        self.versions[v1 as usize] += 1;

        // Only v0 moved and only v0's quadric grew, so only edges incident
        // to v0 need requeueing - and the version bump must be confined to
        // v0 and v1 for the same reason. A first attempt also bumped the
        // one-ring, which invalidated every queued (n, m) edge among the
        // neighbours without requeueing them; the heap drained, decimation
        // stalled far short of the target, and the spike test's tip was
        // removed by one of the few collapses still reachable.
        true
    }

    fn into_primitive(self, source: &CpuPrimitive) -> CpuPrimitive {
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut vertices = Vec::new();
        let mut indices = Vec::with_capacity(self.live_faces * 3);

        for (face_index, face) in self.faces.iter().enumerate() {
            if !self.face_alive[face_index] {
                continue;
            }
            if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                continue;
            }
            for &v in face {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = vertices.len() as u32;
                    vertices.push(self.attributes[v as usize]);
                }
                indices.push(remap[v as usize]);
            }
        }

        CpuPrimitive {
            vertices,
            indices,
            transform: source.transform,
            node_index: source.node_index,
            skin_index: source.skin_index,
            material: source.material.clone(),
            morph_targets: Vec::new(),
            morph_weights: Vec::new(),
        }
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Where `target` falls along the collapsed edge, clamped to it.
///
/// The optimal QEM position is generally OFF the segment, so this is a
/// projection rather than a true parameter - but it still gives the nearer
/// endpoint the larger share, which is what attribute blending wants.
fn interpolation_parameter(p0: DVec3, p1: DVec3, target: DVec3) -> f32 {
    let edge = p1 - p0;
    let length_sq = edge.length_squared();
    if length_sq <= 0.0 {
        return 0.5;
    }
    ((target - p0).dot(edge) / length_sq).clamp(0.0, 1.0) as f32
}

/// Blends vertex attributes for a collapse, `t` = 0 keeps `a`.
///
/// Skin joints are NOT interpolated: joint slots are indices, and the mean
/// of joint 3 and joint 9 is joint 6, which is a different bone. The nearer
/// endpoint's binding is taken whole, together with its weights.
fn blend(a: &Vertex, b: &Vertex, t: f32) -> Vertex {
    let lerp2 = |x: [f32; 2], y: [f32; 2]| [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t];
    let na = Vec3::from_array(a.normal);
    let nb = Vec3::from_array(b.normal);
    let normal = (na + (nb - na) * t).normalize_or_zero();
    let ta = Vec3::new(a.tangent[0], a.tangent[1], a.tangent[2]);
    let tb = Vec3::new(b.tangent[0], b.tangent[1], b.tangent[2]);
    let tangent = (ta + (tb - ta) * t).normalize_or_zero();
    let nearer = if t < 0.5 { a } else { b };

    Vertex {
        position: a.position,
        normal: if normal == Vec3::ZERO {
            nearer.normal
        } else {
            normal.to_array()
        },
        uv: lerp2(a.uv, b.uv),
        tangent: if tangent == Vec3::ZERO {
            nearer.tangent
        } else {
            [
                tangent.x,
                tangent.y,
                tangent.z,
                // Handedness is a sign, not a quantity; averaging +1 and -1
                // yields 0, which is not a valid bitangent direction.
                nearer.tangent[3],
            ]
        },
        joints: nearer.joints,
        weights: nearer.weights,
        color: [
            a.color[0] + (b.color[0] - a.color[0]) * t,
            a.color[1] + (b.color[1] - a.color[1]) * t,
            a.color[2] + (b.color[2] - a.color[2]) * t,
            a.color[3] + (b.color[3] - a.color[3]) * t,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::CpuMaterial;
    use glam::Mat4;

    fn vertex(position: [f32; 3]) -> Vertex {
        Vertex {
            position,
            normal: [0.0, 0.0, 1.0],
            uv: [position[0], position[1]],
            tangent: [1.0, 0.0, 0.0, 1.0],
            joints: [0.0; 4],
            weights: [0.0; 4],
            color: [1.0, 1.0, 1.0, 1.0],
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
            morph_targets: Vec::new(),
            morph_weights: Vec::new(),
        }
    }

    fn grid(n: usize) -> CpuPrimitive {
        let mut vertices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                vertices.push(vertex([x as f32 / n as f32, y as f32 / n as f32, 0.0]));
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
        primitive(vertices, indices)
    }

    #[test]
    fn quadric_of_a_plane_is_zero_on_the_plane() {
        let q = Quadric::from_plane(DVec3::Z, -2.0, 1.0);
        assert!(q.error(DVec3::new(5.0, -3.0, 2.0)).abs() < 1e-12);
        // Squared distance off the plane.
        assert!((q.error(DVec3::new(0.0, 0.0, 5.0)) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn three_orthogonal_planes_solve_to_their_corner() {
        let mut q = Quadric::from_plane(DVec3::X, -1.0, 1.0);
        q.add(&Quadric::from_plane(DVec3::Y, -2.0, 1.0));
        q.add(&Quadric::from_plane(DVec3::Z, -3.0, 1.0));
        let solved = q.optimal_position().expect("well-conditioned");
        assert!((solved - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-9);
    }

    #[test]
    fn coplanar_quadric_is_reported_singular() {
        // The case the fallback exists for: one plane summed many times is
        // still rank 1, and the optimal position is a whole plane of points.
        let mut q = Quadric::default();
        for _ in 0..64 {
            q.add(&Quadric::from_plane(DVec3::Z, 0.0, 1.0));
        }
        assert!(q.optimal_position().is_none());
    }

    #[test]
    fn welding_merges_duplicated_positions() {
        // Two triangles sharing an edge, but with the shared corners stored
        // twice - the split-UV case that makes every edge look like a
        // boundary until welding fixes it.
        let vertices = vec![
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([1.0, 1.0, 0.0]),
        ];
        let prim = primitive(vertices, vec![0, 1, 2, 3, 5, 4]);
        let mesh = WeldedMesh::build(&prim).expect("weldable");
        assert_eq!(mesh.positions.len(), 4);
    }

    #[test]
    fn empty_and_tiny_meshes_do_not_panic() {
        let empty = primitive(Vec::new(), Vec::new());
        assert!(simplify_primitive_qem(&empty, 0.5).indices.is_empty());

        let single = primitive(
            vec![
                vertex([0.0, 0.0, 0.0]),
                vertex([1.0, 0.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            vec![0, 1, 2],
        );
        for ratio in [0.0, 0.5, 1.0, 2.0, -1.0, f32::NAN] {
            let out = simplify_primitive_qem(&single, ratio);
            assert_eq!(out.indices.len() % 3, 0);
        }

        let mut degenerate = grid(4);
        for v in &mut degenerate.vertices {
            v.position = [1.0, 1.0, 1.0];
        }
        let out = simplify_primitive_qem(&degenerate, 0.5);
        assert_eq!(out.vertices.len(), degenerate.vertices.len());
    }

    #[test]
    fn ratio_at_or_above_one_is_a_passthrough() {
        let full = grid(8);
        let out = simplify_primitive_qem(&full, 1.0);
        assert_eq!(out.indices, full.indices);
    }

    #[test]
    fn collapse_that_folds_a_face_is_rejected() {
        // A fan around a centre vertex: dragging the rim vertex across the
        // fan inverts the faces on the far side.
        let mut mesh = WeldedMesh::build(&grid(5)).expect("weldable");
        let far = DVec3::new(-50.0, -50.0, 0.0);
        assert!(
            !mesh.collapse(12, 13, far),
            "a collapse to a wildly out-of-ring position must be rejected"
        );
    }
}
