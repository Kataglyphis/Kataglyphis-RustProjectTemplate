//! Tile-light-grid binning for tiled/clustered lighting.
//!
//! Pure CPU math: given the frame's packed punctual lights and a view-proj
//! matrix, bin each light into the screen-space tiles its footprint
//! overlaps. Binning is **range-aware** — a light covers the screen-space
//! rectangle its `range` subtends, not just the single tile its centre
//! projects to, so a large or off-centre light still lights every tile it
//! actually reaches. `tileW == 0` (an empty grid) is the shader's documented
//! all-lights fallback.

use glam::{Mat4, Vec4Swizzles};

use crate::render::forward::MAX_PUNCTUAL_LIGHTS;

/// Tile size for clustered/tiled lighting. Each tile is TILE_SIZE×TILE_SIZE
/// pixels in screen space. Lights are binned per tile so that each fragment
/// iterates only the lights overlapping its tile.
pub const TILE_SIZE: u32 = 16;
/// Maximum number of lights that can overlap a single tile (conservative).
pub const MAX_LIGHTS_PER_TILE: u32 = 32;

/// Reusable scratch storage for [`build_tile_light_grid`], hoisted onto
/// `ForwardRenderer` so a frame does not make four heap allocations over the
/// screen's ~8100 tiles. `grid` and `indices` are the buffers uploaded to the
/// GPU; `per_tile_counts`/`write_positions` are pass-internal.
#[derive(Default)]
pub(crate) struct TileLightGridScratch {
    pub(crate) per_tile_counts: Vec<u32>,
    pub(crate) grid: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) write_positions: Vec<u32>,
}

/// The inclusive tile range `[min_tx, min_ty, max_tx, max_ty]` a light's
/// screen-space footprint covers, or `None` if it contributes no visible
/// tile (fully behind the camera).
///
/// Directional lights (`kind == 3.0`) have no position and light every
/// pixel, so they cover the whole grid. Point/spot lights are approximated by
/// projecting the light centre plus the six `pos ± range * axis` points and
/// taking the screen-space AABB of whichever of those seven land in front of
/// the camera (`clip.w > 0.0`) — a conservative bound, not an exact
/// cone/sphere-vs-frustum test, which is not worth it at this light count.
fn tile_rect_for_light(
    pos: glam::Vec3,
    range: f32,
    kind: f32,
    tile_x: u32,
    tile_y: u32,
    width: u32,
    height: u32,
    view_proj: &Mat4,
) -> Option<(u32, u32, u32, u32)> {
    if kind == 3.0 {
        return Some((0, 0, tile_x.saturating_sub(1), tile_y.saturating_sub(1)));
    }

    let candidates = [
        pos,
        pos + glam::Vec3::X * range,
        pos - glam::Vec3::X * range,
        pos + glam::Vec3::Y * range,
        pos - glam::Vec3::Y * range,
        pos + glam::Vec3::Z * range,
        pos - glam::Vec3::Z * range,
    ];

    let mut min_uv = glam::Vec2::splat(f32::MAX);
    let mut max_uv = glam::Vec2::splat(f32::MIN);
    let mut any_in_front = false;

    for c in candidates {
        let clip = *view_proj * glam::Vec4::new(c.x, c.y, c.z, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        any_in_front = true;
        let ndc = clip.xy() / clip.w;
        let uv = ndc * 0.5 + 0.5;
        min_uv = min_uv.min(uv);
        max_uv = max_uv.max(uv);
    }

    if !any_in_front {
        return None;
    }

    let min_uv = min_uv.clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
    let max_uv = max_uv.clamp(glam::Vec2::ZERO, glam::Vec2::ONE);

    let last_x = width.saturating_sub(1);
    let last_y = height.saturating_sub(1);
    let min_tx = ((min_uv.x * width as f32) as u32).min(last_x) / TILE_SIZE;
    let min_ty = ((min_uv.y * height as f32) as u32).min(last_y) / TILE_SIZE;
    let max_tx = ((max_uv.x * width as f32) as u32).min(last_x) / TILE_SIZE;
    let max_ty = ((max_uv.y * height as f32) as u32).min(last_y) / TILE_SIZE;

    Some((
        min_tx.min(tile_x.saturating_sub(1)),
        min_ty.min(tile_y.saturating_sub(1)),
        max_tx.min(tile_x.saturating_sub(1)),
        max_ty.min(tile_y.saturating_sub(1)),
    ))
}

/// Build a per-tile light index list for tiled lighting.
/// Each tile gets a list of light indices whose range overlaps its
/// screen-space area (directional lights are in every tile). Results land in
/// `scratch.grid` ([count, offset] per tile) and `scratch.indices` (flat
/// list); both are read back by the caller and uploaded to the GPU.
pub(crate) fn build_tile_light_grid(
    scratch: &mut TileLightGridScratch,
    packed_lights: &[[f32; 4]; MAX_PUNCTUAL_LIGHTS * 4],
    light_count: u32,
    width: u32,
    height: u32,
    view_proj: &Mat4,
) {
    let tile_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tile_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    let total_tiles = (tile_x * tile_y) as usize;

    scratch.per_tile_counts.clear();
    scratch.per_tile_counts.resize(total_tiles, 0);

    for i in 0..light_count as usize {
        let base = i * 4;
        let pos = glam::Vec3::new(
            packed_lights[base][0],
            packed_lights[base][1],
            packed_lights[base][2],
        );
        let kind = packed_lights[base][3];
        let range = packed_lights[base + 1][3];
        let Some((min_tx, min_ty, max_tx, max_ty)) =
            tile_rect_for_light(pos, range, kind, tile_x, tile_y, width, height, view_proj)
        else {
            continue;
        };
        for ty in min_ty..=max_ty {
            let row = ty * tile_x;
            for tx in min_tx..=max_tx {
                let idx = (row + tx) as usize;
                scratch.per_tile_counts[idx] = scratch.per_tile_counts[idx].saturating_add(1);
            }
        }
    }

    // Prefix sum to compute offsets.
    scratch.grid.clear();
    scratch.grid.resize(total_tiles * 2, 0);
    let mut running_offset = 0u32;
    for t in 0..total_tiles {
        let cnt = scratch.per_tile_counts[t].min(MAX_LIGHTS_PER_TILE);
        scratch.grid[t * 2] = cnt; // count
        scratch.grid[t * 2 + 1] = running_offset; // offset
        running_offset += cnt;
    }

    // Fill indices. write_positions tracks the next free slot per tile,
    // bounded by that tile's [offset, offset + count) so a tile whose actual
    // hit count exceeds the MAX_LIGHTS_PER_TILE cap cannot spill into the
    // next tile's block.
    let max_indices = running_offset as usize;
    scratch.indices.clear();
    scratch.indices.resize(max_indices.max(1), 0);
    scratch.write_positions.clear();
    scratch.write_positions.resize(total_tiles, 0);
    for t in 0..total_tiles {
        scratch.write_positions[t] = scratch.grid[t * 2 + 1];
    }

    for i in 0..light_count as usize {
        let base = i * 4;
        let pos = glam::Vec3::new(
            packed_lights[base][0],
            packed_lights[base][1],
            packed_lights[base][2],
        );
        let kind = packed_lights[base][3];
        let range = packed_lights[base + 1][3];
        let Some((min_tx, min_ty, max_tx, max_ty)) =
            tile_rect_for_light(pos, range, kind, tile_x, tile_y, width, height, view_proj)
        else {
            continue;
        };
        for ty in min_ty..=max_ty {
            let row = ty * tile_x;
            for tx in min_tx..=max_tx {
                let idx = (row + tx) as usize;
                let end = scratch.grid[idx * 2 + 1] + scratch.grid[idx * 2];
                let wp = scratch.write_positions[idx];
                if wp < end {
                    scratch.indices[wp as usize] = i as u32;
                    scratch.write_positions[idx] = wp + 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn build_tile_light_grid_bins_lights_into_correct_tiles() {
        // Single light at origin, perspective camera at (0,0,5) looking at origin.
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;

        let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        // One point light at origin with a range large enough to span the
        // whole frustum at this distance - it now covers every tile, not
        // just the centre one, since binning is range-aware. The centre
        // tile containing it is what stays true; an exact count of 1 does
        // not (see build_tile_light_grid_covers_tiles_within_range for the
        // "more than one tile, still a bounded rectangle" case).
        packed[0] = [0.0, 0.0, 0.0, 1.0]; // position + kind=point
        packed[1] = [1.0, 1.0, 1.0, 10.0]; // color*intensity + range

        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 1, 256, 256, &vp);
        // The light at origin should cover the center tile.
        let tx = 128 / TILE_SIZE; // ~8 for 16px tiles
        let ty = 128 / TILE_SIZE;
        let tile_w = 256 / TILE_SIZE; // 16
        let center_tile = (ty * tile_w + tx) as usize;
        assert!(scratch.grid[center_tile * 2] >= 1, "center tile should contain the light");
        let offset = scratch.grid[center_tile * 2 + 1] as usize;
        let count = scratch.grid[center_tile * 2] as usize;
        assert!(
            scratch.indices[offset..offset + count].contains(&0),
            "center tile's light list should contain light index 0"
        );
    }

    #[test]
    fn build_tile_light_grid_handles_zero_lights() {
        let vp = Mat4::IDENTITY;
        let packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 0, 256, 256, &vp);
        assert!(scratch.grid.iter().all(|&c| c == 0), "all tiles should have zero lights");
        assert!(!scratch.indices.is_empty(), "indices buffer should exist");
    }

    #[test]
    fn build_tile_light_grid_skips_lights_behind_camera() {
        // Camera at (5,0,0) looking toward origin; light at (10,0,0) is
        // behind the camera, with a range too small to reach in front of it
        // (a range large enough to reach past the camera position would
        // legitimately light what's in front - see
        // build_tile_light_grid_keeps_offscreen_but_overlapping_lights for
        // that case).
        let view = Mat4::look_at_rh(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;

        let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        packed[0] = [10.0, 0.0, 0.0, 1.0];
        packed[1] = [1.0, 1.0, 1.0, 2.0];

        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 1, 256, 256, &vp);
        // All tiles should have zero lights since the light (and its range)
        // is entirely behind the camera.
        for chunk in scratch.grid.chunks(2) {
            assert_eq!(chunk[0], 0, "no light should reach any tile");
        }
    }

    #[test]
    fn build_tile_light_grid_covers_tiles_within_range() {
        // Point light at the origin with a modest range - large enough to
        // spill past the centre tile, small enough to stay a proper subset
        // of the screen so the "contiguous rectangle" shape is checkable.
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;

        let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        packed[0] = [0.0, 0.0, 0.0, 1.0];
        packed[1] = [1.0, 1.0, 1.0, 1.0]; // range = 1.0

        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 1, 256, 256, &vp);

        let tile_w = 256 / TILE_SIZE;
        let lit_tiles: Vec<(u32, u32)> = (0..tile_w)
            .flat_map(|ty| (0..tile_w).map(move |tx| (tx, ty)))
            .filter(|&(tx, ty)| scratch.grid[((ty * tile_w + tx) * 2) as usize] > 0)
            .collect();

        assert!(
            lit_tiles.len() > 1,
            "a light with nonzero range should cover more than one tile, got {}",
            lit_tiles.len()
        );

        let min_tx = lit_tiles.iter().map(|&(tx, _)| tx).min().unwrap();
        let max_tx = lit_tiles.iter().map(|&(tx, _)| tx).max().unwrap();
        let min_ty = lit_tiles.iter().map(|&(_, ty)| ty).min().unwrap();
        let max_ty = lit_tiles.iter().map(|&(_, ty)| ty).max().unwrap();
        let expected_count = ((max_tx - min_tx + 1) * (max_ty - min_ty + 1)) as usize;
        assert_eq!(
            lit_tiles.len(),
            expected_count,
            "lit tiles should form a contiguous rectangle, got {:?}",
            lit_tiles
        );

        let center_tx = 128 / TILE_SIZE;
        let center_ty = 128 / TILE_SIZE;
        assert!(min_tx <= center_tx && center_tx <= max_tx, "rectangle should surround the centre tile");
        assert!(min_ty <= center_ty && center_ty <= max_ty, "rectangle should surround the centre tile");
    }

    #[test]
    fn build_tile_light_grid_includes_directional_lights_in_every_tile() {
        let vp = Mat4::IDENTITY;
        let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        packed[0] = [0.0, -1.0, 0.0, 3.0]; // kind = directional; position unused
        packed[1] = [1.0, 1.0, 1.0, 0.0]; // color*intensity + range (unused)

        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 1, 256, 256, &vp);

        let total_tiles = scratch.grid.len() / 2;
        for t in 0..total_tiles {
            assert!(scratch.grid[t * 2] >= 1, "tile {} missing the directional light", t);
        }
    }

    #[test]
    fn build_tile_light_grid_keeps_offscreen_but_overlapping_lights() {
        // Light centre is off to the side, beyond the frustum's horizontal
        // extent, but its range reaches back onto the screen.
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;

        let mut packed = [[0.0f32; 4]; MAX_PUNCTUAL_LIGHTS * 4];
        packed[0] = [3.0, 0.0, 0.0, 1.0]; // centre projects outside [0,1]
        packed[1] = [1.0, 1.0, 1.0, 5.0]; // range reaches back onto the screen

        let mut scratch = TileLightGridScratch::default();
        build_tile_light_grid(&mut scratch, &packed, 1, 256, 256, &vp);

        let any_lit = scratch.grid.chunks(2).any(|c| c[0] > 0);
        assert!(
            any_lit,
            "a light whose range reaches the screen must light at least one tile even though its centre is off-screen"
        );
    }
}
