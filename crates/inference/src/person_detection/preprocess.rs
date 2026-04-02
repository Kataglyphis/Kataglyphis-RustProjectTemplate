use anyhow::{bail, Context, Result};

/// Standard YOLO letterbox padding value (114/255 ≈ 0.447).
/// This is the conventional gray value used for padding during letterbox resizing.
const LETTERBOX_FILL_VALUE: f32 = 114.0 / 255.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageMapping {
    pub src_w: u32,
    pub src_h: u32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

#[inline]
fn validate_rgba(rgba: &[u8], src_w: u32, src_h: u32) -> Result<()> {
    let expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|p| p.checked_mul(4))
        .context("Image dimensions overflow")?;
    if rgba.len() != expected {
        bail!(
            "RGBA buffer length mismatch: got {}, expected {} ({}x{}x4)",
            rgba.len(),
            expected,
            src_w,
            src_h
        );
    }
    Ok(())
}

#[inline]
fn write_pixel_chw(chw: &mut [f32], plane: usize, dst_idx: usize, r: f32, g: f32, b: f32) {
    chw[dst_idx] = r;
    chw[plane + dst_idx] = g;
    chw[2 * plane + dst_idx] = b;
}

#[inline]
fn sample_rgba_normalised(src: &[u8], sx: u32, sy: u32, src_w: u32) -> (f32, f32, f32) {
    let idx = ((sy * src_w + sx) * 4) as usize;
    let r = src[idx] as f32 / 255.0;
    let g = src[idx + 1] as f32 / 255.0;
    let b = src[idx + 2] as f32 / 255.0;
    (r, g, b)
}

pub(crate) fn rgba_to_nchw_f32_letterboxed(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    buf: &mut Vec<f32>,
) -> Result<ImageMapping> {
    validate_rgba(rgba, src_w, src_h)?;

    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (src_h as f32 * scale).round().max(1.0) as u32;
    let pad_x_i = (dst_w - new_w) / 2;
    let pad_y_i = (dst_h - new_h) / 2;

    let plane = (dst_w * dst_h) as usize;
    let total = 3 * plane;

    buf.clear();
    buf.resize(total, LETTERBOX_FILL_VALUE);

    let content_x_end = pad_x_i + new_w;
    let content_y_end = pad_y_i + new_h;

    for dy in pad_y_i..content_y_end {
        let sy = ((dy - pad_y_i) * src_h) / new_h;
        for dx in pad_x_i..content_x_end {
            let sx = ((dx - pad_x_i) * src_w) / new_w;
            let dst_idx = (dy * dst_w + dx) as usize;
            let (r, g, b) = sample_rgba_normalised(rgba, sx, sy, src_w);
            write_pixel_chw(buf, plane, dst_idx, r, g, b);
        }
    }

    Ok(ImageMapping {
        src_w,
        src_h,
        scale_x: scale,
        scale_y: scale,
        pad_x: (dst_w - new_w) as f32 / 2.0,
        pad_y: (dst_h - new_h) as f32 / 2.0,
    })
}

pub(crate) fn rgba_to_nchw_f32_stretched(
    rgba: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    buf: &mut Vec<f32>,
) -> Result<ImageMapping> {
    validate_rgba(rgba, src_w, src_h)?;

    let plane = (dst_w * dst_h) as usize;
    let total = 3 * plane;

    buf.clear();
    buf.resize(total, 0.0);

    for dy in 0..dst_h {
        let sy = (dy * src_h) / dst_h;
        for dx in 0..dst_w {
            let sx = (dx * src_w) / dst_w;
            let dst_idx = (dy * dst_w + dx) as usize;
            let (r, g, b) = sample_rgba_normalised(rgba, sx, sy, src_w);
            write_pixel_chw(buf, plane, dst_idx, r, g, b);
        }
    }

    Ok(ImageMapping {
        src_w,
        src_h,
        scale_x: dst_w as f32 / src_w.max(1) as f32,
        scale_y: dst_h as f32 / src_h.max(1) as f32,
        pad_x: 0.0,
        pad_y: 0.0,
    })
}

#[inline]
pub(crate) fn unmap_point(x: f32, y: f32, mapping: ImageMapping) -> (f32, f32) {
    let x = (x - mapping.pad_x) / mapping.scale_x.max(1e-6);
    let y = (y - mapping.pad_y) / mapping.scale_y.max(1e-6);
    (x, y)
}
