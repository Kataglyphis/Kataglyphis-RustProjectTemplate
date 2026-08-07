//! KTX2 loading for block-compressed textures (BCn passthrough).
//!
//! Scope: containers whose payload is already a GPU format we can upload
//! directly (BC1/BC3/BC5/BC7), with no supercompression. Basis
//! ETC1S/UASTC transcoding needs a transcoder dependency and is not
//! handled yet — such files are reported as an error so callers can fall
//! back instead of rendering garbage.

use anyhow::Context as _;

use crate::scene::{CompressedFormat, CompressedTexture, CpuTexture};

/// VkFormat values used by KTX2 for the BCn formats we accept.
const VK_FORMAT_BC1_RGBA_UNORM_BLOCK: u32 = 133;
const VK_FORMAT_BC1_RGBA_SRGB_BLOCK: u32 = 134;
const VK_FORMAT_BC3_UNORM_BLOCK: u32 = 137;
const VK_FORMAT_BC3_SRGB_BLOCK: u32 = 138;
const VK_FORMAT_BC5_UNORM_BLOCK: u32 = 141;
const VK_FORMAT_BC7_UNORM_BLOCK: u32 = 145;
const VK_FORMAT_BC7_SRGB_BLOCK: u32 = 146;

fn map_format(vk_format: u32) -> Option<CompressedFormat> {
    match vk_format {
        VK_FORMAT_BC1_RGBA_UNORM_BLOCK | VK_FORMAT_BC1_RGBA_SRGB_BLOCK => {
            Some(CompressedFormat::Bc1RgbaUnorm)
        }
        VK_FORMAT_BC3_UNORM_BLOCK | VK_FORMAT_BC3_SRGB_BLOCK => {
            Some(CompressedFormat::Bc3RgbaUnorm)
        }
        VK_FORMAT_BC5_UNORM_BLOCK => Some(CompressedFormat::Bc5RgUnorm),
        VK_FORMAT_BC7_UNORM_BLOCK | VK_FORMAT_BC7_SRGB_BLOCK => {
            Some(CompressedFormat::Bc7RgbaUnorm)
        }
        _ => None,
    }
}

/// The colour space a KTX2 vkFormat declares: `Some(true)` for a
/// `*_SRGB_BLOCK` format, `Some(false)` for a colour `*_UNORM_BLOCK` format,
/// `None` for a data format with no colour space to declare (BC5).
fn declared_srgb_for(vk_format: u32) -> Option<bool> {
    match vk_format {
        VK_FORMAT_BC1_RGBA_SRGB_BLOCK | VK_FORMAT_BC3_SRGB_BLOCK | VK_FORMAT_BC7_SRGB_BLOCK => {
            Some(true)
        }
        VK_FORMAT_BC1_RGBA_UNORM_BLOCK | VK_FORMAT_BC3_UNORM_BLOCK | VK_FORMAT_BC7_UNORM_BLOCK => {
            Some(false)
        }
        _ => None,
    }
}

/// Rejects a mip chain that cannot possibly belong to a `width`x`height`
/// texture in `format`: an empty chain, more levels than the full mip
/// pyramid allows, or any level whose data is too small for the block
/// dimensions it is claimed to cover. Pure and adapter-free, so it can run
/// long before any wgpu resource exists.
pub(crate) fn validate_mip_chain(
    width: u32,
    height: u32,
    format: CompressedFormat,
    mips: &[Vec<u8>],
) -> anyhow::Result<()> {
    anyhow::ensure!(!mips.is_empty(), "KTX2 contains no mip levels");

    let max_levels = 32 - width.max(height).leading_zeros();
    anyhow::ensure!(
        mips.len() as u32 <= max_levels,
        "KTX2 declares {} mip levels but a {width}x{height} texture has at most {max_levels}",
        mips.len()
    );

    let block_bytes = format.block_bytes();
    for (level, data) in mips.iter().enumerate() {
        let w = (width >> level).max(1);
        let h = (height >> level).max(1);
        let expected = w.div_ceil(4) * h.div_ceil(4) * block_bytes;
        anyhow::ensure!(
            data.len() as u32 >= expected,
            "KTX2 mip level {level} has {} bytes but a {w}x{h} block-compressed \
             level needs at least {expected}",
            data.len()
        );
    }

    Ok(())
}

/// Parses a KTX2 file into a `CpuTexture` carrying its compressed mip chain.
pub fn load_ktx2(bytes: &[u8]) -> anyhow::Result<CpuTexture> {
    let reader = ktx2::Reader::new(bytes).context("not a valid KTX2 container")?;
    let header = reader.header();

    anyhow::ensure!(
        header.supercompression_scheme.is_none(),
        "KTX2 supercompression {:?} is not supported yet (Basis transcoding pending)",
        header.supercompression_scheme
    );

    anyhow::ensure!(
        header.face_count <= 1,
        "KTX2 cubemaps ({} faces) are not supported yet, only 2D textures",
        header.face_count
    );
    anyhow::ensure!(
        header.layer_count <= 1,
        "KTX2 array textures ({} layers) are not supported yet, only 2D textures",
        header.layer_count
    );
    anyhow::ensure!(
        header.pixel_depth <= 1,
        "KTX2 3D textures (depth {}) are not supported yet, only 2D textures",
        header.pixel_depth
    );

    let vk_format = header
        .format
        .context("KTX2 has no vkFormat (Basis-only files need a transcoder)")?
        .value();
    let format = map_format(vk_format)
        .with_context(|| format!("unsupported KTX2 vkFormat {vk_format} (expected BC1/3/5/7)"))?;
    let declared_srgb = declared_srgb_for(vk_format);

    let mips: Vec<Vec<u8>> = reader.levels().map(|level| level.data.to_vec()).collect();
    anyhow::ensure!(!mips.is_empty(), "KTX2 contains no mip levels");

    let width = header.pixel_width.max(1);
    let height = header.pixel_height.max(1);
    validate_mip_chain(width, height, format, &mips)
        .context("KTX2 mip chain does not match the declared dimensions")?;

    Ok(CpuTexture {
        width,
        height,
        rgba8: Vec::new(),
        compressed: Some(CompressedTexture {
            format,
            mips,
            declared_srgb,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_format_accepts_bcn_and_rejects_others() {
        assert_eq!(
            map_format(VK_FORMAT_BC1_RGBA_SRGB_BLOCK),
            Some(CompressedFormat::Bc1RgbaUnorm)
        );
        assert_eq!(
            map_format(VK_FORMAT_BC3_UNORM_BLOCK),
            Some(CompressedFormat::Bc3RgbaUnorm)
        );
        assert_eq!(
            map_format(VK_FORMAT_BC5_UNORM_BLOCK),
            Some(CompressedFormat::Bc5RgUnorm)
        );
        assert_eq!(
            map_format(VK_FORMAT_BC7_SRGB_BLOCK),
            Some(CompressedFormat::Bc7RgbaUnorm)
        );
        // An uncompressed format (VK_FORMAT_R8G8B8A8_UNORM = 37) is not a BCn
        // passthrough target.
        assert_eq!(map_format(37), None);
        assert_eq!(map_format(0), None);
    }

    #[test]
    fn declared_srgb_follows_the_container_vkformat() {
        assert_eq!(declared_srgb_for(VK_FORMAT_BC7_SRGB_BLOCK), Some(true));
        assert_eq!(declared_srgb_for(VK_FORMAT_BC1_RGBA_SRGB_BLOCK), Some(true));
        assert_eq!(
            declared_srgb_for(VK_FORMAT_BC1_RGBA_UNORM_BLOCK),
            Some(false)
        );
        assert_eq!(declared_srgb_for(VK_FORMAT_BC3_UNORM_BLOCK), Some(false));
        assert_eq!(
            declared_srgb_for(VK_FORMAT_BC5_UNORM_BLOCK),
            None,
            "BC5 is a two-channel data format with no colour space to declare"
        );
    }

    #[test]
    fn loads_a_valid_bc1_container() {
        let bytes = include_bytes!("../../tests/assets/red_bc1.ktx2");
        let tex = load_ktx2(bytes).expect("red_bc1.ktx2 should load");
        assert!(
            tex.width >= 1 && tex.height >= 1,
            "dimensions must be positive"
        );
        assert!(
            tex.rgba8.is_empty(),
            "a compressed texture carries no rgba8"
        );
        let compressed = tex
            .compressed
            .expect("BC1 file must produce a compressed payload");
        assert_eq!(compressed.format, CompressedFormat::Bc1RgbaUnorm);
        assert!(
            !compressed.mips.is_empty(),
            "must have at least one mip level"
        );
        assert!(
            !compressed.mips[0].is_empty(),
            "the base mip must carry block data"
        );
    }

    #[test]
    fn rejects_non_ktx2_bytes_without_panicking() {
        // Garbage in -> a graceful Err, never a panic.
        assert!(load_ktx2(b"not a ktx2 file at all").is_err());
        assert!(load_ktx2(&[]).is_err());
    }

    /// A correctly-sized 4x4 BC1 mip chain: one 4x4 block per level, 8 bytes
    /// (one block) apiece, down to the 1x1 level.
    fn valid_4x4_bc1_chain(levels: usize) -> Vec<Vec<u8>> {
        (0..levels).map(|_| vec![0u8; 8]).collect()
    }

    #[test]
    fn validate_mip_chain_rejects_more_levels_than_the_dimensions_allow() {
        // 4x4 has a full pyramid of 3 levels (4x4, 2x2, 1x1); a 5th level
        // cannot correspond to any real mip of a 4x4 texture.
        let five_levels = valid_4x4_bc1_chain(5);
        assert!(validate_mip_chain(4, 4, CompressedFormat::Bc1RgbaUnorm, &five_levels).is_err());

        let three_levels = valid_4x4_bc1_chain(3);
        assert!(validate_mip_chain(4, 4, CompressedFormat::Bc1RgbaUnorm, &three_levels).is_ok());
    }

    #[test]
    fn validate_mip_chain_rejects_a_truncated_level() {
        let mut chain = valid_4x4_bc1_chain(3);
        chain.last_mut().unwrap().pop();
        assert!(validate_mip_chain(4, 4, CompressedFormat::Bc1RgbaUnorm, &chain).is_err());
    }

    #[test]
    fn validate_mip_chain_accepts_a_base_only_chain() {
        let chain = valid_4x4_bc1_chain(1);
        assert!(validate_mip_chain(4, 4, CompressedFormat::Bc1RgbaUnorm, &chain).is_ok());
    }
}
