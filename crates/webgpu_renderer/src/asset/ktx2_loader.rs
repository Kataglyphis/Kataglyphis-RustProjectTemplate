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

/// Parses a KTX2 file into a `CpuTexture` carrying its compressed mip chain.
pub fn load_ktx2(bytes: &[u8]) -> anyhow::Result<CpuTexture> {
    let reader = ktx2::Reader::new(bytes).context("not a valid KTX2 container")?;
    let header = reader.header();

    anyhow::ensure!(
        header.supercompression_scheme.is_none(),
        "KTX2 supercompression {:?} is not supported yet (Basis transcoding pending)",
        header.supercompression_scheme
    );

    let vk_format = header
        .format
        .context("KTX2 has no vkFormat (Basis-only files need a transcoder)")?
        .value();
    let format = map_format(vk_format)
        .with_context(|| format!("unsupported KTX2 vkFormat {vk_format} (expected BC1/3/5/7)"))?;

    let mips: Vec<Vec<u8>> = reader.levels().map(|level| level.data.to_vec()).collect();
    anyhow::ensure!(!mips.is_empty(), "KTX2 contains no mip levels");

    Ok(CpuTexture {
        width: header.pixel_width.max(1),
        height: header.pixel_height.max(1),
        rgba8: Vec::new(),
        compressed: Some(CompressedTexture { format, mips }),
    })
}
