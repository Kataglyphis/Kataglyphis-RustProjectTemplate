use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use log::warn;

use crate::resource_monitor;

#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub id: u64,
    pub data: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
}

/// Convert a (possibly stride-padded) RGBA buffer into a tightly-packed one.
///
/// When the source buffer is already tightly packed (`src.len() == width * height * 4`),
/// the data is copied directly into an `Arc<[u8]>` — one allocation instead of two
/// (`Vec` + `Arc`).  When stride-stripping is needed, a temporary `Vec` is used.
pub(crate) fn rgba_tightly_packed(src: &[u8], width: u32, height: u32) -> Option<Arc<[u8]>> {
    if width == 0 || height == 0 {
        return None;
    }

    let row_bytes = (width as usize).saturating_mul(4);
    let expected_len = row_bytes.saturating_mul(height as usize);

    // Fast path: already tightly packed — copy straight into Arc.
    if src.len() == expected_len {
        return Some(Arc::from(src));
    }

    // Many GStreamer buffers are padded per row (stride). We conservatively try to
    // interpret the buffer as a single RGBA plane with a constant stride.
    if src.len() < expected_len {
        return None;
    }

    let h = height as usize;
    // The largest valid stride is `src.len() / h` (integer division rounds
    // down), so `stride * h <= src.len()` is guaranteed — no loop needed.
    let stride = src.len() / h;
    if stride < row_bytes {
        return None;
    }

    let mut out = vec![0u8; expected_len];
    for y in 0..h {
        let src_start = y.saturating_mul(stride);
        let src_end = src_start.saturating_add(row_bytes).min(src.len());
        let dst_start = y.saturating_mul(row_bytes);
        let dst_end = dst_start.saturating_add(row_bytes).min(out.len());

        if src_end <= src_start || dst_end <= dst_start {
            return None;
        }

        out[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
    }

    Some(Arc::from(out))
}

pub(crate) fn build_pipeline(
    frame_tx: std::sync::mpsc::SyncSender<Frame>,
) -> Result<gst::Pipeline> {
    let pipeline = gst::Pipeline::new();

    let frame_id = AtomicU64::new(1);

    let src = gst::ElementFactory::make("mfvideosrc")
        .build()
        .or_else(|_| gst::ElementFactory::make("autovideosrc").build())
        .context("Failed to create a video source")?;

    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .context("Failed to create videoconvert")?;

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .build();

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()
        .context("Failed to create capsfilter")?;

    let sink = gst::ElementFactory::make("appsink")
        .property("emit-signals", true)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()
        .context("Failed to create appsink")?;

    pipeline
        .add_many([&src, &convert, &capsfilter, &sink])
        .context("Failed to add elements to pipeline")?;
    gst::Element::link_many([&src, &convert, &capsfilter, &sink])
        .context("Failed to link pipeline elements")?;

    let appsink = sink
        .clone()
        .dynamic_cast::<gst_app::AppSink>()
        .map_err(|_| anyhow::anyhow!("Sink element is not an AppSink"))?;

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                if let Ok(sample) = sink.pull_sample()
                    && let Some(buffer) = sample.buffer()
                {
                    let caps = sample
                        .caps()
                        .and_then(|c| c.structure(0))
                        .map(|s| s.to_owned());
                    if let Some(structure) = caps {
                        let width = structure.get::<i32>("width").unwrap_or_else(|_| {
                            warn!("GStreamer caps missing 'width'; defaulting to 640");
                            640
                        }) as u32;
                        let height = structure.get::<i32>("height").unwrap_or_else(|_| {
                            warn!("GStreamer caps missing 'height'; defaulting to 480");
                            480
                        }) as u32;
                        if let Ok(map) = buffer.map_readable()
                            && let Some(data) = rgba_tightly_packed(map.as_slice(), width, height)
                        {
                            resource_monitor::record_camera_frame();

                            let id = frame_id.fetch_add(1, Ordering::Relaxed);

                            if let Err(e) = frame_tx.try_send(Frame {
                                id,
                                data,
                                width,
                                height,
                            }) {
                                warn!("Frame channel full, dropping frame {id}: {e}");
                            }
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    Ok(pipeline)
}
