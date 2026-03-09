use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

#[cfg(target_os = "windows")]
use crate::resource_monitor;

#[derive(Clone)]
pub struct Frame {
    pub id: u64,
    pub data: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
}

pub fn rgba_tightly_packed_copy(src: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    if width == 0 || height == 0 {
        return None;
    }

    let row_bytes = (width as usize).saturating_mul(4);
    let expected_len = row_bytes.saturating_mul(height as usize);

    if src.len() == expected_len {
        return Some(src.to_vec());
    }

    // Many GStreamer buffers are padded per row (stride). We conservatively try to
    // interpret the buffer as a single RGBA plane with a constant stride.
    if src.len() < expected_len {
        return None;
    }

    let h = height as usize;
    let mut stride = src.len() / h;
    if stride < row_bytes {
        return None;
    }

    // If there is trailing padding after the last row, prefer a stride that still
    // lets us safely read all rows.
    while stride.saturating_mul(h) > src.len() {
        if stride == 0 {
            return None;
        }
        stride -= 1;
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

    Some(out)
}

pub fn build_pipeline(frame_tx: std::sync::mpsc::SyncSender<Frame>) -> gst::Pipeline {
    let pipeline = gst::Pipeline::new();

    let frame_id = AtomicU64::new(1);

    let src = gst::ElementFactory::make("mfvideosrc")
        .build()
        .or_else(|_| gst::ElementFactory::make("autovideosrc").build())
        .expect("Failed to create a video source");

    let convert = gst::ElementFactory::make("videoconvert")
        .build()
        .expect("Failed to create videoconvert");

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .build();

    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()
        .expect("Failed to create capsfilter");

    let sink = gst::ElementFactory::make("appsink")
        .property("emit-signals", true)
        .property("max-buffers", 2u32)
        .property("drop", true)
        .build()
        .expect("Failed to create appsink");

    pipeline
        .add_many([&src, &convert, &capsfilter, &sink])
        .expect("Failed to add elements");
    gst::Element::link_many([&src, &convert, &capsfilter, &sink]).expect("Failed to link elements");

    let appsink = sink
        .clone()
        .dynamic_cast::<gst_app::AppSink>()
        .expect("Sink is not an appsink");

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
                        let width = structure.get::<i32>("width").unwrap_or(640) as u32;
                        let height = structure.get::<i32>("height").unwrap_or(480) as u32;
                        if let Ok(map) = buffer.map_readable()
                            && let Some(data) =
                                rgba_tightly_packed_copy(map.as_slice(), width, height)
                        {
                            #[cfg(target_os = "windows")]
                            resource_monitor::record_camera_frame();

                            let id = frame_id.fetch_add(1, Ordering::Relaxed);

                            let _ = frame_tx.try_send(Frame {
                                id,
                                data: Arc::new(data),
                                width,
                                height,
                            });
                        }
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    pipeline
}
