//! Webcam → RGBA frame capture.
//!
//! A [`CaptureSession`] owns a GStreamer pipeline
//! (`<source> ! videoconvert ! videoscale ! capsfilter(RGBA) ! appsink`) and
//! publishes frames into a single-slot [`FrameSlot`] with overwrite semantics:
//! consumers always see the newest frame and capture is never back-pressured
//! by slow inference.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

use crate::ensure_gst_initialized;

/// Which video source element feeds the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraSource {
    /// Best available webcam source (`mfvideosrc`, else `ksvideosrc`,
    /// else `autovideosrc`), first device.
    Auto,
    /// Webcam by enumeration index (`device-index` property).
    Device(u32),
    /// `videotestsrc` — for containers/CI where no camera device exists.
    Test,
}

#[derive(Debug, Clone, Copy)]
pub struct CaptureConfig {
    pub source: CameraSource,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            source: CameraSource::Auto,
            width: 1280,
            height: 720,
            framerate: 30,
        }
    }
}

/// One RGBA video frame, tightly packed (`width * height * 4` bytes).
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Pipeline timestamp in milliseconds, if the buffer carried one.
    pub pts_ms: Option<u64>,
    /// Monotonic frame counter for this session (starts at 1).
    pub sequence: u64,
}

struct SlotInner {
    latest: Mutex<Option<VideoFrame>>,
    signal: Condvar,
    closed: AtomicBool,
    produced: AtomicU64,
}

/// Latest-frame slot shared between the appsink thread and consumers.
#[derive(Clone)]
pub struct FrameSlot {
    inner: Arc<SlotInner>,
}

impl FrameSlot {
    fn new() -> Self {
        Self {
            inner: Arc::new(SlotInner {
                latest: Mutex::new(None),
                signal: Condvar::new(),
                closed: AtomicBool::new(false),
                produced: AtomicU64::new(0),
            }),
        }
    }

    fn publish(&self, frame: VideoFrame) {
        let mut slot = self.inner.latest.lock().expect("frame slot poisoned");
        *slot = Some(frame);
        self.inner.produced.fetch_add(1, Ordering::Relaxed);
        drop(slot);
        self.inner.signal.notify_all();
    }

    fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.signal.notify_all();
    }

    /// Takes the newest frame, waiting up to `timeout` for one to arrive.
    /// Returns `None` on timeout or after the session closed.
    pub fn take_latest(&self, timeout: Duration) -> Option<VideoFrame> {
        let mut slot = self.inner.latest.lock().expect("frame slot poisoned");
        loop {
            if let Some(frame) = slot.take() {
                return Some(frame);
            }
            if self.inner.closed.load(Ordering::Acquire) {
                return None;
            }
            let (guard, wait) = self
                .inner
                .signal
                .wait_timeout(slot, timeout)
                .expect("frame slot poisoned");
            slot = guard;
            if wait.timed_out() {
                return slot.take();
            }
        }
    }

    /// Total frames produced by the pipeline (including overwritten ones).
    pub fn produced(&self) -> u64 {
        self.inner.produced.load(Ordering::Relaxed)
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

/// Picks the best available webcam source element for this platform/registry.
fn webcam_source_factory() -> Result<&'static str> {
    for name in ["mfvideosrc", "ksvideosrc", "autovideosrc", "v4l2src"] {
        if gst::ElementFactory::find(name).is_some() {
            return Ok(name);
        }
    }
    anyhow::bail!("no webcam source element found in the GStreamer registry")
}

pub struct CaptureSession {
    pipeline: gst::Pipeline,
    frames: FrameSlot,
}

impl CaptureSession {
    /// Builds and starts the capture pipeline.
    pub fn start(config: CaptureConfig) -> Result<Self> {
        ensure_gst_initialized()?;

        let source = match config.source {
            CameraSource::Test => gst::ElementFactory::make("videotestsrc")
                .property_from_str("pattern", "smpte")
                .property("is-live", true)
                .build()
                .context("failed to create videotestsrc")?,
            CameraSource::Auto | CameraSource::Device(_) => {
                let factory = webcam_source_factory()?;
                log::info!("using webcam source element `{factory}`");
                let mut builder = gst::ElementFactory::make(factory);
                if let CameraSource::Device(index) = config.source {
                    // Only mfvideosrc/ksvideosrc expose `device-index`; setting
                    // an unknown property would panic on the fallback sources.
                    if matches!(factory, "mfvideosrc" | "ksvideosrc") {
                        builder = builder.property("device-index", index as i32);
                    } else {
                        log::warn!("source `{factory}` has no device-index; using default device");
                    }
                }
                builder
                    .build()
                    .with_context(|| format!("failed to create `{factory}`"))?
            }
        };

        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .context("failed to create videoconvert")?;
        let videoscale = gst::ElementFactory::make("videoscale")
            .build()
            .context("failed to create videoscale")?;

        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGBA")
            .field("width", config.width as i32)
            .field("height", config.height as i32)
            .build();

        let appsink = gst_app::AppSink::builder()
            .caps(&caps)
            // Latest-frame semantics at the sink too: never queue stale video.
            .max_buffers(1)
            .drop(true)
            .sync(false)
            .build();

        let pipeline = gst::Pipeline::new();
        pipeline
            .add_many([&source, &videoconvert, &videoscale, appsink.upcast_ref()])
            .context("failed to assemble pipeline")?;
        gst::Element::link_many([&source, &videoconvert, &videoscale, appsink.upcast_ref()])
            .context("failed to link pipeline (source may not support requested caps)")?;

        let frames = FrameSlot::new();
        let slot = frames.clone();
        let mut sequence: u64 = 0;
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                    let caps = sample.caps().ok_or(gst::FlowError::Error)?;
                    let info = gstreamer_video::VideoInfo::from_caps(caps)
                        .map_err(|_| gst::FlowError::Error)?;
                    let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                    sequence += 1;
                    slot.publish(VideoFrame {
                        rgba: map.as_slice().to_vec(),
                        width: info.width(),
                        height: info.height(),
                        pts_ms: buffer.pts().map(|t| t.mseconds()),
                        sequence,
                    });
                    Ok(gst::FlowSuccess::Ok)
                })
                .eos({
                    let slot = frames.clone();
                    move |_| slot.close()
                })
                .build(),
        );

        pipeline
            .set_state(gst::State::Playing)
            .context("failed to set pipeline to Playing")?;

        Ok(Self { pipeline, frames })
    }

    /// Shared handle for pulling frames; stays valid until [`stop`](Self::stop).
    pub fn frames(&self) -> FrameSlot {
        self.frames.clone()
    }

    /// Stops the pipeline and wakes all waiting consumers.
    pub fn stop(&self) {
        self.frames.close();
        if let Err(e) = self.pipeline.set_state(gst::State::Null) {
            log::warn!("failed to stop capture pipeline: {e}");
        }
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.stop();
    }
}
