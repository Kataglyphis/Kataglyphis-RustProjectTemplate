//! Webcam/video capture built on GStreamer.
//!
//! Everything is gated behind the `gstreamer` feature; without it this crate
//! compiles to nothing so non-media builds never require GStreamer dev files.

#[cfg(feature = "gstreamer")]
pub mod capture;
#[cfg(feature = "gstreamer")]
pub mod devices;
#[cfg(feature = "gstreamer")]
mod runtime;

#[cfg(feature = "gstreamer")]
pub use capture::{CameraSource, CaptureConfig, CaptureSession, FrameSlot, VideoFrame};
#[cfg(feature = "gstreamer")]
pub use devices::{list_cameras, CameraInfo};
#[cfg(feature = "gstreamer")]
pub use runtime::ensure_gst_initialized;
