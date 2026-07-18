//! Capture pipeline integration test — container/CI safe (videotestsrc only;
//! containers have no camera devices).
//!
//! Run with: `cargo test -p kataglyphis_media --features gstreamer`

#![cfg(feature = "gstreamer")]

use std::time::{Duration, Instant};

use kataglyphis_media::{CameraSource, CaptureConfig, CaptureSession};

#[test]
fn videotestsrc_delivers_rgba_frames() {
    let session = CaptureSession::start(CaptureConfig {
        source: CameraSource::Test,
        width: 320,
        height: 240,
        framerate: 30,
    })
    .expect("capture session should start with videotestsrc");
    let frames = session.frames();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut received = 0u32;
    let mut last_sequence = 0u64;
    while received < 10 && Instant::now() < deadline {
        if let Some(frame) = frames.take_latest(Duration::from_millis(500)) {
            assert_eq!(frame.width, 320);
            assert_eq!(frame.height, 240);
            assert_eq!(frame.rgba.len(), 320 * 240 * 4, "tightly packed RGBA");
            assert!(frame.sequence > last_sequence, "sequence must increase");
            last_sequence = frame.sequence;
            received += 1;
        }
    }
    assert!(
        received >= 10,
        "expected at least 10 frames within deadline, got {received}"
    );

    session.stop();
    assert!(frames.is_closed(), "slot must close on stop");
    assert!(
        frames.produced() >= u64::from(received),
        "produced counter must cover consumed frames"
    );
}

#[test]
fn camera_enumeration_does_not_fail() {
    // Containers report no devices — the call must still succeed.
    let cameras = kataglyphis_media::list_cameras().expect("enumeration should not error");
    println!("cameras: {cameras:?}");
}
