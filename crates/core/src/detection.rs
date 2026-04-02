/// A single detection in *original image pixel coordinates*.
///
/// This struct is always compiled (not feature-gated) so that public API
/// surfaces (`api::onnx`) can reference it regardless of which ONNX backend
/// is enabled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(dead_code)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i64,
}

impl std::fmt::Display for Detection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "class {} ({:.0}%) [{:.1}, {:.1}, {:.1}, {:.1}]",
            self.class_id,
            self.score * 100.0,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )
    }
}
