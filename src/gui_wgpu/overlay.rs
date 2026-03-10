use std::collections::VecDeque;

/// Convert bytes to MiB. Re-exports the canonical implementation from `resource_monitor`.
pub fn bytes_to_mib(bytes: u64) -> f32 {
    crate::resource_monitor::bytes_to_mib(bytes) as f32
}

pub fn draw_cpu_history(ui: &mut egui::Ui, history: &VecDeque<f32>) {
    let desired = egui::vec2(160.0, 48.0);
    let (rect, _response) = ui.allocate_exact_size(desired, egui::Sense::hover());

    let painter = ui.painter();
    painter.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::GRAY),
        egui::StrokeKind::Inside,
    );

    if history.len() < 2 {
        return;
    }

    let max_points = history.len().max(2) as f32 - 1.0;
    let step_x = rect.width() / max_points;
    let mut prev = None;

    for (i, &value) in history.iter().enumerate() {
        let x = rect.left() + (i as f32) * step_x;
        let y_norm = (value / 100.0).clamp(0.0, 1.0);
        let y = rect.bottom() - y_norm * rect.height();
        let pos = egui::pos2(x, y);

        if let Some(prev) = prev {
            painter.line_segment(
                [prev, pos],
                egui::Stroke::new(1.5, egui::Color32::LIGHT_BLUE),
            );
        }
        prev = Some(pos);
    }
}
