use std::collections::VecDeque;

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
    let stroke = egui::Stroke::new(1.5, egui::Color32::LIGHT_BLUE);

    let points: Vec<egui::Pos2> = history
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let x = rect.left() + (i as f32) * step_x;
            let y_norm = (value / 100.0).clamp(0.0, 1.0);
            let y = rect.bottom() - y_norm * rect.height();
            egui::pos2(x, y)
        })
        .collect();

    painter.add(egui::Shape::line(points, stroke));
}
