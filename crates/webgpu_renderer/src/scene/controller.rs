//! Mouse orbit/zoom controls for [`OrbitCamera`], shared by the native
//! viewer and the browser demo (winit delivers the same events on both).

use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

use crate::scene::camera::OrbitCamera;

pub struct OrbitController {
    /// Keep spinning until the user interacts.
    pub auto_orbit: bool,
    pub rotate_speed_deg_per_px: f32,
    pub zoom_step: f32,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            auto_orbit: true,
            rotate_speed_deg_per_px: 0.25,
            zoom_step: 0.1,
            dragging: false,
            last_cursor: None,
        }
    }
}

impl OrbitController {
    /// Feeds a window event; returns true when the event changed the camera.
    pub fn handle_event(&mut self, event: &WindowEvent, camera: &mut OrbitCamera) -> bool {
        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = *state == ElementState::Pressed;
                if self.dragging {
                    self.auto_orbit = false;
                    self.last_cursor = None;
                }
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                let moved = if self.dragging {
                    if let Some((lx, ly)) = self.last_cursor {
                        self.apply_drag(
                            camera,
                            (current.0 - lx) as f32,
                            (current.1 - ly) as f32,
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                self.last_cursor = Some(current);
                moved
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 60.0,
                };
                self.apply_zoom(camera, amount);
                true
            }
            _ => false,
        }
    }

    /// Drag in pixels -> yaw/pitch. Exposed for tests.
    pub fn apply_drag(&self, camera: &mut OrbitCamera, dx_px: f32, dy_px: f32) {
        camera.yaw_deg += dx_px * self.rotate_speed_deg_per_px;
        camera.pitch_deg =
            (camera.pitch_deg + dy_px * self.rotate_speed_deg_per_px).clamp(-89.0, 89.0);
    }

    /// Positive amount zooms in. Exposed for tests.
    pub fn apply_zoom(&self, camera: &mut OrbitCamera, amount: f32) {
        camera.radius = (camera.radius * (1.0 - amount * self.zoom_step)).clamp(0.2, 500.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_rotates_and_clamps_pitch() {
        let controller = OrbitController::default();
        let mut camera = OrbitCamera::default();
        let yaw0 = camera.yaw_deg;

        controller.apply_drag(&mut camera, 40.0, 0.0);
        assert!((camera.yaw_deg - yaw0 - 10.0).abs() < 1e-4);

        controller.apply_drag(&mut camera, 0.0, 100000.0);
        assert!((camera.pitch_deg - 89.0).abs() < 1e-4);
    }

    #[test]
    fn zoom_scales_radius_and_clamps() {
        let controller = OrbitController::default();
        let mut camera = OrbitCamera::default();
        let r0 = camera.radius;

        controller.apply_zoom(&mut camera, 1.0);
        assert!(camera.radius < r0);

        for _ in 0..200 {
            controller.apply_zoom(&mut camera, 1.0);
        }
        assert!(camera.radius >= 0.2);
    }
}
