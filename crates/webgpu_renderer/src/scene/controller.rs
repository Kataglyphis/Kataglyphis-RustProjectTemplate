//! Orbit/zoom controls for [`OrbitCamera`], shared by the native viewer and
//! the browser demo (winit delivers the same events on both).
//!
//! Mouse: left-drag orbits, wheel zooms. Touch: one finger orbits, two
//! fingers pinch to zoom.

use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};

use crate::scene::camera::OrbitCamera;

pub struct OrbitController {
    /// Keep spinning until the user interacts.
    pub auto_orbit: bool,
    pub rotate_speed_deg_per_px: f32,
    pub zoom_step: f32,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    /// Active touch points, most recent position per finger.
    ///
    /// A Vec rather than a map: gestures here use at most two fingers, and a
    /// linear scan over two entries beats hashing. Order is insertion order,
    /// so `touches[0]` is the finger that started the gesture.
    touches: Vec<(u64, (f64, f64))>,
    /// Distance between two fingers on the previous move, for pinch deltas.
    /// Cleared whenever the number of fingers changes.
    last_pinch_distance: Option<f64>,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            auto_orbit: true,
            rotate_speed_deg_per_px: 0.25,
            zoom_step: 0.1,
            dragging: false,
            last_cursor: None,
            touches: Vec::new(),
            last_pinch_distance: None,
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
                        self.apply_drag(camera, (current.0 - lx) as f32, (current.1 - ly) as f32);
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
            WindowEvent::Touch(touch) => self.handle_touch(
                touch.id,
                touch.phase,
                (touch.location.x, touch.location.y),
                camera,
            ),
            _ => false,
        }
    }

    /// One finger orbits, two pinch to zoom. Returns true when the camera moved.
    ///
    /// Takes the touch's fields rather than a `winit::event::Touch` so the
    /// gesture logic can be tested directly: `Touch` embeds a `DeviceId` with
    /// no public constructor, and faking one would mean `mem::zeroed()` in
    /// the tests.
    pub fn handle_touch(
        &mut self,
        id: u64,
        phase: TouchPhase,
        position: (f64, f64),
        camera: &mut OrbitCamera,
    ) -> bool {
        match phase {
            TouchPhase::Started => {
                self.auto_orbit = false;
                // Replace rather than push if the platform re-reports an id:
                // a duplicate entry would make a one-finger gesture look like
                // a pinch against itself, distance 0.
                if let Some(slot) = self
                    .touches
                    .iter_mut()
                    .find(|(existing, _)| *existing == id)
                {
                    slot.1 = position;
                } else {
                    self.touches.push((id, position));
                }
                // The finger count changed, so any stored pinch distance
                // describes a different gesture. Dropping it is what stops
                // the camera jumping when a second finger lands.
                self.last_pinch_distance = None;
                false
            }
            TouchPhase::Moved => {
                let Some(index) = self
                    .touches
                    .iter()
                    .position(|(existing, _)| *existing == id)
                else {
                    // A move for a finger we never saw start. Ignoring it is
                    // deliberate: synthesising a start here would treat the
                    // finger's absolute position as a drag delta and spin the
                    // camera by hundreds of degrees.
                    return false;
                };
                let previous = self.touches[index].1;
                self.touches[index].1 = position;

                match self.touches.len() {
                    1 => {
                        self.apply_drag(
                            camera,
                            (position.0 - previous.0) as f32,
                            (position.1 - previous.1) as f32,
                        );
                        true
                    }
                    2 => {
                        let distance = touch_distance(self.touches[0].1, self.touches[1].1);
                        let moved = match self.last_pinch_distance {
                            // Relative change, so the gesture behaves the same
                            // on a phone and a large tablet - an absolute pixel
                            // delta would zoom far more per centimetre of
                            // finger travel on a high-DPI screen.
                            Some(previous_distance)
                                if previous_distance > 1.0 && distance > 1.0 =>
                            {
                                self.apply_pinch(camera, previous_distance as f32, distance as f32);
                                true
                            }
                            _ => false,
                        };
                        self.last_pinch_distance = Some(distance);
                        moved
                    }
                    // Three or more fingers: track them, but do not guess at
                    // an interpretation.
                    _ => false,
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.retain(|(existing, _)| *existing != id);
                // Same reason as Started: with a different number of fingers
                // the old distance is meaningless, and keeping it would make
                // the camera lurch as the remaining finger continues.
                self.last_pinch_distance = None;
                false
            }
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

    /// Pinch: fingers moving apart zooms in, together zooms out.
    ///
    /// Scales the radius by the inverse ratio of finger separation, which
    /// makes the gesture reversible - pinching out and back in returns to the
    /// starting radius, where an additive step would not.
    pub fn apply_pinch(
        &self,
        camera: &mut OrbitCamera,
        previous_distance: f32,
        current_distance: f32,
    ) {
        if previous_distance.is_nan()
            || previous_distance <= 0.0
            || current_distance.is_nan()
            || current_distance <= 0.0
        {
            return;
        }
        camera.radius = (camera.radius * (previous_distance / current_distance)).clamp(0.2, 500.0);
    }
}

fn touch_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_finger_drag_orbits() {
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera::default();
        let start_yaw = camera.yaw_deg;

        controller.handle_touch(1, TouchPhase::Started, (100.0, 100.0), &mut camera);
        let moved = controller.handle_touch(1, TouchPhase::Moved, (140.0, 100.0), &mut camera);

        assert!(
            moved,
            "a one-finger move must report that the camera changed"
        );
        assert!(
            camera.yaw_deg > start_yaw,
            "dragging right should increase yaw"
        );
        assert!(!controller.auto_orbit, "touching must stop the idle spin");
    }

    #[test]
    fn a_second_finger_does_not_jump_the_camera() {
        // The classic touch bug: the new finger's absolute position gets
        // treated as a drag delta and the camera spins wildly.
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera::default();

        controller.handle_touch(1, TouchPhase::Started, (100.0, 100.0), &mut camera);
        controller.handle_touch(1, TouchPhase::Moved, (110.0, 100.0), &mut camera);
        let yaw_before = camera.yaw_deg;
        let radius_before = camera.radius;

        controller.handle_touch(2, TouchPhase::Started, (700.0, 500.0), &mut camera);

        assert_eq!(
            camera.yaw_deg, yaw_before,
            "a finger landing must not rotate the camera"
        );
        assert_eq!(
            camera.radius, radius_before,
            "a finger landing must not zoom the camera"
        );
    }

    #[test]
    fn pinching_apart_zooms_in_and_together_zooms_out() {
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera {
            radius: 10.0,
            ..OrbitCamera::default()
        };

        controller.handle_touch(1, TouchPhase::Started, (100.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Started, (200.0, 300.0), &mut camera);
        // First move only establishes the baseline distance.
        controller.handle_touch(2, TouchPhase::Moved, (200.0, 300.0), &mut camera);
        let baseline = camera.radius;

        // Fingers apart: 100px -> 300px.
        controller.handle_touch(2, TouchPhase::Moved, (400.0, 300.0), &mut camera);
        assert!(
            camera.radius < baseline,
            "spreading fingers must zoom in (smaller radius)"
        );

        // And back together.
        let after_spread = camera.radius;
        controller.handle_touch(2, TouchPhase::Moved, (200.0, 300.0), &mut camera);
        assert!(
            camera.radius > after_spread,
            "pinching in must zoom out (larger radius)"
        );
    }

    #[test]
    fn pinch_is_reversible() {
        // Ratio-based scaling, not additive steps: out and back must return to
        // where it started, or repeated gestures drift the camera away.
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera {
            radius: 10.0,
            ..OrbitCamera::default()
        };

        controller.handle_touch(1, TouchPhase::Started, (100.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Started, (200.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Moved, (200.0, 300.0), &mut camera);
        let start = camera.radius;

        controller.handle_touch(2, TouchPhase::Moved, (500.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Moved, (200.0, 300.0), &mut camera);

        assert!(
            (camera.radius - start).abs() < 1e-3,
            "pinch out and back should return to {start}, got {}",
            camera.radius
        );
    }

    #[test]
    fn lifting_one_finger_of_two_does_not_jump() {
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera {
            radius: 10.0,
            ..OrbitCamera::default()
        };

        controller.handle_touch(1, TouchPhase::Started, (100.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Started, (300.0, 300.0), &mut camera);
        controller.handle_touch(2, TouchPhase::Moved, (300.0, 300.0), &mut camera);

        controller.handle_touch(2, TouchPhase::Ended, (300.0, 300.0), &mut camera);
        let radius_after_lift = camera.radius;
        let yaw_after_lift = camera.yaw_deg;

        // The remaining finger keeps moving; it must orbit from its own last
        // position, not from the departed finger's.
        controller.handle_touch(1, TouchPhase::Moved, (110.0, 300.0), &mut camera);

        assert_eq!(
            camera.radius, radius_after_lift,
            "lifting a finger must not zoom"
        );
        let yaw_delta = (camera.yaw_deg - yaw_after_lift).abs();
        assert!(
            yaw_delta < 5.0,
            "one finger moving 10px should not swing yaw by {yaw_delta} degrees"
        );
    }

    #[test]
    fn a_move_for_an_unknown_finger_is_ignored() {
        // Happens after Cancelled, or if the first event of a session is lost.
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera::default();
        let yaw = camera.yaw_deg;

        let moved = controller.handle_touch(9, TouchPhase::Moved, (800.0, 600.0), &mut camera);

        assert!(!moved);
        assert_eq!(
            camera.yaw_deg, yaw,
            "an unknown finger must not move the camera"
        );
    }

    #[test]
    fn cancelled_touches_are_forgotten() {
        let mut controller = OrbitController::default();
        let mut camera = OrbitCamera::default();

        controller.handle_touch(1, TouchPhase::Started, (100.0, 100.0), &mut camera);
        controller.handle_touch(1, TouchPhase::Cancelled, (100.0, 100.0), &mut camera);
        let yaw = camera.yaw_deg;

        let moved = controller.handle_touch(1, TouchPhase::Moved, (500.0, 100.0), &mut camera);

        assert!(!moved, "a cancelled finger must not keep orbiting");
        assert_eq!(camera.yaw_deg, yaw);
    }

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
