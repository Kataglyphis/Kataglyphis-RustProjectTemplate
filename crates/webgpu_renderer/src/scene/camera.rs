//! Orbit camera producing wgpu-convention (depth 0..1) view-projection
//! matrices. Yaw/pitch semantics mirror the C++ engine's camera.

use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub radius: f32,
    /// Degrees around the Y axis.
    pub yaw_deg: f32,
    /// Degrees above the horizon, clamped to (-89, 89).
    pub pitch_deg: f32,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            radius: 4.0,
            yaw_deg: 45.0,
            pitch_deg: 25.0,
            fov_y_deg: 45.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let yaw = self.yaw_deg.to_radians();
        let pitch = self.pitch_deg.clamp(-89.0, 89.0).to_radians();
        self.target
            + self.radius
                * Vec3::new(
                    pitch.cos() * yaw.cos(),
                    pitch.sin(),
                    pitch.cos() * yaw.sin(),
                )
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// Projection with wgpu/WebGPU clip space (depth 0..1).
    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_deg.to_radians(),
            aspect.max(1e-6),
            self.near,
            self.far,
        )
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_orbits_at_radius() {
        let cam = OrbitCamera::default();
        assert!((cam.eye().distance(cam.target) - cam.radius).abs() < 1e-4);
    }

    #[test]
    fn view_maps_eye_to_origin() {
        let cam = OrbitCamera::default();
        let origin = cam.view().transform_point3(cam.eye());
        assert!(origin.length() < 1e-4);
    }
}
