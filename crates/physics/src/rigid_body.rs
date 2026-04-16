use crate::math::{Vec3, Quat, Mat3};
use nalgebra::UnitQuaternion;

pub struct RigidBody {
    pub position: Vec3,
    pub orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia_tensor: Mat3,
    pub inv_inertia_tensor: Mat3,
    pub force_accumulator: Vec3,
    pub torque_accumulator: Vec3,
    pub restitution: f64,
    pub friction: f64,
}

impl RigidBody {
    pub fn new(mass: f64, inertia_tensor: Mat3) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inv_inertia = inertia_tensor.try_inverse().unwrap_or(Mat3::zeros());
        Self {
            position: Vec3::zeros(),
            orientation: UnitQuaternion::identity(),
            linear_velocity: Vec3::zeros(),
            angular_velocity: Vec3::zeros(),
            mass,
            inv_mass,
            inertia_tensor,
            inv_inertia_tensor: inv_inertia,
            force_accumulator: Vec3::zeros(),
            torque_accumulator: Vec3::zeros(),
            restitution: 0.6,
            friction: 0.4,
        }
    }

    pub fn apply_force(&mut self, force: Vec3) {
        self.force_accumulator += force;
    }

    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque_accumulator += torque;
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, contact_point: Vec3) {
        self.linear_velocity += impulse * self.inv_mass;
        let r = contact_point - self.position;
        self.angular_velocity += self.world_inv_inertia() * r.cross(&impulse);
    }

    pub fn world_inv_inertia(&self) -> Mat3 {
        let r = self.orientation.to_rotation_matrix();
        r.matrix() * self.inv_inertia_tensor * r.matrix().transpose()
    }

    pub fn clear_accumulators(&mut self) {
        self.force_accumulator = Vec3::zeros();
        self.torque_accumulator = Vec3::zeros();
    }
}
