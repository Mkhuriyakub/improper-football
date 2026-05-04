use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Passive elastic spring between two bodies along an axis fixed in body_a's
/// local frame. Applies an equal-and-opposite torque proportional to the
/// deviation of the relative orientation (body_b w.r.t. body_a) from a target
/// angle along the axis.
///
/// Models the soft elastic preload of joint capsule, ligaments, and unactivated
/// muscle that pulls a joint toward its neutral resting pose. A property of
/// the real joint, not of any controller — it is always present.
pub struct AngularSpring {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame.
    pub axis_local_a: Vec3,
    /// Spring stiffness in N·m/rad.
    pub stiffness: f64,
    /// Resting angle along the axis in rad (relative rotation of body_b w.r.t. body_a).
    pub target_angle: f64,
}

impl AngularSpring {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, stiffness: f64, target_angle: f64) -> Self {
        Self { body_a, body_b, axis_local_a: axis, stiffness, target_angle }
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let q_rel = bodies[self.body_b].orientation
            * bodies[self.body_a].orientation.inverse();
        let angle = q_rel.scaled_axis().dot(&n);
        let torque = n * (-self.stiffness * (angle - self.target_angle));
        bodies[self.body_b].apply_torque(torque);
        bodies[self.body_a].apply_torque(-torque);
    }

    /// Elastic potential energy stored in the spring.
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let q_rel = bodies[self.body_b].orientation
            * bodies[self.body_a].orientation.inverse();
        let angle = q_rel.scaled_axis().dot(&n);
        let dev = angle - self.target_angle;
        0.5 * self.stiffness * dev * dev
    }
}
