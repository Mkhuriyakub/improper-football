use nalgebra::UnitQuaternion;
use crate::math::{Vec3, swing_twist_angle};
use crate::rigid_body::RigidBody;

/// Passive elastic spring between two bodies along an axis fixed in body_a's
/// local frame. Applies an equal-and-opposite torque proportional to the
/// deviation of the relative orientation (body_b w.r.t. body_a) from a target
/// angle along the axis.
///
/// Models the soft elastic preload of joint capsule, ligaments, and unactivated
/// muscle that pulls a joint toward its neutral resting pose. A property of
/// the real joint, not of any controller — it is always present.
///
/// **Angle extraction** uses swing-twist factorization (`swing_twist_angle`)
/// on `q_b_in_a · baseline⁻¹`. This measures the *real* rotation about
/// `axis_local_a` from the rest pose, free of bivector cross-coupling that
/// the older `scaled_axis().dot(axis)` projection introduced for multi-axis
/// joints. See `crates/physics/src/math.rs::swing_twist_angle`.
///
/// **Baseline** is the rest orientation of body_b in body_a's frame. For
/// joints whose bodies are at identity orientation when at anatomical rest
/// (most of our chain), leave `baseline = identity` (the default). For
/// joints where body_b has a non-identity rest pose (e.g. an ankle whose
/// foot is rendered horizontal — body rotated 90° about x relative to the
/// shin), set `baseline = R_x(π/2)` so anatomical neutral still reads as
/// angle = 0.
pub struct AngularSpring {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame.
    pub axis_local_a: Vec3,
    /// Spring stiffness in N·m/rad.
    pub stiffness: f64,
    /// Resting angle along the axis in rad (relative to baseline).
    pub target_angle: f64,
    /// Rest orientation of body_b in body_a's frame. Default identity.
    pub baseline: UnitQuaternion<f64>,
}

impl AngularSpring {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, stiffness: f64, target_angle: f64) -> Self {
        Self {
            body_a, body_b, axis_local_a: axis,
            stiffness, target_angle,
            baseline: UnitQuaternion::identity(),
        }
    }

    /// Set a non-identity rest orientation of body_b in body_a's frame.
    pub fn with_baseline(mut self, baseline: UnitQuaternion<f64>) -> Self {
        self.baseline = baseline;
        self
    }

    fn angle(&self, bodies: &[RigidBody]) -> f64 {
        let q_b_in_a = bodies[self.body_a].orientation.inverse()
                     * bodies[self.body_b].orientation;
        let q_from_rest = q_b_in_a * self.baseline.inverse();
        swing_twist_angle(q_from_rest, self.axis_local_a)
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let n_world = bodies[self.body_a].orientation * self.axis_local_a;
        let angle = self.angle(bodies);
        let torque = n_world * (-self.stiffness * (angle - self.target_angle));
        bodies[self.body_b].apply_torque(torque);
        bodies[self.body_a].apply_torque(-torque);
    }

    /// Elastic potential energy stored in the spring.
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let angle = self.angle(bodies);
        let dev = angle - self.target_angle;
        0.5 * self.stiffness * dev * dev
    }
}
