use nalgebra::UnitQuaternion;
use crate::math::{Mat3, Vec3, swing_twist_angle};
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
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "AngularSpring axis must be unit length (norm = {})",
            axis.norm()
        );
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

    /// Implicit (semi-implicit) spring update: directly modifies angular
    /// velocities instead of routing an explicit torque through the
    /// accumulator. Must be called once per substep, after
    /// `integrate_velocities` (alongside the implicit dampers).
    ///
    /// Backward-Euler on the angle along the axis: with reduced inertia
    /// `w = n·I_inv_a·n + n·I_inv_b·n` and angle error `e = θ − θ_target`,
    ///   ω_rel_new = (ω_rel − k·dt·w·e) / (1 + k·dt²·w),   J = (ω_rel_new − ω_rel)/w.
    /// This is unconditionally stable for any stiffness and dt — an explicit
    /// penalty torque is only stable for `dt < 2·√(I_red/k)`, which the stiff
    /// weld springs (k≈5·10⁴) violate on the foot's few-gram bones.
    pub fn apply_implicit(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let e = self.angle(bodies) - self.target_angle;

        let zero = Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let w = n.dot(&(i_inv_a * n)) + n.dot(&(i_inv_b * n)); // 1/I_red
        if w < 1e-15 {
            return; // both bodies pinned
        }

        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);

        let k = self.stiffness;
        let omega_new = (omega_rel_n - k * dt * w * e) / (1.0 + k * dt * dt * w);
        let j = (omega_new - omega_rel_n) / w;

        bodies[self.body_a].angular_velocity -= i_inv_a * n * j;
        bodies[self.body_b].angular_velocity += i_inv_b * n * j;
    }

    /// Elastic potential energy stored in the spring.
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let angle = self.angle(bodies);
        let dev = angle - self.target_angle;
        0.5 * self.stiffness * dev * dev
    }
}
