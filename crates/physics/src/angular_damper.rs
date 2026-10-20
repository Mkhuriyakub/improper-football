use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Passive viscoelastic damper between two bodies along an axis fixed in body_a's
/// local frame. Models passive joint resistance — capsule, ligaments, tendons,
/// and unactivated muscle.
///
/// Integrates *implicitly* in `apply_implicit`: solves
///   ω_rel_new = ω_rel / (1 + c·dt·w)
/// where w = n·I_inv_a·n + n·I_inv_b·n (1/reduced inertia along axis).
/// This is unconditionally stable for any damping coefficient and time step,
/// fixing the explicit-Euler instability that hit small-inertia bodies
/// (fingers, vertebrae, etc.).
pub struct AngularDamper {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame.
    pub axis_local_a: Vec3,
    /// Damping coefficient in N·m·s/rad.
    pub damping: f64,
}

impl AngularDamper {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, damping: f64) -> Self {
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "AngularDamper axis must be unit length (norm = {})",
            axis.norm()
        );
        Self { body_a, body_b, axis_local_a: axis, damping }
    }

    /// Implicit damping update: directly modifies angular velocities (does not
    /// route through the force/torque accumulator). Must be called once per
    /// substep, after `integrate_velocities`.
    pub fn apply_implicit(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);
        if omega_rel_n.abs() < 1e-15 {
            return;
        }

        // Pinned bodies (inv_mass == 0) should be treated as infinite-inertia
        // for damper impulse splitting too. Otherwise the damper's impulse
        // gets shared with the "phantom" inertia of a pinned parent (whose
        // inv_inertia is computed from a placeholder mass), and the actual
        // damping the moving body experiences is reduced.
        let zero = crate::math::Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let w = n.dot(&(i_inv_a * n)) + n.dot(&(i_inv_b * n));  // = 1/I_red
        if w < 1e-15 {
            return;  // Both bodies pinned — nothing to damp.
        }

        // Implicit Euler: ω_rel_new = ω_rel / (1 + c·dt·w).
        let factor = 1.0 + self.damping * dt * w;
        let omega_rel_new = omega_rel_n / factor;
        let delta_omega_rel = omega_rel_new - omega_rel_n;

        // Required impulse to produce delta_omega_rel: J = delta / w.
        let j = delta_omega_rel / w;

        // Apply: body_a gets -i_inv_a · n · J, body_b gets +i_inv_b · n · J.
        bodies[self.body_a].angular_velocity -= i_inv_a * n * j;
        bodies[self.body_b].angular_velocity += i_inv_b * n * j;
    }

    /// Instantaneous power dissipated by this damper. Always ≤ 0.
    pub fn power(&self, bodies: &[RigidBody]) -> f64 {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);
        -self.damping * omega_rel_n * omega_rel_n
    }
}
