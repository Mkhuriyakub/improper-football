use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Velocity-level angular limit constraint, solved by sequential impulse with
/// Baumgarte stabilization. Inequality (one-sided) constraint: only fires
/// when the relative angle is outside `[min_angle, max_angle]`, and the
/// impulse only pushes the joint *back into range* — never deeper into a
/// limit. This makes it a hard kinematic stop in steady state, but the
/// impulse magnitude is bounded by the velocity-error formula so there's no
/// per-substep amplification.
pub struct AngularLimitConstraint {
    pub body_a: usize,
    pub body_b: usize,
    /// Constraint axis in body_a's local frame.
    pub axis_local_a: Vec3,
    pub min_angle: f64,
    pub max_angle: f64,
    pub compliance: f64,
    pub baumgarte: f64,
}

impl AngularLimitConstraint {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, min: f64, max: f64) -> Self {
        Self {
            body_a,
            body_b,
            axis_local_a: axis,
            min_angle: min,
            max_angle: max,
            compliance: 0.0,
            baumgarte: 0.2,
        }
    }

    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        // Constraint axis in world frame (follows body_a's orientation).
        let n = bodies[self.body_a].orientation * self.axis_local_a;

        // Relative rotation projected onto the axis gives the joint angle.
        let q_rel = bodies[self.body_b].orientation
            * bodies[self.body_a].orientation.inverse();
        let angle = q_rel.scaled_axis().dot(&n);

        // Position error: signed distance into the violated half-space.
        let c = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            angle - self.min_angle
        } else {
            return;
        };

        // Relative angular velocity along the axis.
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let v_rel_n = omega_rel.dot(&n);

        // Baumgarte velocity bias (pushes back toward range).
        let bias = -self.baumgarte / dt * c;
        let velocity_error = v_rel_n - bias;

        // One-sided: only apply impulse if it would push back toward range.
        // Equivalently: skip when velocity_error has the same sign as -c.
        // For c > 0 (above max), we apply only if velocity_error > 0.
        // For c < 0 (below min), we apply only if velocity_error < 0.
        if velocity_error * c <= 0.0 {
            return;
        }

        // Generalized inverse mass along n.
        let i_inv_a = bodies[self.body_a].world_inv_inertia();
        let i_inv_b = bodies[self.body_b].world_inv_inertia();
        let w_a = n.dot(&(i_inv_a * n));
        let w_b = n.dot(&(i_inv_b * n));
        let w = w_a + w_b;
        let cfm = self.compliance / dt;
        let denom = w + cfm;
        if denom < 1e-10 {
            return;
        }

        let lambda = -velocity_error / denom;
        bodies[self.body_a].angular_velocity -= i_inv_a * n * lambda;
        bodies[self.body_b].angular_velocity += i_inv_b * n * lambda;
    }

    /// No-op for the SI solver. Kept for API compatibility.
    pub fn reset_lambda(&mut self) {}
}
