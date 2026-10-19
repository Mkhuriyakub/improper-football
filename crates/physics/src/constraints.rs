use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Velocity-level 3-DOF ball-and-socket joint between two bodies, solved by
/// sequential impulse with Baumgarte stabilization for position drift.
///
/// Implementation: three sequential 1-D scalar constraints along world X, Y,
/// and Z axes. Each iteration of the outer SI solver runs all three axes,
/// and convergence to the true 3-DOF constraint comes from iteration. This
/// avoids any 3×3 matrix inversion and has straightforward numerics.
///
/// `rest_length` is kept as 0 (zero-rest-length ball-and-socket joint).
pub struct PositionalConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub rest_length: f64,
    pub compliance: f64,
    /// Baumgarte position-error feedback coefficient.
    pub baumgarte: f64,
}

impl PositionalConstraint {
    pub fn new(body_a: usize, body_b: usize, rest_length: f64, compliance: f64) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::zeros(),
            local_anchor_b: Vec3::zeros(),
            rest_length,
            compliance,
            baumgarte: 0.2,
        }
    }

    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        // Solve each world axis independently. Convergence to the full
        // 3-DOF constraint emerges from outer SI iterations.
        self.solve_axis(bodies, dt, Vec3::x());
        self.solve_axis(bodies, dt, Vec3::y());
        self.solve_axis(bodies, dt, Vec3::z());
    }

    fn solve_axis(&mut self, bodies: &mut Vec<RigidBody>, dt: f64, n: Vec3) {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;

        // Position error along this axis.
        let c = (pb - pa).dot(&n) - self.rest_length * n.dot(&n).abs();
        // (For zero rest_length, the above is just (pb - pa) · n.)

        // Velocity at each anchor.
        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&r_b);
        let v_rel_n = (v_b_anchor - v_a_anchor).dot(&n);

        // Generalized inverse mass along n.
        let inv_mass_a = bodies[self.body_a].inv_mass;
        let inv_mass_b = bodies[self.body_b].inv_mass;
        let i_inv_a = bodies[self.body_a].world_inv_inertia();
        let i_inv_b = bodies[self.body_b].world_inv_inertia();
        let ra_cross_n = r_a.cross(&n);
        let rb_cross_n = r_b.cross(&n);
        let w_a = inv_mass_a + ra_cross_n.dot(&(i_inv_a * ra_cross_n));
        let w_b = inv_mass_b + rb_cross_n.dot(&(i_inv_b * rb_cross_n));
        let w = w_a + w_b;
        let cfm = self.compliance / dt;
        let denom = w + cfm;
        if denom < 1e-10 {
            return;
        }

        // Baumgarte velocity bias.
        let bias = -self.baumgarte / dt * c;
        let velocity_error = v_rel_n - bias;
        let lambda = -velocity_error / denom;
        if lambda.abs() < 1e-15 {
            return;
        }

        // Apply impulse along n.
        let p = n * lambda;
        bodies[self.body_a].linear_velocity -= p * inv_mass_a;
        bodies[self.body_b].linear_velocity += p * inv_mass_b;
        bodies[self.body_a].angular_velocity -= i_inv_a * ra_cross_n * lambda;
        bodies[self.body_b].angular_velocity += i_inv_b * rb_cross_n * lambda;
    }
}
