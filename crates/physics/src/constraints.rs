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
    /// Baumgarte position-error feedback coefficient, applied per SUBSTEP
    /// (β/sub_dt ≈ 1440/s at sub_dt = 1/7200). Unlike `ContactConstraint`
    /// (which rescaled its β to 0.002 — see its doc), this aggressive rate is
    /// correct for BILATERAL joints and was kept deliberately after a
    /// measured sweep (2026-10-19, β ∈ {0.2, 0.02, 0.002} over a settling
    /// chain, the 26-bone foot, and a 5 mm-assembly-error chain —
    /// docs/findings.md):
    /// - Energy does not track β: tail rates were identical across the sweep
    ///   (chain ≈ −5.6e-5 J/s net dissipation at every β). The bias velocity
    ///   a bilateral constraint grants on the way in is reclaimed by the same
    ///   constraint on the way out (overshoot is pulled back, not released),
    ///   so no steady injection exists — contact's launch failure mode is a
    ///   UNILATERAL artifact (separation deactivates the constraint before
    ///   the energy is reclaimed) and does not transfer here.
    /// - Constraint quality DOES track β: anchor separation scaled inversely
    ///   (chain 4.6e-8 m at 0.2 → 4.7e-6 m at 0.002; foot welds 1.1e-4 m →
    ///   9.6e-3 m, where the assembly went spongy and ejected a bone at
    ///   6.5 m/s). β = 0.2 dominated every scene.
    /// - A 5 mm assembly error does NOT launch: measured peak 2.06 m/s and a
    ///   +0.41 J transient, fully reabsorbed within ~20 substeps (~3 ms). No
    ///   bias clamp is needed; still, assemble joints with coincident anchors
    ///   (zero preload) as `src/anatomy/foot.rs::anchor_at` does.
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

    /// Override the Baumgarte position-feedback coefficient (per substep).
    pub fn with_baumgarte(mut self, b: f64) -> Self { self.baumgarte = b; self }

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
        // Pinned bodies (inv_mass == 0) are infinite-inertia: use a zero
        // inverse inertia for them regardless of what their tensor holds
        // (guards against phantom inertia on bodies pinned by hand). With
        // the zero i_inv in BOTH the effective mass w below and the impulse
        // application, the dynamic partner receives the full impulse rather
        // than a split one, and the pinned body accumulates no velocity.
        let zero = crate::math::Mat3::zeros();
        let i_inv_a = if inv_mass_a == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if inv_mass_b == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
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
