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
    /// Per-substep cache built by `prepare`: the position-dependent
    /// invariants of all three axis rows. Positions are frozen during the
    /// SI loop, so levers, anchors, position errors, biases, and effective
    /// masses are constant across every sweep of a substep; recomputing
    /// them per sweep (2 `world_inv_inertia` per axis × 3 axes) was the
    /// dominant row cost in the 32-sweep assemblies (docs/findings.md
    /// 2026-10-22). BIT-IDENTITY CONTRACT: each cached value is computed by
    /// the exact expression the per-iteration code used, so prepared solves
    /// reproduce the historical trajectories bit-for-bit.
    cache: SolveCache,
}

/// One world-axis row's frozen solve state (see the `cache` field doc).
#[derive(Clone, Copy)]
struct AxisCache {
    /// Baumgarte velocity bias −β/dt·c at the frozen position error.
    bias: f64,
    /// w + compliance/dt. Degenerate (< 1e-10) rows set `skip`.
    denom: f64,
    /// I_a⁻¹·(r_a × n) / I_b⁻¹·(r_b × n): the angular responses to a unit
    /// impulse along the axis (zero matrices for pinned bodies, as the
    /// historical solve used).
    i_inv_a_ra_cross_n: Vec3,
    i_inv_b_rb_cross_n: Vec3,
    skip: bool,
}

#[derive(Clone, Copy)]
struct SolveCache {
    /// World-frame anchor levers at the frozen orientations.
    r_a: Vec3,
    r_b: Vec3,
    inv_mass_a: f64,
    inv_mass_b: f64,
    /// Per world axis (X, Y, Z) in solve order.
    axes: [AxisCache; 3],
}

impl SolveCache {
    /// State before any `prepare`: every axis row skips.
    fn inert() -> Self {
        let axis = AxisCache {
            bias: 0.0,
            denom: 0.0,
            i_inv_a_ra_cross_n: Vec3::zeros(),
            i_inv_b_rb_cross_n: Vec3::zeros(),
            skip: true,
        };
        Self {
            r_a: Vec3::zeros(),
            r_b: Vec3::zeros(),
            inv_mass_a: 0.0,
            inv_mass_b: 0.0,
            axes: [axis; 3],
        }
    }
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
            cache: SolveCache::inert(),
        }
    }

    /// Override the Baumgarte position-feedback coefficient (per substep).
    pub fn with_baumgarte(mut self, b: f64) -> Self { self.baumgarte = b; self }

    /// Build the per-substep cache (see the `cache` field doc). Called once
    /// per substep by `XpbdWorld::substep` before the SI loop; `solve` also
    /// runs it inline for direct callers. Every expression is copied
    /// verbatim from the historical per-iteration `solve_axis` — the
    /// bit-identity contract on the cache.
    pub fn prepare(&mut self, bodies: &[RigidBody], dt: f64) {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;

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
        let cfm = self.compliance / dt;

        let mut axes = [AxisCache {
            bias: 0.0,
            denom: 0.0,
            i_inv_a_ra_cross_n: Vec3::zeros(),
            i_inv_b_rb_cross_n: Vec3::zeros(),
            skip: true,
        }; 3];
        for (k, n) in [Vec3::x(), Vec3::y(), Vec3::z()].into_iter().enumerate() {
            // Position error along this axis.
            let c = (pb - pa).dot(&n) - self.rest_length * n.dot(&n).abs();
            // (For zero rest_length, the above is just (pb - pa) · n.)
            let ra_cross_n = r_a.cross(&n);
            let rb_cross_n = r_b.cross(&n);
            let i_inv_a_ra_cross_n = i_inv_a * ra_cross_n;
            let i_inv_b_rb_cross_n = i_inv_b * rb_cross_n;
            let w_a = inv_mass_a + ra_cross_n.dot(&i_inv_a_ra_cross_n);
            let w_b = inv_mass_b + rb_cross_n.dot(&i_inv_b_rb_cross_n);
            let w = w_a + w_b;
            let denom = w + cfm;
            // Baumgarte velocity bias.
            let bias = -self.baumgarte / dt * c;
            axes[k] = AxisCache {
                bias,
                denom,
                i_inv_a_ra_cross_n,
                i_inv_b_rb_cross_n,
                skip: denom < 1e-10,
            };
        }
        self.cache = SolveCache { r_a, r_b, inv_mass_a, inv_mass_b, axes };
    }

    /// One SI iteration over all three axis rows, self-contained: prepares
    /// the per-substep cache inline, then solves. `XpbdWorld` calls
    /// `solve_prepared` instead (one `prepare` per substep); this composed
    /// form keeps the historical single-call API and is bit-identical.
    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        self.prepare(bodies, dt);
        self.solve_prepared(bodies, dt);
    }

    /// One SI iteration over all three axis rows, reading the frozen
    /// invariants from the cache `prepare` built this substep. Each world
    /// axis is solved independently; convergence to the full 3-DOF
    /// constraint emerges from outer SI iterations.
    pub(crate) fn solve_prepared(&mut self, bodies: &mut Vec<RigidBody>, _dt: f64) {
        self.solve_axis_prepared(bodies, 0, Vec3::x());
        self.solve_axis_prepared(bodies, 1, Vec3::y());
        self.solve_axis_prepared(bodies, 2, Vec3::z());
    }

    fn solve_axis_prepared(&mut self, bodies: &mut Vec<RigidBody>, k: usize, n: Vec3) {
        let ax = self.cache.axes[k];
        if ax.skip {
            return;
        }

        // Velocity at each anchor (the only per-iteration state).
        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&self.cache.r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&self.cache.r_b);
        let v_rel_n = (v_b_anchor - v_a_anchor).dot(&n);

        let velocity_error = v_rel_n - ax.bias;
        let lambda = -velocity_error / ax.denom;
        if lambda.abs() < 1e-15 {
            return;
        }

        // Apply impulse along n.
        let p = n * lambda;
        bodies[self.body_a].linear_velocity -= p * self.cache.inv_mass_a;
        bodies[self.body_b].linear_velocity += p * self.cache.inv_mass_b;
        bodies[self.body_a].angular_velocity -= ax.i_inv_a_ra_cross_n * lambda;
        bodies[self.body_b].angular_velocity += ax.i_inv_b_rb_cross_n * lambda;
    }
}
