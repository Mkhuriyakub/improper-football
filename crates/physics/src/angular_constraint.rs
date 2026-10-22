use crate::math::{Vec3, swing_twist_angle};
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
    /// Baumgarte position-error feedback coefficient, applied per SUBSTEP
    /// (β/sub_dt ≈ 1440/s at sub_dt = 1/7200). Kept at the aggressive 0.2 —
    /// NOT rescaled the way `ContactConstraint` rescaled its β to 0.002 —
    /// on measured evidence (2026-10-19 sweep, docs/findings.md): energy is
    /// β-independent (bilateral bias velocity is reclaimed as the error
    /// closes; the contact launch artifact is unilateral-only), while lock
    /// rigidity tracks β directly. The 26-bone foot's welds ([0,0]-range
    /// locks) held to 2.4e-3 rad at β = 0.2 but degraded to 0.12–0.17 rad at
    /// β = 0.002 — the feedback is what holds welded orientations against
    /// sustained gravity load, so lowering β makes assemblies spongy without
    /// buying any energy improvement.
    pub baumgarte: f64,
    /// Per-substep cache built by `prepare`: the world axis, the swing-twist
    /// angle and its violation, the bias, and the inertia projections —
    /// all functions of the frozen positions/orientations only, so constant
    /// across every SI sweep of a substep. Recomputing the quaternion
    /// swing-twist and two `world_inv_inertia` per row per sweep was a
    /// dominant row cost in the 32-sweep assemblies (docs/findings.md
    /// 2026-10-22). BIT-IDENTITY CONTRACT: every cached value is computed
    /// by the exact expression the per-iteration code used, so prepared
    /// solves reproduce the historical trajectories bit-for-bit.
    cache: SolveCache,
}

/// Frozen per-substep solve state (see the `cache` field doc).
#[derive(Clone, Copy)]
struct SolveCache {
    /// Constraint axis in world frame at the frozen orientation of body_a.
    n: Vec3,
    /// Signed distance into the violated half-space at the frozen angle.
    c: f64,
    /// Baumgarte velocity bias −β/dt·c.
    bias: f64,
    /// I_a⁻¹·n / I_b⁻¹·n (zero for pinned bodies, as the historical solve
    /// used).
    i_inv_a_n: Vec3,
    i_inv_b_n: Vec3,
    /// w + compliance/dt.
    denom: f64,
    /// Angle within [min, max]: iterations are no-ops this substep.
    inactive: bool,
    /// denom < 1e-10 (degenerate pair).
    degenerate: bool,
}

impl SolveCache {
    /// State before any `prepare`: iterations skip.
    fn inert() -> Self {
        Self {
            n: Vec3::zeros(),
            c: 0.0,
            bias: 0.0,
            i_inv_a_n: Vec3::zeros(),
            i_inv_b_n: Vec3::zeros(),
            denom: 0.0,
            inactive: true,
            degenerate: true,
        }
    }
}

impl AngularLimitConstraint {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, min: f64, max: f64) -> Self {
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "AngularLimitConstraint axis must be unit length (norm = {})",
            axis.norm()
        );
        Self {
            body_a,
            body_b,
            axis_local_a: axis,
            min_angle: min,
            max_angle: max,
            compliance: 0.0,
            baumgarte: 0.2,
            cache: SolveCache::inert(),
        }
    }

    /// Override the Baumgarte position-feedback coefficient (per substep).
    pub fn with_baumgarte(mut self, b: f64) -> Self { self.baumgarte = b; self }

    /// Build the per-substep cache (see the `cache` field doc). Called once
    /// per substep by `XpbdWorld::substep` before the SI loop; `solve` also
    /// runs it inline for direct callers. Every expression is copied
    /// verbatim from the historical per-iteration solve — the bit-identity
    /// contract on the cache.
    pub fn prepare(&mut self, bodies: &[RigidBody], dt: f64) {
        // Constraint axis in world frame (follows body_a's orientation).
        let n = bodies[self.body_a].orientation * self.axis_local_a;

        // Joint angle via swing-twist factorization in body_a's local frame
        // — invariant to rotations perpendicular to axis_local_a (no
        // bivector cross-coupling that the older scaled_axis().dot()
        // projection introduced).
        let q_b_in_a = bodies[self.body_a].orientation.inverse()
                     * bodies[self.body_b].orientation;
        let angle = swing_twist_angle(q_b_in_a, self.axis_local_a);

        // Position error: signed distance into the violated half-space.
        let c = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            angle - self.min_angle
        } else {
            self.cache = SolveCache::inert();
            return;
        };

        // Baumgarte velocity bias (pushes back toward range).
        let bias = -self.baumgarte / dt * c;

        // Generalized inverse mass along n. Pinned bodies (inv_mass == 0)
        // are infinite-inertia: use a zero inverse inertia for them
        // regardless of what their tensor holds (guards against phantom
        // inertia on bodies pinned by hand). The zero i_inv flows into
        // both w and the impulse application, so the dynamic partner
        // receives the full impulse and the pinned body stays at rest.
        let zero = crate::math::Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let i_inv_a_n = i_inv_a * n;
        let i_inv_b_n = i_inv_b * n;
        let w_a = n.dot(&i_inv_a_n);
        let w_b = n.dot(&i_inv_b_n);
        let w = w_a + w_b;
        let cfm = self.compliance / dt;
        let denom = w + cfm;

        self.cache = SolveCache {
            n,
            c,
            bias,
            i_inv_a_n,
            i_inv_b_n,
            denom,
            inactive: false,
            degenerate: denom < 1e-10,
        };
    }

    /// One SI iteration, self-contained: prepares the per-substep cache
    /// inline, then solves. `XpbdWorld` calls `solve_prepared` instead (one
    /// `prepare` per substep); this composed form keeps the historical
    /// single-call API and is bit-identical.
    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        self.prepare(bodies, dt);
        self.solve_prepared(bodies, dt);
    }

    /// One SI iteration reading the frozen invariants from the cache
    /// `prepare` built this substep. Only the relative velocity (and the
    /// one-sided gate it feeds) is per-iteration state.
    pub(crate) fn solve_prepared(&mut self, bodies: &mut Vec<RigidBody>, _dt: f64) {
        if self.cache.inactive {
            return;
        }

        // Relative angular velocity along the axis.
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let v_rel_n = omega_rel.dot(&self.cache.n);
        let velocity_error = v_rel_n - self.cache.bias;

        // One-sided: only apply impulse if it would push back toward range.
        // Equivalently: skip when velocity_error has the same sign as -c.
        // For c > 0 (above max), we apply only if velocity_error > 0.
        // For c < 0 (below min), we apply only if velocity_error < 0.
        if velocity_error * self.cache.c <= 0.0 {
            return;
        }
        if self.cache.degenerate {
            return;
        }

        let lambda = -velocity_error / self.cache.denom;
        bodies[self.body_a].angular_velocity -= self.cache.i_inv_a_n * lambda;
        bodies[self.body_b].angular_velocity += self.cache.i_inv_b_n * lambda;
    }
}
