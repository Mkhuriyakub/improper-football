use crate::math::{Vec2, Vec3};
use crate::rigid_body::RigidBody;

/// Unilateral point-on-ground contact, solved as a velocity-level
/// constraint via sequential impulse (Bullet SI-style). Replaces the
/// explicit penalty-spring approach for cases where contact stiffness
/// would force a destabilizing explicit-Euler bound.
///
/// **Why this exists:** with explicit penalty contact (the world-level
/// ground pass and `FootContact`), the contact force is `K·penetration - C·velocity`
/// applied directly to the body's force accumulator. For stiff K and
/// small body inertia (e.g., a 1 kg foot in a chain bearing 56 kg),
/// the bounce mode is underdamped and the system accumulates oscillation
/// energy faster than dampers can dissipate it. This matches Drake's
/// SAP / Bullet's SI / MuJoCo's soft-constraint critique of penalty
/// contacts.
///
/// **Implementation:** each contact constraint is a unilateral
/// position-level constraint `C(x) = -penetration ≥ 0`. The SI solver
/// computes the corrective impulse `λ_n ≥ 0` along the world +Y axis
/// that drives the constraint to satisfaction over the substep.
/// Friction is two velocity-level tangent constraints (world X and Z)
/// whose impulses are jointly clamped to the Coulomb friction cone
/// `|λ_friction| ≤ μ·λ_n`.
///
/// **Compliance** lets the constraint be SOFT (like a stiff spring) —
/// `compliance = 1/stiffness`. Default 0 (rigid). Useful for modeling
/// flesh / shoe sole compressibility.
///
/// Differs from explicit penalty contact in:
/// - No explicit-Euler stability bound: arbitrarily stiff contact OK.
/// - Friction is a true cone constraint, not a viscous-with-cap approximation.
/// - Impulses are computed implicitly each SI iteration; converges over
///   the existing `solver_iterations` outer loop.
pub struct ContactConstraint {
    /// Body this contact attaches to.
    pub body: usize,
    /// Position of the contact point in body-local coordinates.
    pub local_point: Vec3,
    /// World y of the ground plane.
    pub plane_y: f64,
    /// Coulomb friction coefficient.
    pub friction_coefficient: f64,
    /// Compliance = 1/stiffness (m/N). 0 = rigid contact, > 0 = soft.
    pub compliance: f64,
    /// Baumgarte position-error feedback. Drives penetration toward 0
    /// over multiple substeps (0.2 = 20% per substep).
    pub baumgarte: f64,
    /// Accumulated normal impulse this substep (Lagrange multiplier).
    /// Reset before SI iterations begin.
    pub lambda_normal: f64,
    /// Accumulated tangential friction impulse (x, z components).
    pub lambda_friction: Vec2,
    /// Which capsule endpoint this contact belongs to (0 = +axis,
    /// 1 = −axis — `engaged_endpoints`' slot convention). Set by
    /// `generate_ground_manifold`; (body, endpoint_slot) is the manifold's
    /// stable identity for the cross-substep warm-start carry. 0 and
    /// meaningless for user-pushed contacts (their container never
    /// carries).
    pub endpoint_slot: u8,
}

impl ContactConstraint {
    pub fn new(body: usize, local_point: Vec3) -> Self {
        Self {
            body,
            local_point,
            plane_y: 0.0,
            friction_coefficient: 0.6,
            compliance: 0.0,
            // Baumgarte applied per SUBSTEP (sub_dt ~ 0.14 ms), not per
            // frame. The standard β = 0.2 used for joint constraints is
            // tuned for a typical timestep of ~16 ms, where β/dt ≈ 12/s
            // corrects position error in ~80 ms (5 frames). At our
            // sub_dt scale, the same β would give 1400/s — way too
            // aggressive, launching bodies on tiny penetrations. Set
            // β so β/sub_dt ≈ 12/s → β ≈ 0.0017 (sub_dt = 1.39e-4).
            baumgarte: 0.002,
            lambda_normal: 0.0,
            lambda_friction: Vec2::zeros(),
            endpoint_slot: 0,
        }
    }

    pub fn with_plane_y(mut self, y: f64) -> Self { self.plane_y = y; self }
    pub fn with_friction(mut self, mu: f64) -> Self { self.friction_coefficient = mu; self }
    pub fn with_compliance(mut self, c: f64) -> Self { self.compliance = c; self }
    pub fn with_baumgarte(mut self, b: f64) -> Self { self.baumgarte = b; self }
    pub fn with_endpoint_slot(mut self, slot: u8) -> Self { self.endpoint_slot = slot; self }

    /// Reset accumulated impulses. Called once per substep, BEFORE the
    /// SI iteration loop.
    pub fn reset_impulses(&mut self) {
        self.lambda_normal = 0.0;
        self.lambda_friction = Vec2::zeros();
    }

    /// Standard warm start (the coupled solve's L2 lever, design §2.6):
    /// seed this freshly generated constraint with the impulses its
    /// (body, endpoint_slot) identity accumulated LAST substep, and apply
    /// them to the body through the current lever arm — at reset time,
    /// before the SI sweeps. The sweeps then solve incrementally from the
    /// seed: a still-valid seed means iteration deltas near zero (the
    /// cross-substep static memory the near-degenerate joint↔contact
    /// alignments need), a stale one is unwound within the first sweep by
    /// the λ ≥ 0 clamp (normal) and the cone projection (friction) through
    /// this same lever arm — orientation is frozen across the substep's SI
    /// loop, so the removal cancels the application exactly.
    pub fn warm_start(&mut self, bodies: &mut Vec<RigidBody>, lambda_n: f64, lambda_t: Vec2) {
        let body = &bodies[self.body];
        if body.inv_mass == 0.0 {
            return;
        }
        // The unilateral invariant λ_n ≥ 0 holds for any carried value
        // (the carry filters on λ_n > 0; max(0) keeps the invariant local).
        self.lambda_normal = lambda_n.max(0.0);
        self.lambda_friction = lambda_t;
        let impulse = Vec3::new(lambda_t.x, self.lambda_normal, lambda_t.y);
        let inv_mass = body.inv_mass;
        let i_inv = body.world_inv_inertia();
        let r = body.orientation * self.local_point;
        let r_cross_imp = r.cross(&impulse);
        bodies[self.body].linear_velocity += impulse * inv_mass;
        bodies[self.body].angular_velocity += i_inv * r_cross_imp;
    }

    /// One Gauss-Seidel iteration of normal + friction constraint.
    /// Called from `XpbdWorld::solve_constraints` (which is called
    /// `solver_iterations` times per substep).
    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let body = &bodies[self.body];
        if body.inv_mass == 0.0 {
            return;
        }
        let r = body.orientation * self.local_point;
        let contact_world = body.position + r;
        let penetration = self.plane_y - contact_world.y;

        // ─── Normal constraint (world +Y) ─────────────────────────────
        let n = Vec3::new(0.0, 1.0, 0.0);
        let v_contact = body.linear_velocity + body.angular_velocity.cross(&r);
        let v_n = v_contact.dot(&n);

        // Skip when the constraint can produce no impulse this iteration:
        // not yet touching (penetration ≤ 0), nothing accumulated to
        // maintain or unwind, and not approaching fast enough to cross the
        // plane within this substep (v_n ≥ penetration/dt — with that
        // speculative bias the normal solve below would yield
        // delta_lambda ≤ 0, clamped to zero by unilaterality).
        if penetration <= 0.0
            && self.lambda_normal <= 0.0
            && v_n >= penetration / dt
        {
            return;
        }

        let inv_mass = body.inv_mass;
        let i_inv = body.world_inv_inertia();
        let r_cross_n = r.cross(&n);
        let w_n = inv_mass + r_cross_n.dot(&(i_inv * r_cross_n));
        if w_n < 1e-12 {
            return;
        }
        let cfm = self.compliance / dt;

        let bias = if penetration > 0.0 {
            // Baumgarte velocity bias: drives penetration → 0 over time.
            // Want v_n ≥ baumgarte/dt × penetration (separating velocity).
            (self.baumgarte / dt) * penetration
        } else {
            // Speculative contact (not yet touching): allow the body to
            // close the remaining gap within this substep but no more —
            // target floor v_n ≥ penetration/dt (negative: still
            // approaching). The impulse removes exactly the approach speed
            // that would overshoot the plane, so the body lands on the
            // surface instead of embedding one substep deep. Purely
            // dissipative: λ ≥ 0 only ever REDUCES approach speed, and the
            // floor is never positive, so a separated body is never pushed.
            penetration / dt
        };
        let velocity_error = v_n - bias;
        let delta_lambda_n = -velocity_error / (w_n + cfm);

        // Clamp accumulated normal impulse to ≥ 0 (unilateral).
        let old_lambda_n = self.lambda_normal;
        let new_lambda_n = (old_lambda_n + delta_lambda_n).max(0.0);
        let applied_dn = new_lambda_n - old_lambda_n;
        self.lambda_normal = new_lambda_n;

        if applied_dn.abs() > 1e-15 {
            let impulse_n = n * applied_dn;
            bodies[self.body].linear_velocity += impulse_n * inv_mass;
            bodies[self.body].angular_velocity += i_inv * r_cross_n * applied_dn;
        }

        // ─── Friction (Coulomb cone, world XZ tangent plane) ──────────
        //
        // POLICY NOTE (deliberate, not an accident): speculative contacts
        // (generated for a predicted end-of-substep crossing while the point
        // is still a hair above the plane) accumulate λ_n through the
        // approach-speed floor, and therefore ALSO run friction — i.e.
        // touchdown friction fires one substep early, at a point up to
        // v·sub_dt (sub-mm) above the plane. This is the inelastic
        // frictional-impact behavior moved one substep forward, bounded by
        // the gap; NOT a sustained drag on separated bodies (a speculative
        // contact that never develops λ_n > 0 never accumulates friction).
        if self.friction_coefficient <= 0.0 {
            return;
        }
        if self.lambda_normal <= 0.0 {
            // Contact released within this substep: an earlier SI iteration
            // accumulated friction while λ_n > 0, then a later iteration
            // clamped λ_n back to 0. The Coulomb cone is now |λ_t| ≤ μ·λ_n
            // = 0, so any accumulated tangential impulse violates it —
            // leaving it applied puts a stale friction kick on the body for
            // the rest of the substep (and, since both contact containers
            // rebuild/reset impulses every substep, it is never unwound
            // later). Project to the cone: remove the accumulated impulse
            // through the SAME lever arm it was applied with (r is
            // orientation-dependent only, and orientation doesn't change
            // within the SI loop, so the removal cancels exactly).
            if self.lambda_friction.norm() > 1e-15 {
                let body = &bodies[self.body];
                let r = body.orientation * self.local_point;
                let i_inv = body.world_inv_inertia();
                let removal =
                    Vec3::new(-self.lambda_friction.x, 0.0, -self.lambda_friction.y);
                let r_cross_imp = r.cross(&removal);
                bodies[self.body].linear_velocity += removal * inv_mass;
                bodies[self.body].angular_velocity += i_inv * r_cross_imp;
                self.lambda_friction = Vec2::zeros();
            }
            return;
        }
        // Re-read body state after normal impulse.
        let body = &bodies[self.body];
        let r = body.orientation * self.local_point;
        let v_contact = body.linear_velocity + body.angular_velocity.cross(&r);

        let i_inv = body.world_inv_inertia();
        let limit = self.friction_coefficient * self.lambda_normal;

        // Solve world-X tangent then world-Z tangent in sequence,
        // each clamped to the remaining cone budget. Cone is clamped
        // jointly at the end so the (x, z) impulse magnitude stays
        // within μ·λ_n.
        let mut new_lambda_t = self.lambda_friction;

        for (axis_idx, tangent) in [Vec3::x(), Vec3::z()].iter().enumerate() {
            let v_t = v_contact.dot(tangent);
            let r_cross_t = r.cross(tangent);
            let w_t = inv_mass + r_cross_t.dot(&(i_inv * r_cross_t));
            if w_t < 1e-12 {
                continue;
            }
            let delta_lambda_t = -v_t / w_t;
            let component = if axis_idx == 0 { &mut new_lambda_t.x }
                            else { &mut new_lambda_t.y };
            *component += delta_lambda_t;
        }

        // Clamp friction impulse magnitude to the cone.
        let mag = new_lambda_t.norm();
        if mag > limit && mag > 0.0 {
            new_lambda_t *= limit / mag;
        }

        let applied_dt = new_lambda_t - self.lambda_friction;
        self.lambda_friction = new_lambda_t;

        if applied_dt.norm() > 1e-15 {
            let friction_impulse = Vec3::new(applied_dt.x, 0.0, applied_dt.y);
            // Need fresh r_cross calculations.
            let r_cross_imp = r.cross(&friction_impulse);
            bodies[self.body].linear_velocity += friction_impulse * inv_mass;
            bodies[self.body].angular_velocity += i_inv * r_cross_imp;
        }
    }

    /// True if this contact has accumulated normal impulse this substep
    /// (i.e., the ground is currently supporting the body at this point).
    pub fn is_active(&self) -> bool {
        self.lambda_normal > 0.0
    }

    /// World position of the contact point as of the current body pose.
    pub fn contact_point_world(&self, bodies: &[RigidBody]) -> Vec3 {
        let body = &bodies[self.body];
        body.position + body.orientation * self.local_point
    }
}
