use nalgebra::UnitQuaternion;
use crate::math::{Mat3, Vec3, swing_twist_angle};
use crate::rigid_body::RigidBody;

/// Soft range limit configuration for a `JointAxis` — the same nonlinear
/// ligament model as `AngularSoftLimit` (`τ = k·|excess|^exponent` opposing
/// further excess, tearable), carried as data so the joint's combined solve
/// can fold the limit's torque and local stiffness into the shared
/// denominator instead of running it as a separate sequential pass.
#[derive(Clone, Copy, Debug)]
pub struct SoftLimit {
    pub min_angle: f64,
    pub max_angle: f64,
    /// Stiffness coefficient (N·m / rad^exponent). Effective stiffness is
    /// `stiffness * post_tear_factor` once `torn` is true.
    pub stiffness: f64,
    /// Exponent on |excess|. 1 is linear, 2 quadratic (C¹ at the ROM
    /// boundary — torque AND slope are zero there), higher is progressively
    /// stiffer. Real ligament toe/linear regions are roughly 2–3.
    pub exponent: f64,
    /// If |excess| ever exceeds this threshold the ligament tears and the
    /// stiffness is permanently scaled by `post_tear_factor`.
    /// `f64::INFINITY` = no tearing. Same semantics as `AngularSoftLimit`.
    pub tear_threshold: f64,
    /// Stiffness multiplier applied after tearing (0.0 = complete failure).
    pub post_tear_factor: f64,
    /// Latched on the first substep where |excess| exceeds `tear_threshold`.
    /// Persistent — no automatic healing.
    pub torn: bool,
}

impl SoftLimit {
    pub fn new(min_angle: f64, max_angle: f64, stiffness: f64, exponent: f64) -> Self {
        Self {
            min_angle, max_angle, stiffness, exponent,
            tear_threshold: f64::INFINITY,
            post_tear_factor: 0.0,
            torn: false,
        }
    }

    fn effective_stiffness(&self) -> f64 {
        if self.torn { self.stiffness * self.post_tear_factor } else { self.stiffness }
    }

    /// Scalar restoring torque about the joint axis at `angle` (positive =
    /// increases the relative angle). 0 inside [min_angle, max_angle].
    /// Identical formula to `AngularSoftLimit::scalar_torque`.
    pub fn scalar_torque(&self, angle: f64) -> f64 {
        let k = self.effective_stiffness();
        if angle > self.max_angle {
            -k * (angle - self.max_angle).powf(self.exponent)
        } else if angle < self.min_angle {
            k * (self.min_angle - angle).powf(self.exponent)
        } else {
            0.0
        }
    }

    /// Local slope ds/dθ of the restoring torque at `angle` (≤ 0 engaged,
    /// 0 inside the range). The combined solve folds `−slope ≥ 0` into the
    /// shared stiffness denominator.
    fn scalar_torque_slope(&self, angle: f64) -> f64 {
        let k = self.effective_stiffness();
        if angle > self.max_angle {
            let ex = angle - self.max_angle;
            -k * self.exponent * ex.powf(self.exponent - 1.0)
        } else if angle < self.min_angle {
            let ex = self.min_angle - angle;
            -k * self.exponent * ex.powf(self.exponent - 1.0)
        } else {
            0.0
        }
    }

    /// Latch the torn state if |excess| at `angle` exceeds the threshold.
    /// Evaluated at the linearization angle θₙ, on the same substep the
    /// threshold is crossed — identical to `AngularSoftLimit::update_torn`.
    fn update_torn(&mut self, angle: f64) {
        let excess_abs = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            self.min_angle - angle
        } else {
            0.0
        };
        if !self.torn && excess_abs > self.tear_threshold {
            self.torn = true;
        }
    }

    /// Elastic potential energy stored when out of range, with the current
    /// effective (post-tear if torn) stiffness. Same integral as
    /// `AngularSoftLimit::potential_energy`: k·|excess|^(e+1)/(e+1).
    pub fn potential_energy(&self, angle: f64) -> f64 {
        let excess = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            self.min_angle - angle
        } else {
            return 0.0;
        };
        self.effective_stiffness() * excess.powf(self.exponent + 1.0) / (self.exponent + 1.0)
    }
}

/// One joint axis with ALL of its coaxial angular terms — active PD
/// (kp/kd/feedforward/torque limit), passive spring, passive damper, and
/// nonlinear soft ROM limit — solved together in a SINGLE backward-Euler
/// velocity solve with a shared denominator.
///
/// WHY (docs/findings.md 2026-10): the legacy primitives integrate each
/// coaxial term in its own sequential backward-Euler pass, and every such
/// pass divides its own contribution by its own `1 + dt²·w·k` — effective
/// gain `kp_eff = kp/(1 + dt²·w·kp)`, ~94% attenuation on gram-scale toe
/// joints at sub_dt = 1/7200. Consequences measured there: equilibria that
/// do not follow the rigid split kp/(kp+K)·target, a dt-DEPENDENT settle
/// point, and a load-bearing kp-before-dampers pass ordering. The combined
/// solve puts every coaxial stiffness and damping term in ONE denominator,
/// so the equilibrium is the dt-independent physical one and ordering
/// against other passes stops mattering for the joint's own terms.
///
/// INVARIANTS (the stability/fidelity certificate):
/// - The denominator `1 + dt·w·(d_act + d_pas) + dt²·w·(k_act + k_pas)` is
///   ≥ 1 always: every k and d is ≥ 0 (the soft limit contributes
///   `−s'(θ) ≥ 0`; `ext_stiffness`/`ext_damping` are expected ≥ 0), so the
///   solve is unconditionally A-stable for any gains and any dt — the same
///   certificate each legacy implicit pass carries, now shared.
/// - Single-term configurations reduce EXACTLY to the legacy primitives'
///   formulas (spring ⇒ `AngularSpring::apply_implicit`, damper ⇒
///   `AngularDamper::apply_implicit`, soft limit ⇒
///   `AngularSoftLimit::apply_implicit`, kp ⇒
///   `AngularPDController::apply_implicit_kp`'s unsaturated branch).
/// - Linearization is at θₙ, so limit ENGAGEMENT has the same one-substep
///   lag as the legacy pass (C¹ at exponent = 2 — torque and slope are both
///   zero at the ROM boundary, so there is no jump). A predictor-corrector
///   (re-linearizing at the predicted θ⁺) is deliberately deferred.
///
/// `enabled` gates only the ACTIVE terms (kp, kd, feedforward, torque
/// limit) — muscle deactivation. The passive structures (spring, damper,
/// limit) are properties of the joint and stay on.
pub struct JointAxis {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame. Must be unit length.
    pub axis_local_a: Vec3,
    /// Rest orientation of body_b in body_a's frame (default identity).
    /// All angles are measured relative to this baseline via swing-twist
    /// factorization, same as every legacy primitive.
    pub baseline: UnitQuaternion<f64>,

    // ── Passive structures ──
    /// Passive centering stiffness K_sp in N·m/rad (0 = absent).
    pub passive_stiffness: f64,
    /// Rest angle of the passive spring (rad, relative to baseline).
    pub passive_rest_angle: f64,
    /// Passive damping c in N·m·s/rad (0 = absent).
    pub passive_damping: f64,
    /// Nonlinear soft ROM limit (None = absent).
    pub limit: Option<SoftLimit>,

    // ── Active (muscle) terms ──
    /// Proportional gain (N·m/rad).
    pub kp: f64,
    /// Derivative gain (N·m·s/rad).
    pub kd: f64,
    /// Commanded angle along the axis (rad, relative to baseline).
    pub target_angle: f64,
    /// Feedforward torque (N·m) added to the active output.
    pub feedforward_torque: f64,
    /// Cap (N·m) on the controller's TOTAL active output (KP + FF + KD
    /// combined — one muscle's maximum). `f64::INFINITY` = no cap.
    pub torque_limit: f64,
    /// If false, the active terms produce nothing (muscle deactivation).
    pub enabled: bool,

    // ── Per-substep external coaxial injections (future tendons) ──
    /// Extra scalar torque about the axis, consumed (cleared to 0) by each
    /// solve. Lets external actuators join the combined solve instead of
    /// adding another sequential pass.
    pub ext_torque: f64,
    /// Extra stiffness folded into the shared denominator this substep
    /// (cleared each solve). Expected ≥ 0 — see the denominator invariant.
    pub ext_stiffness: f64,
    /// Extra damping folded into the shared denominator this substep
    /// (cleared each solve). Expected ≥ 0.
    pub ext_damping: f64,

    // ── Diagnostics (written every solve) ──
    /// Active (PD + feedforward) torque actually applied last substep,
    /// evaluated at the end-of-substep state; equals ±torque_limit exactly
    /// while saturated.
    pub applied_active_torque: f64,
    /// Total torque (active + passive + ext) actually applied last substep
    /// (= J/dt of the combined impulse).
    pub applied_total_torque: f64,
}

impl JointAxis {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3) -> Self {
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "JointAxis axis must be unit length (norm = {})",
            axis.norm()
        );
        Self {
            body_a, body_b, axis_local_a: axis,
            baseline: UnitQuaternion::identity(),
            passive_stiffness: 0.0,
            passive_rest_angle: 0.0,
            passive_damping: 0.0,
            limit: None,
            kp: 0.0,
            kd: 0.0,
            target_angle: 0.0,
            feedforward_torque: 0.0,
            torque_limit: f64::INFINITY,
            enabled: true,
            ext_torque: 0.0,
            ext_stiffness: 0.0,
            ext_damping: 0.0,
            applied_active_torque: 0.0,
            applied_total_torque: 0.0,
        }
    }

    /// Set a non-identity rest orientation of body_b in body_a's frame.
    pub fn with_baseline(mut self, baseline: UnitQuaternion<f64>) -> Self {
        self.baseline = baseline;
        self
    }

    /// Passive centering spring: stiffness k (N·m/rad) toward `rest` (rad).
    pub fn with_passive_spring(mut self, k: f64, rest: f64) -> Self {
        self.passive_stiffness = k;
        self.passive_rest_angle = rest;
        self
    }

    /// Passive damping c (N·m·s/rad).
    pub fn with_passive_damping(mut self, c: f64) -> Self {
        self.passive_damping = c;
        self
    }

    /// Nonlinear soft ROM limit (same model as `AngularSoftLimit`).
    pub fn with_soft_limit(mut self, min: f64, max: f64, k: f64, exponent: f64) -> Self {
        self.limit = Some(SoftLimit::new(min, max, k, exponent));
        self
    }

    /// Enable ligament tearing on the soft limit (mirror of
    /// `AngularSoftLimit::with_tearing`). Must be called after
    /// `with_soft_limit`.
    pub fn with_tearing(mut self, threshold: f64, post_tear_factor: f64) -> Self {
        let lim = self.limit.as_mut()
            .expect("JointAxis::with_tearing requires with_soft_limit first");
        lim.tear_threshold = threshold;
        lim.post_tear_factor = post_tear_factor;
        self
    }

    /// Active PD gains.
    pub fn with_pd(mut self, kp: f64, kd: f64) -> Self {
        self.kp = kp;
        self.kd = kd;
        self
    }

    /// Commanded target angle.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target_angle = target;
        self
    }

    /// Feedforward torque.
    pub fn with_feedforward(mut self, ff: f64) -> Self {
        self.feedforward_torque = ff;
        self
    }

    /// Cap on the total active output.
    pub fn with_torque_limit(mut self, limit: f64) -> Self {
        self.torque_limit = limit;
        self
    }

    /// Signed joint angle about the axis: swing-twist factorization of
    /// `q_a⁻¹·q_b·baseline⁻¹` — identical to every legacy primitive.
    pub fn angle(&self, bodies: &[RigidBody]) -> f64 {
        let q_b_in_a = bodies[self.body_a].orientation.inverse()
                     * bodies[self.body_b].orientation;
        let q_from_rest = q_b_in_a * self.baseline.inverse();
        swing_twist_angle(q_from_rest, self.axis_local_a)
    }

    /// Elastic potential energy stored in the joint's passive structures:
    /// spring ½·K_sp·(θ − rest)² (mirror of `AngularSpring::potential_energy`)
    /// plus the soft limit's energy integral (mirror of
    /// `AngularSoftLimit::potential_energy`).
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let angle = self.angle(bodies);
        let dev = angle - self.passive_rest_angle;
        let mut e = 0.5 * self.passive_stiffness * dev * dev;
        if let Some(lim) = &self.limit {
            e += lim.potential_energy(angle);
        }
        e
    }

    /// The combined implicit solve. Must be called once per substep, after
    /// `integrate_velocities` (in `XpbdWorld::substep` it runs in the same
    /// slot as the legacy implicit passes, immediately before them).
    ///
    /// Per substep (all quantities scalar, along the world axis
    /// n = q_a·axis_local_a):
    /// 1. Tear latch from θₙ (same semantics as `AngularSoftLimit`).
    /// 2. Shared state: θ = angle(), w = nᵀI_a⁻¹n + nᵀI_b⁻¹n with pinned
    ///    (inv_mass == 0) bodies contributing ZERO; early-out w < 1e-15.
    ///    ω = (ω_b − ω_a)·n.
    /// 3. Torques at the linearization point θₙ:
    ///      τ_act0 = enabled ? kp·(target − θ) + ff : 0
    ///      τ_pas0 = −K_sp·(θ − rest) + s_lim(θ) + ext_torque
    ///      k_act  = enabled ? kp : 0,  k_pas = K_sp − s_lim'(θ) + ext_stiffness
    ///      d_act  = enabled ? kd : 0,  d_pas = c + ext_damping
    /// 4. UNSATURATED combined solve:
    ///      ω⁺ = (ω + dt·w·(τ_act0 + τ_pas0))
    ///         / (1 + dt·w·(d_act + d_pas) + dt²·w·(k_act + k_pas))
    /// 5. Saturation (only when torque_limit is finite and enabled): the
    ///    active torque at the end-of-substep state is
    ///      τ_act⁺ = kp·(target − θ − dt·ω⁺) + ff − kd·ω⁺.
    ///    If |τ_act⁺| ≤ torque_limit, accept the unsaturated solve.
    ///    Otherwise re-solve with the active output held at the cap
    ///    (σ = sign(τ_act⁺)):
    ///      ω⁺ = (ω + dt·w·(σ·torque_limit + τ_pas0))
    ///         / (1 + dt·w·d_pas + dt²·w·k_pas).
    ///    The saturated branch is accepted unconditionally: it replaces the
    ///    active contribution with a constant torque of SMALLER magnitude
    ///    (that is what saturation means), and a bounded constant torque is
    ///    the same regime the legacy clamp already models (`s' = 0`) — it
    ///    only ever reduces the active magnitude, so it cannot inject
    ///    energy or destabilize. The passive terms keep their full implicit
    ///    treatment (their stiffness/damping stay in the denominator):
    ///    "the muscle saturates, the ligaments don't."
    ///
    ///    σ SIGN-FLIP SUB-CASE (intentional): σ is the sign of the
    ///    END-OF-SUBSTEP demand τ_act⁺, not of the position error at θₙ.
    ///    When the joint flies fast TOWARD the target, the velocity terms
    ///    (−kd·ω⁺ and the kp term's −kp·dt·ω⁺ lookahead) can dominate and
    ///    flip σ negative while kp·(target − θₙ) alone is still positive:
    ///    the combined solve then BRAKES at −torque_limit where the legacy
    ///    pass (which clamps the θₙ demand) kept pushing at +torque_limit.
    ///    The muscle spends its bounded budget decelerating into the
    ///    target instead of accelerating past it — more physical, and a
    ///    deliberate divergence from the legacy criterion (see the T2
    ///    saturation-criterion note in tests/joint_axis.rs).
    /// 6. Apply the single impulse J = (ω⁺ − ω)/w to both bodies; record
    ///    `applied_total_torque = J/dt`; clear the ext_* injections.
    pub fn apply_implicit(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        // Consume the per-substep external injections whether or not the
        // solve early-outs — they are per-substep quantities.
        let ext_torque = self.ext_torque;
        let ext_stiffness = self.ext_stiffness;
        let ext_damping = self.ext_damping;
        self.ext_torque = 0.0;
        self.ext_stiffness = 0.0;
        self.ext_damping = 0.0;
        // The denominator-≥-1 A-stability certificate (see the struct docs)
        // requires every stiffness and damping contribution to be ≥ 0; the
        // built-in terms are non-negative by construction, so the external
        // injections are the only way a caller can break it.
        debug_assert!(
            ext_stiffness >= 0.0 && ext_damping >= 0.0,
            "JointAxis ext_stiffness ({ext_stiffness}) and ext_damping ({ext_damping}) must \
             be >= 0: a negative injection voids the denominator->=-1 A-stability certificate"
        );
        self.applied_active_torque = 0.0;
        self.applied_total_torque = 0.0;

        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let theta = self.angle(bodies);

        // 1. Tear latch first, from θₙ — the failure mode fires on the same
        // substep the threshold is crossed, exactly like AngularSoftLimit.
        if let Some(lim) = &mut self.limit {
            lim.update_torn(theta);
        }

        // 2. Shared reduced-inertia state. Pinned bodies contribute zero
        // inverse inertia (post-f11d73f convention of all primitives).
        let zero = Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let w = n.dot(&(i_inv_a * n)) + n.dot(&(i_inv_b * n));
        if w < 1e-15 {
            return; // both bodies pinned
        }

        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega = omega_rel.dot(&n);

        // 3. Torques and local stiffnesses at the linearization point θₙ.
        let active = self.enabled;
        let tau_act0 = if active {
            self.kp * (self.target_angle - theta) + self.feedforward_torque
        } else {
            0.0
        };
        let (s_lim, k_lim) = match &self.limit {
            // k_lim = −s_lim'(θ) ≥ 0 (slope is ≤ 0 in the engaged region).
            Some(lim) => (lim.scalar_torque(theta), -lim.scalar_torque_slope(theta)),
            None => (0.0, 0.0),
        };
        let tau_pas0 = -self.passive_stiffness * (theta - self.passive_rest_angle)
            + s_lim
            + ext_torque;
        let k_act = if active { self.kp } else { 0.0 };
        let k_pas = self.passive_stiffness + k_lim + ext_stiffness;
        let d_act = if active { self.kd } else { 0.0 };
        let d_pas = self.passive_damping + ext_damping;

        // 4. Unsaturated combined solve — ONE shared denominator (≥ 1, the
        // A-stability certificate: every k and d above is ≥ 0).
        let mut omega_new = (omega + dt * w * (tau_act0 + tau_pas0))
            / (1.0 + dt * w * (d_act + d_pas) + dt * dt * w * (k_act + k_pas));

        // 5. Saturation check on the ACTIVE output at the end-of-substep
        // state (θ⁺ = θ + dt·ω⁺).
        if active {
            let tau_act_plus = self.kp * (self.target_angle - theta - dt * omega_new)
                + self.feedforward_torque
                - self.kd * omega_new;
            if self.torque_limit.is_finite() && tau_act_plus.abs() > self.torque_limit {
                let sigma = tau_act_plus.signum();
                omega_new = (omega + dt * w * (sigma * self.torque_limit + tau_pas0))
                    / (1.0 + dt * w * d_pas + dt * dt * w * k_pas);
                self.applied_active_torque = sigma * self.torque_limit;
            } else {
                self.applied_active_torque = tau_act_plus;
            }
        }

        // 6. Apply the single combined impulse.
        let j = (omega_new - omega) / w;
        self.applied_total_torque = j / dt;

        bodies[self.body_a].angular_velocity -= i_inv_a * n * j;
        bodies[self.body_b].angular_velocity += i_inv_b * n * j;
    }
}
