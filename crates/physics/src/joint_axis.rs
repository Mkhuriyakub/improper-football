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
    /// WARNING: under the coupled solve's carried warm seeds, contact
    /// RELEASE fires a documented self-healing one-substep velocity burst
    /// of up to ~3e5 rad/s at toe scale (docs/findings.md 2026-10-21) —
    /// one substep of that is an ω·dt angle excursion of tens of radians,
    /// which would permanently (falsely) latch any finite threshold here.
    /// Size tear thresholds with that burst scale in mind, or keep tearing
    /// off on grounded warm-started assemblies until the L3 escalation
    /// removes the burst.
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
/// TWO EXECUTION MODES, one free-space fixed point: the pre-pass
/// `apply_implicit` below (the executable spec — runs when
/// `XpbdWorld::coupled_joint_solve` is false, the one-flag rollback)
/// solves the joint once per substep before the SI loop; the in-loop
/// lifecycle (`prepare`/`solve_iteration`/`finalize`, flag true — the
/// default since Commit B) runs the SAME
/// backward-Euler solve as an iterated soft constraint inside the SI loop,
/// so joint and contact impulses couple within each sweep. In free space
/// the two are algebraically identical (the in-loop form lands on the
/// pre-pass fixed point on iteration 1); they differ exactly where the
/// pre-pass is wrong — under contact, where the pre-pass books velocity
/// feedback against motion the contacts then annihilate (the measured
/// 1/1880 force-transmission gap). See the design contract:
/// docs/superpowers/specs/2026-10-21-coupled-joint-contact-solve.md.
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

    // ── In-loop coupled-solve state (XpbdWorld::coupled_joint_solve) ──
    /// Accumulated ACTIVE impulse (N·m·s) this substep, built up by
    /// `solve_iteration` inside the SI loop. The torque budget lives here:
    /// |lambda_a| ≤ dt·torque_limit, enforced on the ACCUMULATED value, so
    /// sustained saturated output is exactly ±torque_limit — dt-independent
    /// (the impulse cap scales with dt, the torque cap does not).
    /// WARM-STARTED across substeps (the design's L2 lever): `prepare`
    /// clamps the carried value to the current ±cap and keeps it seeded;
    /// `apply_warm_start` puts the impulse on the bodies at the
    /// contact-reset slot (after the legacy implicit passes — see its
    /// doc), so the cross-substep fixed point under a blocking contact is
    /// the full static demand Λ* = dt·τ₀ for ANY joint↔contact alignment
    /// (design §2.6). In free space the first sweep's solve is independent
    /// of the seed (λ₁ = −(ω+b)/(w+γ), design §2.4), so a stale impulse
    /// unwinds exactly when contact vanishes.
    pub lambda_a: f64,
    /// Accumulated PASSIVE impulse (N·m·s) this substep. Never clamped —
    /// the muscle saturates, the ligaments don't. Warm-started like
    /// `lambda_a` (without the clamp).
    pub lambda_p: f64,
    /// Per-substep cache built by `prepare`. Positions are frozen during
    /// the SI loop, so everything except ω is constant for the substep.
    cache: SolveCache,
}

/// One row of the in-loop 2×2 active/passive mini-block, classified once
/// per substep by `prepare` from the row's stiffness content d + dt·k.
#[derive(Clone, Copy)]
enum Row {
    /// Nothing on this row (active disabled / no passive structures and no
    /// ext injection): contributes nothing to the block.
    Absent,
    /// Finite-γ soft constraint: residual ω⁺ + γ·λ + b = 0 with
    /// γ = 1/(dt·(d + dt·k)) and b = −γ·dt·τ₀ — the exact constraint-space
    /// re-parameterization of the backward-Euler row (its fixed point IS
    /// `apply_implicit`'s impulse; design §2.1).
    Soft { gamma: f64, bias: f64 },
    /// d + dt·k admits no finite γ (pure feedforward / pure ext_torque):
    /// the γ→∞ limit is a CONSTANT impulse dt·τ₀, booked once on the first
    /// iteration and excluded from every later solve — "constant torque
    /// rides outside the denominator".
    Constant { impulse: f64 },
}

/// Per-substep cache for the in-loop solve. Caching the axis, the inertia
/// projections and the row coefficients once per substep (instead of per
/// iteration) is what keeps a block iteration at ~30 flops — no
/// `world_inv_inertia` and no quaternion swing-twist inside the SI loop.
struct SolveCache {
    /// World-frame joint axis n = q_a·axis_local_a at θₙ.
    n: Vec3,
    /// Joint angle at the linearization point (positions are frozen in the
    /// loop, so the linearization is exact for the whole substep).
    #[allow(dead_code)] // diagnostic; the biases already encode it
    theta_n: f64,
    /// Cached I_a⁻¹·n / I_b⁻¹·n (zero for pinned bodies): one body write
    /// per iteration instead of two inertia solves.
    i_inv_a_n: Vec3,
    i_inv_b_n: Vec3,
    /// Reduced inverse inertia along the axis.
    w: f64,
    active: Row,
    passive: Row,
    /// Impulse cap dt·torque_limit on the ACCUMULATED active impulse.
    cap: f64,
    /// Both bodies pinned, or both rows absent: iterations are no-ops.
    skip: bool,
    /// Constant (γ→∞) rows still awaiting their one-time application.
    constants_pending: bool,
    /// Warm-start impulse computed (and clamped) by `prepare`, waiting to
    /// be applied by `apply_warm_start` at the contact-reset slot — AFTER
    /// the legacy implicit passes, exactly where the manifold seeds apply.
    /// Applying it in `prepare` lets a legacy damper/PD on the same or an
    /// adjacent body read the seed velocity as real motion and rectify it
    /// (see `apply_warm_start`).
    pending_warm: f64,
    /// Saturation state of the most recent applied solve, read by
    /// `finalize`: when the loop ENDS clamped, the active diagnostic is
    /// σ·torque_limit assigned directly — dividing the clamped impulse by
    /// dt is not guaranteed bit-exact in f64, and the contract (T3) is bit
    /// equality with ±torque_limit.
    clamped: bool,
    sigma: f64,
}

impl SolveCache {
    /// The state of a joint whose `prepare` has not run: every
    /// `solve_iteration` and `finalize` is a no-op on zeros.
    fn inert() -> Self {
        Self {
            n: Vec3::zeros(),
            theta_n: 0.0,
            i_inv_a_n: Vec3::zeros(),
            i_inv_b_n: Vec3::zeros(),
            w: 0.0,
            active: Row::Absent,
            passive: Row::Absent,
            cap: f64::INFINITY,
            skip: true,
            constants_pending: false,
            pending_warm: 0.0,
            clamped: false,
            sigma: 0.0,
        }
    }
}

/// Axis kinematics shared by the pre-pass solve and `prepare`.
struct AxisState {
    n: Vec3,
    theta: f64,
    i_inv_a_n: Vec3,
    i_inv_b_n: Vec3,
    w: f64,
}

/// The coaxial torques and local stiffness/damping content at θₙ — ONE
/// definition shared by `apply_implicit` and `prepare`, so the two
/// execution modes cannot drift semantically.
struct CoaxialTerms {
    tau_act0: f64,
    tau_pas0: f64,
    k_act: f64,
    k_pas: f64,
    d_act: f64,
    d_pas: f64,
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
            lambda_a: 0.0,
            lambda_p: 0.0,
            cache: SolveCache::inert(),
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
    /// `with_soft_limit`. WARNING: a contact-release burst under carried
    /// warm seeds (a documented one-substep transient up to ~3e5 rad/s at
    /// toe scale) would permanently false-trigger a finite threshold on a
    /// grounded assembly — see the `SoftLimit::tear_threshold` doc before
    /// choosing one.
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
        let (ext_torque, ext_stiffness, ext_damping) = self.consume_ext();
        self.applied_active_torque = 0.0;
        self.applied_total_torque = 0.0;

        let st = self.axis_state(bodies);

        // 1. Tear latch first, from θₙ — the failure mode fires on the same
        // substep the threshold is crossed, exactly like AngularSoftLimit.
        if let Some(lim) = &mut self.limit {
            lim.update_torn(st.theta);
        }

        // 2. Shared reduced-inertia state (see `axis_state`).
        if st.w < 1e-15 {
            return; // both bodies pinned
        }

        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega = omega_rel.dot(&st.n);

        // 3. Torques and local stiffnesses at the linearization point θₙ
        // (shared with `prepare` — see `coaxial_terms`).
        let t = self.coaxial_terms(st.theta, ext_torque, ext_stiffness, ext_damping);

        // 4. Unsaturated combined solve — ONE shared denominator (≥ 1, the
        // A-stability certificate: every k and d above is ≥ 0).
        let mut omega_new = (omega + dt * st.w * (t.tau_act0 + t.tau_pas0))
            / (1.0 + dt * st.w * (t.d_act + t.d_pas) + dt * dt * st.w * (t.k_act + t.k_pas));

        // 5. Saturation check on the ACTIVE output at the end-of-substep
        // state (θ⁺ = θ + dt·ω⁺).
        if self.enabled {
            let tau_act_plus = self.kp * (self.target_angle - st.theta - dt * omega_new)
                + self.feedforward_torque
                - self.kd * omega_new;
            if self.torque_limit.is_finite() && tau_act_plus.abs() > self.torque_limit {
                let sigma = tau_act_plus.signum();
                omega_new = (omega + dt * st.w * (sigma * self.torque_limit + t.tau_pas0))
                    / (1.0 + dt * st.w * t.d_pas + dt * dt * st.w * t.k_pas);
                self.applied_active_torque = sigma * self.torque_limit;
            } else {
                self.applied_active_torque = tau_act_plus;
            }
        }

        // 6. Apply the single combined impulse.
        let j = (omega_new - omega) / st.w;
        self.applied_total_torque = j / dt;

        bodies[self.body_a].angular_velocity -= st.i_inv_a_n * j;
        bodies[self.body_b].angular_velocity += st.i_inv_b_n * j;
    }

    /// Consume the per-substep external injections — they are per-substep
    /// quantities, taken whether or not the solve early-outs — and enforce
    /// their sign invariant. Shared by `apply_implicit` and `prepare`.
    fn consume_ext(&mut self) -> (f64, f64, f64) {
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
        (ext_torque, ext_stiffness, ext_damping)
    }

    /// Axis kinematics shared by `apply_implicit` and `prepare`: the world
    /// axis n, the angle θₙ, the inertia projections I⁻¹·n — zero for
    /// pinned bodies (post-f11d73f convention of all primitives) — and the
    /// reduced inverse inertia w along the axis.
    fn axis_state(&self, bodies: &[RigidBody]) -> AxisState {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let theta = self.angle(bodies);
        let zero = Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let i_inv_a_n = i_inv_a * n;
        let i_inv_b_n = i_inv_b * n;
        let w = n.dot(&i_inv_a_n) + n.dot(&i_inv_b_n);
        AxisState { n, theta, i_inv_a_n, i_inv_b_n, w }
    }

    /// Torques and local stiffness/damping content at the linearization
    /// point θₙ. Shared by `apply_implicit` and `prepare` so the two
    /// execution modes read the identical physics.
    fn coaxial_terms(
        &self,
        theta: f64,
        ext_torque: f64,
        ext_stiffness: f64,
        ext_damping: f64,
    ) -> CoaxialTerms {
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
        CoaxialTerms {
            tau_act0,
            tau_pas0,
            k_act: if active { self.kp } else { 0.0 },
            k_pas: self.passive_stiffness + k_lim + ext_stiffness,
            d_act: if active { self.kd } else { 0.0 },
            d_pas: self.passive_damping + ext_damping,
        }
    }

    // ─── In-loop coupled solve (XpbdWorld::coupled_joint_solve) ─────────
    //
    // The pre-pass above models ITS OWN impulse as the sole determinant of
    // the end-of-substep velocity. When the SI contact loop afterwards
    // annihilates that velocity (a wedged toe), the booked velocity-
    // feedback terms were charged against motion that never happened and
    // the transmitted torque collapses to τ₀/D — the measured 1/1880
    // force-transmission gap (design contract §1). The in-loop form
    // re-parameterizes the SAME backward-Euler solve as an iterated soft
    // constraint inside the SI Gauss-Seidel loop (§2.1), split into an
    // active/passive 2×2 mini-block so the torque budget clamps the
    // ACCUMULATED active impulse (§2.2). In free space, iteration 1 lands
    // exactly on the pre-pass fixed point (GS is exact in one step for an
    // isolated linear constraint) and the convergence skip makes later
    // iterations no-ops; under contact, the block and the contacts
    // exchange impulses within each sweep and the coupled fixed point
    // transmits the full static demand, dt-independently (§2.5).

    /// In-loop step 1/3: once per substep, read-only on the bodies.
    /// Consumes the ext_* injections and runs the tear latch at θₙ — the
    /// identical per-substep semantics as `apply_implicit`, through the
    /// same shared helpers — then caches everything the iterations need
    /// and SEEDS the lambdas (design §2.6 L2, amended 2026-10-21): the
    /// carry criterion is the row's CURRENT class. A row classified Soft
    /// keeps whatever impulse its lambda accumulated last substep (the
    /// active one clamped to the current ±cap) — including an impulse a
    /// then-Constant row booked, if the row changed class to Soft, which
    /// is benign because the first sweep's soft solve is seed-independent
    /// (λ₁ = −(ω+b)/(w+γ), §2.4) and simply re-lands on the fixed point. A
    /// row that is NOW Constant or Absent carries nothing: a Constant row
    /// books its exact impulse dt·τ₀ on iteration 1, so a carried value on
    /// top would double-book.
    ///
    /// The seed impulse is NOT applied here: `prepare` runs before the
    /// legacy implicit passes, and a seed applied that early is read by
    /// legacy dampers/PDs as real motion and rectified (the same channel
    /// the manifold seed had — see `apply_warm_start` and
    /// `XpbdWorld::substep`). It is computed, clamped, and parked in the
    /// cache; `apply_warm_start` puts it on the bodies at the contact-reset
    /// slot. The seed is exactly the impulse the previous substep
    /// ACCUMULATED, so under a blocking contact the pair (joint seed,
    /// manifold seed) relaxes to the true coupled statics across substeps
    /// even where the per-sweep Gauss-Seidel contraction is near-degenerate
    /// (§2.6: the fixed point Λ* = dt·τ₀ is exact for any alignment); in
    /// free space iteration 1's solve is seed-independent and unwinds it
    /// exactly (§2.4).
    pub fn prepare(&mut self, bodies: &[RigidBody], dt: f64) {
        let (ext_torque, ext_stiffness, ext_damping) = self.consume_ext();
        let (prev_la, prev_lp) = (self.lambda_a, self.lambda_p);
        self.lambda_a = 0.0;
        self.lambda_p = 0.0;

        let st = self.axis_state(bodies);

        // Tear latch at θₙ, on the same substep the threshold is crossed —
        // and, like the ext_* consumption above, it runs even when both
        // bodies are pinned (same order as `apply_implicit`).
        if let Some(lim) = &mut self.limit {
            lim.update_torn(st.theta);
        }

        let mut cache = SolveCache {
            n: st.n,
            theta_n: st.theta,
            i_inv_a_n: st.i_inv_a_n,
            i_inv_b_n: st.i_inv_b_n,
            w: st.w,
            active: Row::Absent,
            passive: Row::Absent,
            // The budget clamps the ACCUMULATED impulse: the cap scales
            // with dt so the sustained torque cap does not.
            cap: dt * self.torque_limit,
            skip: true,
            constants_pending: false,
            pending_warm: 0.0,
            clamped: false,
            sigma: 0.0,
        };

        if st.w < 1e-15 {
            // Both bodies pinned: iterations skip, ext_* already consumed.
            self.cache = cache;
            return;
        }

        let t = self.coaxial_terms(st.theta, ext_torque, ext_stiffness, ext_damping);

        // Row classification. γ = 1/(dt·(d + dt·k)) is the exact
        // constraint-space form of the row's backward-Euler solve (§2.1).
        // A row whose γ has no finite value (d + dt·k = 0, or small enough
        // that the division overflows) is the γ→∞ limit: a constant
        // impulse dt·τ₀ — or nothing at all when τ₀ is also zero.
        //
        // GAMMA_MAX guards the overflow band BELOW the is_finite() cut:
        // for stiffness content in ~1e-305..1e-150, γ itself is still
        // finite but its downstream products are not — γ_a·γ_p in the 2×2
        // determinant, (w+γ)·r in the block numerators (r carries the
        // O(γ·dt·τ₀) bias), and the bias itself at large τ₀ — and an
        // inf/NaN there poisons the whole solve (measured: NaN velocities
        // on a two-row config at d = 1e-280 each) while the pre-pass
        // D-form, whose denominator is 1 + dt·w·(d + dt·k) ≈ 1, stays
        // exact. Any γ > 1e100 is therefore classified as the γ→∞
        // constant-impulse limit instead. The guard's working range is
        // finite, not universal, but sits ~100 orders past any physical
        // state in both directions: the surviving products (γ_a·γ_p ≤
        // 1e200, w·γ ≤ ~1e115 at any physical inertia) stay finite, while
        // the 2×2 numerator (w+γ)·r — whose residual carries the
        // O(γ·dt·τ₀) bias — overflows first, at |τ₀| ≳ 1.3e112 with both
        // surviving rows near GAMMA_MAX (self-healing via the clamp only
        // when torque_limit is finite); and the classification error (the
        // dropped soft term, stiffness content d + dt·k < 1/(dt·1e100) ≈
        // 7e-93 N·m·s/rad) stays below one ulp of the constant impulse
        // dt·τ₀ for |ω| up to ~1e75 rad/s, growing only linearly in ω
        // beyond. Within everything a simulation can physically reach,
        // the degraded row is exact to roundoff.
        const GAMMA_MAX: f64 = 1e100;
        let stiff_a = t.d_act + dt * t.k_act;
        let gamma_a = 1.0 / (dt * stiff_a);
        cache.active = if gamma_a.is_finite() && gamma_a <= GAMMA_MAX {
            Row::Soft { gamma: gamma_a, bias: -gamma_a * dt * t.tau_act0 }
        } else if t.tau_act0 != 0.0 {
            // Pure feedforward. The budget still applies: a constant
            // demand past the cap clamps at classification time (σ is the
            // demand's own sign — a constant torque has no velocity
            // feedback to flip it), and `finalize` reports σ·torque_limit.
            let mut impulse = dt * t.tau_act0;
            if self.torque_limit.is_finite() && t.tau_act0.abs() > self.torque_limit {
                cache.sigma = t.tau_act0.signum();
                cache.clamped = true;
                impulse = cache.sigma * cache.cap;
            }
            Row::Constant { impulse }
        } else {
            Row::Absent
        };

        let stiff_p = t.d_pas + dt * t.k_pas;
        let gamma_p = 1.0 / (dt * stiff_p);
        cache.passive = if gamma_p.is_finite() && gamma_p <= GAMMA_MAX {
            Row::Soft { gamma: gamma_p, bias: -gamma_p * dt * t.tau_pas0 }
        } else if t.tau_pas0 != 0.0 {
            // Pure ext_torque with no passive structures: constant and
            // unclamped — neither ligaments nor an injected tendon torque
            // saturate on this joint's budget.
            Row::Constant { impulse: dt * t.tau_pas0 }
        } else {
            Row::Absent
        };

        cache.skip = matches!(cache.active, Row::Absent)
            && matches!(cache.passive, Row::Absent);
        cache.constants_pending = matches!(cache.active, Row::Constant { .. })
            || matches!(cache.passive, Row::Constant { .. });

        // L2 warm start: seed each CURRENTLY-Soft row with its lambda's
        // accumulated impulse from last substep (whatever booked it — a
        // row that changed class Constant→Soft carries the booked constant
        // impulse, benign by the seed-independence of the first sweep's
        // solve; see the method doc). Rows that are NOW Constant or Absent
        // carry nothing (Constant re-books exactly on iteration 1). The
        // active seed is clamped to the CURRENT ±cap, so a change of
        // torque_limit or dt can never smuggle a stale over-budget impulse
        // past the clamp. The combined impulse is parked for
        // `apply_warm_start` — never applied here, where the legacy
        // implicit passes would read it as real motion.
        if let Row::Soft { .. } = cache.active {
            self.lambda_a = prev_la.clamp(-cache.cap, cache.cap);
        }
        if let Row::Soft { .. } = cache.passive {
            self.lambda_p = prev_lp;
        }
        cache.pending_warm = self.lambda_a + self.lambda_p;
        self.cache = cache;
    }

    /// In-loop step 1.5/3: apply the warm-start impulse `prepare` parked —
    /// at the contact-reset slot in `XpbdWorld::substep`, AFTER the legacy
    /// implicit passes and immediately before the SI sweeps, the same slot
    /// where the manifold seeds apply. The seed velocity then exists only
    /// inside the loop that solves incrementally from the seeded lambdas,
    /// which is all the cross-substep fixed point needs. Applied any
    /// earlier, a legacy damper/PD acting on the same or an adjacent body
    /// reads the seed velocity (w·dt·τ₀ — ~1.3e3 rad/s at toe scale under
    /// a 2.4 N·m demand) as real motion and rectifies it every substep:
    /// measured on the mixed-primitive canaries as a stolen ground
    /// reaction (0.21 N arriving of a 101 N demand, same-pair damper into
    /// a pinned parent) and a sustained 151/1441 rad/s limit cycle with a
    /// 5.2e-2 J energy pump (legacy stack on an adjacent free body) —
    /// crates/physics/tests/coupled_joint.rs.
    pub fn apply_warm_start(&mut self, bodies: &mut Vec<RigidBody>) {
        let j = self.cache.pending_warm;
        if j != 0.0 {
            bodies[self.body_a].angular_velocity -= self.cache.i_inv_a_n * j;
            bodies[self.body_b].angular_velocity += self.cache.i_inv_b_n * j;
            self.cache.pending_warm = 0.0;
        }
    }

    /// In-loop step 2/3: one Gauss-Seidel iteration of the joint's
    /// active/passive mini-block. Called from `solve_constraints` after
    /// the `AngularLimitConstraint`s and before the contacts (contacts
    /// keep the last word on non-penetration within each sweep). Reads the
    /// relative velocity fresh from the bodies: it now includes every
    /// contact impulse applied so far this sweep — which is the point.
    ///
    /// The two rows are solved as a DIRECT 2×2 block, never as two
    /// independent GS rows: two near-hard constraints on the identical
    /// axis Gauss-Seidel against each other at ρ = w²/((w+γ_a)(w+γ_p)) ≈ 1
    /// — the SPLIT converges glacially even though the total converges
    /// fast, and the clamp and the diagnostics read the split (§2.2).
    /// det = w·(γ_a + γ_p) + γ_a·γ_p > 0 is the translated A-stability
    /// certificate: every k and d ≥ 0 ⇒ γ > 0 ⇒ det > 0.
    pub fn solve_iteration(&mut self, bodies: &mut Vec<RigidBody>, _dt: f64) {
        if self.cache.skip {
            return;
        }
        let w = self.cache.w;
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let u = omega_rel.dot(&self.cache.n);

        // γ→∞ rows: book the constant impulse on the first iteration only,
        // folded into the velocity the soft rows read below so the whole
        // iteration stays ONE body write.
        let mut j_const = 0.0;
        if self.cache.constants_pending {
            if let Row::Constant { impulse } = self.cache.active {
                self.lambda_a += impulse;
                j_const += impulse;
            }
            if let Row::Constant { impulse } = self.cache.passive {
                self.lambda_p += impulse;
                j_const += impulse;
            }
            self.cache.constants_pending = false;
        }
        let u_eff = u + w * j_const;

        // Convergence skip (§2.3): below 1e-13 RELATIVE to the accumulated
        // impulses, the deltas are roundoff, not physics — skipping them
        // makes free-space iterations 2..n true no-ops (bit-stability for
        // the trajectory equivalences) and cannot mask real coupling,
        // which sits far above 1e-13 relative. The additive tiny keeps the
        // test meaningful at λ = 0 (an exactly-zero delta still skips).
        let converged = |d: f64, lambda_a: f64, lambda_p: f64| -> bool {
            d <= 1e-13 * (lambda_a.abs() + lambda_p.abs() + 1e-300)
        };

        let mut j_soft = 0.0;
        match (self.cache.active, self.cache.passive) {
            (Row::Soft { gamma: ga, bias: ba }, Row::Soft { gamma: gp, bias: bp }) => {
                let r_a = -(u_eff + ga * self.lambda_a + ba);
                let r_p = -(u_eff + gp * self.lambda_p + bp);
                let det = w * (ga + gp) + ga * gp;
                let d_a_unc = ((w + gp) * r_a - w * r_p) / det;
                // The clamp (the torque budget): on the ACCUMULATED active
                // impulse, with σ from the UNCLAMPED end-state demand —
                // λ_a + Δ_a IS dt·τ_act⁺, so this preserves the pre-pass's
                // σ-sign-flip semantics. After clamping, the passive row
                // re-solves alone against the clamped active delta: the
                // muscle saturates, the ligaments don't.
                let la_unc = self.lambda_a + d_a_unc;
                let (d_a, d_p, clamped, sigma);
                if la_unc.abs() > self.cache.cap {
                    sigma = la_unc.signum();
                    d_a = sigma * self.cache.cap - self.lambda_a;
                    d_p = -(u_eff + w * d_a + gp * self.lambda_p + bp) / (w + gp);
                    clamped = true;
                } else {
                    d_a = d_a_unc;
                    d_p = ((w + ga) * r_p - w * r_a) / det;
                    clamped = false;
                    sigma = 0.0;
                }
                if !converged(d_a.abs() + d_p.abs(), self.lambda_a, self.lambda_p) {
                    self.lambda_a += d_a;
                    self.lambda_p += d_p;
                    self.cache.clamped = clamped;
                    self.cache.sigma = sigma;
                    j_soft = d_a + d_p;
                }
            }
            (Row::Soft { gamma: ga, bias: ba }, _) => {
                // 1-D active with the clamp (passive absent, or a constant
                // already booked above — nothing left to re-solve).
                let d_unc = -(u_eff + ga * self.lambda_a + ba) / (w + ga);
                let la_unc = self.lambda_a + d_unc;
                let (d_a, clamped, sigma) = if la_unc.abs() > self.cache.cap {
                    let s = la_unc.signum();
                    (s * self.cache.cap - self.lambda_a, true, s)
                } else {
                    (d_unc, false, 0.0)
                };
                if !converged(d_a.abs(), self.lambda_a, self.lambda_p) {
                    self.lambda_a += d_a;
                    self.cache.clamped = clamped;
                    self.cache.sigma = sigma;
                    j_soft = d_a;
                }
            }
            (_, Row::Soft { gamma: gp, bias: bp }) => {
                // 1-D passive (active absent or constant). Never clamped.
                let d_p = -(u_eff + gp * self.lambda_p + bp) / (w + gp);
                if !converged(d_p.abs(), self.lambda_a, self.lambda_p) {
                    self.lambda_p += d_p;
                    j_soft = d_p;
                }
            }
            // Constant-only (or empty) block: nothing soft to iterate.
            _ => {}
        }

        let j = j_const + j_soft;
        if j != 0.0 {
            bodies[self.body_a].angular_velocity -= self.cache.i_inv_a_n * j;
            bodies[self.body_b].angular_velocity += self.cache.i_inv_b_n * j;
        }
    }

    /// In-loop step 3/3: after the SI loop, before `integrate_positions` —
    /// write the diagnostics from the accumulated impulses. Under the flag
    /// these report the REALIZED transmitted torque including contact
    /// coupling (what the design's receipt experiments read), not the
    /// free-space booking.
    pub fn finalize(&mut self, dt: f64) {
        self.applied_total_torque = (self.lambda_a + self.lambda_p) / dt;
        self.applied_active_torque = if self.cache.clamped {
            // Assigned DIRECTLY, never λ_a/dt: the saturation diagnostic's
            // contract is bit equality with ±torque_limit (T3 asserts it),
            // and (σ·dt·τ_max)/dt is not guaranteed bit-exact in f64.
            self.cache.sigma * self.torque_limit
        } else if self.lambda_a != 0.0 && self.lambda_a.abs() == self.cache.cap {
            // Saturated steady state reached through the convergence skip:
            // the warm seed re-enters at exactly ±cap, the clamped branch's
            // delta is exactly 0, and the skip never writes clamped/sigma —
            // but λ_a sitting bit-exactly at the cap IS saturation, so the
            // report is derived from the lambda itself (the contract's bit
            // equality; λ_a/dt would be up to 1 ulp off). An infinite
            // torque_limit can never match (λ_a is finite), and cap = 0
            // is excluded by λ_a ≠ 0 (the λ_a/dt branch reports 0 exactly).
            self.lambda_a.signum() * self.torque_limit
        } else {
            self.lambda_a / dt
        };
    }
}
