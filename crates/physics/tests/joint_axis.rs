//! Stage 1 of the combined coaxial implicit solve: the `JointAxis` primitive.
//!
//! The legacy engine integrates each coaxial angular term (PD kp, passive
//! spring, soft limit, dampers) in its own sequential backward-Euler pass,
//! and every pass divides its own contribution by its own `1 + dt²·w·k`
//! (docs/findings.md 2026-10): effective gain kp/(1 + dt²·w·kp), a
//! dt-DEPENDENT equilibrium, and a load-bearing kp-before-dampers ordering.
//! `JointAxis` solves every coaxial term in ONE backward-Euler solve with a
//! shared denominator. These tests pin down:
//!   T1 — dt-independence of the combined equilibrium (and the legacy bias,
//!        bite-verified in the same test);
//!   T2 — exact reduction to each legacy primitive in single-term configs;
//!   T3 — torque_limit saturation semantics (active clamped, passive not);
//!   T4 — soft-limit crossing: energy, tear parity, and the airborne-drift
//!        verdict on the full hallux-ray stack;
//!   T5 — worst-case stiffness/inertia stress stays finite and settles.

use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_pd_controller::AngularPDController;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_spring::AngularSpring;
use physics::constraints::PositionalConstraint;
use physics::joint_axis::JointAxis;
use physics::math::{Mat3, Quat, Vec3, swing_twist_angle};
use physics::rigid_body::RigidBody;
use physics::xpbd::XpbdWorld;

/// The engine's standard substep rate (dt = 1/120, 60 substeps) and the
/// half-rate variant used to expose dt-dependence.
const SUB_DT_FINE: f64 = 1.0 / 7200.0;
const SUB_DT_COARSE: f64 = 1.0 / 3600.0;

fn run_substeps(world: &mut XpbdWorld, sub_dt: f64, n: usize) {
    for _ in 0..n {
        world.substep(sub_dt);
    }
}

fn joint_angle(world: &XpbdWorld, a: usize, b: usize, axis: Vec3) -> f64 {
    let q_rel = world.bodies[a].orientation.inverse() * world.bodies[b].orientation;
    swing_twist_angle(q_rel, axis)
}

/// Pinned parent + free small child at the same position (pure rotational
/// joint about `axis`; no positional constraint — with no ground the child's
/// free fall has zero effect on relative orientation dynamics, and the empty
/// constraint set keeps the two compared worlds bit-identical elsewhere).
fn pinned_pair(world: &mut XpbdWorld, child_mass: f64, child_inertia: f64) -> (usize, usize) {
    world.no_ground();
    let parent = RigidBody::new(0.0, Mat3::identity());
    let parent_idx = world.add_body(parent);
    let child = RigidBody::new(child_mass, Mat3::identity() * child_inertia);
    let child_idx = world.add_body(child);
    (parent_idx, child_idx)
}

// ════════════════════════════════════════════════════════════════════════
// T1 — dt-independence of the combined equilibrium (the headline)
// ════════════════════════════════════════════════════════════════════════

/// Full coaxial stack (PD kp = 4 / kd = 0.15, passive spring K = 4 rest 0,
/// passive damper c = 1.0, target 0.8) on a pinned-parent/small-child joint
/// (I = 1e-8 — gram-scale toe territory, where the legacy split bias is
/// ~90%+). Static equilibrium of the real physics: kp·(t − θ) = K·θ ⇒
/// θ* = kp·t/(kp + K) = 0.400, independent of dt.
///
/// The combined solve must land there at BOTH substep rates, because its
/// equilibrium condition is τ_act0 + τ_pas0 = 0 — the shared denominator
/// scales the approach, never the fixed point.
///
/// BITE built in: the identical world built from the raw legacy primitives
/// (sequential kp pass + spring pass + damper passes) must land FAR from
/// 0.400 AND at a DIFFERENT value at the two rates — each sequential pass
/// attenuates its own term by 1/(1 + dt²·w·k), so the split equilibrium is
/// both wrong and dt-dependent.
///
/// MEASURED (this test's build):
///   combined: 0.4000000 at 1/7200, 0.4000000 at 1/3600 (both exactly on
///             the analytic 0.400)
///   legacy:   0.0001212 at 1/7200, 0.0000045 at 1/3600
/// The legacy bias is far MORE severe than the design note's ≈0.038
/// stationary estimate: at I = 1e-8 the shared damper is itself attenuated
/// into a per-substep velocity annihilator (dt·w·c ≈ 1.4e4), which
/// damper-rectifies the already ~90%-attenuated kp impulse to ~0.03% command
/// authority — the same rectification mechanism e9dda0b measured on the
/// pinned hallux, here taken to its small-inertia extreme. Because both
/// legacy values are near zero, "lands at a DIFFERENT value at 1/3600" is
/// asserted as a RATIO (measured 26.7×), not an absolute gap.
#[test]
fn t1_combined_equilibrium_is_dt_independent_legacy_is_not() {
    let kp = 4.0;
    let kd = 0.15;
    let k_sp = 4.0;
    let c = 1.0;
    let target = 0.8;
    let analytic = kp * target / (kp + k_sp); // 0.400
    let axis = Vec3::z();

    let settle = |sub_dt: f64, combined: bool| -> f64 {
        let mut world = XpbdWorld::new();
        let (p, ch) = pinned_pair(&mut world, 0.05, 1e-8);
        if combined {
            world.joint_axes.push(
                JointAxis::new(p, ch, axis)
                    .with_pd(kp, kd)
                    .with_target(target)
                    .with_passive_spring(k_sp, 0.0)
                    .with_passive_damping(c),
            );
        } else {
            world.angular_pd_controllers.push(
                AngularPDController::new(p, ch, axis, kp, kd).with_target(target),
            );
            world.angular_springs.push(AngularSpring::new(p, ch, axis, k_sp, 0.0));
            world.angular_dampers.push(AngularDamper::new(p, ch, axis, c));
        }
        let n = (6.0 / sub_dt).round() as usize; // 6 s — far past settling
        run_substeps(&mut world, sub_dt, n);
        joint_angle(&world, p, ch, axis)
    };

    let comb_fine = settle(SUB_DT_FINE, true);
    let comb_coarse = settle(SUB_DT_COARSE, true);
    let legacy_fine = settle(SUB_DT_FINE, false);
    let legacy_coarse = settle(SUB_DT_COARSE, false);

    println!("T1 combined: {comb_fine:.7} @1/7200, {comb_coarse:.7} @1/3600 (analytic {analytic:.3})");
    println!("T1 legacy:   {legacy_fine:.7} @1/7200, {legacy_coarse:.7} @1/3600");

    // Combined: the analytic equilibrium at BOTH rates.
    assert!(
        (comb_fine - analytic).abs() < 0.005,
        "combined @1/7200 settled at {comb_fine} rad, analytic {analytic} (±0.005)"
    );
    assert!(
        (comb_coarse - analytic).abs() < 0.005,
        "combined @1/3600 settled at {comb_coarse} rad, analytic {analytic} (±0.005)"
    );

    // Bite: legacy is FAR from the analytic value (split-pass gain
    // attenuation ~90%+ at this inertia)...
    assert!(
        (legacy_fine - analytic).abs() > 0.1,
        "legacy @1/7200 settled at {legacy_fine} rad — unexpectedly NEAR the analytic \
         {analytic}; the split bias this primitive exists to remove has vanished"
    );
    // ...and dt-DEPENDENT (the bias scales with dt²·w·k and dt·w·c; both
    // settles are near zero, so the dependence shows as a large ratio —
    // measured 26.7× across the 2× substep change).
    assert!(
        legacy_fine > 5.0 * legacy_coarse,
        "legacy settled at {legacy_fine} vs {legacy_coarse} across a 2× substep change — \
         unexpectedly dt-independent"
    );
}

// ════════════════════════════════════════════════════════════════════════
// T2 — single-term configs reduce exactly to the legacy primitives
// ════════════════════════════════════════════════════════════════════════

/// Run a JointAxis world and a legacy-primitive world in lockstep for
/// `n` substeps; return (max |Δθ|, max |Δω_child|) across the run.
fn lockstep_divergence(
    mut ja_world: XpbdWorld,
    mut legacy_world: XpbdWorld,
    p: usize,
    ch: usize,
    axis: Vec3,
    sub_dt: f64,
    n: usize,
) -> (f64, f64) {
    let mut max_dtheta: f64 = 0.0;
    let mut max_domega: f64 = 0.0;
    for _ in 0..n {
        ja_world.substep(sub_dt);
        legacy_world.substep(sub_dt);
        let da = (joint_angle(&ja_world, p, ch, axis) - joint_angle(&legacy_world, p, ch, axis)).abs();
        let dw = (ja_world.bodies[ch].angular_velocity - legacy_world.bodies[ch].angular_velocity).norm();
        if da > max_dtheta { max_dtheta = da; }
        if dw > max_domega { max_domega = dw; }
    }
    (max_dtheta, max_domega)
}

/// Spring-only, damper-only, soft-limit-only (incl. tear) and kp-only
/// (unlimited and deep-saturated) configurations: the combined solve's
/// formulas reduce to the legacy primitives' EXACTLY (same numerator, same
/// denominator), so the trajectories must agree to floating-point
/// association noise.
///
/// MEASURED over 600 substeps: max|Δθ| = 0.0 (bit-exact) for ALL five
/// sub-cases; the only nonzero divergence anywhere is damper-only
/// max|Δω| = 2.0e-17 (one ulp — the legacy damper skips its solve when
/// |ω_rel| < 1e-15, the combined solve applies its exact-zero impulse).
/// The 1e-12 bounds are kept as generous ceilings over bit-exact reality.
#[test]
fn t2_single_term_configs_match_legacy_primitives() {
    let axis = Vec3::z();
    let sub_dt = SUB_DT_FINE;
    let n = 600;

    // ── spring-only ──
    let mut ja_w = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut ja_w, 0.05, 1e-6);
    ja_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 2.0);
    let mut lg_w = XpbdWorld::new();
    pinned_pair(&mut lg_w, 0.05, 1e-6);
    lg_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 2.0);
    ja_w.joint_axes.push(JointAxis::new(p, ch, axis).with_passive_spring(4.0, 0.3));
    lg_w.angular_springs.push(AngularSpring::new(p, ch, axis, 4.0, 0.3));
    let (dth, dw) = lockstep_divergence(ja_w, lg_w, p, ch, axis, sub_dt, n);
    println!("T2 spring-only: max|Δθ| {dth:.3e}, max|Δω| {dw:.3e}");
    assert!(dth < 1e-12, "spring-only diverged from AngularSpring: max|Δθ| = {dth:e}");

    // ── damper-only ──
    let mut ja_w = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut ja_w, 0.05, 1e-6);
    ja_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 3.0);
    let mut lg_w = XpbdWorld::new();
    pinned_pair(&mut lg_w, 0.05, 1e-6);
    lg_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 3.0);
    ja_w.joint_axes.push(JointAxis::new(p, ch, axis).with_passive_damping(1.0));
    lg_w.angular_dampers.push(AngularDamper::new(p, ch, axis, 1.0));
    let (dth, dw) = lockstep_divergence(ja_w, lg_w, p, ch, axis, sub_dt, n);
    println!("T2 damper-only: max|Δθ| {dth:.3e}, max|Δω| {dw:.3e}");
    assert!(dth < 1e-12, "damper-only diverged from AngularDamper: max|Δθ| = {dth:e}");

    // ── kp-only, torque_limit infinite (bit-exact reduction expected) ──
    let mut ja_w = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut ja_w, 0.05, 1e-6);
    let mut lg_w = XpbdWorld::new();
    pinned_pair(&mut lg_w, 0.05, 1e-6);
    ja_w.joint_axes.push(JointAxis::new(p, ch, axis).with_pd(30.0, 0.0).with_target(1.0));
    lg_w.angular_pd_controllers.push(
        AngularPDController::new(p, ch, axis, 30.0, 0.0).with_target(1.0),
    );
    let (dth, dw) = lockstep_divergence(ja_w, lg_w, p, ch, axis, sub_dt, n);
    println!("T2 kp-only (no limit): max|Δθ| {dth:.3e}, max|Δω| {dw:.3e}");
    assert!(dth < 1e-12, "kp-only diverged from apply_implicit_kp: max|Δθ| = {dth:e}");

    // ── kp-only with finite torque_limit EXERCISED (saturated regime) ──
    //
    // target = 50 rad: the de-saturation band (|target − θ| < τ_max/kp ≈
    // 0.017 rad) is never entered, so BOTH engines apply exactly ±τ_max as
    // a constant torque every substep — the saturated branch of the
    // combined solve and the legacy clamp are the same constant-torque
    // regime, so the trajectories must be identical.
    let mut ja_w = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut ja_w, 0.05, 1e-3);
    let mut lg_w = XpbdWorld::new();
    pinned_pair(&mut lg_w, 0.05, 1e-3);
    ja_w.joint_axes.push(
        JointAxis::new(p, ch, axis).with_pd(30.0, 0.0).with_target(50.0).with_torque_limit(0.5),
    );
    lg_w.angular_pd_controllers.push(
        AngularPDController::new(p, ch, axis, 30.0, 0.0).with_target(50.0).with_torque_limit(0.5),
    );
    let (dth, dw) = lockstep_divergence(ja_w, lg_w, p, ch, axis, sub_dt, n);
    println!("T2 kp-only (limit 0.5, deep-saturated): max|Δθ| {dth:.3e}, max|Δω| {dw:.3e}");
    assert!(
        dth < 1e-12,
        "kp+torque_limit (saturated) diverged from the legacy clamp: max|Δθ| = {dth:e}"
    );

    // KNOWN, DESIGNED divergence outside this regime: the two saturation
    // CRITERIA differ. The legacy pass clamps on the torque at the
    // linearization angle θₙ (pre-solve, plus an impulse post-clamp); the
    // combined solve saturates on the end-of-substep active torque τ_act⁺ —
    // the torque it actually applies. They differ only in substeps where
    // |τ(θₙ)| > τ_max ≥ |τ_act⁺| (the de-saturation window on approach,
    // where the velocity term dt·kp·ω⁺ spans the gap): legacy keeps pushing
    // at the cap one criterion-width longer. MEASURED on an undamped
    // kp = 30 / τ_max = 0.5 / target = 1.0 approach: the window's extra
    // impulses phase-shift the undamped bang-bang transient, max|Δθ| grows
    // to ≈0.81 rad mid-transient; both engines share the same equilibrium
    // (the target) and the same ±τ_max budget on applied torque. This is
    // the combined solve honoring the budget on the torque it APPLIES
    // rather than on a pre-solve estimate — semantic, not float noise, and
    // intentional (design step 5). Not asserted as equivalence; the
    // saturated-regime and unlimited cases above pin down both branches'
    // formulas.

    // ── soft-limit-only, including the tear latch ──
    let mut ja_w = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut ja_w, 0.05, 1e-6);
    ja_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 5.0);
    let mut lg_w = XpbdWorld::new();
    pinned_pair(&mut lg_w, 0.05, 1e-6);
    lg_w.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 5.0);
    // Tear threshold sized to the fling's energy: KE = ½·I·ω² = 1.25e-5 J
    // compresses the k = 1000 exp 2 limit to excess ≈ (3·KE/k)^⅓ ≈ 3.3e-3
    // rad at most, so a 1e-3 threshold is comfortably crossed.
    ja_w.joint_axes.push(
        JointAxis::new(p, ch, axis)
            .with_soft_limit(-0.3, 0.3, 1000.0, 2.0)
            .with_tearing(1e-3, 0.05),
    );
    lg_w.angular_soft_limits.push(
        AngularSoftLimit::new(p, ch, axis, -0.3, 0.3, 1000.0, 2.0).with_tearing(1e-3, 0.05),
    );
    let mut max_dtheta: f64 = 0.0;
    let mut torn_mismatch = false;
    for _ in 0..n {
        ja_w.substep(sub_dt);
        lg_w.substep(sub_dt);
        let da = (joint_angle(&ja_w, p, ch, axis) - joint_angle(&lg_w, p, ch, axis)).abs();
        if da > max_dtheta { max_dtheta = da; }
        let ja_torn = ja_w.joint_axes[0].limit.as_ref().unwrap().torn;
        if ja_torn != lg_w.angular_soft_limits[0].torn {
            torn_mismatch = true;
        }
    }
    println!("T2 soft-limit-only (torn): max|Δθ| {max_dtheta:.3e}");
    assert!(!torn_mismatch, "tear latch state diverged from AngularSoftLimit");
    assert!(
        ja_w.joint_axes[0].limit.as_ref().unwrap().torn,
        "tear was never exercised — raise the fling speed"
    );
    assert!(
        max_dtheta < 1e-12,
        "soft-limit-only diverged from AngularSoftLimit: max|Δθ| = {max_dtheta:e}"
    );
}

// ════════════════════════════════════════════════════════════════════════
// T3 — torque_limit saturation: active clamped exactly, passive untouched
// ════════════════════════════════════════════════════════════════════════

/// kp = 30, τ_max = 0.5, coaxial passive spring K_sp = 2 (rest 0), target
/// 0.8: demand kp·(0.8 − θ) ≫ τ_max all the way to equilibrium, so the
/// active output must sit at EXACTLY ±τ_max (bit equality — the saturated
/// branch assigns σ·τ_max) while the passive spring stays unclamped, and
/// the settled equilibrium is the analytic torque balance
/// τ_max = K_sp·θ* ⇒ θ* = 0.25 — at BOTH substep rates (the saturated
/// active torque is a constant; no gain left to attenuate).
///
/// Moving-away case: with the joint yanked away from the target the
/// end-of-substep active demand is even larger; the recorded active torque
/// must still be within the budget every substep.
#[test]
fn t3_saturation_clamps_active_exactly_and_keeps_passive_free() {
    let axis = Vec3::z();
    let kp = 30.0;
    let kd = 1.0;
    let k_sp = 2.0;
    let tau_max = 0.5;
    let target = 0.8;
    let theta_star = tau_max / k_sp; // 0.25

    for &sub_dt in &[SUB_DT_FINE, SUB_DT_COARSE] {
        let mut world = XpbdWorld::new();
        let (p, ch) = pinned_pair(&mut world, 0.05, 1e-6);
        world.joint_axes.push(
            JointAxis::new(p, ch, axis)
                .with_pd(kp, kd)
                .with_target(target)
                .with_passive_spring(k_sp, 0.0)
                .with_passive_damping(1.0)
                .with_torque_limit(tau_max),
        );

        // Overdemand phase: far from target, active must be clamped to
        // exactly ±τ_max while the TOTAL applied torque differs from the
        // active one (the passive spring's share is alive and unclamped).
        let n_over = (0.05 / sub_dt).round() as usize;
        let mut saw_passive_share = false;
        for _ in 0..n_over {
            world.substep(sub_dt);
            let ja = &world.joint_axes[0];
            assert_eq!(
                ja.applied_active_torque.abs(), tau_max,
                "active torque {} not clamped to exactly ±{tau_max} during overdemand",
                ja.applied_active_torque
            );
            if (ja.applied_total_torque - ja.applied_active_torque).abs() > 1e-6 {
                saw_passive_share = true;
            }
        }
        assert!(
            saw_passive_share,
            "total torque never differed from the clamped active torque — \
             the passive spring's share went missing"
        );

        // Settle and check the analytic saturated equilibrium.
        run_substeps(&mut world, sub_dt, (4.0 / sub_dt).round() as usize);
        let settled = joint_angle(&world, p, ch, axis);
        println!("T3 settled {settled:.6} rad at sub_dt {sub_dt:.6} (analytic {theta_star})");
        assert!(
            (settled - theta_star).abs() < 0.005,
            "saturated equilibrium {settled} rad != analytic τ_max/K_sp = {theta_star} \
             (±0.005) at sub_dt = {sub_dt}"
        );
        assert_eq!(
            world.joint_axes[0].applied_active_torque.abs(), tau_max,
            "active torque must still sit at the cap at the saturated equilibrium"
        );

        // Moving-away case: yank the child away from the target and verify
        // the budget holds on every subsequent substep. (The legacy pass
        // needed a documented post-clamp for exactly this case; here the
        // saturated branch IS the budget.)
        world.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, -50.0);
        for _ in 0..(0.02 / sub_dt).round() as usize {
            world.substep(sub_dt);
            let a = world.joint_axes[0].applied_active_torque.abs();
            assert!(
                a <= tau_max,
                "moving-away active torque {a} exceeded the τ_max = {tau_max} budget"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// T4 — soft-limit crossing: energy, tear parity, airborne-drift verdict
// ════════════════════════════════════════════════════════════════════════

/// Rotational kinetic energy of one body about its CoM (world inertia).
fn rot_ke(world: &XpbdWorld, idx: usize) -> f64 {
    let b = &world.bodies[idx];
    let r = b.orientation.to_rotation_matrix();
    let i_world = r.matrix() * b.inertia_tensor * r.matrix().transpose();
    0.5 * b.angular_velocity.dot(&(i_world * b.angular_velocity))
}

/// T4a — fling a small body through the ROM. Limit engagement must not
/// inject energy: the bounce-back speed of every engagement is at most the
/// incoming speed, and total joint energy (rotational KE + limit potential)
/// never ends above where it started. And the tear latch must fire on the
/// same substep as the legacy primitive (it is evaluated from the same θₙ
/// with the same threshold semantics — and the T2 soft-limit trajectories
/// are bit-equal, so this is parity of the whole failure mode).
///
/// What is deliberately NOT asserted: SUBSTEP-wise monotone non-increase of
/// KE + PE. The design predicted it; measurement says otherwise, for a
/// reason shared with the legacy pass (trajectories are bit-equal): the
/// solve linearizes at θₙ, so on the engagement substep a fast body flies
/// deep into the potential (dt·ω ≈ 4e-3 rad here) while the torque it pays
/// is the near-zero one at θₙ — the PE bookkeeping jumps (measured +1.9e-4 J
/// on entry) before the KE pays for it, and the same lag refunds it on the
/// way out. This is the documented one-substep engagement lag of the
/// θₙ-linearized scheme (see `JointAxis::apply_implicit`'s invariants), not
/// an injector: the per-engagement and end-to-end energy checks below are
/// the physically meaningful statements, and they hold.
///
/// MEASURED: 3 engagements (the post-tear ping-pong between the ±0.5
/// bounds), worst restitution 0.53, E 4.5e-5 → 2.8e-7 J over 0.5 s, tear
/// latched on substep 121 on both engines.
#[test]
fn t4a_limit_crossing_dissipates_and_tears_like_legacy() {
    let axis = Vec3::z();
    let sub_dt = SUB_DT_FINE;

    let build = |combined: bool| -> XpbdWorld {
        let mut world = XpbdWorld::new();
        let (p, ch) = pinned_pair(&mut world, 0.05, 1e-7);
        world.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, 30.0);
        // Tear threshold sized to the fling: KE = ½·1e-7·30² = 4.5e-5 J
        // compresses the k = 8000 exp 2 limit to excess ≈ (3·KE/k)^⅓ ≈
        // 2.6e-3 rad (less after implicit dissipation), so 5e-4 is crossed.
        if combined {
            world.joint_axes.push(
                JointAxis::new(p, ch, axis)
                    .with_soft_limit(-0.5, 0.5, 8000.0, 2.0)
                    .with_tearing(5e-4, 0.05),
            );
        } else {
            world.angular_soft_limits.push(
                AngularSoftLimit::new(p, ch, axis, -0.5, 0.5, 8000.0, 2.0)
                    .with_tearing(5e-4, 0.05),
            );
        }
        world
    };

    let mut ja_w = build(true);
    let mut lg_w = build(false);
    let (p, ch) = (0usize, 1usize);
    let axis = Vec3::z();

    let e_start = rot_ke(&ja_w, ch); // starts inside the ROM: PE = 0
    let mut tear_substep_ja: Option<usize> = None;
    let mut tear_substep_lg: Option<usize> = None;
    let mut max_stepwise_gain: f64 = 0.0;
    let mut e_prev = e_start;
    // Per-engagement speed audit: |ω| entering an out-of-ROM excursion vs
    // |ω| exiting it.
    let mut engaged = false;
    let mut speed_in = 0.0;
    let mut speed_prev = ja_w.bodies[ch].angular_velocity.norm();
    let mut worst_restitution: f64 = 0.0;
    let mut n_engagements = 0usize;
    let n = (0.5 / sub_dt).round() as usize;
    for i in 0..n {
        ja_w.substep(sub_dt);
        lg_w.substep(sub_dt);
        let theta = joint_angle(&ja_w, p, ch, axis);
        let speed = ja_w.bodies[ch].angular_velocity.norm();
        let now_engaged = theta.abs() > 0.5;
        if now_engaged && !engaged {
            speed_in = speed_prev; // speed just before crossing the boundary
        }
        if !now_engaged && engaged {
            let restitution = speed / speed_in;
            if restitution > worst_restitution { worst_restitution = restitution; }
            n_engagements += 1;
        }
        engaged = now_engaged;
        speed_prev = speed;

        let e = rot_ke(&ja_w, ch) + ja_w.joint_axes[0].potential_energy(&ja_w.bodies);
        let gain = e - e_prev;
        if gain > max_stepwise_gain { max_stepwise_gain = gain; }
        e_prev = e;
        if tear_substep_ja.is_none() && ja_w.joint_axes[0].limit.as_ref().unwrap().torn {
            tear_substep_ja = Some(i);
        }
        if tear_substep_lg.is_none() && lg_w.angular_soft_limits[0].torn {
            tear_substep_lg = Some(i);
        }
    }
    let e_end = rot_ke(&ja_w, ch) + ja_w.joint_axes[0].potential_energy(&ja_w.bodies);
    println!(
        "T4a: {n_engagements} engagements, worst restitution {worst_restitution:.6}, \
         E {e_start:.6e} → {e_end:.6e} J (max transient stepwise gain {max_stepwise_gain:.3e} J), \
         tear at ja={tear_substep_ja:?} lg={tear_substep_lg:?}"
    );
    assert!(n_engagements >= 1, "the fling never engaged the limit");
    assert!(
        worst_restitution <= 1.0 + 1e-9,
        "a limit engagement sped the body up: restitution {worst_restitution}"
    );
    assert!(
        e_end <= e_start + 1e-12,
        "net energy gain across the fling: {e_start} J → {e_end} J"
    );
    assert!(tear_substep_ja.is_some(), "the fling never tore the ligament");
    assert_eq!(
        tear_substep_ja, tear_substep_lg,
        "tear latched on a different substep than the legacy AngularSoftLimit"
    );
}

/// Build an AIRBORNE hallux ray (no ground): fixed anchor → mt1 (welded) →
/// hallux proximal phalanx → hallux distal phalanx, with the foot's full
/// articulated-joint stack per joint (positional constraint 1/60000,
/// primary-axis soft limit k = 8000 exp 2, passive spring k = 4, passive
/// damper c = 1.0, active PD, off-axis locks ±0.05 with dampers — exactly
/// `foot.rs::add_articulated` / engine_regression.rs::build_toe_ray, bone
/// and gain values from `foot.rs::BONES`/`JOINTS`). `combined` replaces the
/// four coaxial primary-axis primitives with ONE JointAxis; the positional
/// constraint and the off-axis locks/dampers (different axes — not coaxial)
/// stay identical.
fn hallux_ray_airborne(combined: bool) -> (XpbdWorld, Vec<usize>) {
    struct Link {
        mass: f64,
        half_h: f64,
        radius: f64,
        kp: f64,
        kd: f64,
        rom: (f64, f64),
    }
    let links = [
        // hallux_pp, MTP joint: rom −0.61..1.22, kp 30, kd 1.0
        Link { mass: 0.031, half_h: 0.0082, radius: 0.0073, kp: 30.0, kd: 1.0, rom: (-0.61, 1.22) },
        // hallux_dp, IP joint: rom −1.57..0, kp 6, kd 0.2
        Link { mass: 0.015, half_h: 0.0045, radius: 0.0055, kp: 6.0, kd: 0.2, rom: (-1.57, 0.0) },
    ];
    let mut world = XpbdWorld::new();
    world.no_ground();

    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut anchor = RigidBody::new(0.0, Mat3::identity());
    anchor.orientation = along_z;
    let anchor_idx = world.add_body(anchor);

    // mt1 parent, resting height as if on a ground plane (irrelevant airborne,
    // kept for parity with the grounded ray's geometry).
    let (pm, ph, pr) = (0.094, 0.0245, 0.0075);
    let mut parent = RigidBody::new_capsule(pm, ph, pr);
    parent.orientation = along_z;
    parent.position = Vec3::new(0.0, pr, 0.0);
    let parent_idx = world.add_body(parent);
    // Weld mt1 to the anchor (positional + 3-axis zero-range locks).
    let joint_world = Vec3::new(0.0, pr, 0.0);
    let anchor_local = |world: &XpbdWorld, body: usize, jw: Vec3| -> Vec3 {
        let b = &world.bodies[body];
        b.orientation.inverse() * (jw - b.position)
    };
    let mut c = PositionalConstraint::new(anchor_idx, parent_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = anchor_local(&world, anchor_idx, joint_world);
    c.local_anchor_b = anchor_local(&world, parent_idx, joint_world);
    world.constraints.push(c);
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_constraints.push(AngularLimitConstraint::new(
            anchor_idx, parent_idx, axis, 0.0, 0.0,
        ));
    }

    let mut idxs = vec![parent_idx];
    let mut prev_idx = parent_idx;
    let mut prev_tip_z = ph + pr;
    let mut prev_r = pr;
    for link in &links {
        let joint_world = Vec3::new(0.0, (prev_r + link.radius) * 0.5, prev_tip_z);
        let mut b = RigidBody::new_capsule(link.mass, link.half_h, link.radius);
        b.orientation = along_z;
        b.position = Vec3::new(0.0, link.radius, prev_tip_z + link.half_h + link.radius);
        let idx = world.add_body(b);

        let mut c = PositionalConstraint::new(prev_idx, idx, 0.0, 1.0 / 60_000.0);
        c.local_anchor_a = anchor_local(&world, prev_idx, joint_world);
        c.local_anchor_b = anchor_local(&world, idx, joint_world);
        world.constraints.push(c);
        if combined {
            world.joint_axes.push(
                JointAxis::new(prev_idx, idx, Vec3::x())
                    .with_soft_limit(link.rom.0, link.rom.1, 8000.0, 2.0)
                    .with_passive_spring(4.0, 0.0)
                    .with_passive_damping(1.0)
                    .with_pd(link.kp, link.kd),
            );
        } else {
            world.angular_soft_limits.push(AngularSoftLimit::new(
                prev_idx, idx, Vec3::x(), link.rom.0, link.rom.1, 8000.0, 2.0,
            ));
            world.angular_springs.push(AngularSpring::new(prev_idx, idx, Vec3::x(), 4.0, 0.0));
            world.angular_dampers.push(AngularDamper::new(prev_idx, idx, Vec3::x(), 1.0));
            world.angular_pd_controllers.push(
                AngularPDController::new(prev_idx, idx, Vec3::x(), link.kp, link.kd),
            );
        }
        for off in [Vec3::y(), Vec3::z()] {
            world.angular_constraints.push(AngularLimitConstraint::new(
                prev_idx, idx, off, -0.05, 0.05,
            ));
            world.angular_dampers.push(AngularDamper::new(prev_idx, idx, off, 1.0));
        }

        idxs.push(idx);
        prev_idx = idx;
        prev_tip_z += 2.0 * (link.half_h + link.radius);
        prev_r = link.radius;
    }
    (world, idxs)
}

/// T4b — the airborne-drift verdict. The airborne hallux ray's MTP is
/// commanded hard past its ROM (target 1.5 vs rom_max 1.22) — the sustained
/// PD-vs-soft-limit fight the split passes were suspected to turn into a
/// slow limit cycle past rom_max. Last-second statistics are measured on
/// both engines.
///
/// MEASURED VERDICT — the drift is real and the combined solve cures it,
/// though its legacy shape differs from the suspicion (a stuck settle, not
/// a cycle):
///   legacy:   MTP last-sec mean 1.5340 rad (min 1.5339, max 1.5341),
///             max|ω| 2.8e-4 rad/s — parked 0.31 rad PAST rom_max, nearly
///             at the raw command, because the limit pass's own implicit
///             attenuation (dt²·w·K_lim' ≈ 96 at that excess) starves it of
///             the authority to push back against the (less-attenuated,
///             smaller-k) PD pass.
///   combined: MTP last-sec mean 1.2414 rad (min = max to 4 decimals),
///             max|ω| 2.9e-12 rad/s — exactly the analytic PD-vs-limit
///             torque balance 30·(1.5−θ) = 4θ + 8000(θ−1.22)² ⇒ θ* ≈ 1.2397
///             (the 1.5e-3 offset is the off-axis-lock + positional-anchor
///             coupling of the full stack), inside rom_max + 0.03.
#[test]
fn t4b_airborne_mtp_overdrive_settles_at_the_limit_on_joint_axis() {
    let sub_dt = SUB_DT_FINE;
    let rom_max = 1.22;
    let target = 1.5;

    let run = |combined: bool| -> (f64, f64, f64, f64) {
        let (mut world, idxs) = hallux_ray_airborne(combined);
        // Set the MTP target (first primary-axis joint).
        if combined {
            world.joint_axes[0].target_angle = target;
        } else {
            world.angular_pd_controllers[0].target_angle = target;
        }
        let total = (6.0 / sub_dt).round() as usize;
        let last_sec_from = total - (1.0 / sub_dt).round() as usize;
        let mut mean = 0.0;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut max_w: f64 = 0.0;
        let mut samples = 0usize;
        for i in 0..total {
            world.substep(sub_dt);
            if i >= last_sec_from {
                let a = joint_angle(&world, idxs[0], idxs[1], Vec3::x());
                mean += a;
                if a < mn { mn = a; }
                if a > mx { mx = a; }
                let w = world.bodies[idxs[1]].angular_velocity.norm();
                if w > max_w { max_w = w; }
                samples += 1;
            }
        }
        (mean / samples as f64, mn, mx, max_w)
    };

    let (lg_mean, lg_min, lg_max, lg_w) = run(false);
    let (ja_mean, ja_min, ja_max, ja_w) = run(true);
    println!(
        "T4b legacy:   MTP last-sec mean {lg_mean:.4} min {lg_min:.4} max {lg_max:.4} rad, max|ω| {lg_w:.3e} rad/s"
    );
    println!(
        "T4b combined: MTP last-sec mean {ja_mean:.4} min {ja_min:.4} max {ja_max:.4} rad, max|ω| {ja_w:.3e} rad/s"
    );

    // Bite: the legacy split passes really do let the PD drag the joint far
    // past its ROM (measured 1.534 rad, +0.31 past rom_max).
    assert!(
        lg_max > rom_max + 0.1,
        "legacy MTP stayed at {lg_max} rad — the split-pass drift this test \
         documents (measured 1.534) has vanished; re-measure the verdict"
    );

    // The combined solve: settled at (or just past, by the analytic
    // PD-vs-limit balance) the ROM boundary, and QUIET.
    assert!(
        ja_max <= rom_max + 0.03,
        "combined MTP reached {ja_max} rad in the last second — past rom_max + 0.03 = {}",
        rom_max + 0.03
    );
    assert!(
        ja_w < 0.05,
        "combined MTP not quiet in the last second: max|ω| = {ja_w} rad/s"
    );
}

// ════════════════════════════════════════════════════════════════════════
// T5 — worst-case stress: max limit stiffness, nano-inertia, coarse dt
// ════════════════════════════════════════════════════════════════════════

/// K_lim = 8000 exp 2, I = 1e-9 kg·m², sub_dt = 1/3600, and a
/// torque-limited PD commanding deep into the limit: the stiffest
/// configuration any foot joint composes, on an inertia an order below the
/// smallest phalanx, at HALF the standard substep rate. The shared
/// denominator certifies stability (≥ 1 for any gains); this asserts it:
/// finite throughout, settled at the analytic saturated equilibrium
/// (τ_max = K_sp·θ + K_lim·(θ − rom_max)² ⇒ θ* ≈ 0.1035), quiet at the end.
#[test]
fn t5_worst_case_stiffness_on_nano_inertia_stays_finite_and_settles() {
    let axis = Vec3::z();
    let sub_dt = SUB_DT_COARSE;
    let mut world = XpbdWorld::new();
    let (p, ch) = pinned_pair(&mut world, 0.001, 1e-9);
    world.joint_axes.push(
        JointAxis::new(p, ch, axis)
            .with_soft_limit(-0.1, 0.1, 8000.0, 2.0)
            .with_passive_spring(4.0, 0.0)
            .with_passive_damping(1.0)
            .with_pd(50.0, 0.5)
            .with_target(1.0)
            .with_torque_limit(0.5),
    );

    let n = (3.0 / sub_dt).round() as usize;
    let mut max_theta: f64 = f64::NEG_INFINITY;
    for _ in 0..n {
        world.substep(sub_dt);
        let th = joint_angle(&world, p, ch, axis);
        assert!(
            th.is_finite() && world.bodies[ch].angular_velocity.norm().is_finite(),
            "worst-case stress went non-finite"
        );
        if th > max_theta { max_theta = th; }
    }
    let settled = joint_angle(&world, p, ch, axis);
    let final_w = world.bodies[ch].angular_velocity.norm();
    println!("T5 settled {settled:.5} rad (peak {max_theta:.5}), final |ω| {final_w:.3e} rad/s");
    // Analytic saturated equilibrium: 0.5 = 4·θ + 8000·(θ − 0.1)² ⇒ θ* ≈ 0.1035.
    assert!(
        (settled - 0.1035).abs() < 0.01,
        "worst-case settled at {settled} rad, analytic ≈ 0.1035 (±0.01)"
    );
    assert!(
        final_w < 0.01,
        "worst-case did not settle: final |ω| = {final_w} rad/s"
    );
}
