//! Commit A of the coupled joint+contact solve (design contract:
//! docs/superpowers/specs/2026-10-21-coupled-joint-contact-solve.md, §5
//! item 1): the in-loop JointAxis lifecycle behind
//! `XpbdWorld::coupled_joint_solve` — default FALSE, so a flag-off substep
//! is exactly the legacy pipeline and the whole existing suite guards it.
//! These tests pin the flag-ON free-space guarantee (design §2.4): the
//! in-loop form lands on `apply_implicit`'s fixed point —
//!   A  — single-substep equivalence, unsaturated 2×2 path;
//!   B  — single-substep equivalence, saturated branch (accumulated clamp,
//!        passive re-solve, bit-exact σ·τ_max diagnostic);
//!   C  — degenerate-γ branches (pure feedforward / pure ext_torque
//!        constant-impulse rows, including the clamped constant);
//!   D  — multi-substep free-space trajectory equivalence (E3 precursor);
//!   E6 — #[ignore] convergence-diagnostic harness on a grounded rig.
//!
//! Commit B adds the acceptance gates (design §4), asserted on the flag-ON
//! path with their flag-off receipts recorded in each doc comment:
//!   E1 — static force transmission through a grounded toe tip (~100 N);
//!        landed #[ignore]'d at cold start (its near-degenerate rig missed
//!        the 10% band — miss recorded in-place) and un-ignored once the
//!        L2 warm-start lever closed it;
//!   E2 — heel-lift threshold dt-independence (passes at cold start);
//!   E5 — pressed-phase mechanical-energy monotonicity (no pump under the
//!        sustained contact fight);
//!   plus an #[ignore] receipts/L1-sweep harness.
//!
//! TOLERANCE DERIVATION (single-substep comparisons): the two paths
//! evaluate the SAME rational function of the same inputs — the in-loop
//! γ-form is the pre-pass D-form with numerator and denominator both
//! scaled by dt·(d + dt·k) (design §2.1, verified algebraically) — so the
//! only divergence is floating-point op ORDER. The one genuine
//! cancellation is the γ-form residual r = −(u + γλ + b): its absolute
//! rounding is a few ε·(|u| + |b|), with |b| = |τ₀|/(d + dt·k) ≤ ~25 and
//! |u| ≤ ~45 across these configs — ~1e-13 absolute on ω at worst,
//! attenuated (never amplified) through the solve's ≤ 1 gain factors
//! w/(w + γ). TOL_OMEGA = 1e-11 gives two orders of headroom while
//! sitting ten orders below the velocities compared. The torque
//! diagnostics divide dt-scaled impulses back out: their divergence is the
//! ω divergence divided by w (= 1e6 here) times 1/dt (= 7200), plus a few
//! ulps of the O(≤ 25 N·m) torques themselves — ≤ ~1e-13;
//! TOL_TORQUE = 1e-10 is three orders of headroom and six below the
//! smallest compared torque (~5e-3 N·m).

use physics::constraints::PositionalConstraint;
use physics::joint_axis::JointAxis;
use physics::math::{Mat3, Quat, Vec3, swing_twist_angle};
use physics::rigid_body::RigidBody;
use physics::xpbd::XpbdWorld;

const SUB_DT_FINE: f64 = 1.0 / 7200.0;
const SUB_DT_COARSE: f64 = 1.0 / 3600.0;

const TOL_OMEGA: f64 = 1e-11;
const TOL_TORQUE: f64 = 1e-10;

/// Pinned parent + free small child at the same position (pure rotational
/// joint about the axis) — the tests/joint_axis.rs rig.
fn pinned_pair(world: &mut XpbdWorld, child_mass: f64, child_inertia: f64) -> (usize, usize) {
    world.no_ground();
    let parent = RigidBody::new(0.0, Mat3::identity());
    let parent_idx = world.add_body(parent);
    let child = RigidBody::new(child_mass, Mat3::identity() * child_inertia);
    let child_idx = world.add_body(child);
    (parent_idx, child_idx)
}

fn joint_angle(world: &XpbdWorld, a: usize, b: usize, axis: Vec3) -> f64 {
    let q_rel = world.bodies[a].orientation.inverse() * world.bodies[b].orientation;
    swing_twist_angle(q_rel, axis)
}

/// Build the SAME free-space world twice — flag off (the pre-pass
/// `apply_implicit`, today's default path) and flag on (the in-loop
/// lifecycle) — from identical body states and one identical JointAxis.
fn equivalence_pair(
    make_joint: &dyn Fn(usize, usize) -> JointAxis,
    child_omega_z: f64,
) -> (XpbdWorld, XpbdWorld) {
    let build = |coupled: bool| -> XpbdWorld {
        let mut world = XpbdWorld::new();
        world.coupled_joint_solve = coupled;
        let (p, ch) = pinned_pair(&mut world, 0.05, 1e-6);
        world.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, child_omega_z);
        world.joint_axes.push(make_joint(p, ch));
        world
    };
    (build(false), build(true))
}

/// Max-divergence report after running both worlds one substep: child
/// angular velocity, total-torque diagnostic, active-torque diagnostic.
fn single_substep_divergence(
    off: &mut XpbdWorld,
    on: &mut XpbdWorld,
    sub_dt: f64,
) -> (f64, f64, f64) {
    off.substep(sub_dt);
    on.substep(sub_dt);
    let dv = (on.bodies[1].angular_velocity - off.bodies[1].angular_velocity).norm();
    let d_total =
        (on.joint_axes[0].applied_total_torque - off.joint_axes[0].applied_total_torque).abs();
    let d_active =
        (on.joint_axes[0].applied_active_torque - off.joint_axes[0].applied_active_torque).abs();
    (dv, d_total, d_active)
}

// ════════════════════════════════════════════════════════════════════════
// A — unsaturated 2×2 path
// ════════════════════════════════════════════════════════════════════════

/// Full coaxial stack (both rows Soft), no torque limit: the straight
/// unsaturated 2×2 mini-block. One substep from identical states must land
/// on `apply_implicit`'s fixed point (design §2.4: GS is exact in one step
/// for an isolated linear constraint; iterations 2..8 are suppressed by
/// the convergence skip).
///
/// MEASURED: |Δω| 7.3e-14, |Δτ| ≤ 5.3e-16 — two-plus orders inside the
/// bounds. BITE-VERIFIED (2026-10-20): scaling the in-loop applied impulse
/// by 0.999 fails this test (and B, C, D).
#[test]
fn a_single_substep_unsaturated_matches_apply_implicit() {
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_pd(4.0, 0.15)
                .with_target(0.8)
                .with_passive_spring(4.0, 0.0)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, d_active) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("A: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(dv < TOL_OMEGA, "unsaturated in-loop ω diverged from apply_implicit: {dv:e}");
    assert!(d_total < TOL_TORQUE, "unsaturated total-torque diagnostic diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "unsaturated active-torque diagnostic diverged: {d_active:e}");
}

// ════════════════════════════════════════════════════════════════════════
// B — saturated branch
// ════════════════════════════════════════════════════════════════════════

/// kp·(target − θ) = 24 ≫ τ_max = 0.11: the end-of-substep active demand
/// (~21.7 N·m at the unsaturated fixed point) exceeds the cap by ~70×, so
/// both paths' saturation criteria — pre-pass |τ_act⁺| > τ_max at the
/// unsaturated ω⁺, in-loop |λ_a + Δ_a| > dt·τ_max, the SAME number in
/// different units (§2.2) — agree with ample margin; boundary-marginal
/// configs where fp order could flip the branch are deliberately avoided
/// (they are a semantic knife-edge, not an op-order question).
///
/// τ_max = 0.11 is deliberate: (τ_max·dt)/dt ≠ τ_max in f64 at
/// dt = 1/7200 (verified), so the bit-equality assertions below genuinely
/// discriminate the DIRECT σ·τ_max assignment from dividing the clamped
/// impulse back out — many values (0.2, 0.3, 0.5) happen to round-trip
/// and would let that bug through.
///
/// MEASURED: |Δω| 1.110e-15, |Δτ_total| 8.674e-18 (re-recorded at the
/// landed commit by the adversarial review; the first-draft figures were
/// from an earlier code iteration). BITE-VERIFIED (2026-10-20),
/// both legs: (1) replacing the post-clamp passive re-solve with the
/// unclamped passive delta fails this test alone; (2) making finalize
/// divide the clamped impulse instead of assigning σ·τ_max fails the bit
/// equalities here and in C3.
#[test]
fn b_single_substep_saturated_matches_apply_implicit() {
    let tau_max = 0.11;
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_pd(30.0, 0.1)
                .with_target(0.8)
                .with_passive_spring(2.0, 0.0)
                .with_passive_damping(1.0)
                .with_torque_limit(tau_max)
        },
        2.0,
    );
    let (dv, d_total, _) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("B: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}");
    assert!(dv < TOL_OMEGA, "saturated in-loop ω diverged from apply_implicit: {dv:e}");
    assert!(d_total < TOL_TORQUE, "saturated total-torque diagnostic diverged: {d_total:e}");
    // The saturated diagnostic is ASSIGNED σ·τ_max on both paths — bit
    // equality, not a tolerance (T3's contract; finalize never divides the
    // clamped impulse back out).
    assert_eq!(
        on.joint_axes[0].applied_active_torque, tau_max,
        "in-loop saturated active diagnostic is not bit-equal to τ_max"
    );
    assert_eq!(
        off.joint_axes[0].applied_active_torque, tau_max,
        "pre-pass saturated active diagnostic is not bit-equal to τ_max"
    );
}

// ════════════════════════════════════════════════════════════════════════
// C — degenerate-γ (constant-impulse) branches
// ════════════════════════════════════════════════════════════════════════

/// The γ→∞ rows: d + dt·k = 0 with τ₀ ≠ 0 has no finite soft form; the
/// limit is a constant impulse dt·τ₀ booked once on iteration 1 and
/// excluded afterwards ("constant torque rides outside the denominator").
/// Four compositions, each against the pre-pass:
///   C1 — pure feedforward, nothing else (constant active, empty passive);
///   C2 — pure feedforward + passive damper (constant active + soft
///        passive: the constant applies first, the soft row solves at the
///        updated velocity);
///   C3 — pure feedforward past the cap (|ff| > τ_max): clamped at
///        classification, both paths bit-equal at σ·τ_max (τ_max = 0.11,
///        a non-round-tripping value — see test B's header);
///   C4 — pure ext_torque, no passive structures (constant PASSIVE row,
///        never clamped), consumed by prepare exactly as by the pre-pass.
///
/// BITE-VERIFIED (2026-10-20): never clearing `constants_pending` (the
/// constant impulse re-booked every iteration, 8× the torque) fails this
/// test alone.
#[test]
fn c_degenerate_gamma_branches_match_apply_implicit() {
    // ── C1: pure feedforward ──
    let (mut off, mut on) =
        equivalence_pair(&|p, ch| JointAxis::new(p, ch, Vec3::z()).with_feedforward(0.3), 2.0);
    let (dv, d_total, d_active) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("C1 pure-ff: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(dv < TOL_OMEGA, "C1 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C1 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C1 active torque diverged: {d_active:e}");

    // ── C2: pure feedforward + passive damper ──
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.3)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, d_active) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("C2 ff+damper: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(dv < TOL_OMEGA, "C2 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C2 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C2 active torque diverged: {d_active:e}");

    // ── C3: clamped constant (|ff| = 0.5 > τ_max = 0.11) ──
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.5)
                .with_torque_limit(0.11)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, _) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("C3 clamped-ff: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}");
    assert!(dv < TOL_OMEGA, "C3 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C3 total torque diverged: {d_total:e}");
    assert_eq!(on.joint_axes[0].applied_active_torque, 0.11);
    assert_eq!(off.joint_axes[0].applied_active_torque, 0.11);

    // ── C4: pure ext_torque (constant passive row) ──
    let (mut off, mut on) =
        equivalence_pair(&|p, ch| JointAxis::new(p, ch, Vec3::z()), 2.0);
    off.joint_axes[0].ext_torque = 0.05;
    on.joint_axes[0].ext_torque = 0.05;
    let (dv, d_total, _) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("C4 pure-ext: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}");
    assert!(dv < TOL_OMEGA, "C4 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C4 total torque diverged: {d_total:e}");
    // ext_* is a per-substep quantity on both paths: consumed by prepare
    // (flag on) exactly as by the pre-pass (flag off).
    assert_eq!(on.joint_axes[0].ext_torque, 0.0, "prepare did not consume ext_torque");
    assert_eq!(off.joint_axes[0].ext_torque, 0.0);
    // The pre-pass's torque must have actually arrived (the injection is
    // not a silent no-op): dt·w·τ_ext changes ω by ~69 rad/s here.
    assert!(
        (off.bodies[1].angular_velocity.z - 2.0).abs() > 1.0,
        "ext_torque had no effect — C4 is testing nothing"
    );
}

/// C5 — overflow-scale (unphysical-but-representable) stiffness content
/// degrades to the Constant row instead of poisoning the solve. For
/// damping around 1e-280, γ = 1/(dt·d) ≈ 2e283 is still FINITE — the
/// plain is_finite() classification keeps the row Soft — but the 2×2
/// block's downstream products overflow: the determinant γ_a·γ_p and the
/// numerator (w+γ_p)·r_a both exceed f64 range, inf/inf = NaN lands in
/// the deltas, and the flag-on world's velocities go NaN while the
/// pre-pass D-form (denominator 1 + dt·w·d ≈ 1) stays exact. The
/// GAMMA_MAX = 1e100 guard classifies any such row as the γ→∞
/// constant-impulse limit, whose error is below one ulp of dt·τ₀ (see the
/// derivation at the guard).
///
/// Config: feedforward 0.3 with kd = 1e-280 (active row: nonzero τ₀ +
/// overflow-scale γ) plus passive damping 3e-280 (passive row: zero τ₀ +
/// overflow-scale γ ⇒ Absent under the guard). One substep, flag on vs
/// flag off, the A/B/C tolerances.
///
/// MEASURED: |Δω|, |Δτ_total|, |Δτ_active| all exactly 0.0 — at this
/// scale the constant-impulse limit and the D-form agree bit-for-bit
/// (D = 1 + dt·w·d rounds to exactly 1.0 at d ~ 1e-280).
/// BITE-VERIFIED (2026-10-21): with the GAMMA_MAX guard removed
/// (classification reverted to bare is_finite()), the flag-on child
/// velocity is NaN and this test fails; guard restored, it passes.
#[test]
fn c5_overflow_scale_stiffness_degrades_to_constant_row() {
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.3)
                .with_pd(0.0, 1e-280)
                .with_passive_damping(3e-280)
        },
        2.0,
    );
    let (dv, d_total, d_active) = single_substep_divergence(&mut off, &mut on, SUB_DT_FINE);
    println!("C5 overflow-γ: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(
        on.bodies[1].angular_velocity.z.is_finite(),
        "C5: flag-on velocity went non-finite (overflow-γ row poisoned the block solve)"
    );
    assert!(dv < TOL_OMEGA, "C5 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C5 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C5 active torque diverged: {d_active:e}");
}

// ════════════════════════════════════════════════════════════════════════
// D — multi-substep free-space trajectory (E3 precursor)
// ════════════════════════════════════════════════════════════════════════

/// 600 substeps (1/12 s) of the full unsaturated stack, flag on vs flag
/// off in lockstep.
///
/// TOLERANCE: per-substep op-order divergence is ≤ ~1e-13 absolute on ω
/// (file-header derivation); the stack is strongly damped (contractive),
/// so divergence accumulates at most linearly — 600 × 1e-13 = 6e-11 is the
/// generous linear bound and 1e-10 caps it. This is the same scale the
/// design sanctions for T2's fp-order fallback (§2.4: tolerance
/// re-derivation to ≤ 1e-10, never a semantics change). No torque limit in
/// this config: a saturation-branch boundary could flip between the paths
/// on an fp whisker mid-trajectory and turn op-order noise into a real
/// (bounded, but branch-semantic) divergence — the saturated branch is
/// pinned by test B instead.
///
/// MEASURED: max|Δθ| 7.8e-16, max|Δω| 1.2e-13 over 600 substeps at the
/// cold-start baseline; re-measured with the L2 warm start landed (the
/// current path — the seed re-apply + first-sweep unwind changes op order
/// every substep): max|Δθ| 1.4e-17, max|Δω| 7.3e-14 — still not
/// accumulating, no ulp-walk, so the §2.4 fallback tolerance re-derivation
/// stays unused. BITE-VERIFIED (2026-10-20): the 0.999 impulse
/// scaling (test A's bite) fails this test.
#[test]
fn d_multi_substep_free_space_trajectory_matches() {
    let axis = Vec3::z();
    let (mut off, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, axis)
                .with_pd(4.0, 0.15)
                .with_target(0.8)
                .with_passive_spring(4.0, 0.0)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let mut max_dtheta: f64 = 0.0;
    let mut max_domega: f64 = 0.0;
    for _ in 0..600 {
        off.substep(SUB_DT_FINE);
        on.substep(SUB_DT_FINE);
        let da = (joint_angle(&on, 0, 1, axis) - joint_angle(&off, 0, 1, axis)).abs();
        let dw = (on.bodies[1].angular_velocity - off.bodies[1].angular_velocity).norm();
        if da > max_dtheta { max_dtheta = da; }
        if dw > max_domega { max_domega = dw; }
    }
    println!("D: max|Δθ| {max_dtheta:.3e}, max|Δω| {max_domega:.3e} over 600 substeps");
    assert!(
        max_dtheta < 1e-10,
        "flag-on trajectory diverged from flag-off: max|Δθ| = {max_dtheta:e}"
    );
    assert!(
        max_domega < 1e-10,
        "flag-on trajectory diverged from flag-off: max|Δω| = {max_domega:e}"
    );
}

// ════════════════════════════════════════════════════════════════════════
// E6 — convergence instrumentation (diagnostic only)
// ════════════════════════════════════════════════════════════════════════

/// E6 (design §4): DIAGNOSTIC harness — run with
/// `cargo test -p physics --test coupled_joint -- --ignored --nocapture`.
///
/// Rig: pinned parent; toe-scale child capsule (3 g) lying along +z with
/// both end-sphere bottoms on the plane, hinged at its proximal end
/// (positional anchor, the foot rigs' standard compliance); a PD JointAxis
/// about x presses the distal tip into the ground (positive θ about +x
/// moves the +z tip down), so the contact blocks the axis and the joint
/// must transmit THROUGH it. The table reports the fraction of the
/// state-computed static demand kp·(target − θ) that arrives as
/// `applied_total_torque` (under the flag: the REALIZED transmitted
/// torque including contact coupling) over the last 0.1 s of a 1 s settle,
/// at solver_iterations ∈ {4, 8, 16, 32} × sub_dt ∈ {1/7200, 1/3600},
/// flag on — plus the flag-off pre-pass at 8 iterations as the legacy
/// receipt. This measurement decides the design's escalation levers
/// (L1: 16 iterations; L2: warm-start pair); it asserts nothing beyond
/// running to finite numbers.
///
/// MEASURED (2026-10-20, Commit A baseline — COLD start, historical): this
/// rig is close to the design's WORST-CASE alignment (the parent is
/// pinned, so the tip contact blocks the joint's only freedom almost
/// exactly through the mass matrix — §2.6's a → 1). The cold-start
/// fraction behaved exactly as that analysis predicts: ~linear in
/// iterations and ~linear in dt (legacy pre-pass 0.0090 ≈ 1/D; coupled
/// 0.069 / 0.132 / 0.241 / 0.412 at 4/8/16/32 iters @1/7200, halving at
/// 1/3600). On THIS geometry the residual tracks (D−1)/D — the L2 trigger.
///
/// MEASURED (2026-10-21, L2 warm-start pair landed — the current path):
/// the same sweep transmits the full demand at EVERY point of the grid:
/// fraction 0.9998 / 0.9999 / 0.9999 / 1.0000 at 4/8/16/32 iters @1/7200
/// and 1.0020 / 1.0009 / 1.0003 / 1.0001 @1/3600 (legacy pre-pass
/// unchanged: 0.0090 / 0.0045). The design's §2.6 claim — the
/// cross-substep fixed point Λ* = dt·τ₀ is exact for ANY alignment once
/// BOTH the joint lambdas and the manifold lambdas carry — is verified by
/// measurement on the worst-case geometry.
/// BITE-VERIFIED (2026-10-20): a panic planted at the harness head fails
/// the --ignored run (the harness genuinely executes end to end).
#[test]
#[ignore = "diagnostic harness (design E6) — run with --ignored --nocapture"]
fn e6_transmitted_torque_fraction_vs_solver_iterations() {
    let kp = 2.0;
    let target = 0.6;

    let run = |iters: usize, sub_dt: f64, coupled: bool| -> (f64, f64, f64) {
        let mut world = XpbdWorld::new();
        world.coupled_joint_solve = coupled;
        world.solver_iterations = iters;
        world.set_ground(0.0);

        let r = 0.004;
        let half_h = 0.008;
        let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
        let mut parent = RigidBody::new(0.0, Mat3::identity());
        parent.orientation = along_z;
        parent.position = Vec3::new(0.0, r, -0.03);
        let p = world.add_body(parent);
        let mut child = RigidBody::new_capsule(0.003, half_h, r);
        child.orientation = along_z;
        child.position = Vec3::new(0.0, r, 0.0);
        let ch = world.add_body(child);

        // Hinge at the proximal (−z) endpoint; rotation stays free only
        // about x (planar rig, no off-axis excitation to lock).
        let joint_world = Vec3::new(0.0, r, -half_h);
        let mut c = PositionalConstraint::new(p, ch, 0.0, 1.0 / 60_000.0);
        c.local_anchor_a =
            world.bodies[p].orientation.inverse() * (joint_world - world.bodies[p].position);
        c.local_anchor_b =
            world.bodies[ch].orientation.inverse() * (joint_world - world.bodies[ch].position);
        world.constraints.push(c);

        world.joint_axes.push(
            JointAxis::new(p, ch, Vec3::x())
                .with_pd(kp, 0.05)
                .with_target(target)
                .with_passive_damping(0.01),
        );

        let total = (1.0 / sub_dt).round() as usize;
        let window = (0.1 / sub_dt).round() as usize;
        let mut sum_tx = 0.0;
        let mut sum_demand = 0.0;
        let mut samples = 0usize;
        for i in 0..total {
            world.substep(sub_dt);
            if i >= total - window {
                let theta = joint_angle(&world, p, ch, Vec3::x());
                sum_tx += world.joint_axes[0].applied_total_torque;
                sum_demand += kp * (target - theta);
                samples += 1;
            }
        }
        (
            sum_tx / samples as f64,
            sum_demand / samples as f64,
            world.total_normal_force(ch),
        )
    };

    for &sub_dt in &[SUB_DT_FINE, SUB_DT_COARSE] {
        println!("── sub_dt = 1/{:.0} ──", 1.0 / sub_dt);
        let (tx, demand, f_n) = run(8, sub_dt, false);
        println!(
            "  legacy pre-pass @ 8 iters: transmitted {tx:+.4e} N·m, demand {demand:+.4e} N·m, \
             fraction {:+.4}, ΣF_n {f_n:.2} N",
            tx / demand
        );
        for &iters in &[4usize, 8, 16, 32] {
            let (tx, demand, f_n) = run(iters, sub_dt, true);
            let frac = tx / demand;
            println!(
                "  coupled      @{iters:>2} iters: transmitted {tx:+.4e} N·m, demand \
                 {demand:+.4e} N·m, fraction {frac:+.4}, ΣF_n {f_n:.2} N"
            );
            assert!(frac.is_finite(), "E6 produced a non-finite fraction");
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// E1/E2/E5 — the Commit B acceptance gates (design §4)
// ════════════════════════════════════════════════════════════════════════

/// E1 rig (design §4): pinned proximal body; one JointAxis with kp sized so
/// the pressed static demand is ~2.4 N·m; a toe-scale child capsule whose
/// DISTAL tip rests on the ground, hinged at its proximal end-sphere center
/// through a stiff positional constraint (test-fixture stiffness 1/1e6,
/// the engine_regression weld class, so the hinge sag under the ~100 N
/// reaction is ~0.1 mm and the lever stays crisp); lever = horizontal
/// contact↔hinge distance 2·half_h·cos φ₀ ≈ 23.7 mm, so τ/l ≈ 100 N.
///
/// The proximal end-sphere center is ELEVATED by 4 mm above the distal one
/// (the capsule is pitched nose-down by φ₀ = asin(4/24) ≈ 0.1674 rad), so
/// ONLY the distal tip contacts the ground: the whole transmitted torque
/// reacts through that single contact and the lever statics are exact —
/// a flat-lying toe would put its proximal contact directly under load and
/// split ΔF_n across two points.
///
/// Gains are the foot family's (kp 4 / kd 0.15, passive spring 2 toward
/// φ₀, passive damper c = 1.0). On this child (i_trans ≈ 2.6e-7 kg·m²,
/// w ≈ 3.8e6) the damping term makes the pre-pass denominator
/// D = 1 + dt·w·(kd+c) ≈ 616 at sub_dt = 1/7200 — deep in the measured
/// force-transmission-gap regime.
struct E1Rig {
    world: XpbdWorld,
    parent: usize,
    child: usize,
    /// Initial pitch = the neutral target and the passive-spring rest.
    phi0: f64,
    /// Child-local hinge anchor (for reading the hinge's world position).
    anchor_b: Vec3,
}

const E1_HALF_H: f64 = 0.012;
const E1_R: f64 = 0.004;
const E1_MASS: f64 = 0.005;
const E1_ELEV: f64 = 0.004;
const E1_KP: f64 = 4.0;
const E1_PRESS: f64 = 0.6; // pressed target = φ₀ + 0.6 ⇒ demand ≈ 2.4 N·m

fn e1_rig(coupled: bool, iters: usize) -> E1Rig {
    let phi0 = (E1_ELEV / (2.0 * E1_HALF_H)).asin();
    let mut world = XpbdWorld::new();
    world.coupled_joint_solve = coupled;
    world.solver_iterations = iters;
    world.set_ground(0.0);

    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let pitched = Quat::from_scaled_axis(Vec3::new(phi0, 0.0, 0.0)) * along_z;
    // Capsule long axis (local +Y) in world: nose-down toward +z.
    let axis_dir = Vec3::new(0.0, -phi0.sin(), phi0.cos());
    let hinge_world = Vec3::new(0.0, E1_R + E1_ELEV, 0.0);

    let mut parent = RigidBody::new(0.0, Mat3::identity());
    parent.orientation = along_z;
    parent.position = hinge_world;
    let p = world.add_body(parent);

    let mut child = RigidBody::new_capsule(E1_MASS, E1_HALF_H, E1_R);
    child.orientation = pitched;
    // Center one half-length down-axis from the hinge: the distal
    // end-sphere center lands at y = r exactly (bottom on the plane).
    child.position = hinge_world + axis_dir * E1_HALF_H;
    let ch = world.add_body(child);

    let mut c = PositionalConstraint::new(p, ch, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a =
        world.bodies[p].orientation.inverse() * (hinge_world - world.bodies[p].position);
    c.local_anchor_b =
        world.bodies[ch].orientation.inverse() * (hinge_world - world.bodies[ch].position);
    let anchor_b = c.local_anchor_b;
    world.constraints.push(c);

    world.joint_axes.push(
        JointAxis::new(p, ch, Vec3::x())
            .with_pd(E1_KP, 0.15)
            .with_target(phi0)
            .with_passive_spring(2.0, phi0)
            .with_passive_damping(1.0),
    );

    E1Rig { world, parent: p, child: ch, phi0, anchor_b }
}

/// One settled phase: run `seconds`, then report means over the last 0.1 s
/// of θ, `applied_total_torque`, the state-computed static demand
/// kp·(target − θ) + ff − K_sp·(θ − rest) (+ s_lim(θ) ≡ 0: no limit on this
/// rig), and ΣF_n on the child — plus the lever (horizontal distal-sphere-
/// center ↔ hinge distance) at the end of the window.
struct E1Phase {
    theta: f64,
    tau_tx: f64,
    demand: f64,
    f_n: f64,
    lever: f64,
}

fn e1_run_phase(rig: &mut E1Rig, sub_dt: f64, seconds: f64) -> E1Phase {
    let total = (seconds / sub_dt).round() as usize;
    let window = (0.1 / sub_dt).round() as usize;
    let (mut s_th, mut s_tx, mut s_dm, mut s_fn) = (0.0, 0.0, 0.0, 0.0);
    let mut samples = 0usize;
    for i in 0..total {
        rig.world.substep(sub_dt);
        if i >= total - window {
            let theta = joint_angle(&rig.world, rig.parent, rig.child, Vec3::x());
            let ja = &rig.world.joint_axes[0];
            s_th += theta;
            s_tx += ja.applied_total_torque;
            s_dm += ja.kp * (ja.target_angle - theta) + ja.feedforward_torque
                - ja.passive_stiffness * (theta - ja.passive_rest_angle);
            s_fn += rig.world.total_normal_force(rig.child);
            samples += 1;
        }
    }
    let b = &rig.world.bodies[rig.child];
    let distal = b.position + b.orientation * Vec3::new(0.0, E1_HALF_H, 0.0);
    let hinge = b.position + b.orientation * rig.anchor_b;
    let n = samples as f64;
    E1Phase {
        theta: s_th / n,
        tau_tx: s_tx / n,
        demand: s_dm / n,
        f_n: s_fn / n,
        lever: distal.z - hinge.z,
    }
}

/// Neutral settle (1 s at target φ₀), then pressed settle (1 s at
/// φ₀ + 0.6): the two phases E1 measures.
fn e1_two_phases(coupled: bool, iters: usize, sub_dt: f64) -> (E1Phase, E1Phase, E1Rig) {
    let mut rig = e1_rig(coupled, iters);
    let neutral = e1_run_phase(&mut rig, sub_dt, 1.0);
    let press_target = rig.phi0 + E1_PRESS;
    rig.world.joint_axes[0].target_angle = press_target;
    let pressed = e1_run_phase(&mut rig, sub_dt, 1.0);
    (neutral, pressed, rig)
}

/// E1 — static force transmission (design §4, the ~100 N check), flag ON.
///
/// Assertions, verbatim from the design:
///   (i)   `applied_total_torque` vs the state-computed static demand,
///         within 10% (asserted at both rates);
///   (ii)  Δ(total normal force) between the phases vs Δτ_measured/lever,
///         within 10% (the Δ form cancels the weight share exactly — the
///         neutral-phase moment balance subtracts out — leaving friction
///         cross-talk as the reason this band is 10%, not 2%);
///   (iii) the 1/7200 and 1/3600 runs agree within 2% (on the pressed
///         transmitted torque and on ΔF_n).
///
/// FLAG-OFF RECEIPT (2026-10-21, measured via the receipts harness below):
/// the legacy pre-pass transmits 0.0039 of the 2.4000 N·m demand at 1/7200
/// (fraction 0.0016 ≈ 1/D with D = 1 + dt·w·(kd+c) ≈ 616) and 0.0019 at
/// 1/3600 (0.0008 — halving with dt, the damping-dominated signature);
/// ΔF_n = 0.17 N where the lever statics demand ~101 N. The documented
/// force-transmission gap at full strength; flipping this gate's builders
/// to coupled = false reproduces exactly those numbers and fails (i).
///
/// FLAG-ON COLD-START MISS (2026-10-21, 8 iters): fraction 0.0255 at
/// 1/7200 / 0.0128 at 1/3600 (0.0502/0.0254 at 16 iters, 0.0971/0.0501 at
/// 32) — this rig is in E6's near-degenerate alignment class (pinned
/// proximal + grounded child, a → 1): the per-sweep residual tracks
/// (D−1)/D with D ≈ 616, so the cold-start fraction is ~n_iter/D and L1
/// (16 iterations) does not close the 10% band. The gate landed
/// #[ignore]'d on that miss, pending the L2 warm-start pair (design §2.6),
/// whose cross-substep fixed point is exact for any alignment; L2 is now
/// in and the ignore is lifted. NEVER weaken the bands instead.
///
/// MEASURED AFTER L2 (2026-10-21, joint + manifold warm-start pair):
/// fraction 1.0000 at both rates and at 8/16/32 iterations; ΔF_n 101.42 N
/// = Δτ/l 101.42 N; (iii) cross-rate agreement to the printed precision.
/// BITE-VERIFIED (2026-10-21), both L2 sides: (1) with the MANIFOLD carry
/// disabled (joint seed alone) this test fails at fraction 0.0494 —
/// measuring §2.6's derivation that the joint-side seed cannot accumulate
/// across substeps while the contact side forgets; (2) a warm_start that
/// double-applies the carried contact impulse fails this test (E5 stays
/// green under that bug: the over-application is unwound within the first
/// sweep by the λ ≥ 0 clamp without pumping — E1's statics catch it, not
/// the energy canary).
#[test]
fn e1_static_force_transmission() {
    let (n_f, p_f, _) = e1_two_phases(true, 8, SUB_DT_FINE);
    let (n_c, p_c, _) = e1_two_phases(true, 8, SUB_DT_COARSE);

    for (n, p, rate) in [(&n_f, &p_f, "1/7200"), (&n_c, &p_c, "1/3600")] {
        println!(
            "E1 @{rate}: pressed θ {:+.4} rad, transmitted {:+.4} N·m, demand {:+.4} N·m \
             (fraction {:+.4}), ΔF_n {:+.2} N, Δτ/l {:+.2} N (lever {:.4} m)",
            p.theta, p.tau_tx, p.demand, p.tau_tx / p.demand,
            p.f_n - n.f_n, (p.tau_tx - n.tau_tx) / p.lever, p.lever
        );
        // (i) transmitted torque = static demand within 10%.
        assert!(
            (p.tau_tx - p.demand).abs() <= 0.10 * p.demand.abs(),
            "E1(i) @{rate}: transmitted {} N·m vs demand {} N·m (> 10% off)",
            p.tau_tx, p.demand
        );
        // (ii) the contact carries the lever reaction: ΔF_n = Δτ/l ± 10%.
        let d_fn = p.f_n - n.f_n;
        let expect = (p.tau_tx - n.tau_tx) / p.lever;
        assert!(
            (d_fn - expect).abs() <= 0.10 * expect.abs(),
            "E1(ii) @{rate}: ΔF_n {d_fn} N vs Δτ/l {expect} N (> 10% off)"
        );
    }
    // (iii) cross-sub_dt agreement within 2%.
    assert!(
        (p_f.tau_tx - p_c.tau_tx).abs() <= 0.02 * p_f.tau_tx.abs(),
        "E1(iii): transmitted torque differs across sub_dt: {} vs {} N·m (> 2%)",
        p_f.tau_tx, p_c.tau_tx
    );
    let (dfn_f, dfn_c) = (p_f.f_n - n_f.f_n, p_c.f_n - n_c.f_n);
    assert!(
        (dfn_f - dfn_c).abs() <= 0.02 * dfn_f.abs(),
        "E1(iii): ΔF_n differs across sub_dt: {dfn_f} vs {dfn_c} N (> 2%)"
    );
}

/// Total mechanical energy of E1's rig: child KE (linear + rotational) +
/// gravitational PE + the joint's elastic PE (passive spring; no limit on
/// this rig). The hinge constraint's elastic energy is omitted: its
/// deflection is force-bound (~0.1 mm at 100 N), so its stored energy is
/// ~constant at the settled phase and a pump would have to surface in the
/// terms counted here.
fn e1_mechanical_energy(world: &XpbdWorld, child: usize) -> f64 {
    let b = &world.bodies[child];
    let mut e = 0.5 * b.mass * b.linear_velocity.dot(&b.linear_velocity);
    let r = b.orientation.to_rotation_matrix();
    let i_world = r.matrix() * b.inertia_tensor * r.matrix().transpose();
    e += 0.5 * b.angular_velocity.dot(&(i_world * b.angular_velocity));
    e += b.mass * 9.81 * b.position.y;
    e += world.joint_axes[0].potential_energy(&world.bodies);
    e
}

/// E5 addition (design §4): 5 s of the PRESSED phase on E1's rig with the
/// total mechanical energy sampled every 10 ms — no pump under the
/// sustained honest contact fight. After the 1 s pressed settle the system
/// is static (the tip is wedged; the PD does no work on a stationary
/// joint), so mechanical energy may only dissipate: every sample must stay
/// within E5_BOUND of the running minimum (the running-min form also
/// catches a pump that hides under earlier dissipation).
///
/// BOUND DERIVATION: the house energy canary allows +1 mJ on a ~0.1 J-scale
/// scene (engine_regression::settling_chain_does_not_gain_energy — 1% of
/// scene scale). This rig's mechanical-energy scale is W·l ≈
/// 0.005·9.81·0.024 ≈ 1.2e-3 J; 1% of that is ~1.2e-5 J, so the bound is
/// 1e-5 J (10 µJ) — the same relative tightness as the house canary, and
/// ≥ 10⁶× above the settled micro-jitter KE (½·I·ω² ≈ 1e-13 J at the
/// measured post-speculative jitter scale).
///
/// MEASURED (2026-10-21, flag on, cold start): max rise 0.000e0 J — the
/// sampled energy is non-increasing at every one of the 500 samples.
/// BITE-VERIFIED (2026-10-21): an anti-damper injector (passive_damping
/// set to −1.0 at the window start — energy injection through the
/// joint+contact loop by construction) pumps 10.95 J and fails this test.
/// Note: the manifold-Baumgarte-0.2 injector does NOT trip this rig (one
/// near-coplanar contact; the documented levitation pump needs a
/// non-coplanar multi-contact manifold), hence the anti-damper bite.
const E5_BOUND: f64 = 1e-5;

#[test]
fn e5_pressed_phase_energy_is_monotone() {
    let mut rig = e1_rig(true, 8);
    let sub_dt = SUB_DT_FINE;
    e1_run_phase(&mut rig, sub_dt, 1.0);
    rig.world.joint_axes[0].target_angle = rig.phi0 + E1_PRESS;
    e1_run_phase(&mut rig, sub_dt, 1.0); // pressed settle
    let mut e_min = e1_mechanical_energy(&rig.world, rig.child);
    let mut max_rise: f64 = 0.0;
    let sample_every = (0.01 / sub_dt).round() as usize;
    let total = (5.0 / sub_dt).round() as usize;
    for i in 0..total {
        rig.world.substep(sub_dt);
        if (i + 1) % sample_every == 0 {
            let e = e1_mechanical_energy(&rig.world, rig.child);
            if e - e_min > max_rise {
                max_rise = e - e_min;
            }
            if e < e_min {
                e_min = e;
            }
        }
    }
    println!("E5: max rise above running min {max_rise:.3e} J over 5 s pressed");
    assert!(
        max_rise < E5_BOUND,
        "E5: pressed-phase mechanical energy rose {max_rise} J above its running \
         minimum (> {E5_BOUND} J) — energy pump under the sustained contact fight"
    );
}

/// E2 rig (design §4): pinned "shin"; one foot capsule (m = 1 kg,
/// half_h = 0.1, r = 0.02) lying along +z with both end-sphere bottoms on
/// the plane; ankle JointAxis about x. The ankle's positional anchor is
/// placed at the BALL (distal, +z) end-sphere center: every force acting
/// at that z — the anchor reaction AND the ball's normal — drops out of
/// the x-moment balance, so the heel-unload threshold is exact statics:
///
///     τ* = W·(z_ball − z_com) = m·g·half_h = 0.981 N·m.
///
/// Positive θ about +x presses the ball down / lifts the heel, and the
/// heel contact only blocks the heel-DOWN direction, so once τ > τ* the
/// heel genuinely leaves the ground and the joint settles at the free
/// equilibrium kp·(target − θ) = W·half_h·cos θ.
///
/// Gains kp = 20 / kd = 1 / passive damper c = 4: deliberately sized so
/// the pre-pass denominator D = 1 + dt·w·(kd + c) (w = 1/i_trans ≈ 291)
/// STRADDLES the 1.25 command ratio across the two rates — D ≈ 1.20 at
/// 1/7200, ≈ 1.40 at 1/3600 — the scenario class whose OUTCOME flips with
/// sub_dt on the legacy path.
fn e2_run(coupled: bool, iters: usize, cmd_frac: f64, sub_dt: f64) -> (bool, f64, f64) {
    const M: f64 = 1.0;
    const HALF_H: f64 = 0.1;
    const R: f64 = 0.02;
    let tau_star = M * 9.81 * HALF_H;
    let kp = 20.0;

    let mut world = XpbdWorld::new();
    world.coupled_joint_solve = coupled;
    world.solver_iterations = iters;
    world.set_ground(0.0);
    // The ball contact is the rig's rotation PIVOT: as the foot
    // plantarflexes about the anchor the ball sphere spins in place, and
    // the contact's friction couple (up to ±μ·N·R about x) adds a REAL
    // hysteresis band of ±μ·N·R/kp to the settled lift angle. At the
    // default μ = 0.6 that band is ±5.9e-3 rad — ±48% of the 0.0123 rad
    // lift, and the settled height was MEASURED landing anywhere inside it
    // (2.6 mm vs 1.5 mm depending on rate and iteration count: genuine
    // friction path-dependence, not a solver artifact). The gate's claim
    // is the dt-independence of the joint+contact solve, not the width of
    // a friction-hysteresis band, so the rig shrinks the band to ±4%
    // (μ = 0.05 ⇒ ±4.9e-4 rad) — 2.5× inside the 10% height assertion.
    world.ground.friction_coefficient = 0.05;

    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let ball_center = Vec3::new(0.0, R, HALF_H);

    let mut shin = RigidBody::new(0.0, Mat3::identity());
    shin.orientation = along_z;
    shin.position = Vec3::new(0.0, 0.25, HALF_H);
    let p = world.add_body(shin);

    let mut foot = RigidBody::new_capsule(M, HALF_H, R);
    foot.orientation = along_z;
    foot.position = Vec3::new(0.0, R, 0.0);
    let f = world.add_body(foot);

    let mut c = PositionalConstraint::new(p, f, 0.0, 1.0 / 60_000.0);
    c.local_anchor_a =
        world.bodies[p].orientation.inverse() * (ball_center - world.bodies[p].position);
    c.local_anchor_b =
        world.bodies[f].orientation.inverse() * (ball_center - world.bodies[f].position);
    world.constraints.push(c);

    world.joint_axes.push(
        JointAxis::new(p, f, Vec3::x())
            .with_pd(kp, 1.0)
            .with_passive_damping(4.0)
            .with_target(cmd_frac * tau_star / kp),
    );

    // 2 s settle (slow pole ≈ kp/(kd+c) = 4/s ⇒ 8 time constants), then
    // means over the last 0.2 s.
    let total = (2.0 / sub_dt).round() as usize;
    let window = (0.2 / sub_dt).round() as usize;
    let (mut s_clear, mut s_heel_n) = (0.0, 0.0);
    let mut samples = 0usize;
    for i in 0..total {
        world.substep(sub_dt);
        if i >= total - window {
            let b = &world.bodies[f];
            let heel_center = b.position + b.orientation * Vec3::new(0.0, -HALF_H, 0.0);
            s_clear += heel_center.y - R;
            // Slot 1 = the −local-Y endpoint = the heel (−z).
            s_heel_n += world.engaged_endpoints(f)[1]
                .map(|e| e.normal_force)
                .unwrap_or(0.0);
            samples += 1;
        }
    }
    let clearance = s_clear / samples as f64;
    let heel_n = s_heel_n / samples as f64;
    // Lift classification: the predicted lifted clearance is ~2.4 mm and
    // the grounded one is ±µm, so 0.5 mm splits the outcomes with ≥ 4×
    // margin on either side.
    (clearance > 5e-4, clearance, heel_n)
}

/// E2 — grounded dt-independence (heel-lift threshold), flag ON.
/// Commands realize 0.8·τ* (must NOT lift, heel stays loaded at both
/// rates) and 1.25·τ* (must lift at both rates, settled heel height within
/// 10% across rates). The lift outcomes are also asserted against the
/// statics-derived truth, not merely against each other.
///
/// MEASURED (2026-10-21, flag on, cold start, 8 iters): 0.8·τ* stays
/// grounded with heel load 0.961 N / 0.976 N across rates (statics:
/// (τ* − 0.8τ*)/0.2 m = 0.98 N); 1.25·τ* lifts to 2.377e-3 / 2.376e-3 m —
/// 0.05% apart, and within 3% of the frictionless equilibrium
/// kp·(target − θ) = τ*·cos θ ⇒ 2.45 mm. This rig's alignment is benign
/// (D ≈ 1.2 ⇒ per-sweep residual ≤ 0.17, fully converged in 8 sweeps), so
/// the gate passes at cold start — no lever needed.
///
/// FLAG-OFF RECEIPT (2026-10-21): the outcome flips qualitatively with
/// sub_dt, as the design predicts for this scenario class: at 1/7200 the
/// 1.25·τ* command unloads the heel (0.000 N) and creeps it to 4.6e-4 m
/// (still rising at 2 s), while at 1/3600 the heel stays HARD GROUNDED
/// under 0.543 N — the pre-pass denominator D straddles the 1.25 command
/// ratio (1.20 vs 1.40). BITE-VERIFIED (2026-10-21): with this gate's
/// builders flipped to coupled = false it fails the 1.25·τ* lift
/// assertions at both rates.
#[test]
fn e2_heel_lift_threshold_is_dt_independent() {
    for &(frac, should_lift) in &[(0.8, false), (1.25, true)] {
        let (lift_f, clear_f, heel_f) = e2_run(true, 8, frac, SUB_DT_FINE);
        let (lift_c, clear_c, heel_c) = e2_run(true, 8, frac, SUB_DT_COARSE);
        println!(
            "E2 @{frac}·τ*: fine lift {lift_f} (clear {clear_f:+.3e} m, heel {heel_f:.3} N), \
             coarse lift {lift_c} (clear {clear_c:+.3e} m, heel {heel_c:.3} N)"
        );
        assert_eq!(
            lift_f, should_lift,
            "E2 @{frac}·τ* at 1/7200: lift = {lift_f}, statics says {should_lift} \
             (clearance {clear_f} m)"
        );
        assert_eq!(
            lift_c, should_lift,
            "E2 @{frac}·τ* at 1/3600: lift = {lift_c}, statics says {should_lift} \
             (clearance {clear_c} m)"
        );
        if should_lift {
            assert!(
                (clear_f - clear_c).abs() <= 0.10 * clear_f.abs(),
                "E2: lifted heel height differs across rates: {clear_f} vs {clear_c} m (> 10%)"
            );
        } else {
            // The non-lifting heel stays loaded (λ_n > 0) at both rates.
            assert!(
                heel_f > 0.0 && heel_c > 0.0,
                "E2 @{frac}·τ*: heel unloaded (λ_n = 0) — fine {heel_f} N, coarse {heel_c} N"
            );
        }
    }
}

/// L2-carry friction-seed cone projection: setting ground μ to 0 (or any
/// lower value) mid-sim must re-clamp the CARRIED tangential seed, never
/// ratchet it.
///
/// THE HOLE THIS PINS: `ContactConstraint::warm_start` seeds
/// `lambda_friction` from the cross-substep carry, and `solve` returns at
/// `friction_coefficient <= 0` BEFORE its release-projection branch, while
/// the carry filter checks only `lambda_normal > 0`. Unprojected, a
/// friction impulse carried into a μ = 0 world is applied as a real
/// tangential impulse by warm_start every substep and never unwound or
/// re-clamped by any sweep — unbounded injection (measured below). The fix
/// cone-projects the seed in warm_start against the constraint's CURRENT
/// cone: μ ≤ 0 zeroes it, μ > 0 clamps it to μ·λ_n.
///
/// Rig: a 1 kg capsule slides along its own axis (+x, the friction trio's
/// pure-sliding geometry) at μ = 0.6 for 20 substeps, so the manifold
/// carries a kinetic-friction impulse λ_t ≈ μ·m·g·dt ≈ 8.2e-4 N·s per
/// endpoint; then ground μ is set to 0 and it coasts 0.5 s.
///
/// BOUND DERIVATION: with μ = 0 the ground can exert only +y (normal)
/// impulses and gravity is vertical, so the horizontal speed after the flip
/// is CONSERVED up to fp roundoff — the physics bound is |v| ≤ v_flip
/// exactly; 1% headroom is ~13 orders above the roundoff scale. The bug
/// fails it by construction: re-applying the carried λ_t each substep walks
/// the velocity by λ_t/m ≈ 8.2e-4 m/s per substep, linear in time and
/// unbounded (MEASURED at the pre-fix HEAD: max horizontal speed 1.9594 m/s
/// over the 0.5 s window on a 0.9837 m/s flip speed — the capsule reverses
/// and accelerates backward, 2× the bound in half a second and growing).
///
/// BITE-VERIFIED (2026-10-21): with warm_start's cone projection reverted
/// (seeding `lambda_friction` raw), this test fails at max speed 1.9594 m/s;
/// projection restored, it passes.
#[test]
fn carried_friction_seed_is_cone_projected_when_mu_drops() {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    assert_eq!(world.ground.friction_coefficient, 0.6, "rig assumes default μ = 0.6");

    let radius = 0.03;
    let mut cap = RigidBody::new_capsule(1.0, 0.15, radius);
    // Local +Y onto world +X: slide along the capsule axis (pure sliding).
    cap.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    cap.position = Vec3::new(0.0, radius, 0.0);
    cap.linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    let b = world.add_body(cap);

    // Slide at μ = 0.6: kinetic friction accumulates and the L2 carry
    // seeds it into every next substep.
    for _ in 0..20 {
        world.substep(SUB_DT_FINE);
    }
    let v_flip = world.bodies[b].linear_velocity.x;
    assert!(v_flip > 0.9, "rig broken: capsule stopped before the μ flip (v = {v_flip})");
    assert!(
        world
            .ground_manifold
            .iter()
            .any(|c| c.lambda_normal > 0.0 && c.lambda_friction.norm() > 1e-6),
        "rig broken: no friction impulse is being carried at the μ flip"
    );

    world.ground.friction_coefficient = 0.0;
    let mut max_speed: f64 = 0.0;
    let total = (0.5 / SUB_DT_FINE).round() as usize;
    for _ in 0..total {
        world.substep(SUB_DT_FINE);
        let v = world.bodies[b].linear_velocity;
        let h_speed = (v.x * v.x + v.z * v.z).sqrt();
        if h_speed > max_speed {
            max_speed = h_speed;
        }
    }
    println!("friction-seed ratchet: v_flip {v_flip:.4} m/s, max |v_h| after μ→0 {max_speed:.4} m/s");
    assert!(
        max_speed <= 1.01 * v_flip,
        "carried friction seed injected tangential impulse on a μ = 0 ground: \
         max horizontal speed {max_speed} m/s > v_flip {v_flip} m/s — the warm-start \
         seed must be cone-projected (zeroed at μ ≤ 0)"
    );
}

/// DIAGNOSTIC harness for the E1/E2 receipts and the L1 iteration sweep —
/// run with `cargo test -p physics --test coupled_joint -- --ignored
/// --nocapture`. Prints E1 and E2 at flag off/on × solver_iterations
/// {8, 16, 32} × both sub_dts. The flag-off rows are the Commit B receipts;
/// the 16-iteration rows are the design's L1 trigger measurement.
#[test]
#[ignore = "diagnostic harness (E1/E2 receipts + L1 sweep) — run with --ignored --nocapture"]
fn e1_e2_receipts_and_iteration_sweep() {
    for &sub_dt in &[SUB_DT_FINE, SUB_DT_COARSE] {
        println!("── sub_dt = 1/{:.0} ──", 1.0 / sub_dt);
        for &(coupled, iters) in &[(false, 8usize), (true, 8), (true, 16), (true, 32)] {
            let tag = if coupled { "coupled" } else { "legacy " };
            let (n, p, _) = e1_two_phases(coupled, iters, sub_dt);
            println!(
                "  E1 {tag}@{iters:>2}: τ_tx {:+.4} vs demand {:+.4} N·m (fraction {:+.4}), \
                 ΔF_n {:+.2} N vs Δτ/l {:+.2} N",
                p.tau_tx, p.demand, p.tau_tx / p.demand,
                p.f_n - n.f_n, (p.tau_tx - n.tau_tx) / p.lever
            );
            for &frac in &[0.8, 1.25] {
                let (lift, clear, heel_n) = e2_run(coupled, iters, frac, sub_dt);
                println!(
                    "  E2 {tag}@{iters:>2} @{frac}·τ*: lift {lift} \
                     (clearance {clear:+.3e} m, heel {heel_n:.3} N)"
                );
            }
        }
    }
}
