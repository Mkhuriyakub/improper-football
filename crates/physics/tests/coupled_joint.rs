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
/// MEASURED: max|Δθ| 7.8e-16, max|Δω| 1.2e-13 over 600 substeps — the
/// divergence is NOT accumulating (the contraction wins), five orders
/// inside the linear bound. BITE-VERIFIED (2026-10-20): the 0.999 impulse
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
/// MEASURED (2026-10-20, Commit A baseline): this rig is close to the
/// design's WORST-CASE alignment (the parent is pinned, so the tip contact
/// blocks the joint's only freedom almost exactly through the mass
/// matrix — §2.6's a → 1). The cold-start fraction behaves exactly as
/// that analysis predicts: ~linear in iterations and ~linear in dt
/// (legacy pre-pass 0.0090 ≈ 1/D; coupled 0.069 / 0.132 / 0.241 / 0.412
/// at 4/8/16/32 iters @1/7200, halving at 1/3600). On THIS geometry the
/// residual tracks (D−1)/D, i.e. the L2 warm-start pair — whose fixed
/// point is exact for any alignment — is the lever that closes it; the
/// real foot's chain alignments (a ≈ 0.4–0.5 estimated) are expected far
/// better, which E1/E2 on the Commit B rigs will measure.
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
