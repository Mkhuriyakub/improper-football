//! The coupled joint+contact solve (design contract:
//! docs/superpowers/specs/2026-10-21-coupled-joint-contact-solve.md): the
//! in-loop JointAxis lifecycle — the engine's ONLY joint path since
//! Commit D removed the `coupled_joint_solve` flag and the pre-pass slot
//! (the legacy pre-pass pipeline kept the documented force-transmission
//! gap under contact; its receipts are preserved, re-labeled HISTORICAL,
//! in the doc comments below). `JointAxis::apply_implicit` survives as
//! the documented single-solve primitive, and the A–D tests drive it
//! DIRECTLY on a reference copy of each rig (see `ReferencePath`).
//! The A–D tests pin the free-space guarantee (design §2.4): the
//! in-loop form lands on `apply_implicit`'s fixed point —
//!   A  — single-substep equivalence, unsaturated 2×2 path;
//!   B  — single-substep equivalence, saturated branch (accumulated clamp,
//!        passive re-solve, bit-exact σ·τ_max diagnostic);
//!   C  — degenerate-γ branches (pure feedforward / pure ext_torque
//!        constant-impulse rows, including the clamped constant);
//!   D  — multi-substep free-space trajectory equivalence (E3 precursor);
//!   E6 — #[ignore] convergence-diagnostic harness on a grounded rig.
//!
//! Commit B added the acceptance gates (design §4), asserted on the
//! coupled path with their (now historical) flag-off receipts recorded in
//! each doc comment:
//!   E1 — static force transmission through a grounded toe tip (~100 N);
//!        landed #[ignore]'d at cold start (its near-degenerate rig missed
//!        the 10% band — miss recorded in-place) and un-ignored once the
//!        L2 warm-start lever closed it;
//!   E2 — heel-lift threshold dt-independence (passes at cold start);
//!   E5 — pressed-phase mechanical-energy monotonicity (no pump under the
//!        sustained contact fight);
//!   plus an #[ignore] receipts/L1-sweep harness.
//!
//! The L2 warm-start pair's cross-primitive gates (2026-10-21, the
//! pre-Commit-C set): the two MIXED-PRIMITIVE canaries (a grounded
//! JointAxis press sharing a body with legacy primitives — the warm seed
//! must apply AFTER the legacy implicit passes, at the contact-reset slot)
//! and the carried-friction-seed cone projection.
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

use physics::angular_damper::AngularDamper;
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

/// Reference evolution of a free-space rig that drives
/// `JointAxis::apply_implicit` DIRECTLY — the documented single-solve
/// primitive, the executable spec of the free-space fixed point (design
/// §2.4). The joint lives OUTSIDE the world (the world holds the same
/// bodies and nothing else), and each reference substep calls
/// `apply_implicit` by hand and then advances the empty world. On these
/// rigs that is BIT-IDENTICAL to the deleted pre-pass pipeline (Commit D
/// removed the flag and the in-substep pre-pass slot): there are no
/// torque accumulators, so `integrate_velocities` never touches angular
/// velocity, and positions are frozen until `integrate_positions` — the
/// hand-driven solve reads exactly the state the in-substep slot read, so
/// the A–D comparisons keep their teeth (and their recorded MEASURED
/// values) unchanged.
struct ReferencePath {
    world: XpbdWorld,
    joint: JointAxis,
}

impl ReferencePath {
    fn substep(&mut self, sub_dt: f64) {
        self.joint.apply_implicit(&mut self.world.bodies, sub_dt);
        self.world.substep(sub_dt);
    }
}

/// Build the SAME free-space rig twice from identical body states and one
/// identical JointAxis: the engine path (in-loop lifecycle, the world's
/// only path) and the `apply_implicit` reference (see `ReferencePath`).
fn equivalence_pair(
    make_joint: &dyn Fn(usize, usize) -> JointAxis,
    child_omega_z: f64,
) -> (ReferencePath, XpbdWorld) {
    let build = || -> (XpbdWorld, usize, usize) {
        let mut world = XpbdWorld::new();
        let (p, ch) = pinned_pair(&mut world, 0.05, 1e-6);
        world.bodies[ch].angular_velocity = Vec3::new(0.0, 0.0, child_omega_z);
        (world, p, ch)
    };
    let (ref_world, p, ch) = build();
    let reference = ReferencePath { world: ref_world, joint: make_joint(p, ch) };
    let (mut on, p, ch) = build();
    on.joint_axes.push(make_joint(p, ch));
    (reference, on)
}

/// Max-divergence report after running both paths one substep: child
/// angular velocity, total-torque diagnostic, active-torque diagnostic.
fn single_substep_divergence(
    reference: &mut ReferencePath,
    on: &mut XpbdWorld,
    sub_dt: f64,
) -> (f64, f64, f64) {
    reference.substep(sub_dt);
    on.substep(sub_dt);
    let dv =
        (on.bodies[1].angular_velocity - reference.world.bodies[1].angular_velocity).norm();
    let d_total =
        (on.joint_axes[0].applied_total_torque - reference.joint.applied_total_torque).abs();
    let d_active =
        (on.joint_axes[0].applied_active_torque - reference.joint.applied_active_torque).abs();
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
/// by 0.999 fails this test (and B, C, D). RE-RUN (2026-10-29, Commit D)
/// against the `ReferencePath` restructure (the flag-off world replaced by
/// the direct apply_implicit reference): the same 0.999 scaling fails all
/// four of A/B/C/D; restored, green, with every MEASURED value in this
/// file reproduced exactly (the reference is bit-identical to the deleted
/// pre-pass pipeline on these rigs — see `ReferencePath`).
#[test]
fn a_single_substep_unsaturated_matches_apply_implicit() {
    let (mut reference, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_pd(4.0, 0.15)
                .with_target(0.8)
                .with_passive_spring(4.0, 0.0)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, d_active) =
        single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
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
    let (mut reference, mut on) = equivalence_pair(
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
    let (dv, d_total, _) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
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
        reference.joint.applied_active_torque, tau_max,
        "single-solve saturated active diagnostic is not bit-equal to τ_max"
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
    let (mut reference, mut on) =
        equivalence_pair(&|p, ch| JointAxis::new(p, ch, Vec3::z()).with_feedforward(0.3), 2.0);
    let (dv, d_total, d_active) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
    println!("C1 pure-ff: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(dv < TOL_OMEGA, "C1 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C1 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C1 active torque diverged: {d_active:e}");

    // ── C2: pure feedforward + passive damper ──
    let (mut reference, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.3)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, d_active) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
    println!("C2 ff+damper: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(dv < TOL_OMEGA, "C2 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C2 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C2 active torque diverged: {d_active:e}");

    // ── C3: clamped constant (|ff| = 0.5 > τ_max = 0.11) ──
    let (mut reference, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.5)
                .with_torque_limit(0.11)
                .with_passive_damping(1.0)
        },
        2.0,
    );
    let (dv, d_total, _) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
    println!("C3 clamped-ff: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}");
    assert!(dv < TOL_OMEGA, "C3 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C3 total torque diverged: {d_total:e}");
    assert_eq!(on.joint_axes[0].applied_active_torque, 0.11);
    assert_eq!(reference.joint.applied_active_torque, 0.11);

    // ── C4: pure ext_torque (constant passive row) ──
    let (mut reference, mut on) =
        equivalence_pair(&|p, ch| JointAxis::new(p, ch, Vec3::z()), 2.0);
    reference.joint.ext_torque = 0.05;
    on.joint_axes[0].ext_torque = 0.05;
    let (dv, d_total, _) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
    println!("C4 pure-ext: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}");
    assert!(dv < TOL_OMEGA, "C4 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C4 total torque diverged: {d_total:e}");
    // ext_* is a per-substep quantity on both paths: consumed by prepare
    // (in-loop) exactly as by the single solve (reference).
    assert_eq!(on.joint_axes[0].ext_torque, 0.0, "prepare did not consume ext_torque");
    assert_eq!(reference.joint.ext_torque, 0.0);
    // The reference's torque must have actually arrived (the injection is
    // not a silent no-op): dt·w·τ_ext changes ω by ~69 rad/s here.
    assert!(
        (reference.world.bodies[1].angular_velocity.z - 2.0).abs() > 1.0,
        "ext_torque had no effect — C4 is testing nothing"
    );
}

/// C5 — overflow-scale (unphysical-but-representable) stiffness content
/// degrades to the Constant row instead of poisoning the solve. For
/// damping around 1e-280, γ = 1/(dt·d) ≈ 2e283 is still FINITE — the
/// plain is_finite() classification keeps the row Soft — but the 2×2
/// block's downstream products overflow: the determinant γ_a·γ_p and the
/// numerator (w+γ_p)·r_a both exceed f64 range, inf/inf = NaN lands in
/// the deltas, and the in-loop world's velocities go NaN while the
/// single-solve D-form (denominator 1 + dt·w·d ≈ 1) stays exact. The
/// GAMMA_MAX = 1e100 guard classifies any such row as the γ→∞
/// constant-impulse limit, whose error is below one ulp of dt·τ₀ (see the
/// derivation at the guard).
///
/// Config: feedforward 0.3 with kd = 1e-280 (active row: nonzero τ₀ +
/// overflow-scale γ) plus passive damping 3e-280 (passive row: zero τ₀ +
/// overflow-scale γ ⇒ Absent under the guard). One substep, in-loop vs
/// the apply_implicit reference, the A/B/C tolerances.
///
/// MEASURED: |Δω|, |Δτ_total|, |Δτ_active| all exactly 0.0 — at this
/// scale the constant-impulse limit and the D-form agree bit-for-bit
/// (D = 1 + dt·w·d rounds to exactly 1.0 at d ~ 1e-280).
/// BITE-VERIFIED (2026-10-21): with the GAMMA_MAX guard removed
/// (classification reverted to bare is_finite()), the flag-on child
/// velocity is NaN and this test fails; guard restored, it passes.
#[test]
fn c5_overflow_scale_stiffness_degrades_to_constant_row() {
    let (mut reference, mut on) = equivalence_pair(
        &|p, ch| {
            JointAxis::new(p, ch, Vec3::z())
                .with_feedforward(0.3)
                .with_pd(0.0, 1e-280)
                .with_passive_damping(3e-280)
        },
        2.0,
    );
    let (dv, d_total, d_active) = single_substep_divergence(&mut reference, &mut on, SUB_DT_FINE);
    println!("C5 overflow-γ: |Δω| {dv:.3e}, |Δτ_total| {d_total:.3e}, |Δτ_active| {d_active:.3e}");
    assert!(
        on.bodies[1].angular_velocity.z.is_finite(),
        "C5: in-loop velocity went non-finite (overflow-γ row poisoned the block solve)"
    );
    assert!(dv < TOL_OMEGA, "C5 ω diverged: {dv:e}");
    assert!(d_total < TOL_TORQUE, "C5 total torque diverged: {d_total:e}");
    assert!(d_active < TOL_TORQUE, "C5 active torque diverged: {d_active:e}");
}

// ════════════════════════════════════════════════════════════════════════
// D — multi-substep free-space trajectory (E3 precursor)
// ════════════════════════════════════════════════════════════════════════

/// 600 substeps (1/12 s) of the full unsaturated stack, the in-loop path
/// vs the apply_implicit reference in lockstep.
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
    let (mut reference, mut on) = equivalence_pair(
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
        reference.substep(SUB_DT_FINE);
        on.substep(SUB_DT_FINE);
        let da =
            (joint_angle(&on, 0, 1, axis) - joint_angle(&reference.world, 0, 1, axis)).abs();
        let dw = (on.bodies[1].angular_velocity
            - reference.world.bodies[1].angular_velocity)
            .norm();
        if da > max_dtheta { max_dtheta = da; }
        if dw > max_domega { max_domega = dw; }
    }
    println!("D: max|Δθ| {max_dtheta:.3e}, max|Δω| {max_domega:.3e} over 600 substeps");
    assert!(
        max_dtheta < 1e-10,
        "in-loop trajectory diverged from apply_implicit: max|Δθ| = {max_dtheta:e}"
    );
    assert!(
        max_domega < 1e-10,
        "in-loop trajectory diverged from apply_implicit: max|Δω| = {max_domega:e}"
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
/// `applied_total_torque` (the REALIZED transmitted torque including
/// contact coupling) over the last 0.1 s of a 1 s settle,
/// at solver_iterations ∈ {4, 8, 16, 32} × sub_dt ∈ {1/7200, 1/3600}.
/// The flag-off pre-pass rows this harness printed through Commit C are
/// HISTORICAL (the pre-pass pipeline was deleted in Commit D); their
/// receipts stay recorded below (0.0090 @1/7200, 0.0045 @1/3600). This
/// measurement decided the design's escalation levers
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
/// MEASURED (2026-10-21, L2 warm-start pair landed; re-measured after the
/// warm-seed placement amendment, the current path): the same sweep
/// transmits the full demand at EVERY point of the grid: fraction
/// 0.9999 / 0.9999 / 1.0000 / 1.0000 at 4/8/16/32 iters @1/7200
/// and 1.0002 / 1.0000 / 1.0000 / 1.0000 @1/3600 (legacy pre-pass
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

    let run = |iters: usize, sub_dt: f64| -> (f64, f64, f64) {
        let mut world = XpbdWorld::new();
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
        for &iters in &[4usize, 8, 16, 32] {
            let (tx, demand, f_n) = run(iters, sub_dt);
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

fn e1_rig(iters: usize) -> E1Rig {
    let phi0 = (E1_ELEV / (2.0 * E1_HALF_H)).asin();
    let mut world = XpbdWorld::new();
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
fn e1_two_phases(iters: usize, sub_dt: f64) -> (E1Phase, E1Phase, E1Rig) {
    let mut rig = e1_rig(iters);
    let neutral = e1_run_phase(&mut rig, sub_dt, 1.0);
    let press_target = rig.phi0 + E1_PRESS;
    rig.world.joint_axes[0].target_angle = press_target;
    let pressed = e1_run_phase(&mut rig, sub_dt, 1.0);
    (neutral, pressed, rig)
}

/// E1 — static force transmission (design §4, the ~100 N check).
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
/// HISTORICAL FLAG-OFF RECEIPT (2026-10-21, measured via the receipts
/// harness below while the deleted pre-pass path still existed):
/// the legacy pre-pass transmitted 0.0039 of the 2.4000 N·m demand at 1/7200
/// (fraction 0.0016 ≈ 1/D with D = 1 + dt·w·(kd+c) ≈ 616) and 0.0019 at
/// 1/3600 (0.0008 — halving with dt, the damping-dominated signature);
/// ΔF_n = 0.17 N where the lever statics demand ~101 N. The documented
/// force-transmission gap at full strength; through Commit C, flipping
/// this gate's builders to coupled = false reproduced exactly those
/// numbers and failed (i). Commit D deleted that path; the live bites
/// below (both L2 sides) remain runnable and are what now arm this gate.
///
/// COLD-START MISS (2026-10-21, 8 iters, historical): fraction 0.0255 at
/// 1/7200 / 0.0128 at 1/3600 (0.0502/0.0254 at 16 iters, 0.0971/0.0501 at
/// 32) — this rig is in E6's near-degenerate alignment class (pinned
/// proximal + grounded child, a → 1): the per-sweep residual tracks
/// (D−1)/D with D ≈ 616, so the cold-start fraction is ~n_iter/D and L1
/// (16 iterations) does not close the 10% band. The gate landed
/// #[ignore]'d on that miss, pending the L2 warm-start pair (design §2.6),
/// whose cross-substep fixed point is exact for any alignment; L2 is now
/// in and the ignore is lifted. NEVER weaken the bands instead.
///
/// MEASURED AFTER L2 (2026-10-21, joint + manifold warm-start pair;
/// re-measured after the warm-seed placement amendment, the current path):
/// fraction 1.0000 at both rates and at 8/16/32 iterations; ΔF_n = Δτ/l to
/// the printed precision at each rate (101.34 N @1/7200, 100.82 N @1/3600
/// at 8 iterations); (iii) cross-rate spread 0.56% on τ_tx and 0.51% on
/// ΔF_n, inside the 2% band.
/// BITE-VERIFIED (2026-10-21), both L2 sides: (1) with the MANIFOLD carry
/// disabled (joint seed alone) this test fails at fraction 0.0494 —
/// measuring §2.6's derivation that the joint-side seed cannot accumulate
/// across substeps while the contact side forgets; (2) a warm_start that
/// double-applies the carried contact impulse fails this test (E5 stays
/// green under that bug: the over-application is unwound within the first
/// sweep by the λ ≥ 0 clamp without pumping — E1's statics catch it, not
/// the energy canary). RE-VERIFIED at Commit D (2026-10-29, the flag and
/// pre-pass deleted): with BOTH L2 carries disabled on the live path the
/// gate fails at fraction 0.0255 — bit-for-bit the recorded cold-start
/// miss above — and the JA toe-ray canary fails alongside it (fraction
/// 0.0414); restored, green.
#[test]
fn e1_static_force_transmission() {
    let (n_f, p_f, _) = e1_two_phases(8, SUB_DT_FINE);
    let (n_c, p_c, _) = e1_two_phases(8, SUB_DT_COARSE);

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
/// MEASURED (2026-10-21, cold start): max rise 0.000e0 J — the
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
    let mut rig = e1_rig(8);
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
fn e2_run(iters: usize, cmd_frac: f64, sub_dt: f64) -> (bool, f64, f64) {
    const M: f64 = 1.0;
    const HALF_H: f64 = 0.1;
    const R: f64 = 0.02;
    let tau_star = M * 9.81 * HALF_H;
    let kp = 20.0;

    let mut world = XpbdWorld::new();
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

/// E2 — grounded dt-independence (heel-lift threshold).
/// Commands realize 0.8·τ* (must NOT lift, heel stays loaded at both
/// rates) and 1.25·τ* (must lift at both rates, settled heel height within
/// 10% across rates). The lift outcomes are also asserted against the
/// statics-derived truth, not merely against each other.
///
/// MEASURED (2026-10-21, post-L2 with the warm-seed placement amendment —
/// the current path; 8 iters): 0.8·τ* stays grounded with heel load
/// 0.981 N / 0.981 N across rates (statics: (τ* − 0.8τ*)/0.2 m = 0.98 N);
/// 1.25·τ* lifts to 2.353e-3 / 2.351e-3 m — 0.08% apart, and within 4% of
/// the frictionless equilibrium kp·(target − θ) = τ*·cos θ ⇒ 2.45 mm.
/// (Historical cold-start figures, Commit B: heel 0.961/0.976 N, lift
/// 2.377/2.376 mm.) This rig's alignment is benign
/// (D ≈ 1.2 ⇒ per-sweep residual ≤ 0.17, fully converged in 8 sweeps), so
/// the gate passes at cold start — no lever needed.
///
/// HISTORICAL FLAG-OFF RECEIPT (2026-10-21, the deleted pre-pass path):
/// the outcome flipped qualitatively with
/// sub_dt, as the design predicts for this scenario class: at 1/7200 the
/// 1.25·τ* command unloaded the heel (0.000 N) and crept it to 4.6e-4 m
/// (still rising at 2 s), while at 1/3600 the heel stayed HARD GROUNDED
/// under 0.543 N — the pre-pass denominator D straddled the 1.25 command
/// ratio (1.20 vs 1.40). The original bite flipped this gate's builders
/// to coupled = false and failed the 1.25·τ* lift assertions at both
/// rates; Commit D deleted that path, and the dt-DEPENDENT failure mode
/// (demand throttled by the joint's own denominator) physically cannot
/// recur — the in-loop fixed point is dt-independent by construction
/// (design §2.5). RE-ARMED BITE (2026-10-29, Commit D): with
/// `JointAxis::solve_iteration` made an early-return no-op (the live
/// transmission path severed), the 1.25·τ* command transmits nothing and
/// the lift assertion fails (measured: clearance −3.0e-13 m at 1/7200 —
/// the heel never leaves the ground); restored, green.
#[test]
fn e2_heel_lift_threshold_is_dt_independent() {
    for &(frac, should_lift) in &[(0.8, false), (1.25, true)] {
        let (lift_f, clear_f, heel_f) = e2_run(8, frac, SUB_DT_FINE);
        let (lift_c, clear_c, heel_c) = e2_run(8, frac, SUB_DT_COARSE);
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

/// Append a LEGACY distal link to E1's rig: a dp-scale capsule beyond the
/// child's distal tip, flat on the ground, attached by the foot's full
/// legacy articulated stack exactly as `build_toe_ray`/`add_articulated`
/// compose it (positional constraint at 1/60000, primary-axis soft limit
/// k = 8000 exp 2, passive spring k = 4, damper c = 1.0, PD kp = 4 /
/// kd = 0.15, off-axis ±0.05 locks with dampers). Spring rest and PD
/// target sit at the as-assembled relative angle −φ₀ (zero-preload
/// assembly, the house practice). Returns the new body's index.
fn add_legacy_distal_link(rig: &mut E1Rig) -> usize {
    use physics::angular_pd_controller::AngularPDController;
    use physics::angular_soft_limit::AngularSoftLimit;
    use physics::angular_spring::AngularSpring;
    use physics::angular_constraint::AngularLimitConstraint;

    let (c_mass, c_half_h, c_r) = (0.0024, 0.0010, 0.0030);
    let b = &rig.world.bodies[rig.child];
    // Distal end-sphere center and surface tip of the pitched child.
    let distal = b.position + b.orientation * Vec3::new(0.0, E1_HALF_H, 0.0);
    let tip_z = distal.z + E1_R * rig.phi0.cos();

    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut link = RigidBody::new_capsule(c_mass, c_half_h, c_r);
    link.orientation = along_z;
    link.position = Vec3::new(0.0, c_r, tip_z + c_half_h + c_r);
    let c = rig.world.add_body(link);

    let joint_world = Vec3::new(0.0, (E1_R + c_r) * 0.5, tip_z);
    let mut pc = PositionalConstraint::new(rig.child, c, 0.0, 1.0 / 60_000.0);
    pc.local_anchor_a = rig.world.bodies[rig.child].orientation.inverse()
        * (joint_world - rig.world.bodies[rig.child].position);
    pc.local_anchor_b =
        rig.world.bodies[c].orientation.inverse() * (joint_world - rig.world.bodies[c].position);
    rig.world.constraints.push(pc);

    let rest = -rig.phi0; // as-assembled relative angle (child pitched, link flat)
    rig.world.angular_soft_limits.push(AngularSoftLimit::new(
        rig.child, c, Vec3::x(), rest - 0.70, rest + 0.70, 8000.0, 2.0,
    ));
    rig.world
        .angular_springs
        .push(AngularSpring::new(rig.child, c, Vec3::x(), 4.0, rest));
    rig.world
        .angular_dampers
        .push(AngularDamper::new(rig.child, c, Vec3::x(), 1.0));
    let mut pd = AngularPDController::new(rig.child, c, Vec3::x(), 4.0, 0.15);
    pd.target_angle = rest;
    rig.world.angular_pd_controllers.push(pd);
    for off in [Vec3::y(), Vec3::z()] {
        rig.world.angular_constraints.push(AngularLimitConstraint::new(
            rig.child, c, off, -0.05, 0.05,
        ));
        rig.world
            .angular_dampers
            .push(AngularDamper::new(rig.child, c, off, 1.0));
    }
    c
}

/// MIXED-PRIMITIVE CANARY 1 (pre-Commit-C gate): a grounded JointAxis press
/// with a LEGACY AngularDamper on the SAME body pair — one half of the
/// mid-migration composition a partial foot migration creates.
///
/// THE CHANNEL THIS PINS: the joint-side L2 warm seed is a real velocity
/// change (Δω = w·dt·τ₀ ≈ 1.27e3 rad/s on this rig — measured, carried
/// λ_a+λ_p = 3.33e-4 N·m·s at w ≈ 3.8e6). If it is applied while the
/// legacy implicit passes (apply_pd_kp_implicit, springs, dampers) still
/// run AFTER it, a legacy damper reads the seed velocity as real motion
/// and rectifies it — the identical channel the manifold seed had before
/// it moved to contact-reset time (the measured 26 rad/s legacy-ray limit
/// cycle). The joint seed must apply in the SAME slot: after the legacy
/// passes, before the SI sweeps (`JointAxis::apply_warm_start`).
///
/// Rig: E1's pressed toe (pinned proximal, distal tip wedged, 2.4 N·m
/// sustained demand ⇒ a large warm seed every substep) plus
/// AngularDamper(parent, child, x̂, c = 1.0) — the foot stack's shared
/// damper, exactly as `build_toe_ray`/`add_articulated` attach it. Statics
/// are damping-independent, so the honest settled state matches E1's.
///
/// ON THIS composition (damper into a PINNED parent, everything collinear
/// with the joint axis) the rectification does NOT surface as a limit
/// cycle: the per-substep theft drains into the pin and the collinear
/// joint+contact sweeps re-anchor the lambdas, so the world stays quiet
/// and the λ-based τ diagnostic still reads full demand — but the GROUND
/// REACTION is stolen: the damper removes the seed momentum before the
/// contact ever sees it. Hence the load-bearing assertion here is E1's
/// lever statics, alongside the house invariants:
///   (1) ΔF_n = Δτ_tx/lever within 10% (E1(ii)'s band — NEVER loosen);
///   (2) no sustained velocity limit cycle: max |ω_child| < OMEGA_BOUND;
///   (3) no energy pump: every 10 ms mechanical-energy sample within
///       MIXED_E_BOUND of its running minimum (E5's running-min form).
///
/// OMEGA_BOUND DERIVATION: healthy settled press is static (this rig
/// measures sustained |ω| ≈ 1.0e-13 rad/s); the mechanism scale is the
/// damper-rectified seed velocity, seed Δω/(1 + dt·w·c) ≈ 1.27e3/529 ≈
/// 2.4 rad/s ≈ τ₀/c (and canary 2 measures 151–1441 rad/s when the
/// stolen momentum lands on a free body). 0.05 rad/s sits ≥ 5e11× above
/// the healthy scale and ≥ 48× below the smallest mechanism scale.
/// MIXED_E_BOUND: same scene scale as E5 (the rig is E1's plus one
/// damper): mechanical energy scale W·l ≈ 1.2e-3 J, 1% ⇒ 1e-5 J.
///
/// MEASURED AT THE PRE-FIX HEAD (2026-10-21, seed applied in `prepare`,
/// BEFORE the legacy passes): pressed-phase F_n = 0.21 N against a
/// ~101 N lever demand — ΔF_n/(Δτ/l) ≈ 0.002, assertion (1) bitten by
/// ~500× — while the τ diagnostic reads fraction 1.0000 and the world
/// stays quiet (sustained |ω| 1.9e-13, rise 0.0 J): the theft is silent
/// except in the statics. With the seed at the contact-reset slot:
/// F_n = 101.37 N, ΔF_n = Δτ/l to 4 digits, sustained |ω| 1.0e-13 rad/s,
/// max rise 0.0 J.
/// BITE-VERIFIED (2026-10-21): with `apply_warm_start` reverted to the
/// pre-legacy-pass slot, assertion (1) fails (ΔF_n 0.19 N vs
/// Δτ/l 101.41 N); restored, all three pass.
#[test]
fn mixed_same_pair_legacy_damper_press_transmits_to_ground() {
    const OMEGA_BOUND: f64 = 0.05;
    const MIXED_E_BOUND: f64 = 1e-5;
    let sub_dt = SUB_DT_FINE;

    let mut rig = e1_rig(8);
    rig.world
        .angular_dampers
        .push(AngularDamper::new(rig.parent, rig.child, Vec3::x(), 1.0));

    // Neutral settle, then pressed settle (E1's two phases).
    let neutral = e1_run_phase(&mut rig, sub_dt, 1.0);
    rig.world.joint_axes[0].target_angle = rig.phi0 + E1_PRESS;
    let pressed = e1_run_phase(&mut rig, sub_dt, 1.0);
    let d_fn = pressed.f_n - neutral.f_n;
    let expect = (pressed.tau_tx - neutral.tau_tx) / pressed.lever;
    println!(
        "mixed same-pair: transmitted {:+.4} N·m vs demand {:+.4} N·m (fraction {:+.4}), \
         ΔF_n {d_fn:+.2} N vs Δτ/l {expect:+.2} N",
        pressed.tau_tx, pressed.demand, pressed.tau_tx / pressed.demand
    );
    // (1) The ground actually receives the lever reaction — the silent
    // failure mode of this composition (momentum stolen into the pin).
    assert!(
        (d_fn - expect).abs() <= 0.10 * expect.abs(),
        "mixed same-pair: ΔF_n {d_fn} N vs Δτ/l {expect} N (> 10% off) — the legacy \
         damper is stealing the joint warm seed's momentum before the contact sees it \
         (the seed must apply AFTER the legacy implicit passes, at the contact-reset slot)"
    );

    // 3 s measured window on the settled press.
    let mut e_min = e1_mechanical_energy(&rig.world, rig.child);
    let mut max_rise: f64 = 0.0;
    let mut max_w: f64 = 0.0;
    let sample_every = (0.01 / sub_dt).round() as usize;
    let total = (3.0 / sub_dt).round() as usize;
    for i in 0..total {
        rig.world.substep(sub_dt);
        let w = rig.world.bodies[rig.child].angular_velocity.norm();
        if !w.is_finite() || w > max_w {
            max_w = w;
        }
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
    println!(
        "mixed same-pair: sustained max |ω| {max_w:.3e} rad/s, max energy rise {max_rise:.3e} J"
    );
    // (2)/(3) The house invariants.
    assert!(
        max_w < OMEGA_BOUND,
        "mixed same-pair press sustains |ω| {max_w} rad/s ≥ {OMEGA_BOUND} — seed \
         rectification limit cycle"
    );
    assert!(
        max_rise < MIXED_E_BOUND,
        "mixed same-pair press pumped {max_rise} J above its running minimum \
         (> {MIXED_E_BOUND} J)"
    );
}

/// MIXED-PRIMITIVE CANARY 2 (pre-Commit-C gate): a grounded JointAxis press
/// with a full LEGACY articulated stack on an ADJACENT FREE body — the
/// other half of the mid-migration composition (a JointAxis joint whose
/// child also carries a still-legacy joint to the next bone, exactly what
/// a partial toe-ray migration creates).
///
/// Same channel as canary 1, but here the legacy damper connects the
/// seeded body to a free dp-scale link instead of a pin: the damper
/// equalizes the pair's velocities, so the seed spin (≈ 1.27e3 rad/s,
/// canary 1's derivation) is TRANSFERRED to the link as a real kick every
/// substep — momentum that the SI sweeps cannot re-anchor because the link
/// has no JointAxis. The result at the pre-fix HEAD is categorical, not
/// marginal (see MEASURED below).
///
/// Asserted over 3 s of settled press:
///   (1) no sustained velocity limit cycle: max |ω| over child AND link
///       < OMEGA_BOUND (canary 1's derivation; the measured failure is
///       151–1441 rad/s, ≥ 3000× the bound; healthy is ≤ 2.2e-12);
///   (2) no energy pump: running-min form, MIXED_E_BOUND = 1e-5 J (scene
///       scale unchanged: the 2.4 g link adds < 10% to W·l ≈ 1.2e-3 J;
///       the measured failure pumps 5.2e-2 J, 5200× the bound). The
///       energy sum counts both free bodies + the JointAxis's elastic PE;
///       the legacy stack's elastic terms are omitted like E5's hinge —
///       constant at a settled static press, so a pump must surface in
///       the counted terms;
///   (3) the press still transmits: τ_tx within 10% of the JointAxis's
///       static demand (E1(i)'s band).
///
/// MEASURED AT THE PRE-FIX HEAD (2026-10-21, seed applied in `prepare`):
/// sustained max |ω| 151 rad/s (child) / 1441 rad/s (link), max energy
/// rise 5.23e-2 J, transmitted fraction −2.53 (sign-flipped!), child
/// F_n diagnostic 3.7e6 N — a full-blown limit cycle, pump, and garbage
/// statics. With the seed at the contact-reset slot: max |ω| 9.9e-14 /
/// 2.2e-12 rad/s, max rise 0.0 J, fraction 1.0158.
/// BITE-VERIFIED (2026-10-21): with `apply_warm_start` reverted to the
/// pre-legacy-pass slot the test fails at assertion (3) (fraction −2.53);
/// the measured 151/1441 rad/s and 5.23e-2 J above also violate (1) and
/// (2), each thousands of times over its bound. Restored, green.
#[test]
fn mixed_adjacent_legacy_stack_press_stays_quiet() {
    const OMEGA_BOUND: f64 = 0.05;
    const MIXED_E_BOUND: f64 = 1e-5;
    let sub_dt = SUB_DT_FINE;

    let mut rig = e1_rig(8);
    let link = add_legacy_distal_link(&mut rig);

    e1_run_phase(&mut rig, sub_dt, 1.0);
    rig.world.joint_axes[0].target_angle = rig.phi0 + E1_PRESS;
    let pressed = e1_run_phase(&mut rig, sub_dt, 1.0);
    println!(
        "mixed adjacent-stack: transmitted {:+.4} N·m vs demand {:+.4} N·m (fraction {:+.4})",
        pressed.tau_tx, pressed.demand, pressed.tau_tx / pressed.demand
    );
    // (3) The press transmits through the mixed assembly.
    assert!(
        (pressed.tau_tx - pressed.demand).abs() <= 0.10 * pressed.demand.abs(),
        "mixed adjacent-stack: transmitted {} N·m vs demand {} N·m (> 10% off)",
        pressed.tau_tx, pressed.demand
    );

    // Mechanical energy of both free bodies + the JointAxis's elastic PE.
    let child = rig.child;
    let energy = move |world: &XpbdWorld| -> f64 {
        let mut e = e1_mechanical_energy(world, child);
        let b = &world.bodies[link];
        e += 0.5 * b.mass * b.linear_velocity.dot(&b.linear_velocity);
        let r = b.orientation.to_rotation_matrix();
        let i_w = r.matrix() * b.inertia_tensor * r.matrix().transpose();
        e += 0.5 * b.angular_velocity.dot(&(i_w * b.angular_velocity));
        e += b.mass * 9.81 * b.position.y;
        e
    };

    let mut e_min = energy(&rig.world);
    let mut max_rise: f64 = 0.0;
    let mut max_w: f64 = 0.0;
    let sample_every = (0.01 / sub_dt).round() as usize;
    let total = (3.0 / sub_dt).round() as usize;
    for i in 0..total {
        rig.world.substep(sub_dt);
        for &idx in &[rig.child, link] {
            let w = rig.world.bodies[idx].angular_velocity.norm();
            if !w.is_finite() || w > max_w {
                max_w = w;
            }
        }
        if (i + 1) % sample_every == 0 {
            let e = energy(&rig.world);
            if e - e_min > max_rise {
                max_rise = e - e_min;
            }
            if e < e_min {
                e_min = e;
            }
        }
    }
    println!(
        "mixed adjacent-stack: sustained max |ω| {max_w:.3e} rad/s, \
         max energy rise {max_rise:.3e} J"
    );
    // (1)/(2) The house invariants.
    assert!(
        max_w < OMEGA_BOUND,
        "mixed adjacent-stack press sustains |ω| {max_w} rad/s ≥ {OMEGA_BOUND} — the \
         legacy damper is pumping the joint warm seed into the free link every substep \
         (the seed must apply AFTER the legacy implicit passes, at the contact-reset slot)"
    );
    assert!(
        max_rise < MIXED_E_BOUND,
        "mixed adjacent-stack press pumped {max_rise} J above its running minimum \
         (> {MIXED_E_BOUND} J) — energy injection through the seed-rectification channel"
    );
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

/// Saturation diagnostic under the convergence skip: `applied_active_torque`
/// must equal ±torque_limit BIT-EXACTLY on every substep of a wedged
/// saturated steady state (the T3/§2.3 contract), including the substeps
/// where the solve converges by skipping.
///
/// THE HOLE THIS PINS: `solve_iteration`'s convergence skip does not write
/// `cache.clamped`/`sigma`. In a contact-wedged saturated steady state with
/// an exact warm seed the active lambda sits bit-exactly at the cap
/// (prepare's clamp re-seeds it to ±cap), so the clamped branch computes
/// d_a = σ·cap − λ_a = 0 exactly; with no passive row (d_p absent) the
/// total delta is 0, the skip fires, `clamped` keeps prepare's false, and
/// `finalize` falls through to λ_a/dt — up to 1 ulp off τ_max, because
/// (τ_max·dt)/dt does not round-trip in f64 (τ_max = 0.11 here, the same
/// deliberately non-round-tripping value as test B). `finalize` now derives
/// the saturated report from the lambda itself: |λ_a| at the cap reports
/// sign(λ_a)·torque_limit directly.
///
/// Rig: E6's wedged geometry (pinned parent, toe capsule tip on the
/// ground) with an ACTIVE-ONLY JointAxis (kp = 2 / kd = 0.05, no passive
/// structures ⇒ the passive row is Absent and the skip is reached whenever
/// d_a = 0) and τ_max = 0.11 ≪ the ~1.2 N·m demand. 0.5 s settle, then
/// every substep of the next 0.5 s asserts bit equality.
///
/// MEASURED AT THE UNFIXED HEAD (2026-10-21): the diagnostic reads
/// 0.10999999999999999 (λ_a/dt, 1 ulp low) on the steady-state substeps —
/// red. BITE-VERIFIED (2026-10-21): with finalize's at-cap branch removed
/// the test fails on the first post-settle substep; restored, 3600/3600
/// substeps bit-equal.
#[test]
fn saturated_diagnostic_is_bit_exact_through_convergence_skips() {
    let tau_max = 0.11;
    let kp = 2.0;
    let target = 0.6;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    // E6's wedged rig, active-only joint.
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
            .with_torque_limit(tau_max),
    );

    let settle = (0.5 / SUB_DT_FINE).round() as usize;
    for _ in 0..settle {
        world.substep(SUB_DT_FINE);
    }
    // The rig really is wedged and saturated: tip loaded, demand ≫ τ_max.
    let theta = joint_angle(&world, p, ch, Vec3::x());
    assert!(
        world.total_normal_force(ch) > 0.0 && kp * (target - theta) > 2.0 * tau_max,
        "rig broken: not a wedged saturated press (θ {theta}, F_n {})",
        world.total_normal_force(ch)
    );

    let total = (0.5 / SUB_DT_FINE).round() as usize;
    let mut exact = 0usize;
    for i in 0..total {
        world.substep(SUB_DT_FINE);
        let a = world.joint_axes[0].applied_active_torque;
        assert!(
            a == tau_max,
            "substep {i}: saturated active diagnostic {a:.17} is not bit-equal to \
             τ_max {tau_max:.17} — the convergence skip left the clamped state stale \
             and finalize divided the impulse back out"
        );
        exact += 1;
    }
    println!("saturation diagnostic bit-equal on {exact}/{total} steady-state substeps");
}

/// DIAGNOSTIC harness for the E1/E2 numbers and the L1 iteration sweep —
/// run with `cargo test -p physics --test coupled_joint -- --ignored
/// --nocapture`. Prints E1 and E2 at solver_iterations {8, 16, 32} × both
/// sub_dts. The 16-iteration rows were the design's L1 trigger
/// measurement. The legacy flag-off rows this harness printed through
/// Commit C were the Commit B receipts — that path was deleted in Commit D
/// and the receipts are preserved (HISTORICAL) in E1's and E2's doc
/// comments above.
#[test]
#[ignore = "diagnostic harness (E1/E2 numbers + L1 sweep) — run with --ignored --nocapture"]
fn e1_e2_receipts_and_iteration_sweep() {
    for &sub_dt in &[SUB_DT_FINE, SUB_DT_COARSE] {
        println!("── sub_dt = 1/{:.0} ──", 1.0 / sub_dt);
        for &iters in &[8usize, 16, 32] {
            let (n, p, _) = e1_two_phases(iters, sub_dt);
            println!(
                "  E1 coupled@{iters:>2}: τ_tx {:+.4} vs demand {:+.4} N·m (fraction {:+.4}), \
                 ΔF_n {:+.2} N vs Δτ/l {:+.2} N",
                p.tau_tx, p.demand, p.tau_tx / p.demand,
                p.f_n - n.f_n, (p.tau_tx - n.tau_tx) / p.lever
            );
            for &frac in &[0.8, 1.25] {
                let (lift, clear, heel_n) = e2_run(iters, frac, sub_dt);
                println!(
                    "  E2 coupled@{iters:>2} @{frac}·τ*: lift {lift} \
                     (clearance {clear:+.3e} m, heel {heel_n:.3} N)"
                );
            }
        }
    }
}
