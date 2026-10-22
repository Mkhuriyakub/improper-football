//! Regression suite locking in the *current, accepted* engine behavior after
//! the explicit→implicit integration + impulse-based ground-contact rewrite.
//!
//! Every test asserts a real physical fact (rest height, energy bound, drift
//! tolerance), not merely "doesn't panic". Thresholds are justified inline.
//!
//! Standard timestep used throughout: dt = 1/120 s, 60 substeps/frame,
//! sub_dt = dt/60. This mirrors the known-good caller pattern in
//! `src/anatomy/foot.rs::foot_settles_on_ground_without_exploding`.

use physics::angular_constraint::AngularLimitConstraint;
use physics::contact_constraint::ContactConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_pd_controller::AngularPDController;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::constraints::PositionalConstraint;
use physics::math::{Mat3, Quat, Vec3};
use physics::rigid_body::RigidBody;
use physics::xpbd::{XpbdWorld, GRAVITY};

const DT: f64 = 1.0 / 120.0;
const SUBSTEPS_PER_FRAME: usize = 60;
const SUB_DT: f64 = DT / SUBSTEPS_PER_FRAME as f64;

/// Advance the world by `seconds` of simulated time using the standard
/// substep cadence. Returns the number of substeps run.
fn run_for(world: &mut XpbdWorld, seconds: f64) {
    let frames = (seconds / DT).round() as usize;
    for _ in 0..frames {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
        }
    }
}

fn finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Total mechanical energy of the world: kinetic (linear + rotational) plus
/// gravitational potential. Reference height is y = 0; potential can go
/// negative, which is fine — we only ever compare energy *differences*.
fn total_mechanical_energy(world: &XpbdWorld) -> f64 {
    let mut e = 0.0;
    for b in &world.bodies {
        if b.inv_mass == 0.0 {
            continue; // pinned bodies hold no mechanical energy of interest
        }
        let v2 = b.linear_velocity.dot(&b.linear_velocity);
        e += 0.5 * b.mass * v2;
        // Rotational KE = 1/2 ω·(I_world ω). Use world inertia.
        let r = b.orientation.to_rotation_matrix();
        let i_world = r.matrix() * b.inertia_tensor * r.matrix().transpose();
        let l = i_world * b.angular_velocity;
        e += 0.5 * b.angular_velocity.dot(&l);
        // Gravitational PE: -m (g·x) with g = (0,-9.81,0) ⇒ m*9.81*y.
        e += -b.mass * GRAVITY.dot(&b.position);
    }
    e
}

// ════════════════════════════════════════════════════════════════════════
// Contact model
// ════════════════════════════════════════════════════════════════════════

/// A horizontal capsule (axis along world +X) rests with its center one
/// `radius` above the ground. Dropped from ~5 cm, it must settle within 2 s.
#[test]
fn capsule_drops_and_settles_at_rest() {
    let radius = 0.03;
    let half_h = 0.1;
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    let mut body = RigidBody::new_capsule(1.0, half_h, radius);
    // Lay the capsule horizontal so it has a 2-point footprint and its
    // resting center height is `radius` (both end-spheres touch the floor).
    body.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    body.position = Vec3::new(0.0, radius + 0.05, 0.0); // ~5 cm above rest
    let idx = world.add_body(body);

    run_for(&mut world, 2.0);

    let b = &world.bodies[idx];
    assert!(finite(b.position) && finite(b.linear_velocity), "went non-finite");
    let speed = b.linear_velocity.norm();
    assert!(speed < 0.01, "still moving after settle: |v| = {speed} m/s");
    // Resting center height = radius. Allow a few mm for penalty penetration.
    assert!(
        (b.position.y - radius).abs() < 0.004,
        "rest height {} m, expected ≈ {} m",
        b.position.y,
        radius
    );
}

/// A capsule initialized already at rest on the ground must stay put: it
/// should neither sink nor be ejected. Guards against a contact bias that
/// pushes resting bodies around.
#[test]
fn resting_capsule_stays_at_rest() {
    let radius = 0.03;
    let half_h = 0.1;
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    let mut body = RigidBody::new_capsule(1.0, half_h, radius);
    body.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    body.position = Vec3::new(0.0, radius, 0.0); // exactly at rest
    let idx = world.add_body(body);

    let start = world.bodies[idx].position;
    run_for(&mut world, 2.0);

    let b = &world.bodies[idx];
    assert!(finite(b.position), "went non-finite");
    let drift = (b.position - start).norm();
    assert!(drift < 0.002, "resting capsule drifted {drift} m (> 2 mm)");
    assert!(
        b.linear_velocity.norm() < 0.01,
        "resting capsule speed {} m/s",
        b.linear_velocity.norm()
    );
}

/// Energy guard: a dropped capsule must never rise above its release height
/// after impact. If the contact model pumped energy, the rebound would
/// exceed the drop height. We sample the peak center height across 3 s.
#[test]
fn drop_does_not_gain_height() {
    let radius = 0.03;
    let half_h = 0.1;
    let drop_h = radius + 0.10; // release center 10 cm above rest
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    let mut body = RigidBody::new_capsule(1.0, half_h, radius);
    body.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    body.position = Vec3::new(0.0, drop_h, 0.0);
    let idx = world.add_body(body);

    let mut peak = world.bodies[idx].position.y;
    // Manually loop so we can sample after every frame.
    let frames = (3.0 / DT).round() as usize;
    for _ in 0..frames {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
        }
        let y = world.bodies[idx].position.y;
        if y > peak {
            peak = y;
        }
    }
    assert!(finite(world.bodies[idx].position), "went non-finite");
    assert!(
        peak < drop_h + 0.002,
        "peak height {peak} m exceeded release {drop_h} m by > 2 mm (energy injection on impact)"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Integration stability — the core of the accepted explicit→implicit change
// ════════════════════════════════════════════════════════════════════════

/// Local-frame anchor on `body` for a joint at world point `joint_world`,
/// expressed in the body's local frame: R⁻¹·(joint_world − center).
/// Mirrors `src/anatomy/foot.rs::anchor_at`.
fn anchor_local(world: &XpbdWorld, body: usize, joint_world: Vec3) -> Vec3 {
    let b = &world.bodies[body];
    b.orientation.inverse() * (joint_world - b.position)
}

/// Bilateral weld: lock body_b to body_a at a shared world anchor with all
/// three relative-rotation axes pinned to [0,0]. This is exactly the `weld`
/// recipe from `src/anatomy/foot.rs` (both bodies pass the SAME world point,
/// so each local anchor maps back to it and the rest preload is zero).
fn add_weld(world: &mut XpbdWorld, a: usize, b: usize, joint_world: Vec3) {
    let anchor_a = anchor_local(world, a, joint_world);
    let anchor_b = anchor_local(world, b, joint_world);
    // Near-rigid positional coupling (compliance = 1/k, k = 1e6).
    let mut c = PositionalConstraint::new(a, b, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = anchor_a;
    c.local_anchor_b = anchor_b;
    world.constraints.push(c);
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world
            .angular_constraints
            .push(AngularLimitConstraint::new(a, b, axis, 0.0, 0.0));
    }
}

/// The exact failure mode the explicit→implicit rewrite was made to fix:
/// passive angular damping on a few-gram body. The foot module attaches an
/// `AngularDamper` with coefficient `ARTIC_PASSIVE_C = 1.0` N·m·s/rad to every
/// inter-bone joint, including the ~2 g distal phalanges. Applied with explicit
/// Euler, a damper of coefficient `c` on a body of reduced inertia `I_red`
/// requires `sub_dt < 2·I_red/c` for stability. For a 2 g distal-phalanx
/// capsule the transverse inertia is I ≈ 6e-9 kg·m², so the explicit bound is
/// `sub_dt < 2·6e-9/1.0 ≈ 1.2e-8 s` — about a *million* times smaller than the
/// `sub_dt = 1/7200 s ≈ 1.4e-4 s` used here. Explicit integration therefore
/// over-corrects each step, the relative angular velocity flips sign and grows
/// without bound, and the body explodes (non-finite). The implicit damper
/// (`ω_new = ω/(1 + c·dt·w)`) is unconditionally stable: it can only *remove*
/// angular kinetic energy, never amplify it, for any `c` and `dt`.
///
/// Stressor under test: the ANGULAR DAMPER on the 2 g body. The positional
/// constraint is present only to keep the body located near the anchor (it is
/// NOT the stressor — a bare positional constraint was always stable, which is
/// why the previous version of this test gave false confidence). We seed a
/// non-zero initial angular velocity so the damper has something to act on.
#[test]
fn tiny_mass_body_angular_damper_is_stable() {
    const ARTIC_PASSIVE_C: f64 = 1.0; // N·m·s/rad — the foot's passive joint value.

    let mut world = XpbdWorld::new();
    world.no_ground();

    // Heavy fixed anchor (inv_mass = 0 ⇒ never moves, treated as infinite inertia).
    let anchor = RigidBody::new(0.0, Mat3::identity());
    let anchor_idx = world.add_body(anchor);

    // 2 g distal-phalanx-scale capsule, 1 cm below the anchor.
    let mut tiny = RigidBody::new_capsule(0.002, 0.0035, 0.0028);
    let rest = Vec3::new(0.0, -0.01, 0.0);
    tiny.position = rest;
    // Seed an initial spin about world X so the angular damper has a non-zero
    // relative angular velocity to dissipate. This is the quantity the explicit
    // scheme blows up.
    let initial_omega = Vec3::new(1.0, 0.0, 0.0);
    tiny.angular_velocity = initial_omega;
    let tiny_idx = world.add_body(tiny);

    // Stiff ball-and-socket keeping the tiny body located at the anchor's
    // origin (zero-length, near-rigid). Anchors coincide at rest ⇒ no preload.
    // This only positions the body; the angular damper below is the stressor.
    let mut c = PositionalConstraint::new(anchor_idx, tiny_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = rest; // anchor's local point coinciding with tiny center
    c.local_anchor_b = Vec3::zeros();
    world.constraints.push(c);

    // THE STRESSOR: passive angular damper about world X, foot-strength coeff.
    // On a 2 g body this is ~10⁶× past the explicit-Euler stability bound.
    world.angular_dampers.push(AngularDamper::new(
        anchor_idx,
        tiny_idx,
        Vec3::x(),
        ARTIC_PASSIVE_C,
    ));
    // A passive angular spring on the same axis (also foot-scale) — another
    // implicit-only lever-arm stressor sharing the tiny inertia.
    world.angular_springs.push(physics::angular_spring::AngularSpring::new(
        anchor_idx,
        tiny_idx,
        Vec3::x(),
        50.0,
        0.0,
    ));

    let initial_spin = initial_omega.norm();
    run_for(&mut world, 3.0);

    let b = &world.bodies[tiny_idx];
    assert!(
        finite(b.position) && finite(b.linear_velocity) && finite(b.angular_velocity),
        "tiny body exploded (non-finite) — explicit angular-damper instability"
    );
    // The damper must REMOVE angular energy: final spin strictly below initial,
    // and in fact nearly fully damped over 3 s.
    let final_spin = b.angular_velocity.norm();
    assert!(
        final_spin < initial_spin,
        "angular velocity grew: {final_spin} rad/s ≥ initial {initial_spin} (damper added energy)"
    );
    assert!(
        final_spin < 0.1,
        "angular velocity did not decay: {final_spin} rad/s after 3 s (expected ≈0)"
    );
    // The positional constraint must keep it near rest the whole time.
    let off = (b.position - rest).norm();
    assert!(off < 0.03, "tiny body drifted {off} m from rest (instability)");
    assert!(
        b.linear_velocity.norm() < 0.5,
        "tiny body linear velocity {} m/s (instability)",
        b.linear_velocity.norm()
    );
}

/// Two equal bodies welded together must hold their relative pose under
/// gravity (the whole pair falls freely with no ground, but the *relative*
/// transform between them stays fixed).
#[test]
fn weld_holds_two_bodies_rigid() {
    let mut world = XpbdWorld::new();
    world.no_ground();

    let mut a = RigidBody::new_capsule(0.05, 0.02, 0.01);
    a.position = Vec3::new(0.0, 0.0, 0.0);
    let a_idx = world.add_body(a);

    let mut b = RigidBody::new_capsule(0.05, 0.02, 0.01);
    let offset = Vec3::new(0.0, -0.04, 0.0); // b sits just below a
    b.position = offset;
    let b_idx = world.add_body(b);

    // Weld at the world midpoint between the two centers.
    let mid = (Vec3::zeros() + offset) * 0.5;
    add_weld(&mut world, a_idx, b_idx, mid);

    let rel_pos_start = world.bodies[b_idx].position - world.bodies[a_idx].position;
    let rel_rot_start =
        world.bodies[a_idx].orientation.inverse() * world.bodies[b_idx].orientation;

    run_for(&mut world, 2.0);

    let pa = &world.bodies[a_idx];
    let pb = &world.bodies[b_idx];
    assert!(finite(pa.position) && finite(pb.position), "weld pair went non-finite");

    let rel_pos = pb.position - pa.position;
    let pos_drift = (rel_pos - rel_pos_start).norm();
    // Near-rigid weld (compliance 1e-6) — relative position must barely move.
    assert!(pos_drift < 0.002, "weld relative-position drift {pos_drift} m (> 2 mm)");

    let rel_rot = pa.orientation.inverse() * pb.orientation;
    let rot_drift = rel_rot_start.angle_to(&rel_rot);
    assert!(rot_drift < 0.02, "weld relative-orientation drift {rot_drift} rad");
}

// ════════════════════════════════════════════════════════════════════════
// Primitive behaviors
// ════════════════════════════════════════════════════════════════════════

/// Two free bodies started apart, joined by a zero-rest-length positional
/// constraint, must pull their anchors together (the SI ball-and-socket
/// drives the anchor separation to zero).
#[test]
fn positional_constraint_holds_distance() {
    let mut world = XpbdWorld::new();
    world.no_ground();

    let mut a = RigidBody::new_capsule(0.2, 0.02, 0.01);
    a.position = Vec3::new(0.0, 0.0, 0.0);
    let a_idx = world.add_body(a);

    let mut b = RigidBody::new_capsule(0.2, 0.02, 0.01);
    b.position = Vec3::new(0.05, 0.0, 0.0); // 5 cm apart — anchors start separated
    let b_idx = world.add_body(b);

    // Zero-rest-length constraint between OFF-CENTER local anchors (each 1 cm
    // along the bodies' local +Y), so the constraint must drive two offset
    // anchors — not the bare centers — to coincide. This exercises the lever-arm
    // (rotational) path of the positional constraint, not just the trivial
    // center-to-center case. Stiff (small compliance) so it converges.
    let local_anchor = Vec3::new(0.0, 0.01, 0.0);
    let mut c = PositionalConstraint::new(a_idx, b_idx, 0.0, 1.0 / 100_000.0);
    c.local_anchor_a = local_anchor;
    c.local_anchor_b = local_anchor;
    world.constraints.push(c);

    // Gravity still acts (no_ground only removes the floor), so the pair
    // falls together; what we assert is the *relative* anchor coincidence,
    // which is unaffected by the common free-fall.
    run_for(&mut world, 2.0);

    // World positions of the two off-center anchors. The orientation*offset
    // term is now load-bearing (the offset is non-zero), so this genuinely
    // exercises the rotated-anchor mapping.
    let pa = world.bodies[a_idx].position + world.bodies[a_idx].orientation * local_anchor;
    let pb = world.bodies[b_idx].position + world.bodies[b_idx].orientation * local_anchor;
    let sep = (pb - pa).norm();
    assert!(finite(pa) && finite(pb), "constraint pair went non-finite");
    assert!(sep < 0.002, "anchors did not coincide: separation {sep} m (> 2 mm)");
}

/// An AngularPDController commanded to a non-zero target angle must drive the
/// joint there. body_a is fixed; body_b is free and swings to target.
#[test]
fn angular_pd_drives_to_target() {
    let mut world = XpbdWorld::new();
    world.no_ground();

    // Fixed parent.
    let parent = RigidBody::new(0.0, Mat3::identity());
    let parent_idx = world.add_body(parent);

    // Free child with modest inertia. Keep it at the same position as parent
    // so a positional constraint holds them coincident while it rotates.
    let mut child = RigidBody::new_capsule(0.1, 0.05, 0.02);
    child.position = Vec3::zeros();
    let child_idx = world.add_body(child);

    // Hold the centers coincident (pure rotational joint).
    let mut c = PositionalConstraint::new(parent_idx, child_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = Vec3::zeros();
    c.local_anchor_b = Vec3::zeros();
    world.constraints.push(c);

    // PD about world Z, target 0.5 rad. Add damping so it settles rather than
    // oscillating. Gravity acts on the child but the joint is at its center,
    // so there's no gravity moment about Z to fight at the target pose.
    let target = 0.5;
    let pd = AngularPDController::new(parent_idx, child_idx, Vec3::z(), 20.0, 2.0).with_target(target);
    world.angular_pd_controllers.push(pd);

    run_for(&mut world, 2.0);

    let q_rel = world.bodies[parent_idx].orientation.inverse() * world.bodies[child_idx].orientation;
    let angle = physics::math::swing_twist_angle(q_rel, Vec3::z());
    assert!(angle.is_finite(), "PD joint angle went non-finite");
    assert!(
        (angle - target).abs() < 0.05,
        "PD reached {angle} rad, target {target} (±0.05)"
    );
}

/// A body driven past an AngularSoftLimit's range must be held near the limit,
/// not allowed to pass through freely. We push with a PD target well beyond
/// the limit and assert the steady-state angle stays close to the limit.
#[test]
fn angular_soft_limit_clamps_beyond_rom() {
    let mut world = XpbdWorld::new();
    world.no_ground();

    let parent = RigidBody::new(0.0, Mat3::identity());
    let parent_idx = world.add_body(parent);

    let mut child = RigidBody::new_capsule(0.1, 0.05, 0.02);
    child.position = Vec3::zeros();
    let child_idx = world.add_body(child);

    let mut c = PositionalConstraint::new(parent_idx, child_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = Vec3::zeros();
    c.local_anchor_b = Vec3::zeros();
    world.constraints.push(c);

    // Soft limit: range [-0.2, 0.2] rad about Z, stiff (k = 8000, exp 2) —
    // same scale as the foot's ROM limits.
    let max_angle = 0.2;
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent_idx,
        child_idx,
        Vec3::z(),
        -max_angle,
        max_angle,
        8000.0,
        2.0,
    ));
    // Add a little damping so the chase settles.
    world
        .angular_pd_controllers
        .push(AngularPDController::new(parent_idx, child_idx, Vec3::z(), 5.0, 0.5).with_target(1.5));

    run_for(&mut world, 2.0);

    let q_rel = world.bodies[parent_idx].orientation.inverse() * world.bodies[child_idx].orientation;
    let angle = physics::math::swing_twist_angle(q_rel, Vec3::z());
    assert!(angle.is_finite(), "soft-limit joint angle went non-finite");
    // Driven hard toward 1.5 rad; the soft limit holds it just past 0.2 rad.
    // Equilibrium is where PD push (kp·(target−θ)) balances the nonlinear
    // limit torque (k·(θ−0.2)²): ≈0.23 rad here. The physically meaningful
    // fact is that a 1.5 rad command is arrested near the 0.2 rad limit
    // (within ~25%), NOT allowed to pass freely toward the target.
    assert!(
        angle > max_angle * 0.5 && angle < max_angle * 1.25,
        "soft limit failed to clamp: angle {angle} rad (limit {max_angle}, target 1.5)"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Energy sanity
// ════════════════════════════════════════════════════════════════════════

/// A small chain of capsules joined by stiff constraints, released under
/// gravity onto the ground, must not *gain* mechanical energy as it settles.
///
/// HISTORY: the bound here was originally 0.2 J, budgeted against a
/// "~0.024 J/s constraint-solver injection" measured in 2026-04 on the OLD
/// engine scheme (textbook XPBD position projection with `v += dx/dt`
/// velocity feedback). That scheme — and with it the injection — was removed
/// by the 2026-06 velocity-level SI rewrite. The 2026-10-19 Baumgarte
/// energy-attribution sweep (docs/findings.md) re-measured this exact scene
/// at β ∈ {0.2, 0.02, 0.002}: net DISSIPATION of ~0.150 J over 3 s at every
/// β, steady-tail rate ≈ −5.6e-5 J/s, and peak energy never exceeding the
/// release energy at all. The bound is tightened accordingly: 1 mJ of
/// headroom (~18× the |tail-rate| × 3 s scale) instead of 0.2 J. Never
/// loosen this bound — a regression that needs more than 1 mJ of growth here
/// is a real injector.
#[test]
fn settling_chain_does_not_gain_energy() {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    // Four small capsules stacked vertically, each joined to the next by a
    // stiff positional constraint at the touching ends.
    let n = 4;
    let half_h = 0.03;
    let radius = 0.015;
    let mut idxs = Vec::new();
    for i in 0..n {
        let mut body = RigidBody::new_capsule(0.05, half_h, radius);
        // Lay them horizontal (axis +X) and stack along +Y, starting a few cm
        // up so they drop and settle.
        body.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        body.position = Vec3::new(0.0, radius + 0.05 + i as f64 * (2.0 * radius), 0.0);
        idxs.push(world.add_body(body));
    }
    for i in 0..n - 1 {
        // Anchor at the midpoint between consecutive centers.
        let pa = world.bodies[idxs[i]].position;
        let pb = world.bodies[idxs[i + 1]].position;
        let mid = (pa + pb) * 0.5;
        let mut c = PositionalConstraint::new(idxs[i], idxs[i + 1], 0.0, 1.0 / 200_000.0);
        c.local_anchor_a = world.bodies[idxs[i]].orientation.inverse() * (mid - pa);
        c.local_anchor_b = world.bodies[idxs[i + 1]].orientation.inverse() * (mid - pb);
        world.constraints.push(c);
    }

    let e_start = total_mechanical_energy(&world);
    run_for(&mut world, 3.0);

    for &i in &idxs {
        assert!(finite(world.bodies[i].position), "chain went non-finite");
    }
    let e_end = total_mechanical_energy(&world);
    // Energy must not GROW. Measured behavior (2026-10-19): e_end ≈
    // e_start − 0.150 J, so the 1 mJ headroom is ~150× the distance from the
    // measured value to the bound's physical meaning of "conserved".
    assert!(
        e_end < e_start + 0.001,
        "chain gained energy: start {e_start} J → end {e_end} J (Δ {} J > 1 mJ budget — \
         the engine has no known injector; see docs/findings.md 2026-10-19)",
        e_end - e_start
    );
}

// ════════════════════════════════════════════════════════════════════════
// Multi-point manifold contact
// ════════════════════════════════════════════════════════════════════════

/// Multi-point ground manifold: a single rigid horizontal capsule (axis
/// along world +X, so its two end-spheres form a 2-point footprint) resting
/// with its CoM centered over that base must stay upright and LEVEL over 2 s
/// — AND both endpoints must be simultaneously load-bearing.
///
/// This is the direct guard for "manifold, not single deepest point." With
/// the old per-endpoint penalty pass the body micro-bounced on one endpoint
/// at a time (max 1 simultaneous contact), and any small roll perturbation
/// would not be corrected because only one support point ever pushed back.
/// The SI manifold solves both endpoints together, so the pair shares load
/// and the body levels out.
#[test]
fn manifold_keeps_resting_capsule_level_with_two_active_contacts() {
    let radius = 0.03;
    let half_h = 0.15; // wide base: endpoints 30 cm apart along X
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    let mut body = RigidBody::new_capsule(2.0, half_h, radius);
    // Axis along world +X (rotate local +Y onto +X about world -Z).
    let level = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    // Seed a small initial roll about X (3°) so a single-point contact could
    // NOT recover it; the manifold must drive it back to level.
    let perturb = Quat::from_scaled_axis(Vec3::new(3f64.to_radians(), 0.0, 0.0));
    body.orientation = perturb * level;
    body.position = Vec3::new(0.0, radius, 0.0); // resting height
    let idx = world.add_body(body);

    run_for(&mut world, 2.0);

    let b = &world.bodies[idx];
    assert!(finite(b.position) && finite(b.linear_velocity), "went non-finite");

    // Upright: the capsule axis (local +Y) must remain (anti)parallel to the
    // ground, i.e. its world-Y component must be ~0 (axis stays horizontal).
    let axis_world = b.orientation * Vec3::new(0.0, 1.0, 0.0);
    let tilt_deg = axis_world.y.abs().asin().to_degrees();
    assert!(
        tilt_deg < 2.0,
        "capsule did not stay level: axis tilt {tilt_deg:.2}° (> 2°)"
    );

    // Settled: not drifting/spinning.
    assert!(b.linear_velocity.norm() < 0.02, "still translating: {} m/s", b.linear_velocity.norm());
    assert!(b.angular_velocity.norm() < 0.1, "still rotating: {} rad/s", b.angular_velocity.norm());

    // Rest height preserved (CoM over base, both ends on the floor).
    assert!((b.position.y - radius).abs() < 0.004, "rest height {} m, expected ≈ {radius} m", b.position.y);

    // THE manifold assertion: both endpoints are simultaneously load-bearing.
    let eng = world.engaged_endpoints(idx);
    let active = eng.iter().filter(|e| e.is_some()).count();
    assert!(
        active >= 2,
        "expected >=2 simultaneous active ground contacts (manifold), got {active}"
    );
    // Both should carry positive normal force that together support the weight.
    let total_fn: f64 = eng.iter().filter_map(|e| e.map(|c| c.normal_force)).sum();
    let weight = 2.0 * 9.81;
    assert!(
        (total_fn - weight).abs() < 0.25 * weight,
        "total normal force {total_fn:.2} N should ≈ weight {weight:.2} N"
    );
}


// ════════════════════════════════════════════════════════════════════════
// Pinned bodies — no phantom inertia (the 2026-10 pin fix)
// ════════════════════════════════════════════════════════════════════════

/// Drive a loaded chain hanging off the given pinned anchor and assert the
/// anchor is PERFECTLY inert.
///
/// Rig: pinned anchor at (0, 0.26, 0) → loaded weld (positional constraint
/// with a lever arm on the anchor + three zero-range AngularLimitConstraint
/// locks) → body1 → articulated joint (positional + soft limit + spring +
/// damper + PD + off-axis locks, the foot's add_articulated stack) → body2,
/// which is kicked hard so it swings, strikes the ground plane (contact), and
/// loads every constraint in the chain for the whole run.
///
/// Asserted every frame (120 Hz, 3 s): the anchor's linear AND angular
/// velocity are EXACTLY zero — not small, zero. integrate_velocities and
/// integrate_positions skip inv_mass == 0 bodies and every implicit primitive
/// guards on inv_mass, so the only writers that could deposit velocity into a
/// pinned body are the two SI constraint solvers. Afterwards: the anchor's
/// position and orientation are bit-identical to the start, and body2 has
/// settled at a sane hanging pose (finite, slow, near the rig).
fn assert_pinned_anchor_inert(anchor: RigidBody) {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    let mut anchor = anchor;
    anchor.position = Vec3::new(0.0, 0.26, 0.0);
    let anchor_idx = world.add_body(anchor);

    // body1: welded to the anchor at (0, 0.24, 0) — 2 cm below the anchor
    // center, so the positional constraint has a non-zero lever arm (r_a) on
    // the anchor and its angular-impulse path is exercised, not just the
    // weld's angular locks.
    let mut body1 = RigidBody::new_capsule(0.5, 0.05, 0.02);
    body1.position = Vec3::new(0.0, 0.19, 0.0); // top at 0.26, weld pt inside
    let body1_idx = world.add_body(body1);
    add_weld(&mut world, anchor_idx, body1_idx, Vec3::new(0.0, 0.24, 0.0));

    // body2: hung off body1's bottom end by the articulated stack. Its lowest
    // point starts ~1 cm above the ground, so the kicked swing strikes the
    // ground manifold and keeps contact in the loop.
    let joint_world = Vec3::new(0.0, 0.12, 0.0);
    let mut body2 = RigidBody::new_capsule(0.2, 0.04, 0.015);
    body2.position = Vec3::new(0.0, 0.065, 0.0);
    // The kick: swing hard about the hinge so constraints fire hard.
    body2.linear_velocity = Vec3::new(0.8, 0.0, 0.4);
    body2.angular_velocity = Vec3::new(4.0, 1.0, 0.0);
    let body2_idx = world.add_body(body2);

    let mut c = PositionalConstraint::new(body1_idx, body2_idx, 0.0, 1.0 / 60_000.0);
    c.local_anchor_a = anchor_local(&world, body1_idx, joint_world);
    c.local_anchor_b = anchor_local(&world, body2_idx, joint_world);
    world.constraints.push(c);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        body1_idx, body2_idx, Vec3::x(), -0.7, 0.7, 8000.0, 2.0,
    ));
    world.angular_springs.push(physics::angular_spring::AngularSpring::new(
        body1_idx, body2_idx, Vec3::x(), 4.0, 0.0,
    ));
    world.angular_dampers.push(AngularDamper::new(body1_idx, body2_idx, Vec3::x(), 0.5));
    world.angular_pd_controllers.push(
        AngularPDController::new(body1_idx, body2_idx, Vec3::x(), 8.0, 0.3),
    );
    for off in [Vec3::y(), Vec3::z()] {
        world.angular_constraints.push(AngularLimitConstraint::new(
            body1_idx, body2_idx, off, -0.05, 0.05,
        ));
        world.angular_dampers.push(AngularDamper::new(body1_idx, body2_idx, off, 0.5));
    }

    let start_pos = world.bodies[anchor_idx].position;
    let start_orient = world.bodies[anchor_idx].orientation;
    let body2_start = world.bodies[body2_idx].position;

    let frames = (3.0 / DT).round() as usize;
    for frame in 0..frames {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
        }
        // (a) EXACT zero, every frame. Any phantom-inertia leak through the
        // SI solvers shows up here as a non-zero accumulated velocity.
        let a = &world.bodies[anchor_idx];
        assert_eq!(
            a.angular_velocity,
            Vec3::zeros(),
            "pinned anchor accumulated phantom angular velocity {:?} by frame {frame} \
             (SI constraint applied an impulse through a non-zero inv_inertia_tensor)",
            a.angular_velocity
        );
        assert_eq!(
            a.linear_velocity,
            Vec3::zeros(),
            "pinned anchor accumulated linear velocity {:?} by frame {frame}",
            a.linear_velocity
        );
    }

    // (b) Pose bit-identical: the integrators skip pinned bodies, and with
    // zero velocities nothing else may have moved it.
    let a = &world.bodies[anchor_idx];
    assert_eq!(a.position, start_pos, "pinned anchor position changed");
    assert_eq!(
        a.orientation.coords, start_orient.coords,
        "pinned anchor orientation changed"
    );

    // (c) The hanging chain settled at a sane pose: finite, slow, still near
    // the rig (a weakened weld or phantom-tracking children would drift or
    // keep orbiting the phantom spin).
    let b2 = &world.bodies[body2_idx];
    assert!(
        finite(b2.position) && finite(b2.linear_velocity) && finite(b2.angular_velocity),
        "hanging body went non-finite"
    );
    assert!(
        b2.linear_velocity.norm() < 0.05,
        "hanging body still moving after 3 s: |v| = {} m/s",
        b2.linear_velocity.norm()
    );
    assert!(
        b2.angular_velocity.norm() < 0.5,
        "hanging body still spinning after 3 s: |ω| = {} rad/s",
        b2.angular_velocity.norm()
    );
    let drift = (b2.position - body2_start).norm();
    assert!(
        drift < 0.15,
        "hanging body ended {drift} m from start (chain drifted — weld/joint not holding)"
    );
}

/// REAL BUG GUARDED (found by the 2026-10 full engine review).
/// `RigidBody::new(0.0, inertia)` zeroed `inv_mass` but still inverted the
/// inertia tensor, leaving a pinned body with a full non-zero
/// `inv_inertia_tensor` — a PHANTOM INERTIA. The implicit primitives all
/// guard on `inv_mass == 0.0`, and the integrators skip pinned bodies, but
/// the two SI constraint solvers (`PositionalConstraint::solve_axis`,
/// `AngularLimitConstraint::solve`) applied angular impulses through the
/// phantom tensor: the pinned body's `angular_velocity` accumulated and was
/// NEVER cleared (integrate_positions skips it, so the orientation never
/// moved — the spin was invisible), every `omega_rel` read by dampers,
/// springs, PDs and the constraints' own velocity error was contaminated
/// (children driven to TRACK the phantom spin), and welds to pinned roots
/// were silently weakened (impulse split with the phantom inertia instead of
/// absorbed rigidly).
///
/// Fix (three engine parts, all covered here): `RigidBody::new` zeroes the
/// inverse inertia when mass <= 0; `RigidBody::pin()` is the one correct way
/// to pin an existing body (zeroes both inverse tensors AND velocities); both
/// SI solvers use a zero i_inv for `inv_mass == 0` bodies in the effective
/// mass AND the impulse application (so the dynamic partner receives the
/// FULL impulse, not a split one).
///
/// Both pinning paths are exercised: construction with zero mass, and
/// pinning a live dynamic body mid-flight via `pin()`.
///
/// BITE-VERIFIED (2026-10-19): with the engine fix locally reverted
/// (constructor inverting the tensor again and both SI solvers using the raw
/// `world_inv_inertia()`), this test FAILS in the first frame on the exact-
/// zero phantom angular-velocity assertion for the `RigidBody::new(0.0, ..)`
/// anchor. With the fix restored it passes.
#[test]
fn pinned_body_stays_perfectly_inert_under_si_constraints() {
    // Path 1: pinned at construction. Mat3::identity() is the review's exact
    // reproducer: pre-fix it inverted to a full identity phantom inertia.
    assert_pinned_anchor_inert(RigidBody::new(0.0, Mat3::identity()));

    // Path 2: a live dynamic body (with velocities and a real inertia
    // tensor) pinned via pin(). Must behave identically — pin() zeroes
    // inv_mass, inv_inertia_tensor, AND both velocities.
    let mut live = RigidBody::new_capsule(1.0, 0.05, 0.02);
    live.linear_velocity = Vec3::new(1.0, -2.0, 0.5);
    live.angular_velocity = Vec3::new(3.0, 1.0, -2.0);
    live.pin();
    assert_eq!(live.linear_velocity, Vec3::zeros(), "pin() must zero linear velocity");
    assert_eq!(live.angular_velocity, Vec3::zeros(), "pin() must zero angular velocity");
    assert_pinned_anchor_inert(live);
}

// ════════════════════════════════════════════════════════════════════════
// PD proportional term — implicit integration (the 2026-10 pump fix)
// ════════════════════════════════════════════════════════════════════════

/// One link of a grounded toe ray: capsule params + the joint stack gains
/// connecting it to the previous link.
struct ToeLink {
    mass: f64,
    half_h: f64,
    radius: f64,
    kp: f64,
    kd: f64,
    rom: (f64, f64),
}

/// Build a "toe ray" on the ground plane: a fixed-root parent capsule
/// (metatarsal scale) resting on the ground, plus a chain of toe-bone links,
/// each joined to the previous by the foot's full articulated-joint stack
/// (positional constraint at compliance 1/60000, primary-axis soft limit
/// k = 8000 exp 2, passive spring k = 4, passive damper c = 1.0, active PD,
/// off-axis locks ±0.05 rad with dampers — exactly
/// `foot.rs::add_articulated`). All bones lie along world +Z with their
/// undersides on the ground plane; every hinge is world X (== local X), and
/// a POSITIVE joint angle pitches the child DOWN into the ground.
fn build_toe_ray(
    parent_mhr: (f64, f64, f64),
    links: &[ToeLink],
) -> (XpbdWorld, Vec<usize>) {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    // Capsule long axis (local +Y) along world +Z: rotate +90° about X.
    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Fixed world anchor for the chain root; same orientation as the parent
    // so the weld's zero-relative-rotation locks are satisfied at rest
    // (AngularLimitConstraint has no baseline).
    let mut anchor = RigidBody::new(0.0, Mat3::identity());
    anchor.orientation = along_z;
    let anchor_idx = world.add_body(anchor);

    let (pm, ph, pr) = parent_mhr;
    let mut parent = RigidBody::new_capsule(pm, ph, pr);
    parent.orientation = along_z;
    parent.position = Vec3::new(0.0, pr, 0.0); // resting on the ground
    let parent_idx = world.add_body(parent);
    add_weld(&mut world, anchor_idx, parent_idx, Vec3::new(0.0, pr, 0.0));

    let mut idxs = vec![parent_idx];
    let mut prev_idx = parent_idx;
    let mut prev_tip_z = ph + pr; // world z of the previous capsule surface tip
    let mut prev_r = pr;
    for link in links {
        let joint_world = Vec3::new(0.0, (prev_r + link.radius) * 0.5, prev_tip_z);
        let mut b = RigidBody::new_capsule(link.mass, link.half_h, link.radius);
        b.orientation = along_z;
        b.position = Vec3::new(0.0, link.radius, prev_tip_z + link.half_h + link.radius);
        let idx = world.add_body(b);

        // The foot's articulated-joint stack (mirrors add_articulated).
        let mut c = PositionalConstraint::new(prev_idx, idx, 0.0, 1.0 / 60_000.0);
        c.local_anchor_a = anchor_local(&world, prev_idx, joint_world);
        c.local_anchor_b = anchor_local(&world, idx, joint_world);
        world.constraints.push(c);
        world.angular_soft_limits.push(AngularSoftLimit::new(
            prev_idx, idx, Vec3::x(), link.rom.0, link.rom.1, 8000.0, 2.0,
        ));
        world.angular_springs.push(physics::angular_spring::AngularSpring::new(
            prev_idx, idx, Vec3::x(), 4.0, 0.0,
        ));
        world.angular_dampers.push(AngularDamper::new(prev_idx, idx, Vec3::x(), 1.0));
        world.angular_pd_controllers.push(
            AngularPDController::new(prev_idx, idx, Vec3::x(), link.kp, link.kd),
        );
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

/// The MIGRATED twin of `build_toe_ray`: identical geometry and gains, but
/// each articulated joint's four coaxial legacy primitives (soft limit,
/// spring, damper, PD) are ONE `JointAxis` — exactly the composition
/// `foot.rs::add_articulated` produces since the Stage-2 re-land. Off-axis
/// locks and off-axis dampers stay legacy (different axes, nothing
/// coaxial to combine), as in the migrated foot module.
///
/// ONE deliberate rig deviation from the legacy ray: the metatarsal root
/// is PINNED instead of welded to a pinned anchor while resting on the
/// ground. The welded-grounded root is a REDUNDANT support (weld + ground
/// on one body) and sits exactly on the partial-warm-start preload valve
/// (xpbd.rs `substep` KNOWN LIMITATION): its contact λ_n ratchets to the
/// weld's elastic scale (measured here: 460 N stored on the 0.3 N root at
/// 24+ sweeps, discharging as sustained 615 rad/s bursts at 16). The
/// migrated foot composes no such support — its boundary is a pinned
/// talus behind compliant joints — so the canary pins its root the same
/// way (also the E1/E6 convention). The legacy ray keeps its historical
/// weld (and its silent, bounded preload) untouched.
fn build_toe_ray_ja(
    parent_mhr: (f64, f64, f64),
    links: &[ToeLink],
) -> (XpbdWorld, Vec<usize>) {
    use physics::joint_axis::JointAxis;
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    // The migrated foot's L1 convergence setting (foot.rs::build_foot sets
    // the same): chained JointAxes under contact need ≥ 16 SI sweeps to
    // re-absorb the warm seeds' per-substep velocity injection. At the
    // engine default of 8 this exact rig detonates (see
    // ja_ray_press_dip_velocity_diagnostic: DIP ±2.4 rad past ROM,
    // |ω| 4.8e4 rad/s); the legacy ray keeps the default 8 it was
    // calibrated at.
    world.solver_iterations = 32;

    let along_z = Quat::from_scaled_axis(Vec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let (pm, ph, pr) = parent_mhr;
    let mut parent = RigidBody::new_capsule(pm, ph, pr);
    parent.orientation = along_z;
    parent.position = Vec3::new(0.0, pr, 0.0);
    parent.pin();
    let parent_idx = world.add_body(parent);

    let mut idxs = vec![parent_idx];
    let mut prev_idx = parent_idx;
    let mut prev_tip_z = ph + pr;
    let mut prev_r = pr;
    for link in links {
        let joint_world = Vec3::new(0.0, (prev_r + link.radius) * 0.5, prev_tip_z);
        let mut b = RigidBody::new_capsule(link.mass, link.half_h, link.radius);
        b.orientation = along_z;
        b.position = Vec3::new(0.0, link.radius, prev_tip_z + link.half_h + link.radius);
        let idx = world.add_body(b);

        let mut c = PositionalConstraint::new(prev_idx, idx, 0.0, 1.0 / 60_000.0);
        c.local_anchor_a = anchor_local(&world, prev_idx, joint_world);
        c.local_anchor_b = anchor_local(&world, idx, joint_world);
        world.constraints.push(c);
        world.joint_axes.push(
            JointAxis::new(prev_idx, idx, Vec3::x())
                .with_soft_limit(link.rom.0, link.rom.1, 8000.0, 2.0)
                .with_passive_spring(4.0, 0.0)
                .with_passive_damping(1.0)
                .with_pd(link.kp, link.kd),
        );
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

fn toe5_ray() -> (XpbdWorld, Vec<usize>) {
    // toe5 ray behind a 30 g metatarsal-scale root: pp, mp, dp at the foot's
    // BONES-table values; MTP gains kp = 8 / kd = 0.3, PIP and DIP kp = 4 /
    // kd = 0.15. In this rig + is flexion (down), so the PIP/DIP ROMs
    // [0, 0.61] / [0, 1.05] rest at their straight-toe boundary like the
    // real foot's.
    build_toe_ray((0.030, 0.025, 0.004), &toe5_links())
}

/// toe5_ray's links, shared by the legacy and JointAxis builders.
fn toe5_links() -> [ToeLink; 3] {
    [
        ToeLink { mass: 0.0061, half_h: 0.0060, radius: 0.0040, kp: 8.0, kd: 0.3, rom: (-0.70, 0.70) },
        ToeLink { mass: 0.0022, half_h: 0.0010, radius: 0.0025, kp: 4.0, kd: 0.15, rom: (0.0, 0.61) },
        ToeLink { mass: 0.0024, half_h: 0.0010, radius: 0.0030, kp: 4.0, kd: 0.15, rom: (0.0, 1.05) },
    ]
}

fn toe5_ray_ja() -> (XpbdWorld, Vec<usize>) {
    build_toe_ray_ja((0.030, 0.025, 0.004), &toe5_links())
}

/// DIAGNOSTIC: the JA ray's canary-style three phases with per-joint
/// statics cross-checks, at both sub_dts. This is the measurement the
/// JointAxis pump-canary bands are derived from. Knobs: JA_ITERS=<n>
/// overrides the sweep count, JA_RAMP=1 uses the canary's ramped command
/// protocol (default here: STEP commands — the knife-edge regime, kept
/// steppable precisely so the claw-jam basin stays observable).
/// Run with --ignored --nocapture.
#[test]
#[ignore = "diagnostic harness (Commit C band derivation) — run with --ignored --nocapture"]
fn ja_ray_phase_measurement() {
    for &sub_dt in &[SUB_DT, 1.0 / 3600.0] {
        let (mut world, idxs) = toe5_ray_ja();
        if let Some(it) = std::env::var("JA_ITERS").ok().and_then(|s| s.parse().ok()) {
            world.solver_iterations = it;
        }
        println!("── sub_dt = 1/{:.0}, iterations {} ──", 1.0 / sub_dt, world.solver_iterations);

        let report = |world: &XpbdWorld, m: &ChainMeasure, label: &str| {
            print!("  {label}: max|ω| {:9.3}, last-sec {:9.4} | joint means", m.max_w, m.max_w_last);
            for v in &m.joint_means { print!(" {v:+.4}"); }
            print!(" | τ_tx/demand");
            for (j, ja) in world.joint_axes.iter().enumerate() {
                let theta = ja.angle(&world.bodies);
                let s_lim = ja.limit.as_ref().map_or(0.0, |l| l.scalar_torque(theta));
                let demand = ja.kp * (ja.target_angle - theta)
                    - ja.passive_stiffness * (theta - ja.passive_rest_angle) + s_lim;
                print!(" [{j}] {:+.4}/{:+.4}", ja.applied_total_torque, demand);
            }
            let fn_tot: f64 = idxs.iter().map(|&i| world.total_normal_force(i)).sum();
            println!(" | ΣF_n {fn_tot:.2} N");
        };

        // Ramp the MTP target toward `goal` at 0.3 rad/s (the demo's
        // muscles-don't-step protocol), then hold for the rest of `secs`.
        let run_phase_ramped = |world: &mut XpbdWorld, goal: f64, secs: f64| -> ChainMeasure {
            let frames = (secs / DT).round() as usize;
            let spf = (DT / sub_dt).round() as usize;
            let last_sec_from = frames.saturating_sub((1.0 / DT).round() as usize) * spf;
            let mut max_w: f64 = 0.0;
            let mut max_w_last: f64 = 0.0;
            let n_joints = idxs.len() - 1;
            let mut sums = vec![0.0f64; n_joints];
            let mut n_samples = 0usize;
            let mut substep_i = 0usize;
            for _ in 0..frames {
                let t = world.joint_axes[0].target_angle;
                let step = 0.3 * DT;
                world.joint_axes[0].target_angle = t + (goal - t).clamp(-step, step);
                for _ in 0..spf {
                    world.substep(sub_dt);
                    for &i in &idxs[1..] {
                        let w = world.bodies[i].angular_velocity.norm();
                        if !w.is_finite() || w > max_w { max_w = w; }
                        if substep_i >= last_sec_from && (!w.is_finite() || w > max_w_last) {
                            max_w_last = w;
                        }
                    }
                    if substep_i >= last_sec_from {
                        for (j, pair) in idxs.windows(2).enumerate() {
                            sums[j] += joint_angle(world, pair[0], pair[1]);
                        }
                        n_samples += 1;
                    }
                    substep_i += 1;
                }
            }
            ChainMeasure {
                max_w, max_w_last,
                joint_means: sums.iter().map(|s| s / n_samples as f64).collect(),
            }
        };

        let run_phase = |world: &mut XpbdWorld, secs: f64| -> ChainMeasure {
            // run_chain_measured's cadence but at this sub_dt.
            let frames = (secs / DT).round() as usize;
            let spf = (DT / sub_dt).round() as usize;
            let last_sec_from = frames.saturating_sub((1.0 / DT).round() as usize) * spf;
            let mut max_w: f64 = 0.0;
            let mut max_w_last: f64 = 0.0;
            let n_joints = idxs.len() - 1;
            let mut sums = vec![0.0f64; n_joints];
            let mut n_samples = 0usize;
            let mut substep_i = 0usize;
            for _ in 0..frames {
                for _ in 0..spf {
                    world.substep(sub_dt);
                    for &i in &idxs[1..] {
                        let w = world.bodies[i].angular_velocity.norm();
                        if !w.is_finite() || w > max_w { max_w = w; }
                        if substep_i >= last_sec_from && (!w.is_finite() || w > max_w_last) {
                            max_w_last = w;
                        }
                    }
                    if substep_i >= last_sec_from {
                        for (j, pair) in idxs.windows(2).enumerate() {
                            sums[j] += joint_angle(world, pair[0], pair[1]);
                        }
                        n_samples += 1;
                    }
                    substep_i += 1;
                }
            }
            ChainMeasure {
                max_w, max_w_last,
                joint_means: sums.iter().map(|s| s / n_samples as f64).collect(),
            }
        };

        let p0 = run_phase(&mut world, 3.0);
        report(&world, &p0, "P0 rest ");
        let ramped = std::env::var("JA_RAMP").is_ok();
        if ramped {
            let p1 = run_phase_ramped(&mut world, 0.3, 4.0);
            report(&world, &p1, "P1 press");
            let p2 = run_phase_ramped(&mut world, -0.3, 4.0);
            report(&world, &p2, "P2 lift ");
        } else {
            world.joint_axes[0].target_angle = 0.3;
            let p1 = run_phase(&mut world, 4.0);
            report(&world, &p1, "P1 press");
            world.joint_axes[0].target_angle = -0.3;
            let p2 = run_phase(&mut world, 3.0);
            report(&world, &p2, "P2 lift ");
        }
    }
}

/// DIAGNOSTIC: per-0.1 s trace of the ray's LIFT phase (the sustained
/// limit cycle at 16 iterations; the preload-valve ratchet on the legacy
/// ray). Knobs: JA_ITERS=<n>, JA_LEGACY=1 (trace `toe5_ray` instead of
/// the JointAxis ray). Run with --ignored --nocapture.
#[test]
#[ignore = "diagnostic harness (Commit C isolation) — run with --ignored --nocapture"]
fn ja_ray_lift_trace() {
    let iters: usize = std::env::var("JA_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let legacy = std::env::var("JA_LEGACY").is_ok();
    let (mut world, idxs) = if legacy { toe5_ray() } else { toe5_ray_ja() };
    world.solver_iterations = iters;
    println!("── iters = {iters}, legacy = {legacy} ──");
    let set_target = |world: &mut XpbdWorld, t: f64| {
        if legacy {
            world.angular_pd_controllers[0].target_angle = t;
        } else {
            world.joint_axes[0].target_angle = t;
        }
    };
    // settle + press, as in the canary.
    for _ in 0..((3.0 / SUB_DT) as usize) { world.substep(SUB_DT); }
    set_target(&mut world, 0.3);
    for _ in 0..((4.0 / SUB_DT) as usize) { world.substep(SUB_DT); }
    set_target(&mut world, -0.3);
    let window = (0.1 / SUB_DT) as usize;
    let mut wmax = vec![0.0f64; idxs.len()];
    for i in 0..((3.0 / SUB_DT) as usize) {
        world.substep(SUB_DT);
        for (k, &b) in idxs.iter().enumerate() {
            wmax[k] = wmax[k].max(world.bodies[b].angular_velocity.norm());
        }
        if (i + 1) % window == 0 {
            print!("t={:4.2}", (i + 1) as f64 * SUB_DT);
            print!(" ω[");
            for w in &mut wmax { print!(" {w:9.2}"); *w = 0.0; }
            print!(" ] Fn[");
            for &b in &idxs { print!(" {:7.2}", world.total_normal_force(b)); }
            print!(" ] θ[");
            for pair in idxs.windows(2) {
                print!(" {:+.3}", joint_angle(&world, pair[0], pair[1]));
            }
            print!(" ] y_dp={:.4}", world.bodies[idxs[3]].position.y);
            println!();
        }
    }
}

/// THE PARTIAL-WARM-START PRELOAD VALVE — living reproducer of the KNOWN
/// LIMITATION documented at `XpbdWorld::substep` (found during Commit C).
/// The LEGACY toe ray's root is a capsule WELDED to a pinned anchor while
/// resting on the ground — a redundant support. The ground-manifold rows
/// carry λ across substeps (the L2 lever); the weld's bilateral rows do
/// not. Each substep the carried λ_n re-applies as an up-kick that the
/// seedless weld rows (earlier in every sweep) absorb before the manifold
/// row (last) can see the separating velocity that would unwind it, while
/// every downward residual reaching the manifold row becomes +Δλ_n — a
/// one-way integrator, fed by the chain's per-substep impulse traffic (a
/// constant external torque alone relaxes cleanly — measured). The stored
/// contact force ratchets toward the weld's ELASTIC scale instead of the
/// physical share, discharging as velocity bursts at some sweep counts
/// (the legacy canary's lift phase limit-cycles at 20.9 rad/s under 16
/// default sweeps; the pre-pin JA ray burst at 615 rad/s at 16 and
/// ratcheted quietly to ~460 N at 24/48).
///
/// MEASURED here (legacy ray, default 8 sweeps, lift phase after a press):
/// root F_n climbs 46 → 77 → 95 → 106 → 113 → 117 N across 0.5–3 s on a
/// 0.30 N body, world kinematically quiet throughout. Carrying the
/// positional rows symmetrically was tried and measured WORSE on loop
/// topologies (the foot's weld loops co-inflate through the still-
/// uncarried angular locks: ΣF_n → 2e5 N on the demo foot); the complete
/// fix is UNIFORM warm start across every row family (angular locks
/// included, with unwind-capable accumulation for ranged limits) —
/// deferred. Until then JointAxis assemblies must not compose rigid
/// weld-to-pin supports with ground contact on one body (the migrated
/// foot's boundary — a pinned talus behind compliant joints — does not,
/// and `toe5_ray_ja` pins its root for the same reason).
/// Run with --ignored --nocapture.
#[test]
#[ignore = "known-limitation reproducer (partial-warm-start preload valve) — run with --ignored --nocapture"]
fn legacy_ray_welded_root_preload_valve() {
    let (mut world, idxs) = toe5_ray();
    for _ in 0..((3.0 / SUB_DT) as usize) { world.substep(SUB_DT); }
    world.angular_pd_controllers[0].target_angle = 0.3;
    for _ in 0..((4.0 / SUB_DT) as usize) { world.substep(SUB_DT); }
    world.angular_pd_controllers[0].target_angle = -0.3;
    let root = idxs[0];
    let mut next = 0.5f64;
    for i in 0..((3.0 / SUB_DT) as usize) {
        world.substep(SUB_DT);
        let t = (i + 1) as f64 * SUB_DT;
        if t >= next {
            next += 0.5;
            println!(
                "valve t={t:3.1}s root F_n={:7.2} N (root weight 0.30 N), |v_root|={:.2e} m/s",
                world.total_normal_force(root),
                world.bodies[root].linear_velocity.norm()
            );
        }
    }
    // The world stays quiet and finite — the defect is the stored preload,
    // deliberately NOT asserted (it awaits the uniform-carry fix).
    assert!(world.bodies[root].linear_velocity.norm() < 0.01);
    assert!(world.bodies[root].position.y.is_finite());
}

/// DIAGNOSTIC: the JA ray in FREE SPACE (no ground): does the chained-seed
/// limit cycle need contact, or do the chained joint rows alone sustain
/// it? Run with --ignored --nocapture.
#[test]
#[ignore = "diagnostic harness (Commit C isolation) — run with --ignored --nocapture"]
fn ja_ray_free_space_diagnostic() {
    let (mut world, idxs) = toe5_ray_ja();
    world.no_ground();
    // Hold the chain against gravity by its welded root; command the MTP
    // through the same ±0.3 cycle as the grounded canary.
    for phase in [0.0f64, 0.3, -0.3] {
        world.joint_axes[0].target_angle = phase;
        let m = run_chain_measured(&mut world, &idxs, 2.0);
        println!(
            "free-space target {phase:+.1}: max|ω| {:9.3}, last-sec {:9.5} | joint means {:+.4} {:+.4} {:+.4}",
            m.max_w, m.max_w_last, m.joint_means[0], m.joint_means[1], m.joint_means[2]);
    }
}

/// DIAGNOSTIC: per-0.1 s stats of the DIP hinge velocity on the JointAxis
/// toe ray through a commanded press — the minimal reproducer for the
/// seed-scale limit cycle seen on the migrated demo foot (the L1
/// solver_iterations measurement). Run with --ignored --nocapture.
#[test]
#[ignore = "diagnostic harness (Commit C limit-cycle isolation) — run with --ignored --nocapture"]
fn ja_ray_press_dip_velocity_diagnostic() {
    for iters in [8usize, 16, 32] {
        let (mut world, idxs) = toe5_ray_ja();
        world.solver_iterations = iters;
        println!("── solver_iterations = {iters} ──");
        // 3 s settle.
        for _ in 0..((3.0 / SUB_DT) as usize) {
            world.substep(SUB_DT);
        }
        world.joint_axes[0].target_angle = 0.3;
        let (mp, dp) = (idxs[2], idxs[3]);
        let mut n_sub = 0usize;
        let (mut s, mut mx, mut flips) = (0.0f64, 0.0f64, 0u64);
        let mut last_sign = 0i8;
        let window = (0.1 / SUB_DT) as usize;
        for i in 0..((2.0 / SUB_DT) as usize) {
            world.substep(SUB_DT);
            let n = world.bodies[mp].orientation * Vec3::x();
            let w_rel = (world.bodies[dp].angular_velocity
                - world.bodies[mp].angular_velocity).dot(&n);
            let sign = if w_rel > 0.5 { 1 } else if w_rel < -0.5 { -1 } else { 0 };
            if sign != 0 {
                if last_sign != 0 && sign != last_sign { flips += 1; }
                last_sign = sign;
            }
            s += w_rel.abs();
            mx = mx.max(w_rel.abs());
            n_sub += 1;
            if (i + 1) % window == 0 {
                let dip = joint_angle(&world, mp, dp);
                let lam = world.joint_axes[2].lambda_a + world.joint_axes[2].lambda_p;
                println!(
                    "  t={:4.2} mean|w|={:10.4} max|w|={:10.3} flips={:4} dipθ={:+.4} λ_dip={:+.3e} F_n(dp)={:6.2}",
                    (i + 1) as f64 * SUB_DT, s / n_sub as f64, mx, flips, dip, lam,
                    world.total_normal_force(dp));
                s = 0.0; mx = 0.0; flips = 0; n_sub = 0;
            }
        }
    }
}

fn joint_angle(world: &XpbdWorld, a: usize, b: usize) -> f64 {
    let q_rel = world.bodies[a].orientation.inverse() * world.bodies[b].orientation;
    physics::math::swing_twist_angle(q_rel, Vec3::x())
}

/// Per-phase measurements over a toe ray.
struct ChainMeasure {
    /// max |ω| over every toe bone and every substep of the phase.
    max_w: f64,
    /// max |ω| over every toe bone, last simulated second only (the
    /// sustained-jitter ceiling, free of the command-step transient).
    max_w_last: f64,
    /// Per joint (root-most first): mean joint angle over the last second.
    joint_means: Vec<f64>,
}

/// Advance `seconds` and measure. `idxs` is the full chain (parent first).
fn run_chain_measured(world: &mut XpbdWorld, idxs: &[usize], seconds: f64) -> ChainMeasure {
    let frames = (seconds / DT).round() as usize;
    let last_sec_from = frames.saturating_sub((1.0 / DT).round() as usize) * SUBSTEPS_PER_FRAME;
    let mut max_w: f64 = 0.0;
    let mut max_w_last: f64 = 0.0;
    let n_joints = idxs.len() - 1;
    let mut sums = vec![0.0f64; n_joints];
    let mut n_samples = 0usize;
    let mut substep_i = 0usize;
    for _ in 0..frames {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
            for &i in &idxs[1..] {
                let w = world.bodies[i].angular_velocity.norm();
                if !w.is_finite() || w > max_w {
                    max_w = w;
                }
                if substep_i >= last_sec_from && (!w.is_finite() || w > max_w_last) {
                    max_w_last = w;
                }
            }
            if substep_i >= last_sec_from {
                for (j, pair) in idxs.windows(2).enumerate() {
                    sums[j] += joint_angle(world, pair[0], pair[1]);
                }
                n_samples += 1;
            }
            substep_i += 1;
        }
    }
    ChainMeasure {
        max_w,
        max_w_last,
        joint_means: sums.iter().map(|s| s / n_samples as f64).collect(),
    }
}

/// The 2026-10 PD-proportional-term fix (`AngularPDController::apply_implicit_kp`),
/// guarded from BOTH directions it can regress.
///
/// REAL FAILURE GUARDED. The PD proportional term was the last stiff explicit
/// term left after the 2026-06 implicit rewrite. On the foot's gram-scale
/// phalanges its per-substep kick is past the explicit-integration bound
/// (√(kp/I_red)·sub_dt ≈ 4.4–6.1 on the lesser-toe PIP/DIP joints, where > 2
/// is unstable; I_red ≈ 4·10⁻⁹ kg·m²). In the full 26-bone foot that pumped
/// energy through the ground manifold (toe jitter; a toppled bare foot;
/// detonating loaded toe chains). In this minimal single-ray world the
/// ~90×-supracritical shared c = 1.0 dampers contain the kick instead and
/// RECTIFY it into loss of command authority — the same mechanism e9dda0b
/// measured on a pinned hallux ray that crept back to 0 against 24 N·m of PD
/// torque. Both faces are asserted here: bounded sustained motion (no pump)
/// and retained command authority (no rectification).
///
/// Scenario: the toe5 ray above (metatarsal root welded to the world, pp–mp–dp
/// chain with the foot's full per-joint stack) resting on the ground plane.
/// Three phases: neutral rest, MTP commanded +0.3 rad (pressing the toe into
/// the manifold — the sustained PD-vs-contact fight), MTP flipped to −0.3 rad
/// (lift). The commanded press is realized down the chain: the MTP itself is
/// wedged by the manifold/friction loop, and the DIP — the joint with the
/// smallest inertia, the pump's epicenter — carries the response.
///
/// MEASURED (pre-speculative-contact engine, the values the bounds were
/// originally derived from, ~3× headroom):
///   P0 rest:   max|ω| 4.8e-2, last-sec 3.3e-7 rad/s; joints ≈ 0.
///   P1 press:  max|ω| 42.7 (transient), last-sec 0.167 rad/s;
///              DIP mean +0.2100 rad, PIP −0.086, MTP ≈ 0 (wedged).
///   P2 lift:   max|ω| 208.8 (transient), last-sec 8.0e-3 rad/s;
///              DIP mean −0.1557 rad.
/// RE-MEASURED after speculative ground contact (ea16cf1): transients and
/// settles nearly unchanged (P1 max|ω| 41.4, DIP press +0.2085, DIP lift
/// −0.1606), but sustained jitter COLLAPSED by ~5 orders (P0 3e-12,
/// P1 1.2e-11, P2 2.2e-12 rad/s) — settle-phase micro-embeds were a
/// low-grade excitation source. Bounds kept at the pre-speculative
/// calibration: they still discriminate the failure regimes by orders of
/// magnitude and stay valid if speculative admission is ever tuned.
///
/// EQUILIBRIUM BAND. The DIP band [0.10, 0.35] is around the MEASURED settle,
/// not an analytic prediction: the sequential one-term-at-a-time implicit
/// passes attenuate each term's effective gain by 1/(1 + dt²·w·k) (review
/// formula kp_eff = kp/(1 + dt²·w·kp)): on a gram-scale toe joint
/// (w ≈ 2.5·10⁸, kp = 4, sub_dt = 1/7200) that is ≈ 95% attenuation, so the
/// classic rigid-split prediction kp/(kp + K_spring)·target does NOT hold and
/// the settle also includes manifold support. This dt-dependent effective-gain
/// bias is a KNOWN, documented limitation of the LEGACY sequential passes
/// this ray is built from; its removal is migration onto JointAxis under the
/// coupled in-loop solve + warm-start pair (docs/findings.md 2026-10-21 — the
/// pass-based combined solve preserved free space but throttled under
/// contact, so the fix had to move into the SI loop). Commit C (2026-10-21)
/// migrated the foot and added the parallel `ja_toe_ray_press_canary` below,
/// whose bands ARE the honest wedge statics (see the findings Stage-2
/// re-land entry); this legacy ray stays on the legacy path, keeps the
/// bias, and keeps this band calibrated to it — it is the flag-off
/// pipeline's canary now. Its welded-grounded root also sits on the
/// partial-warm-start preload valve (silently stored contact preload, see
/// `legacy_ray_welded_root_preload_valve`), which its bands predate and
/// tolerate at the default 8 sweeps.
///
/// BITE-VERIFIED against the pre-fix explicit-kp engine (checkout of
/// e9dda0b~1 angular_pd_controller.rs + xpbd.rs): P1 DIP mean 0.0141 rad and
/// P2 DIP mean −0.0345 rad — both outside their bands (kick damper-rectified
/// to ~5–20% authority) — the test FAILS against the old engine.
///
/// ORDERING CANARY (this test doubles as it). `apply_pd_kp_implicit` must run
/// BEFORE the damper passes in `XpbdWorld::substep` so the dampers dissipate
/// its velocity impulse within the same substep. Measured with the call
/// temporarily moved AFTER the dampers: the undamped excursion reaches the SI
/// contact loop and the ray limit-cycles — P1 last-sec ω = 264 rad/s
/// (vs 0.167), P2 last-sec ω = 32.0 rad/s (vs 0.008), P1 max|ω| 272
/// (vs 42.7), P2 max|ω| 1104 (vs 209). The sustained-ω and max-ω bounds
/// below sit between the regimes and FAIL the flipped ordering.
#[test]
fn gram_scale_pd_in_ground_contact_does_not_pump() {
    let (mut world, idxs) = toe5_ray();

    // Phase 0 — 3 s at neutral targets (resting stance).
    let p0 = run_chain_measured(&mut world, &idxs, 3.0);
    for &i in &idxs {
        let b = &world.bodies[i];
        assert!(
            finite(b.position) && finite(b.linear_velocity) && finite(b.angular_velocity),
            "phase 0 (rest) went non-finite"
        );
    }
    assert!(
        p0.max_w_last < 0.01,
        "resting toe chain is not quiet: sustained |ω| {} rad/s (measured 3.3e-7)",
        p0.max_w_last
    );

    // Phase 1 — 4 s with the MTP commanded +0.3 (pressing the toe into the
    // ground): the sustained PD-vs-manifold fight that pumped pre-fix.
    world.angular_pd_controllers[0].target_angle = 0.3;
    let p1 = run_chain_measured(&mut world, &idxs, 4.0);
    for &i in &idxs {
        let b = &world.bodies[i];
        assert!(
            finite(b.position) && finite(b.linear_velocity) && finite(b.angular_velocity),
            "phase 1 (press) went non-finite — PD-vs-manifold energy pump"
        );
    }
    assert!(
        p1.max_w < 130.0,
        "press transient |ω| {} rad/s > 130 (measured 42.7; flipped kp-after-dampers ordering gives 272)",
        p1.max_w
    );
    assert!(
        p1.max_w_last < 0.5,
        "press phase sustains |ω| {} rad/s > 0.5 (measured 0.167; flipped ordering limit-cycles at 264)",
        p1.max_w_last
    );
    let dip_press = p1.joint_means[2];
    assert!(
        (0.10..=0.35).contains(&dip_press),
        "press response lost: DIP mean {dip_press} rad outside [0.10, 0.35] \
         (measured 0.210; the pre-fix explicit-kp engine damper-rectifies to 0.014)"
    );

    // Phase 2 — flip the MTP command to lift (−0.3), 3 s.
    world.angular_pd_controllers[0].target_angle = -0.3;
    let p2 = run_chain_measured(&mut world, &idxs, 3.0);
    for &i in &idxs {
        let b = &world.bodies[i];
        assert!(
            finite(b.position) && finite(b.linear_velocity) && finite(b.angular_velocity),
            "phase 2 (lift) went non-finite — PD-vs-manifold energy pump"
        );
    }
    assert!(
        p2.max_w < 650.0,
        "lift transient |ω| {} rad/s > 650 (measured 208.8; flipped ordering gives 1104)",
        p2.max_w
    );
    assert!(
        p2.max_w_last < 0.05,
        "lift phase sustains |ω| {} rad/s > 0.05 (measured 0.008; flipped ordering limit-cycles at 32)",
        p2.max_w_last
    );
    let dip_lift = p2.joint_means[2];
    assert!(
        (-0.35..=-0.08).contains(&dip_lift),
        "lift response lost: DIP mean {dip_lift} rad outside [−0.35, −0.08] \
         (measured −0.156; the pre-fix engine gives −0.035, flipped ordering +0.0003)"
    );
}

/// Advance `seconds` on the JA ray, ramping the MTP (joint_axes[0]) target
/// toward `goal` at 0.3 rad/s (the demo's muscles-don't-step protocol — a
/// STEP command rams the honest full-transmission chain onto the
/// flat-press/claw-jam buckling knife edge; measured: step presses land in
/// either basin depending non-monotonically on sweeps × sub_dt, while
/// ramped presses are clean and reproducible across the whole
/// {16,32,64,128} × {1/7200, 1/3600} grid). Measures like
/// `run_chain_measured`.
fn run_ja_chain_ramped(
    world: &mut XpbdWorld,
    idxs: &[usize],
    goal: f64,
    seconds: f64,
) -> ChainMeasure {
    let frames = (seconds / DT).round() as usize;
    let last_sec_from = frames.saturating_sub((1.0 / DT).round() as usize) * SUBSTEPS_PER_FRAME;
    let mut max_w: f64 = 0.0;
    let mut max_w_last: f64 = 0.0;
    let n_joints = idxs.len() - 1;
    let mut sums = vec![0.0f64; n_joints];
    let mut n_samples = 0usize;
    let mut substep_i = 0usize;
    for _ in 0..frames {
        let t = world.joint_axes[0].target_angle;
        let step = 0.3 * DT;
        world.joint_axes[0].target_angle = t + (goal - t).clamp(-step, step);
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
            for &i in &idxs[1..] {
                let w = world.bodies[i].angular_velocity.norm();
                if !w.is_finite() || w > max_w { max_w = w; }
                if substep_i >= last_sec_from && (!w.is_finite() || w > max_w_last) {
                    max_w_last = w;
                }
            }
            if substep_i >= last_sec_from {
                for (j, pair) in idxs.windows(2).enumerate() {
                    sums[j] += joint_angle(world, pair[0], pair[1]);
                }
                n_samples += 1;
            }
            substep_i += 1;
        }
    }
    ChainMeasure {
        max_w,
        max_w_last,
        joint_means: sums.iter().map(|s| s / n_samples as f64).collect(),
    }
}

/// Total mechanical + joint-elastic energy of the JA ray's free links.
fn ja_chain_energy(world: &XpbdWorld, idxs: &[usize]) -> f64 {
    let mut e = 0.0;
    for &i in &idxs[1..] {
        let b = &world.bodies[i];
        e += 0.5 * b.mass * b.linear_velocity.dot(&b.linear_velocity);
        let r = b.orientation.to_rotation_matrix();
        let i_world = r.matrix() * b.inertia_tensor * r.matrix().transpose();
        e += 0.5 * b.angular_velocity.dot(&(i_world * b.angular_velocity));
        e += b.mass * 9.81 * b.position.y;
    }
    for ja in &world.joint_axes {
        e += ja.potential_energy(&world.bodies);
    }
    e
}

/// THE MIGRATED PUMP CANARY (Commit C, design E4): the JointAxis twin of
/// `gram_scale_pd_in_ground_contact_does_not_pump`, on `toe5_ray_ja` —
/// same bones, same gains, the coupled in-loop solve + warm-start pair,
/// the migrated foot's 32-sweep setting, ramped commands (see
/// `run_ja_chain_ramped` for why steps are not used). The legacy canary
/// above stays untouched on the legacy path.
///
/// THE RIGID-SPLIT-WITH-CONTACT PREDICTION (each band's derivation):
///
/// P1 ramped press (MTP target +0.3): the flat chain is kinematically
/// WEDGED — the ground blocks flexion of every joint (the links lie on the
/// plane) and the PIP/DIP ROM floors at 0 block extension — so every joint
/// holds its rest angle, θ ≈ 0 (bands ±0.05 rad, ~25× the measured
/// residuals), and the MTP must transmit its FULL static demand
/// kp·(0.3 − 0) = 2.4 N·m through the contacts: `applied_total_torque` =
/// state demand within E1(i)'s 10% band (measured 2.402 vs 2.399, 0.1%).
/// The x-moment balance of the chain about the MTP hinge puts a hard floor
/// under the total contact force: ΣF_i·Δz_i = τ + W·Δz_com ≈ 2.4 N·m with
/// every lever Δz_i ≤ the 45 mm chain length ⇒ ΣF_n ≥ 2.4/0.045 ≈ 53 N
/// (assert ≥ 50; measured 361–386 N across engine-knob states — the
/// distribution is proximal-heavy, and its exact split is statically
/// indeterminate and NOT asserted; see the xpbd.rs KNOWN LIMITATION). The legacy pre-pass transmits ~1/D of the
/// demand, so under it ΣF_n collapses to the ~0.1 N chain weight — this
/// floor is the force-transmission bite. Contrast the LEGACY ray's press:
/// its DIP settles at +0.21 rad — a burst-transient artifact of the
/// throttled sequential passes (its own doc comment); the honest coupled
/// chain does not move, it pushes.
///
/// P2 ramped lift (MTP target −0.3): the chain leaves the ground and the
/// free-space equilibrium is exact statics: θ_MTP = kp/(kp + K_sp)·(−0.3)
/// = 8/12·(−0.3) = −0.200 (band ±0.02 = 10%; measured −0.2000), PIP/DIP
/// at their 0 rest (bands ±0.02; measured −0.0007, +0.0000).
///
/// Quiet bands: sustained (last-second) max |ω| < 0.5 rad/s in the press
/// and < 0.05 in the lift — the LEGACY canary's own untouchable bounds,
/// reused unchanged (measured here: press 0.026, lift 0.0000; the
/// claw-chatter failure regime sustains 200+ rad/s). The press transient
/// is bounded < 10 rad/s (measured 0.059 at the ramp protocol — two
/// orders of headroom — vs 100s–1000s in the chatter regime); the LIFT
/// transient is NOT banded: contact release under carried seeds fires a
/// self-healing velocity burst whose magnitude varies non-monotonically
/// with sweeps × sub_dt (measured 1.3–3.3e5 rad/s across the grid at a
/// clean end state; one substep's un-cancelled seed is w·dt·τ ≈ 8e4 rad/s
/// at toe scale) — documented in docs/findings.md 2026-10-21, awaiting
/// the design's L3 escalation; finiteness and the sustained/means bands
/// still catch every sticky failure.
///
/// ENERGY (E5's running-min form, frame-cadence samples over one settled
/// second after 4 s of press): bound 1e-5 J — the house 1% of the scene's
/// W·l ≈ 2e-3 J scale, same derivation as E5_BOUND in
/// tests/coupled_joint.rs (measured rise: 3.4e-7 J, 30× inside).
///
/// DUAL-RATE (E1(iii) at assembly level): the pressed transmitted torque
/// at 1/3600 with 64 sweeps — MATCHED CONVERGENCE EFFORT, 230k sweeps per
/// simulated second at both rates, because the warm seeds' per-substep
/// velocity injection scales ∝ dt (full derivation at
/// foot.rs::grounded_toe_press_is_substep_rate_independent) — agrees with
/// the 1/7200×32 value within 2% (measured 2.4084 vs 2.4060, 0.1%).
///
/// BITE-VERIFIED (2026-10-21), three independent legs:
/// (1) `coupled_joint_solve = false` (the legacy pre-pass on the same
///     rig): fails the P1 transmission band at fraction 0.0005 (= 1/D at
///     this w), with ΣF_n 0.2 N far under the 50 N floor;
/// (2) `solver_iterations = 8` (the under-converged chain regime): fails
///     at P0 already — the RESTING chain limit-cycles under the seeds;
/// (3) an anti-damper injector (passive_damping → −1.0 on the MTP after
///     the settled hold) pumps 8.5e4 J against the 1e-5 J bound and fails
///     the run.
#[test]
fn ja_toe_ray_press_canary() {
    let (mut world, idxs) = toe5_ray_ja();

    // P0 — rest.
    let p0 = run_ja_chain_ramped(&mut world, &idxs, 0.0, 3.0);
    assert!(
        p0.max_w_last < 0.01,
        "resting JA toe chain is not quiet: sustained |ω| {} rad/s (measured 0.0000)",
        p0.max_w_last
    );

    // P1 — ramped press to +0.3 (1 s ramp + 3 s hold), with the energy
    // monitor over the settled hold (last 2 s).
    let p1a = run_ja_chain_ramped(&mut world, &idxs, 0.3, 2.0);
    let mut e_min = ja_chain_energy(&world, &idxs);
    let mut max_rise: f64 = 0.0;
    let p1b = run_ja_chain_ramped(&mut world, &idxs, 0.3, 2.0);
    // Frame-cadence energy samples over one further settled second.
    for _ in 0..120 {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.substep(SUB_DT);
        }
        let e = ja_chain_energy(&world, &idxs);
        if e - e_min > max_rise {
            max_rise = e - e_min;
        }
        if e < e_min {
            e_min = e;
        }
    }
    let max_w = p1a.max_w.max(p1b.max_w);
    let theta_mtp = joint_angle(&world, idxs[0], idxs[1]);
    let ja = &world.joint_axes[0];
    let s_lim = ja.limit.as_ref().map_or(0.0, |l| l.scalar_torque(theta_mtp));
    let demand = ja.kp * (ja.target_angle - theta_mtp)
        - ja.passive_stiffness * (theta_mtp - ja.passive_rest_angle)
        + s_lim;
    let tau_tx = ja.applied_total_torque;
    let fn_total: f64 = idxs.iter().map(|&i| world.total_normal_force(i)).sum();
    println!(
        "JA canary P1: τ_tx {tau_tx:+.4} vs demand {demand:+.4} N·m (fraction {:+.4}), \
         ΣF_n {fn_total:.1} N, max|ω| {max_w:.3}, last-sec {:.4}, joint means \
         {:+.4} {:+.4} {:+.4}, energy rise {max_rise:.3e} J",
        tau_tx / demand, p1b.max_w_last,
        p1b.joint_means[0], p1b.joint_means[1], p1b.joint_means[2]
    );
    assert!(
        (tau_tx - demand).abs() <= 0.10 * demand.abs(),
        "P1 press transmission lost: τ_tx {tau_tx} vs demand {demand} N·m (> 10% off; \
         the legacy pre-pass transmits 1/D ≈ 0.05% here)"
    );
    assert!(
        fn_total >= 50.0,
        "P1 press ground reaction {fn_total} N below the moment-balance floor \
         τ/l_max ≈ 53 N — the chain is not pushing on the world"
    );
    for (j, mean) in p1b.joint_means.iter().enumerate() {
        assert!(
            mean.abs() <= 0.05,
            "P1 wedged joint {j} mean {mean} rad outside ±0.05 (prediction: the flat \
             chain is kinematically wedged at θ ≈ 0; the claw-jam regime gives +0.5)"
        );
    }
    assert!(
        max_w < 10.0,
        "P1 ramped-press transient |ω| {max_w} rad/s > 10 (measured 0.059; the \
         chatter regime gives 100s–1000s)"
    );
    assert!(
        p1b.max_w_last < 0.5,
        "P1 press sustains |ω| {} rad/s > 0.5 (the legacy canary's own band; \
         measured 0.026)",
        p1b.max_w_last
    );
    assert!(
        max_rise < 1e-5,
        "P1 settled press pumped {max_rise} J above its running minimum (> 1e-5 J)"
    );

    // P2 — ramped lift to −0.3 (2 s ramp + 2 s hold).
    let p2 = run_ja_chain_ramped(&mut world, &idxs, -0.3, 4.0);
    for &i in &idxs {
        let b = &world.bodies[i];
        assert!(
            finite(b.position) && finite(b.linear_velocity) && finite(b.angular_velocity),
            "P2 lift went non-finite"
        );
    }
    println!(
        "JA canary P2: max|ω| {:.3} (transient, unbanded — see doc), last-sec {:.4}, \
         joint means {:+.4} {:+.4} {:+.4}",
        p2.max_w, p2.max_w_last, p2.joint_means[0], p2.joint_means[1], p2.joint_means[2]
    );
    assert!(
        p2.max_w_last < 0.05,
        "P2 lift sustains |ω| {} rad/s > 0.05 (the legacy canary's own band; the \
         claw-chatter regime sustains 200+)",
        p2.max_w_last
    );
    assert!(
        (p2.joint_means[0] + 0.200).abs() <= 0.02,
        "P2 lifted MTP mean {} rad vs the exact free equilibrium kp/(kp+K)·(−0.3) = \
         −0.200 (±0.02)",
        p2.joint_means[0]
    );
    for (j, mean) in p2.joint_means.iter().enumerate().skip(1) {
        assert!(
            mean.abs() <= 0.02,
            "P2 lifted joint {j} mean {mean} rad outside ±0.02 of its 0 rest"
        );
    }

    // DUAL-RATE press at matched convergence effort (1/3600 × 64 sweeps).
    let (mut wc, idxs_c) = toe5_ray_ja();
    wc.solver_iterations = 64;
    let sub_dt_c = 1.0 / 3600.0;
    let mut target = 0.0f64;
    for _ in 0..((4.0 / DT) as usize) {
        let step = 0.3 * DT;
        target = (target + (0.3 - target).clamp(-step, step)).min(0.3);
        wc.joint_axes[0].target_angle = target;
        for _ in 0..((DT / sub_dt_c).round() as usize) {
            wc.substep(sub_dt_c);
        }
    }
    let tau_c = wc.joint_axes[0].applied_total_torque;
    println!("JA canary dual-rate: τ_tx {tau_tx:+.4} @1/7200×32 vs {tau_c:+.4} @1/3600×64");
    let _ = &idxs_c;
    assert!(
        (tau_tx - tau_c).abs() <= 0.02 * tau_tx.abs(),
        "pressed transmitted torque differs across rates at matched convergence \
         effort: {tau_tx} vs {tau_c} N·m (> 2%)"
    );
}

/// The assembly convergence-effort guard (2026-10-22 review fix F2): a
/// world whose assembly recorded a validated sweeps-per-simulated-second
/// floor (`min_si_sweep_rate`, e.g. foot::build_foot's 32 × 7200) must not
/// silently run below it — chained JointAxes under contact are the
/// measured 27%-wrong-statics regime at half the product and detonate at a
/// quarter (docs/findings.md 2026-10-21, Stage-2 re-land finding 1).
/// This drives the guard directly: 32 iterations at sub_dt 1/3600 is half
/// the recorded 230,400 floor and must trip the debug_assert. (In release
/// builds the same condition prints a once-per-world stderr warning
/// instead; the probe binaries surface it.)
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "sweeps-per-simulated-second")]
fn under_converged_sweep_rate_trips_the_assembly_guard() {
    let mut world = XpbdWorld::new();
    world.min_si_sweep_rate = 32.0 * 7200.0;
    world.solver_iterations = 32;
    world.substep(1.0 / 3600.0);
}

/// Companion positive leg: at the validated product itself — at BOTH rate
/// pairings of it — the guard stays quiet. (The real-assembly coverage is
/// the foot's matched-effort dual-rate test; this pins the arithmetic,
/// including the exact-equality boundary the 1e-9 slack protects.)
#[test]
fn validated_sweep_rate_passes_the_assembly_guard() {
    for (iters, sub_dt) in [(32usize, 1.0 / 7200.0), (64, 1.0 / 3600.0)] {
        let mut world = XpbdWorld::new();
        world.min_si_sweep_rate = 32.0 * 7200.0;
        world.solver_iterations = iters;
        world.substep(sub_dt); // must not panic
    }
}

// ════════════════════════════════════════════════════════════════════════
// Coulomb friction — cone threshold, kinetic dissipation, release unwind
// ════════════════════════════════════════════════════════════════════════

/// A 1 kg capsule lying with its long axis along world +X (2-point footprint)
/// at rest height. Pushed ALONG its axis so the response is pure sliding —
/// a push perpendicular to the axis would let it roll instead of testing the
/// friction cone.
fn resting_axial_capsule(world: &mut XpbdWorld) -> usize {
    let radius = 0.03;
    let half_h = 0.15;
    let mut body = RigidBody::new_capsule(1.0, half_h, radius);
    // Local +Y onto world +X (same rotation as the manifold level test).
    body.orientation = Quat::from_scaled_axis(Vec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    body.position = Vec3::new(0.0, radius, 0.0);
    world.add_body(body)
}

/// Advance `seconds` while applying a constant external force to `idx` every
/// substep. Force accumulators clear at the end of every substep, so a
/// sustained load must be re-applied inside the substep loop (applying it
/// once per frame would deliver 1/60 of the intended authority).
fn run_with_constant_force(world: &mut XpbdWorld, idx: usize, force: Vec3, seconds: f64) {
    let frames = (seconds / DT).round() as usize;
    for _ in 0..frames {
        for _ in 0..SUBSTEPS_PER_FRAME {
            world.bodies[idx].apply_force(force);
            world.substep(SUB_DT);
        }
    }
}

/// Static friction cone threshold, pinned from BOTH sides. The engine's
/// ground plane is horizontal-only, so slope loading is emulated by a
/// constant lateral force F_t on a resting capsule of weight W = mg = 9.81 N
/// with μ = 0.6 (the ground default): the Coulomb condition is hold for
/// F_t < μW = 5.886 N, slide for F_t > μW.
///
/// Below the cone (F_t = 0.7·μW = 4.12 N): net horizontal displacement over
/// 2 s must stay under 1 mm. The friction impulse needed per substep
/// (F_t·sub_dt) is 70% of the cone budget (μ·λ_n ≈ μ·W·sub_dt), so the SI
/// solver must cancel the push exactly every substep — any systematic creep
/// is a cone leak.
///
/// Above the cone (F_t = 1.3·μW = 7.65 N): kinetic sliding at
/// a = (1.3 − 1.0)·μ·g ≈ 1.766 m/s² ⇒ ideal displacement ½at² ≈ 3.53 m and
/// ideal final speed ≈ 3.53 m/s over 2 s.
#[test]
fn static_friction_holds_below_cone_and_slides_above() {
    let mu = 0.6;
    let weight = 1.0 * 9.81;

    // ── Below the cone: must NOT slide. ──────────────────────────────────
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    assert_eq!(world.ground.friction_coefficient, mu, "test assumes ground μ = 0.6");
    let idx = resting_axial_capsule(&mut world);
    let start = world.bodies[idx].position;

    run_with_constant_force(&mut world, idx, Vec3::new(0.7 * mu * weight, 0.0, 0.0), 2.0);

    let b = &world.bodies[idx];
    assert!(finite(b.position), "held capsule went non-finite");
    let slip = ((b.position.x - start.x).powi(2) + (b.position.z - start.z).powi(2)).sqrt();
    println!("below-cone horizontal displacement = {:.3e} m", slip);
    assert!(
        slip < 0.001,
        "capsule slid {slip} m under sub-cone load 0.7·μW (static friction must hold; < 1 mm)"
    );

    // ── Above the cone: MUST slide, at the Coulomb kinetic rate. ─────────
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let idx = resting_axial_capsule(&mut world);
    let start = world.bodies[idx].position;

    run_with_constant_force(&mut world, idx, Vec3::new(1.3 * mu * weight, 0.0, 0.0), 2.0);

    let b = &world.bodies[idx];
    assert!(finite(b.position), "sliding capsule went non-finite");
    let travel = b.position.x - start.x;
    let v_end = b.linear_velocity.x;
    println!("above-cone travel = {travel:.4} m, end speed = {v_end:.4} m/s (ideal 3.53 m, 3.53 m/s)");
    // Ideal Coulomb values: 3.53 m and 3.53 m/s. Band ±25%: friction can
    // only act through the cone, so travel below 75% of ideal means extra
    // tangential resistance beyond μ·λ_n (cone violation from above), and
    // travel beyond 125% means friction under-delivering (cone leak).
    assert!(
        (2.65..=4.42).contains(&travel),
        "supra-cone slide traveled {travel} m; Coulomb kinetics predict 3.53 m (±25%)"
    );
    assert!(
        v_end > 2.65,
        "supra-cone slide end speed {v_end} m/s; Coulomb kinetics predict 3.53 m/s"
    );
}

/// Kinetic friction dissipates a velocity kick at a = μg and nothing else.
/// A resting capsule kicked to 1 m/s along its axis must decelerate at
/// μg ≈ 5.886 m/s²: stop time v/μg ≈ 0.17 s, stopping distance
/// v²/(2μg) ≈ 0.0849 m. Asserted on BOTH sides so the test pins the scale,
/// not just "eventually stops": still moving at 0.05 s (Coulomb friction
/// cannot decelerate faster than μg, so v(0.05 s) ≥ 1 − μg·0.05 ≈ 0.71 m/s),
/// and at rest (< 0.01 m/s) by 0.5 s (≈ 3× the ideal stop time).
#[test]
fn kinetic_friction_stops_slide_at_coulomb_rate() {
    let mu = 0.6;
    let g = 9.81;
    let v0 = 1.0;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let idx = resting_axial_capsule(&mut world);
    world.bodies[idx].linear_velocity = Vec3::new(v0, 0.0, 0.0);
    let start = world.bodies[idx].position;

    // 0.05 s = 6 frames: must NOT have stopped yet.
    run_for(&mut world, 0.05);
    let v_mid = world.bodies[idx].linear_velocity.norm();
    println!("speed at 0.05 s = {v_mid:.4} m/s (ideal 0.706)");
    assert!(
        v_mid > 0.5,
        "speed at 0.05 s is {v_mid} m/s — friction decelerating faster than the cone \
         allows (μg bound predicts ≥ 0.71 m/s; > 0.5 with transient headroom)"
    );

    // By 0.5 s (≈ 3× the ideal 0.17 s stop time): at rest.
    run_for(&mut world, 0.45);
    let b = &world.bodies[idx];
    assert!(finite(b.position), "slide-stop capsule went non-finite");
    let v_end = b.linear_velocity.norm();
    assert!(
        v_end < 0.01,
        "still sliding at 0.5 s: |v| = {v_end} m/s (μg = 5.886 m/s² stops 1 m/s in ~0.17 s)"
    );

    // Net displacement ≈ v²/(2μg) = 84.9 mm.
    let travel = b.position.x - start.x;
    let ideal = v0 * v0 / (2.0 * mu * g);
    println!("slide-stop travel = {travel:.4} m (ideal {ideal:.4})");
    assert!(
        (ideal * 0.75..=ideal * 1.25).contains(&travel),
        "slide-stop travel {travel} m; Coulomb kinetics predict v²/(2μg) = {ideal:.4} m (±25%)"
    );
}

/// THE RELEASE-UNWIND BUG (2026-10 engine review). Within one substep's SI
/// iterations, an early iteration can accumulate friction impulse
/// (λ_n > 0, λ_t ≠ 0) and a later iteration can clamp λ_n back to 0 — the
/// contact releasing mid-substep, e.g. another constraint in the Gauss-Seidel
/// sweep yanking the body upward (foot roll-off / CoP migration regime).
/// Pre-fix, `ContactConstraint::solve` early-returned from the friction
/// section on `lambda_normal <= 0.0` BEFORE the cone projection, so the stale
/// tangential impulse stayed applied for the rest of the substep: a
/// systematic Coulomb-cone violation |λ_t| > μ·λ_n = 0. Both contact
/// containers (`ground_manifold`, regenerated per substep, and the
/// user-pushed `contact_constraints`, impulse-reset per substep) run this
/// same solve path, so the stale impulse is never unwound later — the error
/// lands on the body, bounded by one substep's friction impulse per release.
///
/// This test drives `ContactConstraint::solve` through exactly that
/// iteration sequence, deterministically (the mid-sweep upward velocity
/// change stands in for the interleaved constraint):
///   iter 1: penetrating, moving down + sideways ⇒ accumulates λ_n and λ_t;
///   between iters: +2 m/s upward velocity (Gauss-Seidel neighbor);
///   iter 2: separating ⇒ λ_n clamps to 0 — the cone is now |λ_t| ≤ 0;
///   iters 3..8: no-ops (default solver_iterations = 8).
/// The contact geometry puts the contact point straight below the CoM
/// (r ∥ n ⇒ r×n = 0), so the normal impulse cannot touch vx or ω: ANY net
/// change in either is tangential impulse. After release the cone allows
/// exactly zero net tangential impulse.
///
/// MEASURED (2026-10-20): the solve accumulates λ_t = 0.17393 N·s in
/// iteration 1. Pre-fix the release leaves ALL of it on the body (net
/// m·Δvx = −0.17393 N·s, stale ω = 6.354 rad/s from the friction torque).
/// Post-fix the unwind applies −λ_t through the same lever arm, cancelling
/// to float rounding. Bound 1e-9 N·s: ~10⁶× above f64 rounding of these
/// O(0.1) impulses, ~10⁸× below the stale impulse.
///
/// BITE-VERIFIED (2026-10-20): with the cone projection reverted to the
/// plain `lambda_normal <= 0.0` early-return, this test FAILS on the
/// net-tangential-impulse assertion with the full −0.17393 N·s stale
/// impulse. Tests `static_friction_holds_below_cone_and_slides_above` and
/// `kinetic_friction_stops_slide_at_coulomb_rate` pass pre-fix too — they
/// guard the cone generally, not this release path.
#[test]
fn contact_release_unwinds_accumulated_friction_to_the_cone() {
    let mass = 1.0;
    let half_h = 0.1;
    let radius = 0.03;

    // Upright capsule, contact point at the bottom tip, straight below the
    // CoM: local_point ∥ -Y so r×n = 0 (normal impulses can't alter vx/ω).
    let mut body = RigidBody::new_capsule(mass, half_h, radius);
    let tip = half_h + radius;
    body.position = Vec3::new(0.0, tip - 0.001, 0.0); // 1 mm penetration
    let vx0 = 1.0;
    body.linear_velocity = Vec3::new(vx0, -0.5, 0.0); // sliding + pressing
    let mut bodies = vec![body];

    let mut c = ContactConstraint::new(0, Vec3::new(0.0, -tip, 0.0))
        .with_friction(0.6)
        .with_baumgarte(0.0); // manifold constraints run β = 0
    c.reset_impulses();

    // Iteration 1: accumulate normal + friction.
    c.solve(&mut bodies, SUB_DT);
    assert!(c.lambda_normal > 0.0, "setup failed: no normal impulse accumulated");
    let lambda_t_accum = c.lambda_friction.norm();
    assert!(
        lambda_t_accum > 0.05,
        "setup failed: no meaningful friction accumulated (λ_t = {lambda_t_accum} N·s)"
    );
    assert!(
        bodies[0].linear_velocity.x < vx0,
        "setup failed: friction did not engage in iteration 1"
    );

    // Between iterations: a neighboring constraint in the Gauss-Seidel sweep
    // yanks the body upward (release mid-substep).
    bodies[0].linear_velocity.y += 2.0;

    // Iterations 2..8 (default solver_iterations = 8): λ_n clamps to 0 in
    // iter 2; the remaining iterations must leave the released state alone.
    for _ in 0..7 {
        c.solve(&mut bodies, SUB_DT);
    }

    assert_eq!(c.lambda_normal, 0.0, "contact did not release (λ_n = {})", c.lambda_normal);
    // Net tangential impulse actually delivered to the body over the substep,
    // and the stale angular velocity the friction torque would leave behind.
    let net_tangential = mass * (bodies[0].linear_velocity.x - vx0);
    let stale_omega = bodies[0].angular_velocity.norm();
    println!(
        "release substep: accumulated λ_t = {lambda_t_accum:.4} N·s, \
         net delivered = {net_tangential:.3e} N·s, residual ω = {stale_omega:.3e} rad/s"
    );
    assert!(
        net_tangential.abs() < 1e-9,
        "released contact left {net_tangential} N·s of stale tangential impulse on the body \
         (iteration-1 accumulation was {lambda_t_accum} N·s; cone with λ_n = 0 allows none)"
    );
    // The Coulomb cone with λ_n = 0 admits NO tangential impulse.
    assert!(
        c.lambda_friction.norm() <= 1e-12,
        "released contact still holds friction accumulator |λ_t| = {} N·s (cone: |λ_t| ≤ μ·λ_n = 0)",
        c.lambda_friction.norm()
    );
    // The unwind goes through the same lever arm, so the angular kick must
    // cancel too (r×n = 0 ⇒ nothing else touched ω).
    assert!(
        stale_omega < 1e-9,
        "released contact left stale angular velocity {stale_omega} rad/s from friction torque"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Speculative ground contact — fast impacts land on the plane
// ════════════════════════════════════════════════════════════════════════

/// Lowest point (world y) across both end-spheres of a capsule.
fn capsule_bottom_y(world: &XpbdWorld, idx: usize) -> f64 {
    let b = &world.bodies[idx];
    let s = b.capsule_shape.unwrap();
    let axis = b.orientation * Vec3::new(0.0, 1.0, 0.0);
    let e0 = b.position + axis * s.half_h;
    let e1 = b.position - axis * s.half_h;
    (e0.y - s.radius).min(e1.y - s.radius)
}

/// Speculative contact: a fast-dropping capsule must land ON the ground
/// plane, not embed one substep deep into it.
///
/// Positions are frozen during the velocity-level SI loop, so without
/// speculative generation (`generate_ground_manifold` predicting the
/// end-of-substep crossing) a contact is only created AFTER the body has
/// already penetrated — by up to v·sub_dt, which the manifold's Baumgarte=0
/// (load-bearing, see generate_ground_manifold) then never recovers:
/// a permanent embed. MEASURED with speculative generation reverted to
/// penetration>0-only (these exact initial conditions, deterministic):
/// max/settled penetration 0.0150 mm at 3 m/s, 0.5627 mm at 5 m/s,
/// 0.0032 mm at 7.2 m/s (depth is phase-of-crossing dependent, bounded by
/// v·sub_dt = 0.42/0.69/1.0 mm). MEASURED with speculative contact:
/// 0.0002/0.0003/0.0004 mm. The 0.02 mm bound sits ~50× above the
/// speculative result and ~28× below the worst reverted embed — it FAILS
/// the reverted engine at 5 m/s (bite-verified).
#[test]
fn fast_impact_lands_on_plane_without_embedding() {
    for impact_speed in [3.0, 5.0, 7.2] {
        let mut world = XpbdWorld::new();
        world.set_ground(0.0);

        // Foot-scale capsule, long axis horizontal (local +y → world +z):
        // both endpoints strike together, a 2-point manifold like a sole.
        let mut cap = RigidBody::new_capsule(1.0, 0.10, 0.03);
        cap.orientation = Quat::from_scaled_axis(
            Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
        // 5 mm above contact, already moving at impact speed (the end of a
        // free fall without simulating the fall).
        cap.position = Vec3::new(0.0, 0.03 + 0.005, 0.0);
        cap.linear_velocity = Vec3::new(0.0, -impact_speed, 0.0);
        let idx = world.add_body(cap);

        let mut max_pen: f64 = 0.0;
        for _ in 0..((0.3 / SUB_DT) as usize) {
            world.substep(SUB_DT);
            max_pen = max_pen.max(-capsule_bottom_y(&world, idx));
        }

        const PEN_BOUND: f64 = 2.0e-5; // 20 µm — see header for derivation
        assert!(
            max_pen < PEN_BOUND,
            "{impact_speed} m/s impact embedded {:.4} mm > 0.02 mm \
             (speculative contact measured 0.0002–0.0004 mm; one-substep-late \
             contact embeds up to v·sub_dt = {:.2} mm)",
            max_pen * 1e3,
            impact_speed * SUB_DT * 1e3
        );
        // Settles cleanly: resting on the plane, supported by its weight,
        // no residual motion (no bounce survives 300 ms).
        let settled = capsule_bottom_y(&world, idx);
        assert!(
            settled.abs() < PEN_BOUND,
            "{impact_speed} m/s impact settled at {settled} m, not on the plane"
        );
        let v_end = world.bodies[idx].linear_velocity.norm();
        assert!(
            v_end < 1e-3,
            "{impact_speed} m/s impact still moving at {v_end} m/s after 300 ms"
        );
        let support = world.total_normal_force(idx);
        assert!(
            (support - 9.81).abs() < 0.5,
            "{impact_speed} m/s impact settled with ground support {support} N \
             (expected ≈ weight 9.81 N)"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// AngularPDController torque_limit — one budget across KP + FF + KD
// ════════════════════════════════════════════════════════════════════════

/// `torque_limit` caps the controller's TOTAL output. The KP+FF pass clamps
/// its own applied torque and records it (`applied_kp_torque`); the KD pass
/// (a later pass of the same substep) clamps its torque into the remaining
/// budget. Scenario where BOTH terms individually overspend a 0.5 N·m limit:
/// angle error 0.5 rad at kp = 50 (raw 25 N·m) and ω_rel = 10 rad/s at
/// kd = 5 (implicit KD torque ≈ 40 N·m). The total applied torque is
/// reconstructed exactly from the velocity delta across the two passes:
/// τ_total = Δ(ω·n)/(w·sub_dt) with w = n·I_inv·n (partner pinned).
///
/// BITE-VERIFIED (both legs of the budget):
/// - KD clamp disabled: τ_total = −42.25 N·m — fails the |τ| ≤ limit assert.
/// - KP recording zeroed (budget blind to the KP spend): KD clamps to the
///   full ±limit on its own, τ_total = −1.0 N·m = 2× limit — fails.
/// With `torque_limit` left at the default INFINITY the same scenario
/// delivers |τ| ≫ limit (asserted below), so the capped leg cannot pass
/// vacuously.
#[test]
fn pd_torque_limit_caps_total_kp_plus_kd_torque() {
    const LIMIT: f64 = 0.5;
    let axis = Vec3::x();

    let run = |torque_limit: f64| -> (f64, f64) {
        let mut anchor = RigidBody::new_capsule(1.0, 0.1, 0.03);
        anchor.pin();
        let mut arm = RigidBody::new_capsule(1.0, 0.1, 0.03);
        // 0.5 rad from the target (0) about the axis, swinging away at
        // 10 rad/s — both PD terms pull/brake in the same (−) direction.
        arm.orientation = Quat::from_scaled_axis(Vec3::new(0.5, 0.0, 0.0));
        arm.angular_velocity = Vec3::new(10.0, 0.0, 0.0);
        let mut bodies = vec![anchor, arm];

        let mut ctrl = AngularPDController::new(0, 1, axis, 50.0, 5.0)
            .with_torque_limit(torque_limit);

        let n = bodies[0].orientation * axis;
        let w = n.dot(&(bodies[1].world_inv_inertia() * n)); // body 0 pinned
        let omega_before = bodies[1].angular_velocity.dot(&n);

        // The two engine passes of one substep, in substep order.
        ctrl.apply_implicit_kp(&mut bodies, SUB_DT);
        ctrl.apply_implicit_damping(&mut bodies, SUB_DT);

        let omega_after = bodies[1].angular_velocity.dot(&n);
        let tau_total = (omega_after - omega_before) / (w * SUB_DT);
        (tau_total, ctrl.applied_kp_torque)
    };

    // Uncapped: the same scenario really would overspend the budget.
    let (tau_uncapped, _) = run(f64::INFINITY);
    assert!(
        tau_uncapped.abs() > 10.0 * LIMIT,
        "scenario does not overspend uncapped (τ = {tau_uncapped} N·m) — \
         the capped assertion below would be vacuous"
    );

    // Capped: total respects the limit, and the controller is actually
    // producing torque at the limit (not a no-op).
    let (tau_total, applied_kp) = run(LIMIT);
    assert!(
        applied_kp.abs() <= LIMIT * (1.0 + 1e-12),
        "KP pass overspent the budget on its own: {applied_kp} N·m"
    );
    assert!(
        tau_total.abs() <= LIMIT * (1.0 + 1e-9),
        "total PD torque {tau_total} N·m exceeds torque_limit {LIMIT} \
         (KP applied {applied_kp}; the KD term must fit the remaining budget)"
    );
    assert!(
        tau_total < -0.9 * LIMIT,
        "controller under-delivers: τ = {tau_total} N·m, expected saturation \
         near −{LIMIT} (both terms demand − torque far beyond the limit)"
    );
}
