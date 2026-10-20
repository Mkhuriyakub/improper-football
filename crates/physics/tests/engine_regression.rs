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

fn toe5_ray() -> (XpbdWorld, Vec<usize>) {
    // toe5 ray behind a 30 g metatarsal-scale root: pp, mp, dp at the foot's
    // BONES-table values; MTP gains kp = 8 / kd = 0.3, PIP and DIP kp = 4 /
    // kd = 0.15. In this rig + is flexion (down), so the PIP/DIP ROMs
    // [0, 0.61] / [0, 1.05] rest at their straight-toe boundary like the
    // real foot's.
    build_toe_ray(
        (0.030, 0.025, 0.004),
        &[
            ToeLink { mass: 0.0061, half_h: 0.0060, radius: 0.0040, kp: 8.0, kd: 0.3, rom: (-0.70, 0.70) },
            ToeLink { mass: 0.0022, half_h: 0.0010, radius: 0.0025, kp: 4.0, kd: 0.15, rom: (0.0, 0.61) },
            ToeLink { mass: 0.0024, half_h: 0.0010, radius: 0.0030, kp: 4.0, kd: 0.15, rom: (0.0, 1.05) },
        ],
    )
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
/// MEASURED (current engine, the values the bounds derive from, ~3× headroom):
///   P0 rest:   max|ω| 4.8e-2, last-sec 3.3e-7 rad/s; joints ≈ 0.
///   P1 press:  max|ω| 42.7 (transient), last-sec 0.167 rad/s;
///              DIP mean +0.2100 rad, PIP −0.086, MTP ≈ 0 (wedged).
///   P2 lift:   max|ω| 208.8 (transient), last-sec 8.0e-3 rad/s;
///              DIP mean −0.1557 rad.
///
/// EQUILIBRIUM BAND. The DIP band [0.10, 0.35] is around the MEASURED settle,
/// not an analytic prediction: the sequential one-term-at-a-time implicit
/// passes attenuate each term's effective gain by 1/(1 + dt²·w·k) (review
/// formula kp_eff = kp/(1 + dt²·w·kp)): on a gram-scale toe joint
/// (w ≈ 2.5·10⁸, kp = 4, sub_dt = 1/7200) that is ≈ 95% attenuation, so the
/// classic rigid-split prediction kp/(kp + K_spring)·target does NOT hold and
/// the settle also includes manifold support. This dt-dependent effective-gain
/// bias is a KNOWN, documented limitation (docs/findings.md 2026-10), to be
/// removed by a combined coaxial-stiffness solve when Phase 3 needs accurate
/// toe tracking.
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
