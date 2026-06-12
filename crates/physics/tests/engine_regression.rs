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

/// The exact failure mode the rewrite was made to fix: a ~2 g distal-phalanx-
/// mass body tied by a stiff constraint to a heavy fixed anchor, under gravity.
/// Explicit integration of a stiff coupling exploded such tiny bodies; the
/// implicit scheme must keep it finite and pinned near rest.
#[test]
fn tiny_mass_body_with_stiff_constraint_is_stable() {
    let mut world = XpbdWorld::new();
    world.no_ground();

    // Heavy fixed anchor (inv_mass = 0 ⇒ never moves).
    let anchor = RigidBody::new(0.0, Mat3::identity());
    let anchor_idx = world.add_body(anchor);

    // 2 g distal-phalanx-scale body, 1 cm below the anchor.
    let mut tiny = RigidBody::new_capsule(0.002, 0.0035, 0.0028);
    let rest = Vec3::new(0.0, -0.01, 0.0);
    tiny.position = rest;
    let tiny_idx = world.add_body(tiny);

    // Stiff ball-and-socket joining the anchor origin to the tiny body's
    // origin (zero-length, near-rigid). Anchors coincide at rest ⇒ no preload.
    let mut c = PositionalConstraint::new(anchor_idx, tiny_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = rest; // anchor's local point that coincides with tiny center
    c.local_anchor_b = Vec3::zeros();
    world.constraints.push(c);

    run_for(&mut world, 3.0);

    let b = &world.bodies[tiny_idx];
    assert!(finite(b.position) && finite(b.linear_velocity), "tiny body exploded (non-finite)");
    // Must stay within a few cm of rest — no blow-up, no runaway drift.
    let off = (b.position - rest).norm();
    assert!(off < 0.03, "tiny body drifted {off} m from rest (instability)");
    assert!(b.linear_velocity.norm() < 0.5, "tiny body velocity {} m/s (instability)", b.linear_velocity.norm());
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

    // Zero-rest-length constraint between the two centers. Stiff (compliance
    // small) so it converges to coincident anchors.
    let mut c = PositionalConstraint::new(a_idx, b_idx, 0.0, 1.0 / 100_000.0);
    c.local_anchor_a = Vec3::zeros();
    c.local_anchor_b = Vec3::zeros();
    world.constraints.push(c);

    // Gravity still acts (no_ground only removes the floor), so the pair
    // falls together; what we assert is the *relative* anchor coincidence,
    // which is unaffected by the common free-fall.
    run_for(&mut world, 2.0);

    let pa = world.bodies[a_idx].position + world.bodies[a_idx].orientation * Vec3::zeros();
    let pb = world.bodies[b_idx].position + world.bodies[b_idx].orientation * Vec3::zeros();
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
/// NOTE: this engine has a documented XPBD energy artifact (~0.024 J/s
/// injection from the constraint solver — see memory
/// `project_engine_energy_artifact`). So we assert energy growth is BOUNDED,
/// not exactly conserved: total energy at t = 3 s must not exceed the start
/// energy by more than 0.2 J (≈ 8× the documented 0.024 J/s × 3 s artifact
/// budget, leaving headroom; a real instability would blow far past this).
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
    // Energy must not GROW beyond the documented-artifact budget. A genuine
    // explicit-Euler instability would push e_end orders of magnitude higher.
    assert!(
        e_end < e_start + 0.2,
        "chain gained energy: start {e_start} J → end {e_end} J (Δ {} J > 0.2 J budget)",
        e_end - e_start
    );
}
