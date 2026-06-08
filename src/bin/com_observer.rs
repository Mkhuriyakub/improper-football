// Phase 3 step 2 prep: validate the CoM observer primitives on
// `XpbdWorld` (`com_position`, `com_velocity`, `com_position_of`,
// `com_velocity_of`).
//
// These are pure observation — no control logic. They're the first
// piece needed for whole-body coordinated balance: the support-polygon
// / ZMP estimator reads CoM, the multi-actuator dispatcher uses CoM
// as feedback. Validating them in isolation before building anything
// on top.
//
// Three scenarios:
//   1. Static two-body system — CoM math vs analytical answer.
//   2. Free-fall single body — CoM trajectory under gravity.
//   3. Balanced 1c chain — CoM stays stationary while balanced.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

fn place_child(
    parent: &RigidBody, parent_distal: Vec3, child_proximal: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: Vec3, ca: Vec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = pa; s.local_anchor_b = ca;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = pa; d.local_anchor_b = ca;
    world.positional_dampers.push(d);
}

fn approx_eq(a: Vec3, b: Vec3, tol: f64) -> bool {
    (a - b).norm() < tol
}

// ─── Scenario 1: Static two-body system ────────────────────────────────

fn scenario_static() {
    println!("=== Scenario 1: static two-body CoM ===");
    let mut world = XpbdWorld::new();

    // Body A: mass 1.0 at (0, 0, 0)
    let mut a = RigidBody::new(1.0, Mat3::identity());
    a.position = Vec3::new(0.0, 0.0, 0.0);
    let a_idx = world.add_body(a);

    // Body B: mass 3.0 at (4, 0, 0). 3:1 mass ratio puts CoM at x=3.
    let mut b = RigidBody::new(3.0, Mat3::identity());
    b.position = Vec3::new(4.0, 0.0, 0.0);
    let b_idx = world.add_body(b);

    let total_mass = world.total_mass();
    let com = world.com_position();
    let expected = Vec3::new(3.0, 0.0, 0.0);
    let pass_total = (total_mass - 4.0).abs() < 1e-12;
    let pass_com = approx_eq(com, expected, 1e-12);

    println!("  total_mass: {} (expected 4.0) — {}",
             total_mass, if pass_total { "✓" } else { "✗" });
    println!("  com_position: ({:.3}, {:.3}, {:.3}) (expected ({:.3}, {:.3}, {:.3})) — {}",
             com.x, com.y, com.z, expected.x, expected.y, expected.z,
             if pass_com { "✓" } else { "✗" });

    // Subset queries.
    let com_a = world.com_position_of(&[a_idx]);
    let com_b = world.com_position_of(&[b_idx]);
    let pass_subset_a = approx_eq(com_a, Vec3::zeros(), 1e-12);
    let pass_subset_b = approx_eq(com_b, Vec3::new(4.0, 0.0, 0.0), 1e-12);
    println!("  com_of([a]) = single body's position — {}", if pass_subset_a { "✓" } else { "✗" });
    println!("  com_of([b]) = single body's position — {}", if pass_subset_b { "✓" } else { "✗" });

    let com_velocity = world.com_velocity();
    let pass_v = approx_eq(com_velocity, Vec3::zeros(), 1e-12);
    println!("  com_velocity at rest = 0 — {}", if pass_v { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 2: Free-fall single body ─────────────────────────────────

fn scenario_freefall() {
    println!("=== Scenario 2: free-fall single body ===");
    let mut world = XpbdWorld::new();

    // One body, no constraints, no contacts.
    let mut body = RigidBody::new(2.0, Mat3::identity());
    body.position = Vec3::new(0.0, 10.0, 0.0);
    world.add_body(body);

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;

    let mut t = 0.0;
    let initial_y = world.com_position().y;
    let mut max_error = 0.0_f64;

    while t < 1.0 {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        t += dt;

        let com = world.com_position();
        let com_v = world.com_velocity();
        // Analytical: y(t) = y0 - 0.5*g*t², vy(t) = -g*t.
        let expected_y = initial_y - 0.5 * 9.81 * t * t;
        let expected_vy = -9.81 * t;
        let err_y = (com.y - expected_y).abs();
        let err_vy = (com_v.y - expected_vy).abs();
        if err_y > max_error { max_error = err_y; }

        if (t - 0.25).abs() < dt * 0.6 || (t - 0.5).abs() < dt * 0.6
            || (t - 0.75).abs() < dt * 0.6 || (t - 1.0).abs() < dt * 0.6 {
            println!("  t={:.2}s  com.y={:+.4} (expected {:+.4})  com_v.y={:+.4} (expected {:+.4})  err_y={:.5}  err_vy={:.5}",
                     t, com.y, expected_y, com_v.y, expected_vy, err_y, err_vy);
        }
    }

    let pass = max_error < 0.001;
    println!("  max |com.y - analytical| over 1s: {:.6} m — {}",
             max_error, if pass { "✓ (within 1 mm)" } else { "✗" });
    println!();
}

// ─── Scenario 3: Balanced 1c chain ─────────────────────────────────────

fn scenario_balanced_chain() {
    println!("=== Scenario 3: balanced 1c chain — CoM stays stationary ===");

    // Geometry/joint constants from balance_torso.rs.
    const FOOT_MASS: f64 = 1.0;
    const FOOT_HALF_H: f64 = 0.090;
    const FOOT_RADIUS: f64 = 0.032;
    let foot_anchor = Vec3::new(0.0, FOOT_HALF_H, 0.0);
    const SHIN_MASS: f64 = 3.0; const SHIN_HALF_H: f64 = 0.21; const SHIN_RADIUS: f64 = 0.05;
    const THIGH_MASS: f64 = 7.0; const THIGH_HALF_H: f64 = 0.225; const THIGH_RADIUS: f64 = 0.07;
    const PELVIS_MASS: f64 = 8.0; const PELVIS_HALF_H: f64 = 0.10; const PELVIS_RADIUS: f64 = 0.13;
    const TORSO_MASS: f64 = 40.0; const TORSO_HALF_H: f64 = 0.30; const TORSO_RADIUS: f64 = 0.13;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    let mut torso = RigidBody::new_capsule(TORSO_MASS, TORSO_HALF_H, TORSO_RADIUS);
    torso.position = Vec3::new(0.0, 1.402, 0.0);
    let torso_idx = world.add_body(torso);

    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    let (pos, orient) = place_child(&world.bodies[torso_idx],
        Vec3::new(0.0, -TORSO_HALF_H, 0.0),
        Vec3::new(0.0,  PELVIS_HALF_H, 0.0), id);
    pelvis.position = pos; pelvis.orientation = orient;
    let pelvis_idx = world.add_body(pelvis);

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        Vec3::new(0.0, -PELVIS_HALF_H, 0.0),
        Vec3::new(0.0,  THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), foot_anchor, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    // Joints (lifted from balance_torso.rs).
    add_elastic_joint(&mut world, torso_idx, pelvis_idx,
        Vec3::new(0.0, -TORSO_HALF_H, 0.0), Vec3::new(0.0, PELVIS_HALF_H, 0.0),
        300_000.0, 600.0);
    let lumbar_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(torso_idx, pelvis_idx, lumbar_axis,
        -0.44, 1.05, 200.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(torso_idx, pelvis_idx, lumbar_axis, 25.0));
    world.angular_springs.push(AngularSpring::new(torso_idx, pelvis_idx, lumbar_axis, 150.0, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        torso_idx, pelvis_idx, lumbar_axis, 1500.0, 150.0));

    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        Vec3::new(0.0, -PELVIS_HALF_H, 0.0), Vec3::new(0.0, THIGH_HALF_H, 0.0),
        200_000.0, 400.0);
    let hip_axis = Vec3::new(-1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_axis,
        -0.35, 2.09, 120.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_axis, 7.0));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_axis, 10.0, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, thigh_idx, hip_axis, 2500.0, 250.0));

    add_elastic_joint(&mut world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0), Vec3::new(0.0, SHIN_HALF_H, 0.0),
        80_000.0, 200.0);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
        -0.087, 2.44, 80.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_axis, 1.5));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_axis, 5.0, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        thigh_idx, shin_idx, knee_axis, 3000.0, 300.0));

    add_elastic_joint(&mut world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), foot_anchor, 60_000.0, 150.0);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
        -0.44, 0.87, 80.0, 2.0).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, 1.2));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, pf_axis, 4.0, 0.0)
        .with_baseline(foot_baseline));

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. Pre-migration shin/thigh/pelvis/torso had
    // per-body disable hacks; now they collide if the chain tips far enough.

    // Body subset for "everything that's part of the standing chain."
    // (All 5 here, but the subset API matters for cases where a separate
    // pinned base or external object is also in the world.)
    let chain_indices = [foot_idx, shin_idx, thigh_idx, pelvis_idx, torso_idx];

    let bal_kp: f64 = 1000.0;
    let bal_kd: f64 = 235.0;
    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;

    let initial_com = world.com_position();
    let initial_chain_com = world.com_position_of(&chain_indices);
    let mut max_drift = 0.0_f64;

    let mut t = 0.0;
    while t < 5.0 {
        for _ in 0..sub_iters {
            let thigh = &world.bodies[thigh_idx];
            let tilt_x  = thigh.orientation.scaled_axis().x;
            let omega_x = thigh.angular_velocity.x;
            let tau_x = -bal_kp * tilt_x - bal_kd * omega_x;
            let tau_world = Vec3::new(tau_x, 0.0, 0.0);
            world.bodies[shin_idx].apply_torque(tau_world);
            world.bodies[foot_idx].apply_torque(-tau_world);
            world.substep(sub_dt);
        }
        t += dt;

        let drift = (world.com_position() - initial_com).norm();
        if drift > max_drift { max_drift = drift; }
    }

    let final_com = world.com_position();
    let final_chain_com = world.com_position_of(&chain_indices);
    let total_mass = world.total_mass();
    let chain_mass = world.mass_of(&chain_indices);

    println!("  total mass: {:.2} kg, chain subset mass: {:.2} kg (should be equal — all bodies are in chain) — {}",
             total_mass, chain_mass,
             if (total_mass - chain_mass).abs() < 1e-9 { "✓" } else { "✗" });
    println!("  initial CoM:   ({:+.4}, {:+.4}, {:+.4})",
             initial_com.x, initial_com.y, initial_com.z);
    println!("  final CoM:     ({:+.4}, {:+.4}, {:+.4})",
             final_com.x, final_com.y, final_com.z);
    println!("  initial chain CoM matches whole-world CoM — {}",
             if approx_eq(initial_com, initial_chain_com, 1e-12) { "✓" } else { "✗" });
    println!("  final chain CoM matches whole-world CoM — {}",
             if approx_eq(final_com, final_chain_com, 1e-12) { "✓" } else { "✗" });
    // Drift threshold of 5 cm — generous enough to absorb the posture-PD
    // settle (each PD has equilibrium bend ≈ gravity_moment/KP rad, so the
    // chain sags slightly compared to the analytical neutral pose) plus
    // the ground-contact penalty equilibrium (~7 mm). The ESSENTIAL check
    // is that CoM doesn't wander — i.e. the body remains roughly balanced
    // over the support. A monotonic drift would indicate the controller
    // failing.
    println!("  max |CoM drift| over 5s: {:.5} m — {}",
             max_drift, if max_drift < 0.05 { "✓ (under 5 cm — within posture-PD settle)" } else { "✗" });
    println!();
}

fn main() {
    println!("CoM observer validation\n");
    scenario_static();
    scenario_freefall();
    scenario_balanced_chain();
}
