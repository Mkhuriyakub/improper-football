// Validates the `ContactConstraint` primitive — implicit velocity-level
// contact via sequential impulse, replacing explicit penalty springs.
//
// Five scenarios:
//   1. Single body, single contact point at body bottom — settles on ground
//      with correct equilibrium position and force.
//   2. Multiple contact points on one body (tripod) — distributes load.
//   3. Heavy-load chain stress test — the exact scenario that broke
//      explicit FootContact at 56 kg load. ContactConstraint should
//      handle it without instability.
//   4. Friction — sliding body decelerates and stops.
//   5. Lifted body — no impulse, body falls freely.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::contact_constraint::ContactConstraint;
use physics::constraints::PositionalConstraint;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Mat3, Vec3};
use nalgebra::UnitQuaternion;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(i_trans, 0.0, 0.0, 0.0, i_axial, 0.0, 0.0, 0.0, i_trans)
}

fn settle(world: &mut XpbdWorld, sub_dt: f64, sub_iters: usize, frames: usize) {
    for _ in 0..frames {
        for _ in 0..sub_iters { world.substep(sub_dt); }
    }
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

// ─── Scenario 1: single body, single contact ────────────────────────────

fn scenario_single_contact() {
    println!("=== Scenario 1: single body, single contact at bottom ===");
    let mut world = XpbdWorld::new();
    let mut body = RigidBody::new(1.0, capsule_inertia(1.0, 0.05, 0.10));
    body.position = Vec3::new(0.0, 0.30, 0.0);  // drop from 0.20 m above eq
    let body_idx = world.add_body(body);
    // Contact point at body's local -Y end (the bottom).
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new(0.0, -0.15, 0.0)));  // 0.10 half_h + 0.05 radius

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let body = &world.bodies[body_idx];
    let final_y = body.position.y;
    let final_vy = body.linear_velocity.y;
    let lambda = world.contact_constraints[0].lambda_normal;

    // At equilibrium with a unilateral constraint, the body's bottom
    // sits at y=0 → body center at y = 0.15.
    let pass_position = approx_eq(final_y, 0.15, 0.001);
    let pass_velocity = final_vy.abs() < 0.01;
    // Normal impulse per substep ≈ m·g·sub_dt = 1·9.81·sub_dt.
    let expected_lambda = 1.0 * 9.81 * sub_dt;
    let pass_lambda = approx_eq(lambda, expected_lambda, expected_lambda * 0.20);

    println!("  final body.y = {:.4} (expected 0.150) — {}",
             final_y, if pass_position { "✓" } else { "✗" });
    println!("  final body.v.y = {:.6} (expected ≈ 0) — {}",
             final_vy, if pass_velocity { "✓" } else { "✗" });
    println!("  per-substep normal impulse λ = {:.6} (expected ≈ {:.6}) — {}",
             lambda, expected_lambda, if pass_lambda { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 2: tripod on one body ─────────────────────────────────────

fn scenario_tripod() {
    println!("=== Scenario 2: tripod (3 contact points on one body) ===");
    let mut world = XpbdWorld::new();
    let mut body = RigidBody::new(1.0, capsule_inertia(1.0, 0.05, 0.10));
    body.position = Vec3::new(0.0, 0.30, 0.0);
    let body_idx = world.add_body(body);

    // 3 contact points at the body's bottom, in a triangular layout.
    let r = 0.04;
    let bottom_y = -0.15;
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new(0.0,  bottom_y, -r)));     // back
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new(-r,   bottom_y,  r*0.5))); // front-left
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new( r,   bottom_y,  r*0.5))); // front-right

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let body = &world.bodies[body_idx];
    let final_y = body.position.y;
    let total_lambda: f64 = world.contact_constraints.iter().map(|c| c.lambda_normal).sum();
    let expected_total = 1.0 * 9.81 * sub_dt;

    let pass_y = approx_eq(final_y, 0.15, 0.002);
    let pass_lambda = approx_eq(total_lambda, expected_total, expected_total * 0.20);
    println!("  final body.y = {:.4} (expected ≈ 0.150) — {}",
             final_y, if pass_y { "✓" } else { "✗" });
    println!("  total normal impulse = {:.6} (expected ≈ {:.6}) — {}",
             total_lambda, expected_total, if pass_lambda { "✓" } else { "✗" });
    println!("  per-point λ: {:.6}, {:.6}, {:.6}",
             world.contact_constraints[0].lambda_normal,
             world.contact_constraints[1].lambda_normal,
             world.contact_constraints[2].lambda_normal);
    println!();
}

// ─── Scenario 3: heavy-load chain stress test ───────────────────────────

fn scenario_heavy_load_chain() {
    println!("=== Scenario 3: foot + shin + thigh + 56 kg load (the case that broke explicit FootContact) ===");

    const FOOT_MASS: f64 = 1.0;
    const FOOT_HALF_H: f64 = 0.130;
    const FOOT_RADIUS: f64 = 0.032;
    const SHIN_MASS: f64 = 3.0;
    const SHIN_HALF_H: f64 = 0.21;
    const SHIN_RADIUS: f64 = 0.05;
    const THIGH_MASS: f64 = 7.0;
    const THIGH_HALF_H: f64 = 0.225;
    const THIGH_RADIUS: f64 = 0.07;
    const LOAD_MASS: f64 = 10.0;

    let mut world = XpbdWorld::new();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    thigh.position = Vec3::new(0.0, FOOT_RADIUS + 2.0 * SHIN_HALF_H + THIGH_HALF_H, 0.0);
    let thigh_idx = world.add_body(thigh);

    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    shin.position = Vec3::new(0.0, FOOT_RADIUS + SHIN_HALF_H, 0.0);
    let shin_idx = world.add_body(shin);

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    foot.position = Vec3::new(0.0, FOOT_RADIUS, 0.0);
    foot.orientation = foot_baseline;
    let foot_idx = world.add_body(foot);

    let mut load = RigidBody::new(LOAD_MASS, capsule_inertia(LOAD_MASS, 0.13, 0.5));
    load.position = Vec3::new(0.0, world.bodies[thigh_idx].position.y + THIGH_HALF_H + 0.50, 0.0);
    let load_idx = world.add_body(load);

    // Knee joint (PositionalSpring, the existing pattern).
    let mut s = PositionalSpring::new(thigh_idx, shin_idx, 80_000.0);
    s.local_anchor_a = Vec3::new(0.0, -THIGH_HALF_H, 0.0);
    s.local_anchor_b = Vec3::new(0.0,  SHIN_HALF_H, 0.0);
    world.positional_springs.push(s);
    // Keep the damper for now (it's explicit but should be fine).
    let mut d = PositionalDamper::new(thigh_idx, shin_idx, 200.0);
    d.local_anchor_a = Vec3::new(0.0, -THIGH_HALF_H, 0.0);
    d.local_anchor_b = Vec3::new(0.0,  SHIN_HALF_H, 0.0);
    world.positional_dampers.push(d);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
        -0.087, 2.44, 80.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_axis, 1.5));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_axis, 5.0, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        thigh_idx, shin_idx, knee_axis, 3000.0, 300.0));

    // Ankle joint.
    let mut s = PositionalSpring::new(shin_idx, foot_idx, 60_000.0);
    s.local_anchor_a = Vec3::new(0.0, -SHIN_HALF_H, 0.0);
    s.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(shin_idx, foot_idx, 150.0);
    d.local_anchor_a = Vec3::new(0.0, -SHIN_HALF_H, 0.0);
    d.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_dampers.push(d);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
        -0.44, 0.87, 80.0, 2.0).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, 1.2));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, pf_axis, 4.0, 0.0)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0), 400.0));

    // Load-thigh connection.
    let mut s = PositionalSpring::new(load_idx, thigh_idx, 200_000.0);
    s.local_anchor_a = Vec3::new(0.0, -0.50, 0.0);
    s.local_anchor_b = Vec3::new(0.0,  THIGH_HALF_H, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(load_idx, thigh_idx, 400.0);
    d.local_anchor_a = Vec3::new(0.0, -0.50, 0.0);
    d.local_anchor_b = Vec3::new(0.0,  THIGH_HALF_H, 0.0);
    world.positional_dampers.push(d);

    // FOOT CONTACT — 3 ContactConstraints in tripod layout, instead of
    // FootContact's penalty springs.
    let half_h = FOOT_HALF_H;
    let radius = FOOT_RADIUS;
    let medial_x = -radius * 0.85;
    let lateral_x = radius * 0.85;
    let heel   = Vec3::new(0.0,         half_h * 0.85, -radius);
    let ball_m = Vec3::new(medial_x,   -half_h * 0.65, -radius);
    let ball_l = Vec3::new(lateral_x,  -half_h * 0.65, -radius);
    for local in [heel, ball_m, ball_l] {
        world.contact_constraints.push(ContactConstraint::new(foot_idx, local));
    }

    println!("  setup: 4-body chain, total mass {} kg, FootContact replaced with 3 ContactConstraints",
             (FOOT_MASS + SHIN_MASS + THIGH_MASS + LOAD_MASS) as i32);
    println!();
    println!("  {:<6} {:<10} {:<10} {:<10} {:<10}", "t", "load.y", "thigh.y", "shin.y", "foot.y");

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let load_y_init = world.bodies[load_idx].position.y;

    while t < 3.0 {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        t += dt;
        if t >= next_print {
            println!("  {:<6.2} {:<+10.4} {:<+10.4} {:<+10.4} {:<+10.4}",
                     t, world.bodies[load_idx].position.y,
                     world.bodies[thigh_idx].position.y,
                     world.bodies[shin_idx].position.y,
                     world.bodies[foot_idx].position.y);
            next_print += 0.25;
        }
    }

    let load_y_final = world.bodies[load_idx].position.y;
    let stable = load_y_final.is_finite()
        && (load_y_final - load_y_init).abs() < 0.1;
    println!();
    println!("  load.y started {:.4}, final {:.4} — {}",
             load_y_init, load_y_final,
             if stable { "✓ stable (within 10 cm)" } else { "✗ unstable" });
    println!();
}

// ─── Scenario 4: friction ───────────────────────────────────────────────

fn scenario_friction() {
    println!("=== Scenario 4: friction stops a sliding body ===");
    let mut world = XpbdWorld::new();
    let mut body = RigidBody::new(1.0, capsule_inertia(1.0, 0.05, 0.10));
    body.position = Vec3::new(0.0, 0.15, 0.0);
    body.linear_velocity = Vec3::new(2.0, 0.0, 0.0);  // sliding in +x at 2 m/s
    let body_idx = world.add_body(body);
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new(0.0, -0.15, 0.0)).with_friction(0.5));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    let mut t = 0.0;
    let mut last_vx = 2.0;
    while t < 2.0 {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        last_vx = world.bodies[body_idx].linear_velocity.x;
        t += dt;
    }

    // With μ=0.5, expected deceleration = μg = 4.9 m/s². Initial 2 m/s
    // stops in ~0.4 s. After 2 s, velocity should be 0.
    let pass = last_vx.abs() < 0.05;
    println!("  initial v.x = 2.0, after 2.0 s v.x = {:.4} (expected ≈ 0) — {}",
             last_vx, if pass { "✓" } else { "✗" });
    println!("  body.x = {:.4} (expected ≈ d = v²/(2μg) = {:.4})",
             world.bodies[body_idx].position.x, 2.0_f64 * 2.0 / (2.0 * 0.5 * 9.81));
    println!();
}

// ─── Scenario 5: lifted body ────────────────────────────────────────────

fn scenario_lifted() {
    println!("=== Scenario 5: lifted body — no impulse, falls freely ===");
    let mut world = XpbdWorld::new();
    let mut body = RigidBody::new(1.0, capsule_inertia(1.0, 0.05, 0.10));
    body.position = Vec3::new(0.0, 5.0, 0.0);  // way above ground
    let body_idx = world.add_body(body);
    world.contact_constraints.push(ContactConstraint::new(
        body_idx, Vec3::new(0.0, -0.15, 0.0)));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;

    // After 0.5 s of free fall from rest: y = 5 - 0.5×9.81×0.25 = 3.77625
    let target_t = 0.5;
    let mut t = 0.0;
    while t < target_t - dt {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        t += dt;
    }
    let body_y = world.bodies[body_idx].position.y;
    let body_vy = world.bodies[body_idx].linear_velocity.y;
    let expected_y = 5.0 - 0.5 * 9.81 * t * t;
    let expected_vy = -9.81 * t;
    let pass_y = approx_eq(body_y, expected_y, 0.01);
    let pass_vy = approx_eq(body_vy, expected_vy, 0.05);
    let lambda = world.contact_constraints[0].lambda_normal;
    println!("  after {:.3} s: body.y = {:.4} (expected {:.4}) — {}",
             t, body_y, expected_y, if pass_y { "✓" } else { "✗" });
    println!("  body.v.y = {:.4} (expected {:.4}) — {}",
             body_vy, expected_vy, if pass_vy { "✓" } else { "✗" });
    println!("  contact impulse λ = {:.6} (expected 0, no contact) — {}",
             lambda, if lambda == 0.0 { "✓" } else { "✗" });
    println!();
}

fn main() {
    println!("ContactConstraint (implicit contact) validation\n");
    scenario_single_contact();
    scenario_tripod();
    scenario_friction();
    scenario_lifted();
    scenario_heavy_load_chain();
}
