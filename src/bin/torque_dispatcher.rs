// Phase 3 step 2 item 3: validate the `TorqueDispatcher` primitive.
//
// Splits a scalar torque demand across multiple torque-pair actuator
// routes about a common world axis. Pure dispatch — no sensing logic.
// Validated in isolation here; later wired into a balance controller
// alongside the CoM observer (item 1) and ground-contact force info
// (item 2.5).
//
// Six scenarios:
//   1. Single-route, demand zero — no body responds.
//   2. Single-route, positive demand — action body spins +x, reaction -x.
//   3. Two equal-weight routes — each gets half the demand.
//   4. Two unequal-weight routes (3:1) — unequal split.
//   5. Per-route limit clamps an oversized demand.
//   6. Disabled dispatcher applies nothing.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::torque_dispatcher::{DispatchRoute, TorqueDispatcher};
use physics::math::{Mat3, Vec3};

fn unit_inertia_body() -> RigidBody {
    // Body with identity inertia tensor → angular accel = torque (in
    // body frame, identity orientation). Makes algebra trivial for tests.
    let mut body = RigidBody::new(1.0, Mat3::identity());
    body.position = Vec3::new(0.0, 100.0, 0.0);  // far above ground (no contact)
    body
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

// Run ONE substep with the dispatcher applied. Bodies are placed far
// from the ground so no contacts engage.
fn run_one_substep(world: &mut XpbdWorld, sub_dt: f64) {
    world.substep(sub_dt);
}

// ─── Scenario 1: zero demand ────────────────────────────────────────────

fn scenario_zero_demand() {
    println!("=== Scenario 1: zero demand ===");
    let mut world = XpbdWorld::new();
    let a = world.add_body(unit_inertia_body());
    let b = world.add_body(unit_inertia_body());
    let dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![DispatchRoute::new(a, b)],
    );
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 0.0;

    let sub_dt = 1.0 / 7200.0;
    run_one_substep(&mut world, sub_dt);

    let omega_a = world.bodies[a].angular_velocity.norm();
    let omega_b = world.bodies[b].angular_velocity.norm();
    let pass = omega_a < 1e-12 && omega_b < 1e-12;
    println!("  body A omega: {:.6}, body B omega: {:.6} — {}",
             omega_a, omega_b, if pass { "✓ both zero" } else { "✗" });
    println!();
}

// ─── Scenario 2: single-route torque pair ───────────────────────────────

fn scenario_single_route() {
    println!("=== Scenario 2: single-route, +100 N·m demand ===");
    let mut world = XpbdWorld::new();
    let a = world.add_body(unit_inertia_body());
    let b = world.add_body(unit_inertia_body());
    let dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![DispatchRoute::new(a, b)],
    );
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 100.0;

    let sub_dt = 1.0 / 7200.0;
    run_one_substep(&mut world, sub_dt);

    // Body A: +100 N·m about +x → α=(100,0,0), Δω = 100 * sub_dt.
    // Body B: -100 N·m → Δω = -100 * sub_dt.
    let expected = 100.0 * sub_dt;
    let dx_a = world.bodies[a].angular_velocity.x;
    let dx_b = world.bodies[b].angular_velocity.x;
    let pass_a = approx_eq(dx_a,  expected, 1e-9);
    let pass_b = approx_eq(dx_b, -expected, 1e-9);
    println!("  A.ω.x = {:+.6} (expected {:+.6}) — {}",
             dx_a, expected, if pass_a { "✓" } else { "✗" });
    println!("  B.ω.x = {:+.6} (expected {:+.6}) — {}",
             dx_b, -expected, if pass_b { "✓" } else { "✗" });
    println!("  per_route_torques: {:?}", world.torque_dispatchers[0].per_route_torques());
    println!();
}

// ─── Scenario 3: two equal-weight routes ────────────────────────────────

fn scenario_two_equal_weight() {
    println!("=== Scenario 3: two equal-weight routes split demand ===");
    let mut world = XpbdWorld::new();
    let a1 = world.add_body(unit_inertia_body());
    let r1 = world.add_body(unit_inertia_body());
    let a2 = world.add_body(unit_inertia_body());
    let r2 = world.add_body(unit_inertia_body());
    let dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![DispatchRoute::new(a1, r1), DispatchRoute::new(a2, r2)],
    );
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 200.0;

    let sub_dt = 1.0 / 7200.0;
    run_one_substep(&mut world, sub_dt);

    // Each route gets 100 N·m. Δω = 100 * sub_dt.
    let expected_each = 100.0 * sub_dt;
    let pass_a1 = approx_eq(world.bodies[a1].angular_velocity.x,  expected_each, 1e-9);
    let pass_r1 = approx_eq(world.bodies[r1].angular_velocity.x, -expected_each, 1e-9);
    let pass_a2 = approx_eq(world.bodies[a2].angular_velocity.x,  expected_each, 1e-9);
    let pass_r2 = approx_eq(world.bodies[r2].angular_velocity.x, -expected_each, 1e-9);
    println!("  per_route_torques: {:?}", world.torque_dispatchers[0].per_route_torques());
    println!("  route 1 (a1, r1) split correctly: {}", if pass_a1 && pass_r1 { "✓" } else { "✗" });
    println!("  route 2 (a2, r2) split correctly: {}", if pass_a2 && pass_r2 { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 4: unequal weights (3:1) ──────────────────────────────────

fn scenario_unequal_weights() {
    println!("=== Scenario 4: routes with weights 3:1 — 75/25 split ===");
    let mut world = XpbdWorld::new();
    let a1 = world.add_body(unit_inertia_body());
    let r1 = world.add_body(unit_inertia_body());
    let a2 = world.add_body(unit_inertia_body());
    let r2 = world.add_body(unit_inertia_body());
    let dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![
            DispatchRoute::new(a1, r1).with_weight(3.0),
            DispatchRoute::new(a2, r2).with_weight(1.0),
        ],
    );
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 100.0;

    let torques = world.torque_dispatchers[0].per_route_torques();
    let pass_split = approx_eq(torques[0], 75.0, 1e-9) && approx_eq(torques[1], 25.0, 1e-9);
    println!("  per_route_torques: {:?} — {}",
             torques, if pass_split { "✓ 75/25 as expected" } else { "✗" });
    println!();
}

// ─── Scenario 5: per-route limit clamps ─────────────────────────────────

fn scenario_per_route_limit() {
    println!("=== Scenario 5: per-route limit clamps the demand ===");
    let mut world = XpbdWorld::new();
    let a = world.add_body(unit_inertia_body());
    let b = world.add_body(unit_inertia_body());
    let dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![DispatchRoute::new(a, b)],
    ).with_per_route_limit(50.0);
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 200.0;  // would be 200 unclamped

    let torques = world.torque_dispatchers[0].per_route_torques();
    let pass = approx_eq(torques[0], 50.0, 1e-9);
    println!("  per_route_torques: {:?} (clamped to {:+.1}) — {}",
             torques, 50.0, if pass { "✓" } else { "✗" });

    // And verify negative demand clamps to -50.
    world.torque_dispatchers[0].demand = -300.0;
    let torques_neg = world.torque_dispatchers[0].per_route_torques();
    let pass_neg = approx_eq(torques_neg[0], -50.0, 1e-9);
    println!("  with demand=-300, clamps to -50: {:?} — {}",
             torques_neg, if pass_neg { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 6: disabled dispatcher ────────────────────────────────────

fn scenario_disabled() {
    println!("=== Scenario 6: disabled dispatcher applies nothing ===");
    let mut world = XpbdWorld::new();
    let a = world.add_body(unit_inertia_body());
    let b = world.add_body(unit_inertia_body());
    let mut dispatcher = TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![DispatchRoute::new(a, b)],
    );
    dispatcher.enabled = false;
    world.torque_dispatchers.push(dispatcher);
    world.torque_dispatchers[0].demand = 1000.0;

    let sub_dt = 1.0 / 7200.0;
    run_one_substep(&mut world, sub_dt);

    let omega_a = world.bodies[a].angular_velocity.norm();
    let omega_b = world.bodies[b].angular_velocity.norm();
    let pass = omega_a < 1e-12 && omega_b < 1e-12;
    println!("  bodies unaffected by disabled dispatcher: {}",
             if pass { "✓" } else { "✗" });

    let torques = world.torque_dispatchers[0].per_route_torques();
    let pass_torques = torques[0] == 0.0;
    println!("  per_route_torques returns 0 when disabled: {:?} — {}",
             torques, if pass_torques { "✓" } else { "✗" });
    println!();
}

fn main() {
    println!("TorqueDispatcher validation\n");
    scenario_zero_demand();
    scenario_single_route();
    scenario_two_equal_weight();
    scenario_unequal_weights();
    scenario_per_route_limit();
    scenario_disabled();
}
