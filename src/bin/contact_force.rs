// Phase 3 step 2.5: validate the ground-contact force-surfacing
// primitives on `GroundContact`.
//
// Methods being validated:
//   - `engaged_endpoints`: which endpoints are pressing, with contact
//     point and normal-force magnitude
//   - `total_normal_force`: sum of normal forces across endpoints
//   - `is_in_contact`: any endpoint in active contact
//
// Five scenarios:
//   1. Vertical capsule at rest on ground — only bottom endpoint
//      engaged; total normal force ≈ m·g.
//   2. Vertical capsule lifted off ground — no engaged endpoints,
//      `is_in_contact` = false.
//   3. Horizontal capsule lying flat — both endpoints engaged equally;
//      total normal force ≈ m·g.
//   4. Horizontal capsule lifted off ground — both endpoints not engaged.
//   5. Foot-shaped capsule on ground — endpoint positions match
//      `axis_endpoints_world` (cross-check between primitives).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::Vec3;
use nalgebra::UnitQuaternion;

fn settle(world: &mut XpbdWorld, sub_dt: f64, sub_iters: usize, frames: usize) {
    for _ in 0..frames {
        for _ in 0..sub_iters { world.substep(sub_dt); }
    }
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

// ─── Scenario 1: vertical capsule at rest ───────────────────────────────

fn scenario_vertical_at_rest() {
    println!("=== Scenario 1: vertical capsule at rest on ground ===");
    const MASS: f64 = 1.0;
    const HALF_H: f64 = 0.20;
    const RADIUS: f64 = 0.05;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    body.position = Vec3::new(0.0, 0.30, 0.0);  // bottom at 0.05 above ground; settle to ~RADIUS
    let body_idx = world.add_body(body);

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);  // 5 s

    let endpoints = world.engaged_endpoints(body_idx);
    let positive_engaged = endpoints[0].is_some();
    let negative_engaged = endpoints[1].is_some();

    println!("  positive Y endpoint engaged: {} (expected false — top of vertical capsule)",
             if positive_engaged { "yes" } else { "no" });
    println!("  negative Y endpoint engaged: {} (expected true — bottom)",
             if negative_engaged { "yes" } else { "no" });
    let pass_engagement = !positive_engaged && negative_engaged;
    println!("  engagement pattern correct: {}", if pass_engagement { "✓" } else { "✗" });

    let f_total = world.total_normal_force(body_idx);
    let expected = MASS * 9.81;
    let pass_force = approx_eq(f_total, expected, 0.5);  // 0.5 N tolerance for residual oscillation
    println!("  total normal force: {:.3} N (expected ≈ {:.3} N) — {}",
             f_total, expected, if pass_force { "✓" } else { "✗" });

    let is_in = world.is_in_contact(body_idx);
    println!("  is_in_contact: {} — {}",
             is_in, if is_in { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 2: vertical capsule lifted off ────────────────────────────

fn scenario_vertical_lifted() {
    println!("=== Scenario 2: vertical capsule lifted above ground ===");
    const MASS: f64 = 1.0;
    const HALF_H: f64 = 0.20;
    const RADIUS: f64 = 0.05;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    body.position = Vec3::new(0.0, 2.0, 0.0);  // 2 m above ground
    let body_idx = world.add_body(body);

    let endpoints = world.engaged_endpoints(body_idx);
    let any_engaged = endpoints.iter().any(|e| e.is_some());
    let f_total = world.total_normal_force(body_idx);
    let is_in = world.is_in_contact(body_idx);

    println!("  any endpoint engaged: {} — {}",
             any_engaged, if !any_engaged { "✓" } else { "✗" });
    println!("  total normal force: {:.4} N — {}",
             f_total, if f_total == 0.0 { "✓" } else { "✗" });
    println!("  is_in_contact: {} — {}",
             is_in, if !is_in { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 3: horizontal capsule lying flat ──────────────────────────

fn scenario_horizontal_at_rest() {
    println!("=== Scenario 3: horizontal capsule lying flat on ground ===");
    const MASS: f64 = 1.0;
    const HALF_H: f64 = 0.090;
    const RADIUS: f64 = 0.032;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    // Foot-like capsule: oriented horizontal, centered at y = RADIUS.
    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    body.position = Vec3::new(0.0, 0.040, 0.0);  // slightly above; settle to RADIUS
    body.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let body_idx = world.add_body(body);

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let endpoints = world.engaged_endpoints(body_idx);
    let positive_engaged = endpoints[0].is_some();
    let negative_engaged = endpoints[1].is_some();
    println!("  positive Y endpoint (heel) engaged: {} (expected true)",
             if positive_engaged { "yes" } else { "no" });
    println!("  negative Y endpoint (toe) engaged:  {} (expected true)",
             if negative_engaged { "yes" } else { "no" });
    let pass_eng = positive_engaged && negative_engaged;
    println!("  both endpoints engaged: {}", if pass_eng { "✓" } else { "✗" });

    if let (Some(heel), Some(toe)) = (endpoints[0], endpoints[1]) {
        let f_each_expected = 0.5 * MASS * 9.81;
        let pass_balance = (heel.normal_force - toe.normal_force).abs() < 0.1;
        println!("  heel force: {:.3} N, toe force: {:.3} N (each expected ≈ {:.3}) — {}",
                 heel.normal_force, toe.normal_force, f_each_expected,
                 if pass_balance { "✓ balanced" } else { "✗ imbalanced" });

        let f_total = world.total_normal_force(body_idx);
        let pass_total = approx_eq(f_total, MASS * 9.81, 0.5);
        println!("  total normal force: {:.3} N (expected ≈ {:.3}) — {}",
                 f_total, MASS * 9.81, if pass_total { "✓" } else { "✗" });

        // Contact points should be at world ±FOOT_HALF_H along z, at y = 0.
        let pass_heel_pos = approx_eq(heel.contact_point.y, 0.0, 1e-3)
            && approx_eq(heel.contact_point.z.abs(), HALF_H, 1e-3);
        let pass_toe_pos = approx_eq(toe.contact_point.y, 0.0, 1e-3)
            && approx_eq(toe.contact_point.z.abs(), HALF_H, 1e-3);
        println!("  heel contact point: ({:+.3}, {:+.3}, {:+.3}) — {}",
                 heel.contact_point.x, heel.contact_point.y, heel.contact_point.z,
                 if pass_heel_pos { "✓ on ground at ±half_h along z" } else { "✗" });
        println!("  toe contact point:  ({:+.3}, {:+.3}, {:+.3}) — {}",
                 toe.contact_point.x, toe.contact_point.y, toe.contact_point.z,
                 if pass_toe_pos { "✓ on ground at ±half_h along z" } else { "✗" });
    }
    println!();
}

// ─── Scenario 4: horizontal capsule lifted ──────────────────────────────

fn scenario_horizontal_lifted() {
    println!("=== Scenario 4: horizontal capsule lifted above ground ===");
    const MASS: f64 = 1.0;
    const HALF_H: f64 = 0.090;
    const RADIUS: f64 = 0.032;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    body.position = Vec3::new(0.0, 1.5, 0.0);
    body.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let body_idx = world.add_body(body);

    let any_engaged = world.engaged_endpoints(body_idx).iter().any(|e| e.is_some());
    let f_total = world.total_normal_force(body_idx);
    let is_in = world.is_in_contact(body_idx);
    println!("  any endpoint engaged: {} — {}", any_engaged, if !any_engaged { "✓" } else { "✗" });
    println!("  total normal force: {:.4} — {}", f_total, if f_total == 0.0 { "✓" } else { "✗" });
    println!("  is_in_contact: {} — {}", is_in, if !is_in { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 5: cross-check with axis_endpoints_world ──────────────────

fn scenario_endpoint_consistency() {
    println!("=== Scenario 5: contact-point consistency with axis_endpoints_world ===");
    const MASS: f64 = 1.0;
    const HALF_H: f64 = 0.090;
    const RADIUS: f64 = 0.032;

    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    body.position = Vec3::new(0.0, 0.040, 0.0);
    body.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let body_idx = world.add_body(body);

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let endpoints = world.engaged_endpoints(body_idx);
    let (axis_pos, axis_neg) = world.bodies[body_idx].axis_endpoints_world(HALF_H);

    if let Some(positive) = endpoints[0] {
        // Contact point is one radius below the endpoint center, on the
        // ground plane. So contact_point.y = 0 and (x, z) match the
        // endpoint center's (x, z).
        let dx = axis_pos.x - positive.contact_point.x;
        let dz = axis_pos.z - positive.contact_point.z;
        let xz_match = dx.abs() < 1e-4 && dz.abs() < 1e-4;
        println!("  +Y endpoint (axis): ({:+.4}, {:+.4}, {:+.4})",
                 axis_pos.x, axis_pos.y, axis_pos.z);
        println!("  +Y contact point:   ({:+.4}, {:+.4}, {:+.4})",
                 positive.contact_point.x, positive.contact_point.y, positive.contact_point.z);
        println!("  contact (x,z) matches axis (x,z): {}", if xz_match { "✓" } else { "✗" });
    }
    if let Some(negative) = endpoints[1] {
        let dx = axis_neg.x - negative.contact_point.x;
        let dz = axis_neg.z - negative.contact_point.z;
        let xz_match = dx.abs() < 1e-4 && dz.abs() < 1e-4;
        println!("  -Y contact point matches axis (x,z): {}", if xz_match { "✓" } else { "✗" });
    }
    println!();
}

fn main() {
    println!("Ground-contact force-surfacing validation\n");
    scenario_vertical_at_rest();
    scenario_vertical_lifted();
    scenario_horizontal_at_rest();
    scenario_horizontal_lifted();
    scenario_endpoint_consistency();
}
