// Validates the `FootContact` primitive — multi-point ground contact
// modeling the biomechanical tripod (heel + medial ball + lateral ball).
//
// Six scenarios:
//   1. Flat foot at rest, tripod config — all 3 points engaged, force ≈ m·g
//      distributed across them.
//   2. Lifted foot — no engaged points, force 0.
//   3. Tilted foot (heel up) — only the ball points engaged.
//   4. Tilted foot (toe up) — only heel engaged.
//   5. Pressure centroid computation — agrees with force-weighted average.
//   6. Support points → convex hull is a triangle (3-vertex polygon).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::foot_contact::FootContact;
use physics::math::{Mat3, Vec2, Vec3, convex_hull_2d};
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

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.130;
const FOOT_RADIUS: f64 = 0.032;

fn horizontal_foot_body() -> RigidBody {
    let mut body = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    body.position = Vec3::new(0.0, FOOT_RADIUS + 0.005, 0.0);  // slightly above ground
    body.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    body
}

// ─── Scenario 1: flat foot at rest ──────────────────────────────────────

fn scenario_flat_at_rest() {
    println!("=== Scenario 1: flat foot at rest, tripod ===");
    let mut world = XpbdWorld::new();
    let body = horizontal_foot_body();
    let body_idx = world.add_body(body);
    // side_sign = +1 (right foot)
    world.foot_contacts.push(FootContact::tripod(body_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let fc = &world.foot_contacts[0];
    let engaged = fc.engaged_points(&world.bodies);
    println!("  engaged points: {} (expected 3)", engaged.len());
    let pass_count = engaged.len() == 3;
    println!("  all three tripod points pressing: {}", if pass_count { "✓" } else { "✗" });

    let total = fc.total_normal_force(&world.bodies);
    let expected = FOOT_MASS * 9.81;
    let pass_total = approx_eq(total, expected, 0.5);
    println!("  total normal force: {:.3} N (expected ≈ {:.3} N) — {}",
             total, expected, if pass_total { "✓" } else { "✗" });

    for (i, p) in engaged.iter().enumerate() {
        println!("    point {}: world ({:+.3}, {:+.3}, {:+.3}) — force {:.3} N",
                 i, p.world_position.x, p.world_position.y, p.world_position.z, p.normal_force);
    }
    println!();
}

// ─── Scenario 2: lifted foot ────────────────────────────────────────────

fn scenario_lifted() {
    println!("=== Scenario 2: lifted foot ===");
    let mut world = XpbdWorld::new();
    let mut body = horizontal_foot_body();
    body.position = Vec3::new(0.0, 1.5, 0.0);
    let body_idx = world.add_body(body);
    world.foot_contacts.push(FootContact::tripod(body_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let fc = &world.foot_contacts[0];
    let in_contact = fc.is_in_contact(&world.bodies);
    let total = fc.total_normal_force(&world.bodies);
    println!("  is_in_contact: {} — {}", in_contact, if !in_contact { "✓" } else { "✗" });
    println!("  total normal force: {:.4} N — {}",
             total, if total == 0.0 { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 3: heel raised, ball pressing ─────────────────────────────

fn scenario_heel_raised() {
    println!("=== Scenario 3: heel raised (foot pitched forward) — ball points engaged ===");
    let mut world = XpbdWorld::new();
    let mut body = horizontal_foot_body();
    // Pitch foot forward about world +x by 0.2 rad: the back (+local Y →
    // world -Z) rotates upward. Foot's baseline already has local +Y → -Z,
    // so an additional +x rotation tilts the heel up. Add the rotation
    // ON TOP of the baseline.
    let extra = UnitQuaternion::from_scaled_axis(Vec3::new(0.2, 0.0, 0.0));
    body.orientation = extra * body.orientation;
    // Position so the ball points (forward, world +Z side) touch ground.
    body.position = Vec3::new(0.0, FOOT_RADIUS, 0.0);
    let body_idx = world.add_body(body);
    world.foot_contacts.push(FootContact::tripod(body_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let fc = &world.foot_contacts[0];
    let engaged = fc.engaged_points(&world.bodies);
    println!("  engaged points after settle: {}", engaged.len());
    let any_heel_engaged = engaged.iter().any(|p| p.world_position.z < 0.0);
    let any_ball_engaged = engaged.iter().any(|p| p.world_position.z > 0.0);
    println!("  heel pressing? {}, ball pressing? {} — {}",
             any_heel_engaged, any_ball_engaged,
             if !any_heel_engaged && any_ball_engaged { "✓ heel up, ball down" } else { "✗" });
    println!();
}

// ─── Scenario 4: pressure centroid ─────────────────────────────────────

fn scenario_pressure_centroid() {
    println!("=== Scenario 4: pressure centroid agrees with force-weighted average ===");
    let mut world = XpbdWorld::new();
    let body = horizontal_foot_body();
    let body_idx = world.add_body(body);
    world.foot_contacts.push(FootContact::tripod(body_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let fc = &world.foot_contacts[0];
    let engaged = fc.engaged_points(&world.bodies);

    let total: f64 = engaged.iter().map(|p| p.normal_force).sum();
    let mut weighted = Vec2::new(0.0, 0.0);
    for p in &engaged {
        weighted += Vec2::new(p.world_position.x, p.world_position.z) * p.normal_force;
    }
    let manual = if total > 0.0 { weighted / total } else { Vec2::zeros() };
    let centroid = fc.pressure_centroid(&world.bodies).unwrap();

    let pass = approx_eq(centroid.x, manual.x, 1e-9) && approx_eq(centroid.y, manual.y, 1e-9);
    println!("  pressure_centroid:    ({:+.4}, {:+.4})", centroid.x, centroid.y);
    println!("  force-weighted manual:({:+.4}, {:+.4})", manual.x, manual.y);
    println!("  matches: {}", if pass { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 5: support points form a triangle ─────────────────────────

fn scenario_support_polygon() {
    println!("=== Scenario 5: support points form a triangular polygon ===");
    let mut world = XpbdWorld::new();
    let body = horizontal_foot_body();
    let body_idx = world.add_body(body);
    world.foot_contacts.push(FootContact::tripod(body_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    settle(&mut world, sub_dt, sub_iters, 600);

    let fc = &world.foot_contacts[0];
    let pts = fc.support_points_xz(&world.bodies);
    let hull = convex_hull_2d(&pts);
    println!("  support points: {}, hull vertices: {}",
             pts.len(), hull.len());
    println!("  hull is a triangle: {}",
             if hull.len() == 3 { "✓" } else { "✗" });

    // Print the triangle vertices for visualization.
    for (i, v) in hull.iter().enumerate() {
        println!("    vertex {}: ({:+.4}, {:+.4})", i, v.x, v.y);
    }
    println!();
}

// ─── Scenario 6: left foot mirroring ───────────────────────────────────

fn scenario_left_foot_mirror() {
    println!("=== Scenario 6: left foot tripod mirrors right ===");
    let mut world = XpbdWorld::new();
    let body_r = horizontal_foot_body();
    let body_l = horizontal_foot_body();
    let r_idx = world.add_body(body_r);
    let l_idx = world.add_body(body_l);
    world.foot_contacts.push(FootContact::tripod(r_idx, FOOT_HALF_H, FOOT_RADIUS,  1.0));
    world.foot_contacts.push(FootContact::tripod(l_idx, FOOT_HALF_H, FOOT_RADIUS, -1.0));

    // Compare local points directly.
    let r_pts = &world.foot_contacts[0].local_points;
    let l_pts = &world.foot_contacts[1].local_points;
    // The 1st point is heel (centered, no side-sign). 2nd is medial ball,
    // 3rd is lateral ball. For right foot medial = -x; for left, medial = +x.
    let mirror_correct = (r_pts[1].x - (-l_pts[1].x)).abs() < 1e-12
                      && (r_pts[2].x - (-l_pts[2].x)).abs() < 1e-12;
    println!("  R ball-medial x: {:+.4}, L ball-medial x: {:+.4} (should be opposite signs)",
             r_pts[1].x, l_pts[1].x);
    println!("  R ball-lateral x: {:+.4}, L ball-lateral x: {:+.4}",
             r_pts[2].x, l_pts[2].x);
    println!("  mirroring correct: {}", if mirror_correct { "✓" } else { "✗" });
    println!();
}

fn main() {
    println!("FootContact (tripod) validation\n");
    scenario_flat_at_rest();
    scenario_lifted();
    scenario_heel_raised();
    scenario_pressure_centroid();
    scenario_support_polygon();
    scenario_left_foot_mirror();
}
