// Phase 3 step 2 prep: validate the 2D support-polygon primitives.
//
// Pure observation primitives — no control logic. They're the second
// piece needed for whole-body coordinated balance: combined with the
// CoM observer (item 1), they let a controller answer "is the CoM
// over the support polygon? if not, by how much?"
//
// Engine pieces being validated:
//   - `Vec2` type alias
//   - `convex_hull_2d`: Andrew's monotone chain
//   - `point_in_convex_polygon_2d`
//   - `signed_distance_to_convex_polygon_2d` (positive inside, negative outside)
//   - `RigidBody::axis_endpoints_world`: world-frame capsule endpoints
//   - `XpbdWorld::com_ground_projection`: CoM projected to y=0
//
// Six scenarios:
//   1. Convex hull of a square's 4 corners → unchanged.
//   2. Convex hull with redundant interior points → only hull returned.
//   3. Point-in-polygon: inside / on edge / outside / strictly outside.
//   4. Signed distance: matches Euclidean distance, sign is correct.
//   5. Axis endpoints: foot at horizontal pose → heel and toe in world.
//   6. Bilateral feet → support polygon, CoM projection inside.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::{Vec2, Vec3, Mat3, convex_hull_2d, point_in_convex_polygon_2d,
    signed_distance_to_convex_polygon_2d};
use nalgebra::UnitQuaternion;

fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }
fn approx_eq_v2(a: Vec2, b: Vec2, tol: f64) -> bool { (a - b).norm() < tol }
fn approx_eq_v3(a: Vec3, b: Vec3, tol: f64) -> bool { (a - b).norm() < tol }

// ─── Scenario 1: Convex hull of a square ────────────────────────────────

fn scenario_hull_square() {
    println!("=== Scenario 1: convex hull of a square's corners ===");
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let hull = convex_hull_2d(&pts);
    let pass = hull.len() == 4;
    println!("  hull has 4 vertices: {}", if pass { "✓" } else { "✗" });
    // CCW ordering: starts at lowest-x lowest-y → (0,0), (1,0), (1,1), (0,1).
    let expected = [
        Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
    ];
    let order_match = (0..4).all(|i| approx_eq_v2(hull[i], expected[i], 1e-12));
    println!("  CCW order from lower-left corner: {}", if order_match { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 2: Hull with interior points ──────────────────────────────

fn scenario_hull_interior_pts() {
    println!("=== Scenario 2: hull strips interior points ===");
    let pts = [
        Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0),
        Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0),
        Vec2::new(1.0, 1.0),  // interior — should be stripped
        Vec2::new(0.5, 0.5),  // interior — should be stripped
    ];
    let hull = convex_hull_2d(&pts);
    let pass = hull.len() == 4;
    println!("  hull has 4 vertices (interior 2 stripped): {}", if pass { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 3: Point-in-polygon ───────────────────────────────────────

fn scenario_point_in_polygon() {
    println!("=== Scenario 3: point-in-polygon ===");
    let square = [
        Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
    ];
    let cases = [
        (Vec2::new(0.5, 0.5), true,  "center"),
        (Vec2::new(0.0, 0.5), true,  "on edge (boundary considered inside)"),
        (Vec2::new(0.0, 0.0), true,  "on vertex"),
        (Vec2::new(0.5, 1.1), false, "just outside"),
        (Vec2::new(2.0, 2.0), false, "far outside"),
        (Vec2::new(-0.1, 0.5), false, "just outside other side"),
    ];
    for (p, expected, desc) in cases {
        let got = point_in_convex_polygon_2d(p, &square);
        let pass = got == expected;
        println!("  point ({:+.1}, {:+.1}) — {} → {} {}", p.x, p.y, desc,
                 if got { "inside" } else { "outside" },
                 if pass { "✓" } else { "✗" });
    }

    // Degenerate polygon (segment, single point, empty).
    let segment = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)];
    let in_seg = point_in_convex_polygon_2d(Vec2::new(0.5, 0.0), &segment);
    println!("  degenerate (2-pt segment) → reported outside: {}",
             if !in_seg { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 4: Signed distance ────────────────────────────────────────

fn scenario_signed_distance() {
    println!("=== Scenario 4: signed distance to polygon ===");
    let square = [
        Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
    ];
    let cases = [
        (Vec2::new(0.5, 0.5),  0.5,  "center, distance to nearest edge = 0.5"),
        (Vec2::new(0.5, 0.0),  0.0,  "on edge → 0"),
        (Vec2::new(2.0, 0.5), -1.0,  "1.0 units outside right edge"),
        (Vec2::new(0.5, -0.3), -0.3, "0.3 units outside bottom edge"),
        (Vec2::new(-1.0, -1.0), -(2.0_f64).sqrt(), "outside corner — sqrt(2) away"),
    ];
    for (p, expected, desc) in cases {
        let d = signed_distance_to_convex_polygon_2d(p, &square);
        let pass = approx_eq_f64(d, expected, 1e-9);
        println!("  ({:+.1}, {:+.1}) — {}: d={:+.4} (expected {:+.4}) {}",
                 p.x, p.y, desc, d, expected,
                 if pass { "✓" } else { "✗" });
    }
    println!();
}

// ─── Scenario 5: Foot axis endpoints ────────────────────────────────────

fn scenario_foot_endpoints() {
    println!("=== Scenario 5: foot axis endpoints in world ===");
    const FOOT_HALF_H: f64 = 0.090;

    // Foot at horizontal anatomical pose. baseline = R_x(-π/2):
    //   local +Y → world -Z (heel)
    //   local -Y → world +Z (toe)
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new(1.0, Mat3::identity());
    foot.position = Vec3::new(0.0, 0.032, 0.0);  // foot center at ground+radius
    foot.orientation = foot_baseline;

    // axis_endpoints_world returns (positive_Y_end, negative_Y_end).
    let (heel, toe) = foot.axis_endpoints_world(FOOT_HALF_H);
    let expected_heel = Vec3::new(0.0, 0.032, -FOOT_HALF_H);
    let expected_toe  = Vec3::new(0.0, 0.032,  FOOT_HALF_H);
    let pass_heel = approx_eq_v3(heel, expected_heel, 1e-9);
    let pass_toe  = approx_eq_v3(toe,  expected_toe, 1e-9);
    println!("  heel (local +Y → world -Z): ({:+.3}, {:+.3}, {:+.3}) {}",
             heel.x, heel.y, heel.z, if pass_heel { "✓" } else { "✗" });
    println!("  toe  (local -Y → world +Z): ({:+.3}, {:+.3}, {:+.3}) {}",
             toe.x, toe.y, toe.z, if pass_toe { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 6: Bilateral support polygon + CoM projection ─────────────

fn scenario_bilateral_support() {
    println!("=== Scenario 6: bilateral support polygon and CoM projection ===");
    const FOOT_HALF_H: f64 = 0.090;
    const HIP_LATERAL: f64 = 0.10;  // each foot offset 10 cm laterally

    let mut world = XpbdWorld::new();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Right foot at x=+0.10, left foot at x=-0.10. Each lying flat
    // along z with half-length 0.090 m.
    let mut r_foot = RigidBody::new(1.0, Mat3::identity());
    r_foot.position = Vec3::new( HIP_LATERAL, 0.032, 0.0);
    r_foot.orientation = foot_baseline;
    let r_idx = world.add_body(r_foot);

    let mut l_foot = RigidBody::new(1.0, Mat3::identity());
    l_foot.position = Vec3::new(-HIP_LATERAL, 0.032, 0.0);
    l_foot.orientation = foot_baseline;
    let l_idx = world.add_body(l_foot);

    // Single "torso lump" representing the upper body, placed centered.
    let mut torso = RigidBody::new(50.0, Mat3::identity());
    torso.position = Vec3::new(0.0, 1.0, 0.0);
    world.add_body(torso);

    // Build support polygon from foot endpoints projected to ground.
    // axis_endpoints_world returns (+Y_end, -Y_end) which for the foot
    // baseline are (heel, toe) respectively.
    let (r_heel, r_toe) = world.bodies[r_idx].axis_endpoints_world(FOOT_HALF_H);
    let (l_heel, l_toe) = world.bodies[l_idx].axis_endpoints_world(FOOT_HALF_H);
    let footprint_pts = [
        Vec2::new(r_toe.x, r_toe.z), Vec2::new(r_heel.x, r_heel.z),
        Vec2::new(l_toe.x, l_toe.z), Vec2::new(l_heel.x, l_heel.z),
    ];
    let support = convex_hull_2d(&footprint_pts);
    println!("  4 foot-endpoint inputs → hull has {} vertices (expected 4): {}",
             support.len(), if support.len() == 4 { "✓" } else { "✗" });

    // CoM ground projection. With most mass on the centered torso, CoM_xz ≈ (0, 0).
    let com_xz = world.com_ground_projection();
    println!("  CoM ground projection: ({:+.4}, {:+.4})", com_xz.x, com_xz.y);

    let inside = point_in_convex_polygon_2d(com_xz, &support);
    let dist = signed_distance_to_convex_polygon_2d(com_xz, &support);
    println!("  CoM inside support polygon: {} (signed distance {:+.4} m)",
             if inside { "✓ yes" } else { "✗ no" }, dist);

    // Now lean the torso forward (move it +z by 0.5 m, well outside the
    // 0.18 m foot length) — should now be outside the support polygon.
    world.bodies[2].position = Vec3::new(0.0, 1.0, 0.5);
    let com_xz_leaned = world.com_ground_projection();
    let inside_leaned = point_in_convex_polygon_2d(com_xz_leaned, &support);
    let dist_leaned = signed_distance_to_convex_polygon_2d(com_xz_leaned, &support);
    println!("  After leaning torso to z=+0.5:");
    println!("    CoM ground projection: ({:+.4}, {:+.4})",
             com_xz_leaned.x, com_xz_leaned.y);
    println!("    inside polygon: {} (signed distance {:+.4} m) — {}",
             if inside_leaned { "yes" } else { "no" }, dist_leaned,
             if !inside_leaned && dist_leaned < 0.0 { "✓" } else { "✗" });
    println!();
}

fn main() {
    println!("Support-polygon primitive validation\n");
    scenario_hull_square();
    scenario_hull_interior_pts();
    scenario_point_in_polygon();
    scenario_signed_distance();
    scenario_foot_endpoints();
    scenario_bilateral_support();
}
