// First Phase 2 validation: drop a capsule onto an infinite ground plane
// at y=0 and watch it settle. Tests:
//   1. Detection — does the GroundContact know when the capsule penetrates?
//   2. Constraint — does the penalty spring push it back to the surface?
//   3. Damping — does the capsule actually come to rest, or oscillate forever?
//   4. Off-axis correctness — does a tilted capsule fall flat over time
//      (the lower endpoint takes the contact, producing a torque that rotates
//      the body until both endpoints touch)?
//
// Three test cases run in sequence:
//   A. Vertical capsule (axis = world +y), no rotation. Both endpoints
//      penetrate equally; expected resting body.y = half_h + radius - sink,
//      where sink ≈ m·g/K is sub-mm at K=1e5.
//   B. Horizontal capsule (axis = world +x). Both endpoints sit on the
//      ground at body.y = radius - sink; no torque (contact symmetric
//      about body center).
//   C. Tilted capsule (axis at 45° in xy plane). Lower endpoint takes the
//      first contact, body rotates from 45° toward horizontal as it falls,
//      and finally settles flat at body.y = radius.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.10;

const MASS:    f64 = 1.0;
const HALF_H:  f64 = 0.05;
const RADIUS:  f64 = 0.05;
const DROP_HEIGHT: f64 = 1.0;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(
        i_trans, 0.0,     0.0,
        0.0,     i_axial, 0.0,
        0.0,     0.0,     i_trans,
    )
}

fn run_case(label: &str, initial_orient: UnitQuaternion<f64>, expected_y: f64) {
    let mut world = XpbdWorld::new();

    let mut body = RigidBody::new(MASS, capsule_inertia(MASS, RADIUS, HALF_H));
    body.position = Vec3::new(0.0, DROP_HEIGHT, 0.0);
    body.orientation = initial_orient;
    let body_idx = world.add_body(body);

    world.ground_contacts.push(GroundContact::new(body_idx, HALF_H, RADIUS));

    println!("=== {} ===", label);
    println!("Drop height = {:.2}, expected resting y = {:.4}", DROP_HEIGHT, expected_y);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<12}",
             "t", "y", "vy", "tilt(rad)", "lowest_y");
    println!("{}", "-".repeat(54));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let b = &world.bodies[body_idx];
            // Compute lowest world y of the capsule for diagnostic.
            let axis_world = b.orientation * Vec3::new(0.0, 1.0, 0.0);
            let p1 = b.position + axis_world * HALF_H;
            let p2 = b.position - axis_world * HALF_H;
            let lowest = p1.y.min(p2.y) - RADIUS;
            let tilt = b.orientation.scaled_axis().norm();
            println!("{:<6.2} {:<+10.5} {:<+10.4} {:<+10.4} {:<+12.5}",
                     t, b.position.y, b.linear_velocity.y, tilt, lowest);
            next_print += PRINT_INTERVAL;
        }
    }

    let b = &world.bodies[body_idx];
    let axis_world = b.orientation * Vec3::new(0.0, 1.0, 0.0);
    let p1 = b.position + axis_world * HALF_H;
    let p2 = b.position - axis_world * HALF_H;
    let lowest = p1.y.min(p2.y) - RADIUS;
    let tilt = b.orientation.scaled_axis().norm();
    println!();
    println!("Final: y = {:+.5}, vy = {:+.5}, tilt = {:.4} rad ({:.2}°), lowest_y = {:+.5}",
             b.position.y, b.linear_velocity.y, tilt, tilt.to_degrees(), lowest);
    let err = b.position.y - expected_y;
    println!("       error from expected = {:+.5} m ({:+.3} mm)", err, err * 1000.0);
    println!();
}

fn main() {
    println!("CAPSULE DROP VALIDATION");
    println!("XPBD + GroundContact (penalty spring), {:.1}s sim, dt={:.5}s",
             SIM_DURATION, DT);
    println!("Mass = {} kg, half_h = {} m, radius = {} m",
             MASS, HALF_H, RADIUS);
    println!();

    // Case A: vertical capsule. Settles with both endpoints near the ground.
    // Body center at y = HALF_H + RADIUS (lower endpoint touches at y=0).
    run_case(
        "Case A: vertical capsule (axis = world +y)",
        UnitQuaternion::identity(),
        HALF_H + RADIUS,
    );

    // Case B: horizontal capsule (rotate +π/2 about z so axis becomes +x).
    // Body center at y = RADIUS.
    run_case(
        "Case B: horizontal capsule (axis = world +x)",
        UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2)),
        RADIUS,
    );

    // Case C: tilted at 45° about +z. Should fall, rotate to horizontal,
    // and settle flat. Final tilt should be near π/2 (rotated to horizontal),
    // y near RADIUS.
    run_case(
        "Case C: tilted 45° (initial axis 45° from vertical, in xy plane)",
        UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_4)),
        RADIUS,
    );
}
