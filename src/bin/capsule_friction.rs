// Phase 2 step 2: validate Coulomb friction.
//
// Test setup: place a horizontal capsule already at its resting height
// (y = radius), give it horizontal velocity vx = 1 m/s, watch it slide.
// We skip the drop — during a hard impact, F_n spikes 100× above mg
// briefly and friction (proportional to F_n) ablates most of the
// horizontal velocity in 1-2 ms. The classical sliding result
// d = v² / (2μg) only holds under steady normal force, which the drop-
// then-slide scenario doesn't have.
//
// Three test cases:
//   Case A: μ = 0    (frictionless ice)   — horizontal motion never stops
//   Case B: μ = 0.5  (typical sports turf) — slides 0.102 m and stops
//   Case C: μ = 1.0  (very high)          — slides 0.051 m and stops
//
// Analytical: deceleration = μ·g, slide distance = v²/(2μg).
//   μ = 0.5: d = 1²/(2·0.5·9.81) = 0.102 m
//   μ = 1.0: d = 1²/(2·1.0·9.81) = 0.051 m

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::Vec3;
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.10;

const MASS:    f64 = 1.0;
const HALF_H:  f64 = 0.05;
const RADIUS:  f64 = 0.05;
const INITIAL_VX:  f64 = 1.0;

fn run_case(label: &str, friction: f64) {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    world.ground.friction_coefficient = friction;

    let mut body = RigidBody::new_capsule(MASS, HALF_H, RADIUS);
    // Start AT the resting height (y = radius) with no vertical velocity —
    // skip the drop so we measure pure sliding behavior.
    body.position = Vec3::new(0.0, RADIUS, 0.0);
    // Horizontal capsule (axis = world +x), with initial +x velocity.
    body.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    body.linear_velocity = Vec3::new(INITIAL_VX, 0.0, 0.0);
    let body_idx = world.add_body(body);

    let analytical_slide = if friction > 0.0 {
        INITIAL_VX * INITIAL_VX / (2.0 * friction * 9.81)
    } else {
        f64::INFINITY
    };

    println!("=== {} (μ = {:.2}) ===", label, friction);
    if friction > 0.0 {
        println!("Analytical slide distance: {:.4} m  (d = v²/(2μg))", analytical_slide);
    } else {
        println!("Frictionless: horizontal motion should continue indefinitely");
    }
    println!();
    println!("{:<6} {:<10} {:<10} {:<10}",
             "t", "x", "vx", "y");
    println!("{}", "-".repeat(40));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let x_start = 0.0;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let b = &world.bodies[body_idx];
            println!("{:<6.2} {:<+10.5} {:<+10.4} {:<+10.5}",
                     t, b.position.x, b.linear_velocity.x, b.position.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let b = &world.bodies[body_idx];
    let measured_slide = b.position.x - x_start;
    println!();
    println!("Final: x = {:+.5}, vx = {:+.5}, y = {:+.5}",
             b.position.x, b.linear_velocity.x, b.position.y);
    println!("Slid {:.4} m total", measured_slide);
    if friction > 0.0 {
        let err = measured_slide - analytical_slide;
        let pct = err / analytical_slide * 100.0;
        println!("Analytical = {:.4} m, error = {:+.4} m ({:+.1}%)",
                 analytical_slide, err, pct);
    }
    println!();
}

fn main() {
    println!("CAPSULE FRICTION VALIDATION");
    println!("Drop horizontal capsule with vx={} m/s; vary μ", INITIAL_VX);
    println!("({:.1}s sim, mass={} kg, radius={} m)", SIM_DURATION, MASS, RADIUS);
    println!();

    run_case("Case A: frictionless ice", 0.0);
    run_case("Case B: typical turf",    0.5);
    run_case("Case C: very sticky",     1.0);
}
