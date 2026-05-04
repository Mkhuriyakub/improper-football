// Step 9 of joint-by-joint physics validation.
//
// Adds passive elastic stiffness to the biomechanical elbow. Real unactivated
// elbow joints don't rest at exactly 0° flex — joint capsule, ligaments, and
// resting muscle tension produce a soft preload toward a slightly flexed
// neutral pose. With both gravity (pulling toward 0°) and stiffness (pulling
// toward the neutral pose) active, the resting equilibrium is the angle where
// the two restoring torques balance.
//
// Parameters in literature range:
//   stiffness  = 1.5 N·m/rad   (elbow passive stiffness, mid-range)
//   target     = 0.20 rad     (~11.5° neutral resting flex)
//   damping    = 0.05 N·m·s/rad   (same as previous step)
//
// Predicted equilibrium for a hanging arm at this elbow:
//   m·g·L·sin(θ_eq) + k·(θ_eq − target) = 0
//   1.836·sin(θ_eq) + 1.5·(θ_eq − 0.20) = 0
//   solving numerically: θ_eq ≈ 0.090 rad ≈ 5.15°
//
// Validation: starting from 0.30 rad flex, the lower arm should oscillate
// around 0.090 rad (NOT 0.0 as in elbow_pendulum_realistic.rs) and settle
// there within ~5 s. The presence of a non-zero rest angle is the validation
// that passive stiffness is functioning correctly.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 10.0;
const PRINT_INTERVAL: f64 = 0.10;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;  // N·m·s/rad
const ELBOW_STIFFNESS: f64 = 1.5;   // N·m/rad
const ELBOW_NEUTRAL:   f64 = 0.20;  // rad — slightly-flexed resting pose

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

/// Numerically solve for the equilibrium elbow angle: gravity + stiffness = 0.
fn predicted_equilibrium(m: f64, g: f64, l: f64, k: f64, target: f64) -> f64 {
    // Newton iteration on f(θ) = m·g·L·sin(θ) + k·(θ − target)
    let mut theta: f64 = 0.05;
    for _ in 0..30 {
        let f  = m * g * l * theta.sin() + k * (theta - target);
        let fp = m * g * l * theta.cos() + k;
        theta -= f / fp;
    }
    theta
}

fn main() {
    let mut world = XpbdWorld::new();

    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(upper_def.mass, upper_def.inertia_tensor());
    upper.position = Vec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.orientation = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_FLEX));
    let rotated_anchor = lower.orientation * Vec3::new(0.0, 0.13, 0.0);
    lower.position = Vec3::new(0.0, ELBOW_WORLD_Y, 0.0) - rotated_anchor;
    let lower_idx = world.add_body(lower);

    let mut elbow = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow);

    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::z(), -1.57, 2.20));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));

    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, Vec3::z(), ELBOW_DAMPING));

    world.angular_springs.push(
        AngularSpring::new(upper_idx, lower_idx, Vec3::z(), ELBOW_STIFFNESS, ELBOW_NEUTRAL));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let theta_eq = predicted_equilibrium(
        lower_def.mass, 9.81, lower_def.half_height,
        ELBOW_STIFFNESS, ELBOW_NEUTRAL);

    println!("ELBOW PENDULUM, BIOMECHANICAL ELBOW + STIFFNESS  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("damping = {}   stiffness = {}   neutral = {}",
             ELBOW_DAMPING, ELBOW_STIFFNESS, ELBOW_NEUTRAL);
    println!("predicted equilibrium angle = {:.5} rad ({:.2}°)",
             theta_eq, theta_eq.to_degrees());
    println!();
    println!("{:<6} {:<10} {:<10} {:<12}",
             "t", "elbow_z", "elbow_z−eq", "approaching?");
    println!("{}", "-".repeat(50));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut last_dev: f64 = INITIAL_FLEX - theta_eq;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            let body = &mut world.bodies[upper_idx];
            let pos_err = body.position - upper_rest_pos;
            let force   = -ANCHOR_KP_LIN * pos_err;
            let orient_err = body.orientation.scaled_axis();
            let torque  = -ANCHOR_KP_ANG * orient_err;
            body.apply_force(force);
            body.apply_torque(torque);

            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            let dev = elbow_z - theta_eq;
            let approaching = if dev.abs() < last_dev.abs() { "✓" } else { " " };
            println!("{:<6.2} {:<10.5} {:<10.5} {:<12}",
                     t, elbow_z, dev, approaching);
            last_dev = dev;
            next_print += PRINT_INTERVAL;
        }
    }

    let final_elbow = {
        let larm = &world.bodies[lower_idx];
        let uarm = &world.bodies[upper_idx];
        (larm.orientation * uarm.orientation.inverse()).scaled_axis().z
    };
    println!();
    println!("Final elbow angle:    {:.5} rad ({:.2}°)",
             final_elbow, final_elbow.to_degrees());
    println!("Predicted equilibrium: {:.5} rad ({:.2}°)",
             theta_eq, theta_eq.to_degrees());
    println!("Deviation:             {:+.5} rad",
             final_elbow - theta_eq);
}
