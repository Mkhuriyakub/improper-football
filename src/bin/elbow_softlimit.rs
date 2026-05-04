// Step 10 of joint-by-joint physics validation.
//
// Replaces the elbow's Z hard-stop angular limit with a nonlinear soft limit.
// Real ligaments engage progressively as the joint approaches its anatomical
// max — they do not act as a kinematic wall. Pushed past their elastic capacity
// they tear, which is a real failure mode the model must be able to represent.
// A hard stop precludes that failure mode by construction; a soft limit
// supports it because the joint can be driven past the anatomical range under
// extreme load (here we just observe the soft engagement; tear is a later
// step).
//
// Setup: lower arm released from 2.50 rad initial flex — well past the
// anatomical max of +2.20 rad. The soft limit should push the arm back into
// range progressively (stiffness 500 N·m/rad², quadratic), the arm should
// retreat past 2.20 into the natural range, and after some oscillation
// settle at the gravity+stiffness equilibrium ≈ 0.090 rad.
//
// Validation observables:
//   1. Peak Z exceeds 2.20 only by a finite amount (not a wall, not a runaway).
//   2. Once in range, the soft limit applies zero torque (clean handoff).
//   3. Final rest angle is the gravity+stiffness equilibrium, the same as
//      elbow_pendulum_with_stiffness.rs found.
//   4. Peak torque from the limit stays bounded — no impulsive spike.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 10.0;
const PRINT_INTERVAL: f64 = 0.05;

const INITIAL_FLEX: f64 = 2.50;  // past the +2.20 anatomical max

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;
const ELBOW_STIFFNESS: f64 = 1.5;
const ELBOW_NEUTRAL:   f64 = 0.20;

const ELBOW_LIMIT_MIN: f64 = -0.40;
const ELBOW_LIMIT_MAX: f64 =  2.20;
const ELBOW_LIMIT_STIFFNESS: f64 = 500.0;
const ELBOW_LIMIT_EXPONENT:  f64 =   2.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

fn predicted_equilibrium(m: f64, g: f64, l: f64, k: f64, target: f64) -> f64 {
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

    // X and Y still hard locks (this binary is not testing those).
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));
    // Z range now governed by the soft limit instead of a hard stop.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        upper_idx, lower_idx, Vec3::z(),
        ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX,
        ELBOW_LIMIT_STIFFNESS, ELBOW_LIMIT_EXPONENT,
    ));

    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, Vec3::z(), ELBOW_DAMPING));
    world.angular_springs.push(
        AngularSpring::new(upper_idx, lower_idx, Vec3::z(), ELBOW_STIFFNESS, ELBOW_NEUTRAL));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let theta_eq = predicted_equilibrium(
        lower_def.mass, 9.81, lower_def.half_height,
        ELBOW_STIFFNESS, ELBOW_NEUTRAL);

    println!("ELBOW SOFT LIMIT  ({:.1}s, dt={:.5}s)", SIM_DURATION, DT);
    println!("limit Z range = [{}, {}]   stiffness = {}   exponent = {}",
             ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX, ELBOW_LIMIT_STIFFNESS, ELBOW_LIMIT_EXPONENT);
    println!("initial flex = {} rad    in-range equilibrium = {:.5} rad ({:.2}°)",
             INITIAL_FLEX, theta_eq, theta_eq.to_degrees());
    println!();
    println!("{:<6} {:<10} {:<10} {:<12} {:<10}",
             "t", "elbow_z", "excess", "limit_τ", "in_range");
    println!("{}", "-".repeat(58));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_excess: f64 = 0.0;
    let mut peak_limit_torque: f64 = 0.0;
    let mut first_in_range_time: Option<f64> = None;

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

            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            let excess = (elbow_z - ELBOW_LIMIT_MAX).max(0.0)
                .max((ELBOW_LIMIT_MIN - elbow_z).max(0.0));
            if excess > max_excess { max_excess = excess; }
            let lt = world.angular_soft_limits[0].scalar_torque(elbow_z).abs();
            if lt > peak_limit_torque { peak_limit_torque = lt; }

            if first_in_range_time.is_none() && elbow_z <= ELBOW_LIMIT_MAX
                && elbow_z >= ELBOW_LIMIT_MIN
            {
                first_in_range_time = Some(t);
            }
        }
        t += DT;

        if t >= next_print {
            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            let excess = if elbow_z > ELBOW_LIMIT_MAX {
                elbow_z - ELBOW_LIMIT_MAX
            } else if elbow_z < ELBOW_LIMIT_MIN {
                elbow_z - ELBOW_LIMIT_MIN
            } else { 0.0 };
            let lt = world.angular_soft_limits[0].scalar_torque(elbow_z);
            let in_range = excess == 0.0;
            println!("{:<6.3} {:<10.5} {:<+10.5} {:<+12.4} {:<10}",
                     t, elbow_z, excess, lt,
                     if in_range { "✓" } else { " " });
            next_print += PRINT_INTERVAL;
        }
    }

    let final_elbow = {
        let larm = &world.bodies[lower_idx];
        let uarm = &world.bodies[upper_idx];
        (larm.orientation * uarm.orientation.inverse()).scaled_axis().z
    };

    println!();
    println!("Initial flex:                 {:.5} rad", INITIAL_FLEX);
    println!("Max excess past limit:        {:.5} rad ({:.3}°)",
             max_excess, max_excess.to_degrees());
    println!("Peak limit torque magnitude:  {:.3} N·m", peak_limit_torque);
    println!("First time back in range:     {}",
             first_in_range_time.map_or("never".to_string(),
                 |t| format!("{:.3} s", t)));
    println!("Final elbow angle:            {:.5} rad ({:.2}°)",
             final_elbow, final_elbow.to_degrees());
    println!("Predicted in-range equilib:   {:.5} rad ({:.2}°)",
             theta_eq, theta_eq.to_degrees());
    println!("Deviation from equilibrium:   {:+.5} rad",
             final_elbow - theta_eq);
}
