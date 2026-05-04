// Step 13 of joint-by-joint physics validation.
//
// Step-response test for the active controller. Same biomechanical elbow as
// elbow_active_hold.rs, but the commanded target changes on a schedule so we
// can watch the controller track moving commands. Validates that the joint
// follows successive willed targets without overshoot, settles cleanly
// between steps, and produces realistic muscular motion (rise time, no
// oscillation, natural offset from gravity + passive elements).
//
// Schedule:
//   t = 0.0 → 1.0:  target = 0.30 rad  (initial hold)
//   t = 1.0 → 2.5:  target = 0.50 rad
//   t = 2.5 → 4.0:  target = 1.00 rad
//   t = 4.0 → 5.5:  target = 0.30 rad  (return)
//   t = 5.5 → 7.0:  target = 0.00 rad  (extend toward straight arm)
//
// At each target, the joint should settle at the analytical equilibrium that
// balances active drive, passive spring, and gravity. Equilibrium values for
// the parameters here:
//   target 0.30 → eq ≈ 0.286 rad
//   target 0.50 → eq ≈ 0.460 rad
//   target 1.00 → eq ≈ 0.918 rad
//   target 0.00 → eq ≈ 0.009 rad

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 7.0;
const PRINT_INTERVAL: f64 = 0.10;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;
const ELBOW_STIFFNESS: f64 = 1.5;
const ELBOW_NEUTRAL:   f64 = 0.20;

const ELBOW_LIMIT_MIN: f64 = -0.40;
const ELBOW_LIMIT_MAX: f64 =  2.20;
const ELBOW_LIMIT_STIFFNESS: f64 = 500.0;
const ELBOW_LIMIT_EXPONENT:  f64 =   2.0;

const CTRL_KP: f64 = 30.0;
const CTRL_KD: f64 =  2.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

fn target_at(t: f64) -> f64 {
    if      t < 1.0 { 0.30 }
    else if t < 2.5 { 0.50 }
    else if t < 4.0 { 1.00 }
    else if t < 5.5 { 0.30 }
    else            { 0.00 }
}

fn predicted_equilibrium(target: f64) -> f64 {
    // Solve kp·(target − θ) + kp_passive·(neutral − θ) − m·g·L·sin(θ) = 0
    let m = segment_def(BodyType::Receiver, SegmentId::LowerArmL).mass;
    let l = segment_def(BodyType::Receiver, SegmentId::LowerArmL).half_height;
    let mut theta: f64 = (target + ELBOW_NEUTRAL) * 0.5;
    for _ in 0..40 {
        let f  = CTRL_KP * (target - theta)
                 + ELBOW_STIFFNESS * (ELBOW_NEUTRAL - theta)
                 - m * 9.81 * l * theta.sin();
        let fp = -CTRL_KP - ELBOW_STIFFNESS - m * 9.81 * l * theta.cos();
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

    let mut elbow_pos = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow_pos.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow_pos.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow_pos);

    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        upper_idx, lower_idx, Vec3::z(),
        ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX,
        ELBOW_LIMIT_STIFFNESS, ELBOW_LIMIT_EXPONENT,
    ));
    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, Vec3::z(), ELBOW_DAMPING));
    world.angular_springs.push(
        AngularSpring::new(upper_idx, lower_idx, Vec3::z(), ELBOW_STIFFNESS, ELBOW_NEUTRAL));
    world.angular_pd_controllers.push(
        AngularPDController::new(upper_idx, lower_idx, Vec3::z(), CTRL_KP, CTRL_KD)
            .with_target(INITIAL_FLEX));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    println!("ELBOW ACTIVE STEP RESPONSE  ({:.1}s, dt={:.5}s)", SIM_DURATION, DT);
    println!("ctrl gains: kp = {} N·m/rad, kd = {} N·m·s/rad", CTRL_KP, CTRL_KD);
    println!();
    println!("Schedule and predicted equilibria:");
    for &(t_lo, t_hi, tgt) in &[
        (0.0, 1.0, 0.30),
        (1.0, 2.5, 0.50),
        (2.5, 4.0, 1.00),
        (4.0, 5.5, 0.30),
        (5.5, 7.0, 0.00),
    ] {
        let eq = predicted_equilibrium(tgt);
        println!("  t = [{:.1}, {:.1}]:  target = {:.2}  →  predicted eq = {:.5} rad ({:+.3}° from target)",
                 t_lo, t_hi, tgt, eq, (eq - tgt).to_degrees());
    }
    println!();
    println!("{:<6} {:<8} {:<10} {:<10} {:<10} {:<10}",
             "t", "target", "elbow_z", "error", "ctrl_τ", "ω_z");
    println!("{}", "-".repeat(56));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        let target = target_at(t);
        world.angular_pd_controllers[0].target_angle = target;

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
            let error = elbow_z - target;
            let ctrl_torque = world.angular_pd_controllers[0].scalar_torque(&world.bodies);
            let omega_z = larm.angular_velocity.z - uarm.angular_velocity.z;
            println!("{:<6.3} {:<8.2} {:<10.5} {:<+10.5} {:<+10.4} {:<+10.4}",
                     t, target, elbow_z, error, ctrl_torque, omega_z);
            next_print += PRINT_INTERVAL;
        }
    }
}
