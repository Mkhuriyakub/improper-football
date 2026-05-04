// Step 12 of joint-by-joint physics validation. First active control test.
//
// Same biomechanical elbow rig as elbow_softlimit_tear.rs (passive spring,
// damper, soft limit, hard X/Y locks), now with an `AngularPDController`
// that commands the joint to hold at a non-zero target angle. This is the
// smallest possible test of active drive: a single fixed target, no step
// changes, just "the muscle wants to hold the elbow at 0.5 rad — does it?"
//
// The actual settling angle won't be exactly 0.5 rad. Passive elements at
// the joint pull elsewhere (passive spring → 0.20 rad neutral) and gravity
// pulls toward 0 rad equivalent. The active controller, the passive spring,
// and gravity all balance to produce the equilibrium:
//
//   kp_active·(target − θ_eq) + kp_passive·(neutral − θ_eq) − m·g·L·sin(θ_eq) = 0
//
// For our parameters this solves to θ_eq ≈ 0.460 rad. Validation: the joint
// approaches this analytic equilibrium smoothly, settles, and holds there.
// The deviation between commanded target (0.500) and actual rest (0.460) is
// realistic — real muscle drive operates against the same passive resistance
// and gravity, and never lands exactly on the commanded angle without
// integral control or a higher-level inverse model.

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
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.05;

const INITIAL_FLEX: f64 = 0.30;
const COMMAND_TARGET: f64 = 0.50;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;
const ELBOW_STIFFNESS: f64 = 1.5;
const ELBOW_NEUTRAL:   f64 = 0.20;

const ELBOW_LIMIT_MIN: f64 = -0.40;
const ELBOW_LIMIT_MAX: f64 =  2.20;
const ELBOW_LIMIT_STIFFNESS: f64 = 500.0;
const ELBOW_LIMIT_EXPONENT:  f64 =   2.0;

const CTRL_KP: f64 = 30.0;  // active proportional gain (N·m/rad)
const CTRL_KD: f64 =  2.0;  // active derivative gain (N·m·s/rad)

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

/// Solve for the joint's settling angle including the active controller.
///   kp_active·(target − θ) + kp_passive·(neutral − θ) − m·g·L·sin(θ) = 0
fn predicted_equilibrium(
    m: f64, g: f64, l: f64,
    kp_active: f64, target: f64,
    kp_passive: f64, neutral: f64,
) -> f64 {
    let mut theta: f64 = (target + neutral) * 0.5;
    for _ in 0..40 {
        let f  = kp_active * (target - theta) + kp_passive * (neutral - theta) - m * g * l * theta.sin();
        let fp = -kp_active - kp_passive - m * g * l * theta.cos();
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
            .with_target(COMMAND_TARGET));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let theta_eq = predicted_equilibrium(
        lower_def.mass, 9.81, lower_def.half_height,
        CTRL_KP, COMMAND_TARGET,
        ELBOW_STIFFNESS, ELBOW_NEUTRAL,
    );

    println!("ELBOW ACTIVE HOLD  ({:.1}s, dt={:.5}s)", SIM_DURATION, DT);
    println!("commanded target = {:.3} rad ({:.2}°)",
             COMMAND_TARGET, COMMAND_TARGET.to_degrees());
    println!("ctrl gains:  kp = {} N·m/rad   kd = {} N·m·s/rad",
             CTRL_KP, CTRL_KD);
    println!("predicted equilibrium = {:.5} rad ({:.2}°)",
             theta_eq, theta_eq.to_degrees());
    println!("(commanded target − equilibrium = {:+.4} rad — natural offset)",
             COMMAND_TARGET - theta_eq);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10}",
             "t", "elbow_z", "error", "ctrl_τ", "ω_z");
    println!("{}", "-".repeat(50));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

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
            let error = elbow_z - COMMAND_TARGET;
            let ctrl_torque = world.angular_pd_controllers[0].scalar_torque(&world.bodies);
            let omega_z = larm.angular_velocity.z - uarm.angular_velocity.z;
            println!("{:<6.3} {:<10.5} {:<+10.5} {:<+10.4} {:<+10.4}",
                     t, elbow_z, error, ctrl_torque, omega_z);
            next_print += PRINT_INTERVAL;
        }
    }

    let final_elbow = {
        let larm = &world.bodies[lower_idx];
        let uarm = &world.bodies[upper_idx];
        (larm.orientation * uarm.orientation.inverse()).scaled_axis().z
    };
    println!();
    println!("Final elbow angle:        {:.5} rad ({:.3}°)",
             final_elbow, final_elbow.to_degrees());
    println!("Predicted equilibrium:    {:.5} rad ({:.3}°)",
             theta_eq, theta_eq.to_degrees());
    println!("Deviation from analytic:  {:+.5} rad",
             final_elbow - theta_eq);
    println!("Commanded target:         {:.5} rad ({:.3}°)",
             COMMAND_TARGET, COMMAND_TARGET.to_degrees());
    println!("Final tracking error:     {:+.5} rad ({:+.3}°)",
             final_elbow - COMMAND_TARGET,
             (final_elbow - COMMAND_TARGET).to_degrees());
}
