// Step 1 of joint-by-joint physics validation.
//
// Rig: upper arm + lower arm only, joined at the elbow.
// The upper arm is held in place by a soft 6-DOF anchor spring (linear + angular PD)
// — not pinned with inv_mass = 0 — so reaction torques from the elbow propagate into
// the parent body realistically. The lower arm is free, subject only to gravity and
// the elbow constraint.
//
// First observable: with the elbow at its rest angle (zero flex) and no driving,
// the lower arm hangs in static equilibrium. We expect no drift in position, no
// orientation deviation, and no accumulating angular velocity over the run.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;          // matches physics::xpbd
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;

// Soft anchor on the upper arm: stiff enough to hold near rest, finite so the
// parent still reacts when we eventually drive the elbow.
const ANCHOR_KP_LIN: f64 = 50_000.0;  // N/m
const ANCHOR_KD_LIN: f64 =  1_000.0;  // N·s/m
const ANCHOR_KP_ANG: f64 =  5_000.0;  // N·m/rad
const ANCHOR_KD_ANG: f64 =    100.0;  // N·m·s/rad

// Initial geometry — picked so every joint and anchor starts at zero error.
// Upper-arm capsule: half-height 0.15m, axis = local Y, +Y at shoulder, -Y at elbow.
// Lower-arm capsule: half-height 0.13m, +Y at elbow, -Y at wrist.
const UPPER_REST_Y: f64 = 1.33;        // matches skeleton.rs segment_y_offset
const LOWER_REST_Y: f64 = 1.33 - 0.15 - 0.13;  // = 1.05

fn main() {
    let mut world = XpbdWorld::new();

    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(upper_def.mass, upper_def.inertia_tensor());
    upper.position = Vec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.position = Vec3::new(0.0, LOWER_REST_Y, 0.0);
    let lower_idx = world.add_body(lower);

    let mut elbow = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);  // upper-arm distal end
    elbow.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);  // lower-arm proximal end
    world.constraints.push(elbow);

    // Hinge enforced via three angular limits: Z free over the anatomical range,
    // X and Y locked to a small dead band.
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::z(), -0.40, 2.20));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    println!("ELBOW REST EQUILIBRIUM TEST  ({:.1}s, dt={:.5}s, {} substeps/frame)",
             SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("upper-arm rest (0, {:.4}, 0)   lower-arm rest (0, {:.4}, 0)",
             UPPER_REST_Y, LOWER_REST_Y);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "t", "uarm_y", "uarm_dev", "larm_y", "larm_dev", "elbow_z", "larm_|w|");
    println!("{}", "-".repeat(72));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        // 30 substeps per frame, with the upper-arm anchor force re-applied each one
        // (force/torque accumulators are cleared at the end of every substep).
        for _ in 0..SOLVER_ITERATIONS {
            {
                let body = &mut world.bodies[upper_idx];
                let pos_err = body.position - upper_rest_pos;
                let force   = -ANCHOR_KP_LIN * pos_err - ANCHOR_KD_LIN * body.linear_velocity;
                body.apply_force(force);

                let orient_err = body.orientation.scaled_axis();
                let torque     = -ANCHOR_KP_ANG * orient_err - ANCHOR_KD_ANG * body.angular_velocity;
                body.apply_torque(torque);
            }
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let uarm = &world.bodies[upper_idx];
            let larm = &world.bodies[lower_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse())
                              .scaled_axis().z;
            println!("{:<6.2} {:<10.5} {:<10.5} {:<10.5} {:<10.5} {:<10.5} {:<10.5}",
                     t,
                     uarm.position.y,
                     uarm.orientation.scaled_axis().norm(),
                     larm.position.y,
                     larm.orientation.scaled_axis().norm(),
                     elbow_z,
                     larm.angular_velocity.norm());
            next_print += 0.5;
        }
    }
}
