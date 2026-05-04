// Diagnostic: minimal test with just the proximal phalanx attached to the
// hand. Isolates whether the finger_rest instability is from chain length
// or from the very large hand-to-phalanx mass ratio (~67:1) that the
// single-iteration constraint solver may not handle.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 2.0;
const PRINT_INTERVAL: f64 = 0.05;

const HAND_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);
const HAND_MASS:    f64 = 0.54;
const HAND_RADIUS:  f64 = 0.030;
const HAND_HALF_H:  f64 = 0.050;

const PROX_MASS:    f64 = 0.008;
const PROX_RADIUS:  f64 = 0.010;
const PROX_HALF_H:  f64 = 0.022;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KD_ANG: f64 =     50.0;

const INITIAL_MCP: f64 = 0.5;

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

fn main() {
    let mut world = XpbdWorld::new();

    let mut hand = RigidBody::new(HAND_MASS, capsule_inertia(HAND_MASS, HAND_RADIUS, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

    let mut prox = RigidBody::new(PROX_MASS, capsule_inertia(PROX_MASS, PROX_RADIUS, PROX_HALF_H));
    let prox_orient = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_MCP));
    let joint_world = HAND_REST_POS + Vec3::new(0.0, -HAND_HALF_H, 0.0);
    let prox_proximal_world = prox_orient * Vec3::new(0.0, PROX_HALF_H, 0.0);
    prox.position = joint_world - prox_proximal_world;
    prox.orientation = prox_orient;
    let prox_idx = world.add_body(prox);

    let mut joint = PositionalConstraint::new(hand_idx, prox_idx, 0.0, 0.0);
    joint.local_anchor_a = Vec3::new(0.0, -HAND_HALF_H, 0.0);
    joint.local_anchor_b = Vec3::new(0.0,  PROX_HALF_H, 0.0);
    world.constraints.push(joint);

    world.angular_soft_limits.push(AngularSoftLimit::new(
        hand_idx, prox_idx, Vec3::z(), -0.52, 1.57, 0.5, 2.0));
    world.angular_dampers.push(
        AngularDamper::new(hand_idx, prox_idx, Vec3::z(), 0.001));
    world.angular_springs.push(
        AngularSpring::new(hand_idx, prox_idx, Vec3::z(), 0.005, 0.20));

    println!("ONE PHALANX  ({:.1}s, dt={:.5}s, {} substeps)", SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("Hand mass {} / Phalanx mass {} = ratio {:.0}:1",
             HAND_MASS, PROX_MASS, HAND_MASS/PROX_MASS);
    println!();
    println!("{:<6} {:<10} {:<10}", "t", "mcp_z", "prox_y");
    println!("{}", "-".repeat(36));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            let body = &mut world.bodies[hand_idx];
            let pos_err = body.position - HAND_REST_POS;
            let force = -ANCHOR_KP_LIN * pos_err - ANCHOR_KD_LIN * body.linear_velocity;
            let orient_err = body.orientation.scaled_axis();
            let torque = -ANCHOR_KP_ANG * orient_err - ANCHOR_KD_ANG * body.angular_velocity;
            body.apply_force(force);
            body.apply_torque(torque);
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let hand = &world.bodies[hand_idx];
            let prox = &world.bodies[prox_idx];
            let mcp = (prox.orientation * hand.orientation.inverse()).scaled_axis().z;
            println!("{:<6.2} {:<+10.5} {:<+10.5}", t, mcp, prox.position.y);
            next_print += PRINT_INTERVAL;
        }
    }
}
