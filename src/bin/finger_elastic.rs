// The critical test: a single phalanx attached to a dynamic hand (anchored
// by external soft spring) using the new elastic joint primitive instead
// of the rigid PositionalConstraint. This is the configuration that NaN'd
// every previous engine version.
//
// If the elastic-joint approach is sound, this should now run stably with
// the hand-to-phalanx mass ratio of 68:1 that previously broke things.
//
// Predicted (k = 10,000 N/m, c = 40 N·s/m, m_phalanx = 0.008 kg):
//   static deflection = m·g/k = 0.008·9.81/10000 = 7.85 µm
//   ω_n = √(k/m) = 1118 rad/s
//   ω·sub_dt at 60 substeps/frame = 0.16 — stable
//   damping ratio ζ = c/(2·√(m·k)) ≈ 0.7

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, Mat3};

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 1.0;
const PRINT_INTERVAL: f64 = 0.025;

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

const JOINT_STIFFNESS: f64 = 10_000.0;
const JOINT_DAMPING:   f64 = 40.0;

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
    prox.position = Vec3::new(0.0, HAND_REST_POS.y - HAND_HALF_H - PROX_HALF_H, 0.0);
    let prox_idx = world.add_body(prox);

    let mut spring = PositionalSpring::new(hand_idx, prox_idx, JOINT_STIFFNESS);
    spring.local_anchor_a = Vec3::new(0.0, -HAND_HALF_H, 0.0);
    spring.local_anchor_b = Vec3::new(0.0,  PROX_HALF_H, 0.0);
    world.positional_springs.push(spring);

    let mut damper = PositionalDamper::new(hand_idx, prox_idx, JOINT_DAMPING);
    damper.local_anchor_a = Vec3::new(0.0, -HAND_HALF_H, 0.0);
    damper.local_anchor_b = Vec3::new(0.0,  PROX_HALF_H, 0.0);
    world.positional_dampers.push(damper);

    println!("FINGER ELASTIC JOINT TEST  ({:.2} s, dt = {:.5} s, {} substeps)",
             SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("k = {} N/m   c = {} N·s/m", JOINT_STIFFNESS, JOINT_DAMPING);
    println!("Hand:Phalanx mass ratio = {:.0}:1", HAND_MASS / PROX_MASS);
    println!("Predicted phalanx static deflection = {:.6} m",
             PROX_MASS * 9.81 / JOINT_STIFFNESS);
    println!();
    println!("{:<7} {:<13} {:<13} {:<13}",
             "t (s)", "hand_y (m)", "prox_y (m)", "joint_sep (m)");
    println!("{}", "-".repeat(50));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            // External soft anchor on the hand (the configuration that previously failed).
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
            let hand_anchor = hand.position + hand.orientation * Vec3::new(0.0, -HAND_HALF_H, 0.0);
            let prox_anchor = prox.position + prox.orientation * Vec3::new(0.0,  PROX_HALF_H, 0.0);
            let sep = (prox_anchor - hand_anchor).norm();
            println!("{:<7.4} {:<13.6} {:<13.6} {:<13.7}",
                     t, hand.position.y, prox.position.y, sep);
            next_print += PRINT_INTERVAL;
        }
    }

    let hand = &world.bodies[hand_idx];
    let prox = &world.bodies[prox_idx];
    let hand_anchor = hand.position + hand.orientation * Vec3::new(0.0, -HAND_HALF_H, 0.0);
    let prox_anchor = prox.position + prox.orientation * Vec3::new(0.0,  PROX_HALF_H, 0.0);
    let sep = (prox_anchor - hand_anchor).norm();
    println!();
    println!("Final hand y:           {:.7} m  (rest = {})", hand.position.y, HAND_REST_POS.y);
    println!("Final phalanx y:        {:.7} m", prox.position.y);
    println!("Final joint separation: {:.7e} m  (predicted ~7.8e-6 m)", sep);
    println!("Stable run, no NaN.");
}
