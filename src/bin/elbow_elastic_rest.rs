// First test of the elastic joint approach.
//
// Setup: upper arm is pinned (mass = 0). Lower arm hangs from it via a
// PositionalSpring + PositionalDamper at the elbow anchor — no rigid
// constraint, no Baumgarte, no constraint solver. Just elastic-spring
// physics with viscous damping.
//
// This is the smallest test of "biomechanically faithful joints" — does
// the elastic joint hold the lower arm against gravity at the predicted
// equilibrium, and settle within the predicted time?
//
// Predicted (k=100,000 N/m, m_lower=1.44 kg):
//   static deflection = m·g/k = 1.44·9.81/100000 = 0.000141 m = 0.14 mm
//   ω_n = √(k/m) = 263 rad/s (period 0.024 s)
//   c for ζ=0.7: 2·ζ·√(m·k) = 530 N·s/m
//   settling time (1% envelope) ≈ 4/(ζ·ω_n) = 22 ms
//
// Expected lower arm Y position at equilibrium: 1.05 - 0.000141 = 1.04986 m.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 1.0;
const PRINT_INTERVAL: f64 = 0.005;  // 5 ms — fine enough to see the settling transient

const JOINT_STIFFNESS: f64 = 100_000.0;
const JOINT_DAMPING:   f64 = 530.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

fn main() {
    let mut world = XpbdWorld::new();

    // Upper arm — pinned (mass 0 → never moves), placed at standard rest.
    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(0.0, upper_def.inertia_tensor());
    upper.position = Vec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    // Lower arm — placed at start so the elbow joint is at zero error.
    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.position = Vec3::new(0.0, 1.05, 0.0);  // 0.13 below elbow world position
    let lower_idx = world.add_body(lower);

    // Elastic joint: spring + damper at elbow anchors.
    let mut spring = PositionalSpring::new(upper_idx, lower_idx, JOINT_STIFFNESS);
    spring.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    spring.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.positional_springs.push(spring);

    let mut damper = PositionalDamper::new(upper_idx, lower_idx, JOINT_DAMPING);
    damper.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    damper.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.positional_dampers.push(damper);

    println!("ELBOW ELASTIC JOINT REST TEST  ({:.2} s, dt = {:.5} s)", SIM_DURATION, DT);
    println!("k = {} N/m   c = {} N·s/m", JOINT_STIFFNESS, JOINT_DAMPING);
    println!("Predicted deflection = {:.6} m   ω_n = {:.1} rad/s   settling = ~22 ms",
             1.44 * 9.81 / JOINT_STIFFNESS,
             (JOINT_STIFFNESS / 1.44_f64).sqrt());
    println!("Predicted equilibrium lower-arm y = {:.6} m", 1.05 - 1.44 * 9.81 / JOINT_STIFFNESS);
    println!();
    println!("{:<7} {:<14} {:<14} {:<14}",
             "t (s)", "lower_y (m)", "y_offset (m)", "v_y (m/s)");
    println!("{}", "-".repeat(54));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let lower = &world.bodies[lower_idx];
            let y = lower.position.y;
            let offset = y - 1.05;  // signed offset from rest position
            println!("{:<7.4} {:<14.7} {:<+14.7} {:<+14.6}",
                     t, y, offset, lower.linear_velocity.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let lower = &world.bodies[lower_idx];
    let final_y = lower.position.y;
    let final_offset = final_y - 1.05;
    let predicted_offset = -1.44 * 9.81 / JOINT_STIFFNESS;
    println!();
    println!("Final lower arm y:       {:.7} m", final_y);
    println!("Final offset from rest:  {:+.7} m  (predicted {:+.7})",
             final_offset, predicted_offset);
    println!("Deviation from prediction: {:+.4e} m",
             final_offset - predicted_offset);
}
