// Step in the elastic-joint hand build: wrist on its own (no fingers yet).
//
// Bodies:
//   forearm stub  — pinned to world (mass = 0), held horizontal so gravity
//                   exerts a real torque on the wrist.
//   palm          — full mass + inertia, hanging off the forearm via the wrist.
//
// Joint:
//   wrist — elastic positional anchor (PositionalSpring + PositionalDamper)
//           plus per-axis angular biomech:
//             Z (flexion / extension)         range [-1.10, +1.30] rad
//             X (radial / ulnar deviation)    range [-0.35, +0.60] rad
//             Y (pronation: kept tight)       range [-0.05, +0.05] rad
//           Each axis: AngularSoftLimit + AngularSpring (neutral 0) + AngularDamper.
//
// Initial state: simultaneous palmar flex (Z = +0.40) and ulnar deviation
// (X = +0.30). Gravity acts on the palm at a 0.04 m lever arm below the wrist.
// Expected: both Z and X decay to their gravity-loaded equilibria within a
// second or two, no NaN, no drift.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.05;

const FOREARM_HALF_H: f64 = 0.135;
const FOREARM_RADIUS: f64 = 0.040;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

// Elastic joint: ~60 µm static deflection at palm weight (well under 1 mm).
const JOINT_K: f64 = 50_000.0;
const JOINT_C: f64 =    100.0;

// Per-axis angular biomech.
const WRIST_Z_K:        f64 = 1.0;
const WRIST_X_K:        f64 = 1.5;
const WRIST_Y_K:        f64 = 5.0;
const WRIST_Z_C:        f64 = 0.02;
const WRIST_X_C:        f64 = 0.02;
const WRIST_Z_MIN:      f64 = -1.10;  // dorsi extension
const WRIST_Z_MAX:      f64 =  1.30;  // palmar flexion
const WRIST_X_MIN:      f64 = -0.35;  // radial deviation
const WRIST_X_MAX:      f64 =  0.60;  // ulnar deviation
const WRIST_Y_LIM:      f64 =  0.05;
const WRIST_LIMIT_K:    f64 = 200.0;
const WRIST_Y_LIMIT_K:  f64 = 500.0;
const WRIST_LIMIT_EXP:  f64 =   2.0;

const INITIAL_WRIST_Z: f64 = 0.40;
const INITIAL_WRIST_X: f64 = 0.30;

const FOREARM_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);

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

fn place_child(
    parent: &RigidBody,
    parent_distal_local: Vec3,
    child_proximal_local: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint_world = parent.position + parent.orientation * parent_distal_local;
    let child_pos = joint_world - child_orient * child_proximal_local;
    (child_pos, child_orient)
}

fn main() {
    let mut world = XpbdWorld::new();

    // Forearm pinned (mass = 0). Held horizontal: capsule's local +y axis is
    // rotated to world +x, so the palm hangs out to the right and gravity
    // exerts a clear torque on the wrist.
    let mut forearm = RigidBody::new(
        0.0,
        capsule_inertia(1.5, FOREARM_RADIUS, FOREARM_HALF_H),
    );
    forearm.position = FOREARM_REST_POS;
    forearm.orientation = UnitQuaternion::from_scaled_axis(
        Vec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2),
    );
    let forearm_idx = world.add_body(forearm);

    // Palm: simultaneous Z and X deflection at the wrist.
    let mut palm = RigidBody::new(
        PALM_MASS,
        capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H),
    );
    let wrist_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_WRIST_Z))
                  * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_WRIST_X, 0.0, 0.0));
    let (pos, orient) = place_child(
        &world.bodies[forearm_idx],
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0,  PALM_HALF_H, 0.0),
        wrist_rel,
    );
    palm.position = pos;
    palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    // Wrist elastic joint (positional anchor).
    let mut spring = PositionalSpring::new(forearm_idx, palm_idx, JOINT_K);
    spring.local_anchor_a = Vec3::new(0.0, -FOREARM_HALF_H, 0.0);
    spring.local_anchor_b = Vec3::new(0.0,  PALM_HALF_H,    0.0);
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(forearm_idx, palm_idx, JOINT_C);
    damp.local_anchor_a = Vec3::new(0.0, -FOREARM_HALF_H, 0.0);
    damp.local_anchor_b = Vec3::new(0.0,  PALM_HALF_H,    0.0);
    world.positional_dampers.push(damp);

    // Per-axis angular biomech.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, Vec3::z(),
        WRIST_Z_MIN, WRIST_Z_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(forearm_idx, palm_idx, Vec3::z(), WRIST_Z_C));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, Vec3::z(), WRIST_Z_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, Vec3::x(),
        WRIST_X_MIN, WRIST_X_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(forearm_idx, palm_idx, Vec3::x(), WRIST_X_C));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, Vec3::x(), WRIST_X_K, 0.0));

    // Y axis (axial twist of the hand vs forearm) gets a tight soft limit
    // and a stiff spring — anatomically there's almost no rotation here at
    // the carpus itself.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, Vec3::y(),
        -WRIST_Y_LIM, WRIST_Y_LIM, WRIST_Y_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, Vec3::y(), WRIST_Y_K, 0.0));

    println!("WRIST COMPLIANT (elastic joint + full biomech)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("forearm pinned horizontal; palm hangs off the wrist under gravity.");
    println!("initial: wrist_z = {:.2} rad, wrist_x = {:.2} rad",
             INITIAL_WRIST_Z, INITIAL_WRIST_X);
    println!();
    println!("{:<6} {:<11} {:<11} {:<11}",
             "t", "wrist_z", "wrist_x", "wrist_y");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let f = &world.bodies[forearm_idx];
            let p = &world.bodies[palm_idx];
            let rel = (p.orientation * f.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.z, rel.x, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let f = &world.bodies[forearm_idx];
    let p = &world.bodies[palm_idx];
    let rel = (p.orientation * f.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: wrist_z = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       wrist_x = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       wrist_y = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());

    // Analytical equilibrium check on Z (dominant gravity axis).
    // Lever from wrist to palm CoM (in world) = −palm.orientation·(0, PALM_HALF_H, 0).
    // Forearm rotated −90° about z places its distal end at world −x, so the
    // lever points in world +x and gravity torque about world z is
    // (−PALM_HALF_H, 0, 0) × (0, −m·g, 0) = (0, 0, +m·g·PALM_HALF_H). The
    // wrist's z-axis is forearm-local z = world z, so θ_eq is positive.
    let predicted_z = PALM_MASS * 9.81 * PALM_HALF_H / WRIST_Z_K;
    println!("Predicted wrist_z equilibrium under gravity: {:+.5} rad ({:+.2}°)",
             predicted_z, predicted_z.to_degrees());
}
