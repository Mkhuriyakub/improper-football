// First step in finger validation. A single index finger as a 3-phalanx chain
// (proximal, middle, distal) attached to a fixed hand by three 1-DOF flexion
// joints (MCP, PIP, DIP). All passive biomechanics — no active control yet.
//
// Dimensions and biomech values are anthropometric estimates for an adult
// index finger:
//   proximal phalanx  ≈  44 mm long, 10 mm radius,  8 g
//   middle phalanx    ≈  24 mm long,  9 mm radius,  5 g
//   distal phalanx    ≈  16 mm long,  8 mm radius,  3 g
//
// Anatomical joint ranges (Z axis = flexion):
//   MCP: extension −30° → flexion 90°    [−0.52, 1.57] rad
//   PIP: nearly no extension, flexion 120°  [−0.05, 2.09] rad
//   DIP: nearly no extension, flexion 80°   [−0.05, 1.40] rad
//
// Test: finger released from a deeply-curled pose (MCP 1.0, PIP 1.5, DIP 1.0)
// — like a half-fist. Should relax toward the resting "slightly curled" pose
// that an unactivated finger naturally assumes (passive springs target small
// flex angles at each joint).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::tendon::{Tendon, TendonSpan};
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.05;

const HAND_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);
const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KD_ANG: f64 =     50.0;

// Hand (the finger's parent — kept fixed in this test by anchor).
const HAND_MASS:      f64 = 0.54;
const HAND_RADIUS:    f64 = 0.030;
const HAND_HALF_H:    f64 = 0.050;

// Phalanges
const PROX_MASS:    f64 = 0.008;
const PROX_RADIUS:  f64 = 0.010;
const PROX_HALF_H:  f64 = 0.022;
const MID_MASS:     f64 = 0.005;
const MID_RADIUS:   f64 = 0.009;
const MID_HALF_H:   f64 = 0.012;
const DIST_MASS:    f64 = 0.003;
const DIST_RADIUS:  f64 = 0.008;
const DIST_HALF_H:  f64 = 0.008;

// Joint biomech
struct JointBiomech {
    z_min: f64, z_max: f64,
    damping: f64,
    stiffness: f64,
    neutral: f64,
    limit_stiffness: f64,
    limit_exponent: f64,
}

// Per-joint passive parameters bumped to represent soft tissue (skin, fat,
// joint capsule) — substantially higher damping than the bare-ligament values
// would suggest, because a real finger is wrapped in flesh that resists
// motion. The neutral angles reflect resting-tone curl, not anatomical
// neutral.
const MCP: JointBiomech = JointBiomech {
    z_min: -0.52, z_max: 1.57,
    damping: 0.05, stiffness: 0.05, neutral: 0.30,
    limit_stiffness: 0.5, limit_exponent: 2.0,
};
const PIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 2.09,
    damping: 0.04, stiffness: 0.03, neutral: 0.40,
    limit_stiffness: 0.4, limit_exponent: 2.0,
};
const DIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 1.40,
    damping: 0.025, stiffness: 0.02, neutral: 0.30,
    limit_stiffness: 0.3, limit_exponent: 2.0,
};

// Flexor tendon parameters — represents the flexor digitorum profundus that
// crosses all three joints. Moment arms ~5 mm at each joint match the
// biomech literature for the index finger.
const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 200.0;  // N per m of excursion
const TENDON_DAMPING:    f64 = 30.0;   // N·s per m of excursion
// Rest excursion: tendon length consumed by the natural resting curl
// (sum over joints of moment_arm × neutral_angle).
const TENDON_REST_EXCURSION: f64 = TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);

// Stay within all anatomical ranges so no limit fires at t=0; observe purely
// passive spring + damping decay first.
const INITIAL_MCP: f64 = 0.6;
const INITIAL_PIP: f64 = 0.7;
const INITIAL_DIP: f64 = 0.5;

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
    joint_rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * joint_rel_rot;
    let joint_world = parent.position + parent.orientation * parent_distal_local;
    let child_pos = joint_world - child_orient * child_proximal_local;
    (child_pos, child_orient)
}

fn add_finger_joint(
    world: &mut XpbdWorld,
    parent_idx: usize, child_idx: usize,
    parent_distal_local: Vec3, child_proximal_local: Vec3,
    biomech: &JointBiomech,
) {
    let mut joint = PositionalConstraint::new(parent_idx, child_idx, 0.0, 0.0);
    joint.local_anchor_a = parent_distal_local;
    joint.local_anchor_b = child_proximal_local;
    world.constraints.push(joint);

    // X and Y use soft limits with low stiffness — fingers don't deviate
    // laterally or twist much, but a hard lock with phalanx-scale inertia
    // is numerically unstable, so we use a gentle soft cage instead.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent_idx, child_idx, Vec3::x(),
        -0.10, 0.10, 0.3, 2.0,
    ));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent_idx, child_idx, Vec3::y(),
        -0.10, 0.10, 0.3, 2.0,
    ));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent_idx, child_idx, Vec3::z(),
        biomech.z_min, biomech.z_max,
        biomech.limit_stiffness, biomech.limit_exponent,
    ));
    world.angular_dampers.push(
        AngularDamper::new(parent_idx, child_idx, Vec3::z(), biomech.damping));
    world.angular_springs.push(
        AngularSpring::new(parent_idx, child_idx, Vec3::z(),
                           biomech.stiffness, biomech.neutral));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Hand (anchored)
    let mut hand = RigidBody::new(HAND_MASS, capsule_inertia(HAND_MASS, HAND_RADIUS, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

    // Proximal phalanx — joined at MCP at the hand's distal (-Y) end.
    let mut prox = RigidBody::new(PROX_MASS, capsule_inertia(PROX_MASS, PROX_RADIUS, PROX_HALF_H));
    let mcp_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_MCP));
    let (pos, orient) = place_child(
        &world.bodies[hand_idx],
        Vec3::new(0.0, -HAND_HALF_H, 0.0),
        Vec3::new(0.0,  PROX_HALF_H, 0.0),
        mcp_rel,
    );
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    // Middle phalanx — PIP joint
    let mut mid = RigidBody::new(MID_MASS, capsule_inertia(MID_MASS, MID_RADIUS, MID_HALF_H));
    let pip_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_PIP));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        Vec3::new(0.0, -PROX_HALF_H, 0.0),
        Vec3::new(0.0,  MID_HALF_H, 0.0),
        pip_rel,
    );
    mid.position = pos;
    mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    // Distal phalanx — DIP joint
    let mut dist = RigidBody::new(DIST_MASS, capsule_inertia(DIST_MASS, DIST_RADIUS, DIST_HALF_H));
    let dip_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_DIP));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        Vec3::new(0.0, -MID_HALF_H, 0.0),
        Vec3::new(0.0,  DIST_HALF_H, 0.0),
        dip_rel,
    );
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_finger_joint(&mut world, hand_idx, prox_idx,
                     Vec3::new(0.0, -HAND_HALF_H, 0.0),
                     Vec3::new(0.0,  PROX_HALF_H, 0.0),
                     &MCP);
    add_finger_joint(&mut world, prox_idx, mid_idx,
                     Vec3::new(0.0, -PROX_HALF_H, 0.0),
                     Vec3::new(0.0,  MID_HALF_H, 0.0),
                     &PIP);
    add_finger_joint(&mut world, mid_idx, dist_idx,
                     Vec3::new(0.0, -MID_HALF_H, 0.0),
                     Vec3::new(0.0,  DIST_HALF_H, 0.0),
                     &DIP);

    // Flexor tendon spanning all three joints. This is the biological
    // constraint that prevents finger joints from moving independently —
    // they're physically tied together by the shared tendon.
    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: hand_idx, body_b: prox_idx,
                         axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,
                         axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx,
                         axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    println!("FINGER REST  ({:.1}s, dt={:.5}s, {} substeps)",
             SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("Initial: MCP={}, PIP={}, DIP={}", INITIAL_MCP, INITIAL_PIP, INITIAL_DIP);
    println!("Neutral targets (passive springs): MCP={}, PIP={}, DIP={}",
             MCP.neutral, PIP.neutral, DIP.neutral);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10}",
             "t", "mcp_z", "pip_z", "dip_z");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            // Hand anchor — full 6-DOF spring + damper, holds hand fixed.
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
            let mid  = &world.bodies[mid_idx];
            let dist = &world.bodies[dist_idx];
            let mcp_z = (prox.orientation * hand.orientation.inverse()).scaled_axis().z;
            let pip_z = (mid.orientation  * prox.orientation.inverse()).scaled_axis().z;
            let dip_z = (dist.orientation * mid.orientation.inverse()).scaled_axis().z;
            println!("{:<6.2} {:<+10.5} {:<+10.5} {:<+10.5}",
                     t, mcp_z, pip_z, dip_z);
            next_print += PRINT_INTERVAL;
        }
    }

    let hand = &world.bodies[hand_idx];
    let prox = &world.bodies[prox_idx];
    let mid  = &world.bodies[mid_idx];
    let dist = &world.bodies[dist_idx];
    let mcp_z = (prox.orientation * hand.orientation.inverse()).scaled_axis().z;
    let pip_z = (mid.orientation  * prox.orientation.inverse()).scaled_axis().z;
    let dip_z = (dist.orientation * mid.orientation.inverse()).scaled_axis().z;
    println!();
    println!("Final state:");
    println!("  MCP = {:+.5} rad ({:+.2}°)", mcp_z, mcp_z.to_degrees());
    println!("  PIP = {:+.5} rad ({:+.2}°)", pip_z, pip_z.to_degrees());
    println!("  DIP = {:+.5} rad ({:+.2}°)", dip_z, dip_z.to_degrees());
}
