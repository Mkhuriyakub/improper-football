// Diagnostic: 3-phalanx finger with PINNED hand (mass=0). Removes the hand-anchor
// explicit-damping path entirely. Adds biomech primitives one at a time to
// pinpoint which triggers the chain explosion.
//
// Set DIAG_LEVEL via the const to vary what's added:
//   0 = bare elastic chain (control — should match finger_diagnostic.rs)
//   1 = + AngularSpring on z (very low stiffness)
//   2 = + AngularDamper (implicit) on z
//   3 = + AngularSoftLimit on z
//   4 = + X/Y soft limits

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
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 0.5;
const PRINT_INTERVAL: f64 = 0.025;

const DIAG_LEVEL: u32 = 4;

const HAND_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);
const HAND_RADIUS:  f64 = 0.030;
const HAND_HALF_H:  f64 = 0.050;

const PROX_MASS:    f64 = 0.008;
const PROX_RADIUS:  f64 = 0.010;
const PROX_HALF_H:  f64 = 0.022;

const MID_MASS:     f64 = 0.005;
const MID_RADIUS:   f64 = 0.009;
const MID_HALF_H:   f64 = 0.012;

const DIST_MASS:    f64 = 0.003;
const DIST_RADIUS:  f64 = 0.008;
const DIST_HALF_H:  f64 = 0.008;

const JOINT_K: f64 = 1_000.0;
const C_HAND_PROX: f64 = 4.0;
const C_PROX_MID:  f64 = 2.5;
const C_MID_DIST:  f64 = 2.0;

const ANG_K: f64 = 0.0001;
const ANG_C: f64 = 0.0001;
const ANG_LIMIT_K: f64 = 0.05;

const NEUTRAL_MCP: f64 = 0.30;
const NEUTRAL_PIP: f64 = 0.40;
const NEUTRAL_DIP: f64 = 0.30;

const INITIAL_MCP: f64 = 0.35;
const INITIAL_PIP: f64 = 0.45;
const INITIAL_DIP: f64 = 0.35;

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

fn add_elastic_joint(
    world: &mut XpbdWorld,
    parent_idx: usize, child_idx: usize,
    parent_distal_local: Vec3, child_proximal_local: Vec3,
    damping: f64,
) {
    let mut spring = PositionalSpring::new(parent_idx, child_idx, JOINT_K);
    spring.local_anchor_a = parent_distal_local;
    spring.local_anchor_b = child_proximal_local;
    world.positional_springs.push(spring);

    let mut damper = PositionalDamper::new(parent_idx, child_idx, damping);
    damper.local_anchor_a = parent_distal_local;
    damper.local_anchor_b = child_proximal_local;
    world.positional_dampers.push(damper);
}

fn add_angular_biomech(
    world: &mut XpbdWorld, parent: usize, child: usize, neutral: f64,
    z_min: f64, z_max: f64,
) {
    if DIAG_LEVEL >= 1 {
        world.angular_springs.push(
            AngularSpring::new(parent, child, Vec3::z(), ANG_K, neutral));
    }
    if DIAG_LEVEL >= 2 {
        world.angular_dampers.push(
            AngularDamper::new(parent, child, Vec3::z(), ANG_C));
    }
    if DIAG_LEVEL >= 3 {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            parent, child, Vec3::z(), z_min, z_max, ANG_LIMIT_K, 2.0));
    }
    if DIAG_LEVEL >= 4 {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            parent, child, Vec3::x(), -0.10, 0.10, 0.3, 2.0));
        world.angular_soft_limits.push(AngularSoftLimit::new(
            parent, child, Vec3::y(), -0.10, 0.10, 0.3, 2.0));
    }
}

fn main() {
    let mut world = XpbdWorld::new();

    // PINNED hand: mass = 0. Removes any need for an anchor controller.
    let mut hand = RigidBody::new(0.0, capsule_inertia(0.54, HAND_RADIUS, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

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

    add_elastic_joint(&mut world, hand_idx, prox_idx,
                      Vec3::new(0.0, -HAND_HALF_H, 0.0),
                      Vec3::new(0.0,  PROX_HALF_H, 0.0),
                      C_HAND_PROX);
    add_angular_biomech(&mut world, hand_idx, prox_idx, NEUTRAL_MCP, -0.52, 1.57);

    add_elastic_joint(&mut world, prox_idx, mid_idx,
                      Vec3::new(0.0, -PROX_HALF_H, 0.0),
                      Vec3::new(0.0,  MID_HALF_H, 0.0),
                      C_PROX_MID);
    add_angular_biomech(&mut world, prox_idx, mid_idx, NEUTRAL_PIP, -0.05, 2.09);

    add_elastic_joint(&mut world, mid_idx, dist_idx,
                      Vec3::new(0.0, -MID_HALF_H, 0.0),
                      Vec3::new(0.0,  DIST_HALF_H, 0.0),
                      C_MID_DIST);
    add_angular_biomech(&mut world, mid_idx, dist_idx, NEUTRAL_DIP, -0.05, 1.40);

    println!("FINGER PINNED DIAG (level {}, pinned hand)", DIAG_LEVEL);
    println!("  level 0 = bare elastic chain only");
    println!("  level 1 = + ang springs");
    println!("  level 2 = + ang dampers");
    println!("  level 3 = + Z soft limits");
    println!("  level 4 = + X/Y soft limits");
    println!("Initial: MCP={}, PIP={}, DIP={}", INITIAL_MCP, INITIAL_PIP, INITIAL_DIP);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10}",
             "t (s)", "mcp_z", "pip_z", "dip_z");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut nan_reported = false;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        let h = &world.bodies[hand_idx];
        let p = &world.bodies[prox_idx];
        let m = &world.bodies[mid_idx];
        let d = &world.bodies[dist_idx];
        let mcp = (p.orientation * h.orientation.inverse()).scaled_axis().z;
        let pip = (m.orientation * p.orientation.inverse()).scaled_axis().z;
        let dip = (d.orientation * m.orientation.inverse()).scaled_axis().z;

        if !nan_reported && (!mcp.is_finite() || !pip.is_finite() || !dip.is_finite()) {
            println!("DIVERGED at t={:.4}", t);
            nan_reported = true;
        }

        if t >= next_print {
            println!("{:<6.3} {:<+10.5} {:<+10.5} {:<+10.5}", t, mcp, pip, dip);
            next_print += PRINT_INTERVAL;
        }
    }
}
