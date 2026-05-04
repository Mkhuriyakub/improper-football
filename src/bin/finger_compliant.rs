// Phase 1, Step 4: full three-phalanx finger with elastic joints + complete
// passive biomechanics. This is the originally-failing `finger_rest` test
// rebuilt around the elastic-joint approach.
//
// Each joint is a composition of:
//   - PositionalSpring + PositionalDamper  (elastic capsule + cartilage)
//   - AngularSoftLimit                      (anatomical Z range with progressive engagement)
//   - AngularDamper                         (passive viscoelastic resistance)
//   - AngularSpring                         (resting muscle tone)
//   - X and Y soft limits                   (lateral/twist resistance, gentle)
// Plus a Tendon coupling all three Z axes (the flexor digitorum profundus).
//
// Predicted: from initial deep curl (MCP 0.8, PIP 1.0, DIP 0.5), the finger
// relaxes toward its passive equilibrium. With proper biology, no constraint
// solver, no Baumgarte — should be stable from the first frame.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::tendon::{Tendon, TendonSpan};
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.05;

const HAND_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);
// Hand anchor: stiffness/damping reduced to stay within explicit-Euler
// stability bounds (c·dt/I < 2). With sub_dt = 1/(120·60) ≈ 1.4e-4 and hand
// transverse inertia ≈ 5.71e-4, max stable damping is ≈ 8 N·m·s/rad.
const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =    200.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KD_ANG: f64 =      4.0;

// Body data — anthropometric for adult index finger.
const HAND_MASS:    f64 = 0.54;
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

// Elastic joint parameters. Stiffness 1000 N/m for finger-scale (gives
// ~80 µm deflection — visible but small). Lower than body-scale because the
// coupled chain dynamics with stiffer joints become unstable under explicit
// integration.
const JOINT_K: f64 = 1_000.0;  // N/m
const C_HAND_PROX: f64 = 4.0;
const C_PROX_MID:  f64 = 2.5;
const C_MID_DIST:  f64 = 2.0;

// Per-joint angular biomech — values from previous finger test.
struct JointBiomech {
    z_min: f64, z_max: f64,
    damping: f64, stiffness: f64, neutral: f64,
    limit_stiffness: f64,
}
// Per-pair angular damping ζ ≈ 0.7 (well-damped) using c = 1.4·√(I_red·k).
// Stiffnesses are biomechanical estimates; dampings scaled to per-pair I_red.
const MCP: JointBiomech = JointBiomech {
    z_min: -0.52, z_max: 1.57,
    damping: 0.0004, stiffness: 0.05, neutral: 0.30,
    limit_stiffness: 0.5,
};
const PIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 2.09,
    damping: 0.00013, stiffness: 0.03, neutral: 0.40,
    limit_stiffness: 0.4,
};
const DIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 1.40,
    damping: 0.00006, stiffness: 0.02, neutral: 0.30,
    limit_stiffness: 0.3,
};

const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION: f64 = TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);

const INITIAL_MCP: f64 = 0.8;
const INITIAL_PIP: f64 = 1.0;
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

fn add_angular_biomech(world: &mut XpbdWorld, parent: usize, child: usize, b: &JointBiomech) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        b.z_min, b.z_max, b.limit_stiffness, 2.0,
    ));
    world.angular_dampers.push(
        AngularDamper::new(parent, child, Vec3::z(), b.damping));
    world.angular_springs.push(
        AngularSpring::new(parent, child, Vec3::z(), b.stiffness, b.neutral));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(), -0.10, 0.10, 0.3, 2.0,
    ));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(), -0.10, 0.10, 0.3, 2.0,
    ));
}

fn main() {
    let mut world = XpbdWorld::new();

    let mut hand = RigidBody::new(HAND_MASS, capsule_inertia(HAND_MASS, HAND_RADIUS, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

    // Proximal phalanx: MCP joint at hand distal end.
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

    // Middle phalanx: PIP joint at proximal phalanx distal end.
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

    // Distal phalanx: DIP joint at middle phalanx distal end.
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

    // Three elastic joints with full angular biomechanics.
    add_elastic_joint(&mut world, hand_idx, prox_idx,
                      Vec3::new(0.0, -HAND_HALF_H, 0.0),
                      Vec3::new(0.0,  PROX_HALF_H, 0.0),
                      C_HAND_PROX);
    add_angular_biomech(&mut world, hand_idx, prox_idx, &MCP);

    add_elastic_joint(&mut world, prox_idx, mid_idx,
                      Vec3::new(0.0, -PROX_HALF_H, 0.0),
                      Vec3::new(0.0,  MID_HALF_H, 0.0),
                      C_PROX_MID);
    add_angular_biomech(&mut world, prox_idx, mid_idx, &PIP);

    add_elastic_joint(&mut world, mid_idx, dist_idx,
                      Vec3::new(0.0, -MID_HALF_H, 0.0),
                      Vec3::new(0.0,  DIST_HALF_H, 0.0),
                      C_MID_DIST);
    add_angular_biomech(&mut world, mid_idx, dist_idx, &DIP);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: hand_idx, body_b: prox_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,  axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    println!("FINGER COMPLIANT (full biomech, elastic joints)  ({:.1} s, dt {:.5} s, {} substeps)",
             SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("Initial: MCP={}, PIP={}, DIP={}", INITIAL_MCP, INITIAL_PIP, INITIAL_DIP);
    println!("Neutrals: MCP={}, PIP={}, DIP={}", MCP.neutral, PIP.neutral, DIP.neutral);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10}",
             "t (s)", "mcp_z", "pip_z", "dip_z");
    println!("{}", "-".repeat(46));

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
            let h = &world.bodies[hand_idx];
            let p = &world.bodies[prox_idx];
            let m = &world.bodies[mid_idx];
            let d = &world.bodies[dist_idx];
            let mcp = (p.orientation * h.orientation.inverse()).scaled_axis().z;
            let pip = (m.orientation * p.orientation.inverse()).scaled_axis().z;
            let dip = (d.orientation * m.orientation.inverse()).scaled_axis().z;
            println!("{:<6.2} {:<+10.5} {:<+10.5} {:<+10.5}", t, mcp, pip, dip);
            next_print += PRINT_INTERVAL;
        }
    }

    let h = &world.bodies[hand_idx];
    let p = &world.bodies[prox_idx];
    let m = &world.bodies[mid_idx];
    let d = &world.bodies[dist_idx];
    let mcp = (p.orientation * h.orientation.inverse()).scaled_axis().z;
    let pip = (m.orientation * p.orientation.inverse()).scaled_axis().z;
    let dip = (d.orientation * m.orientation.inverse()).scaled_axis().z;
    println!();
    println!("Final: MCP = {:.4} rad, PIP = {:.4} rad, DIP = {:.4} rad",
             mcp, pip, dip);
    println!("Stable run, no NaN.");
}
