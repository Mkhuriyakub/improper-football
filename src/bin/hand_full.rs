// Step in the elastic-joint hand build: wrist + four flat fingers + thumb.
//
// Bodies (13 total): forearm + palm + 4 × (prox, mid, dist) finger phalanges
// + 3 × thumb segments (metacarpal, prox, dist).
//
// The thumb differs from the four flat fingers in three structural ways:
//   1. Three bones (meta + prox + dist) vs three phalanges. The thumb's
//      metacarpal is its own body because the CMC joint moves it as a unit.
//   2. CMC joint sits at the proximal-radial corner of the palm, not the
//      distal end like the MCPs of the flat fingers.
//   3. The metacarpal's natural rest pose extends *radially* (world +x in
//      our setup), so the CMC's Z-axis spring neutral is π/2 rather than
//      a small angle. Anatomically this represents the resting-tone radial
//      abduction of the thumb.
//
// CMC is modeled as 1-DOF flexion (Z) only for this step. Real CMC is a
// 2-DOF saddle (flex + abd), but the simplified version validates the
// thumb's place in the hand before adding the second DOF.
//
// Validation: wrist cocked, all four fingers curled, thumb at deep-flex
// initial pose (CMC = 0.5, MCP = 0.6, IP = 0.8). Expected: stable
// equilibrium, every chain settles independently.

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
const PRINT_INTERVAL: f64 = 0.10;

const FOREARM_HALF_H: f64 = 0.135;
const FOREARM_RADIUS: f64 = 0.040;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

const WRIST_JOINT_K: f64 = 50_000.0;
const WRIST_JOINT_C: f64 =    100.0;
const WRIST_Z_K:     f64 = 0.4;
const WRIST_X_K:     f64 = 0.6;
const WRIST_Y_K:     f64 = 0.5;
const WRIST_Z_C:     f64 = 0.05;
const WRIST_X_C:     f64 = 0.06;
const WRIST_Y_C:     f64 = 0.015;
const WRIST_Z_MIN:   f64 = -1.10;
const WRIST_Z_MAX:   f64 =  1.30;
const WRIST_X_MIN:   f64 = -0.35;
const WRIST_X_MAX:   f64 =  0.60;
const WRIST_Y_LIM:   f64 =  0.30;
const WRIST_LIMIT_K: f64 = 200.0;
const WRIST_Y_LIMIT_K:f64 = 20.0;
const WRIST_LIMIT_EXP:f64 =  2.0;

const FJOINT_K: f64 = 1_000.0;
const C_PALM_PROX:  f64 = 4.0;
const C_PROX_MID:   f64 = 2.5;
const C_MID_DIST:   f64 = 2.0;

struct JointBiomech {
    z_min: f64, z_max: f64,
    damping: f64, stiffness: f64, neutral: f64,
    limit_stiffness: f64,
    off_axis_k: f64,
    off_axis_c: f64,
}
const MCP: JointBiomech = JointBiomech {
    z_min: -0.52, z_max: 1.57,
    damping: 0.0004, stiffness: 0.05, neutral: 0.30,
    limit_stiffness: 0.5,
    off_axis_k: 0.05, off_axis_c: 0.0004,
};
const PIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 2.09,
    damping: 0.00013, stiffness: 0.03, neutral: 0.40,
    limit_stiffness: 0.4,
    off_axis_k: 0.04, off_axis_c: 0.0002,
};
const DIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 1.40,
    damping: 0.00006, stiffness: 0.02, neutral: 0.30,
    limit_stiffness: 0.3,
    off_axis_k: 0.03, off_axis_c: 0.0001,
};

// Thumb joints — 2-bone variant (prox + dist phalanges, no separate
// metacarpal body, in line with the visible 2-segment look of a real thumb).
// MCP neutral is large (~1.45 rad ≈ 83°) because the prox phalanx in its
// rest pose extends radially; flexion drives the angle DOWN toward 0 to
// swing the prox phalanx clockwise toward the across-palm direction.
// IP behaves like a finger DIP — small neutral, flex still decreases it.
const THUMB_MCP: JointBiomech = JointBiomech {
    z_min:  0.40, z_max:  2.00,
    damping: 0.0006, stiffness: 0.06, neutral: 1.45,
    limit_stiffness: 0.5,
    off_axis_k: 0.05, off_axis_c: 0.0004,
};
const THUMB_IP: JointBiomech = JointBiomech {
    z_min: -1.30, z_max:  0.10,
    damping: 0.00015, stiffness: 0.04, neutral: -0.10,
    limit_stiffness: 0.4,
    off_axis_k: 0.04, off_axis_c: 0.0002,
};

const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION_FINGER: f64 =
    TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);
// Thumb (FPL) tendon: rest at the thumb's resting tone (MCP = 1.45,
// IP = -0.10). Sum at rest = 1.35.
const TENDON_REST_EXCURSION_THUMB: f64 =
    TENDON_MOMENT_ARM * (1.45 + (-0.10));

const INITIAL_WRIST_Z: f64 = 0.40;
const INITIAL_WRIST_X: f64 = 0.30;
const INITIAL_MCP:     f64 = 0.80;
const INITIAL_PIP:     f64 = 1.00;
const INITIAL_DIP:     f64 = 0.50;
// Thumb starts in deep flexion (across-palm grip pose). With the 2-bone
// thumb and large MCP neutral, "deep flex" means LOW MCP and LOW IP.
const INITIAL_THUMB_MCP: f64 = 0.60;  // mostly across-palm
const INITIAL_THUMB_IP:  f64 = -1.10; // distal phalanx tucked further

const FOREARM_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);

struct FingerSpec {
    name: &'static str,
    palm_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    mid_mass: f64,  mid_radius: f64,  mid_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}

const INDEX: FingerSpec = FingerSpec {
    name: "index",
    palm_offset_x:  0.035,
    prox_mass: 0.008,  prox_radius: 0.0080, prox_half_h: 0.022,
    mid_mass:  0.005,  mid_radius:  0.0072, mid_half_h:  0.012,
    dist_mass: 0.003,  dist_radius: 0.0065, dist_half_h: 0.0085,
};
const MIDDLE: FingerSpec = FingerSpec {
    name: "middle",
    palm_offset_x:  0.012,
    prox_mass: 0.010,  prox_radius: 0.0090, prox_half_h: 0.026,
    mid_mass:  0.007,  mid_radius:  0.0080, mid_half_h:  0.014,
    dist_mass: 0.004,  dist_radius: 0.0070, dist_half_h: 0.0085,
};
const RING: FingerSpec = FingerSpec {
    name: "ring",
    palm_offset_x: -0.012,
    prox_mass: 0.009,  prox_radius: 0.0085, prox_half_h: 0.023,
    mid_mass:  0.006,  mid_radius:  0.0075, mid_half_h:  0.0135,
    dist_mass: 0.0035, dist_radius: 0.0067, dist_half_h: 0.0085,
};
const LITTLE: FingerSpec = FingerSpec {
    name: "little",
    palm_offset_x: -0.035,
    prox_mass: 0.006,  prox_radius: 0.0070, prox_half_h: 0.019,
    mid_mass:  0.004,  mid_radius:  0.0062, mid_half_h:  0.011,
    dist_mass: 0.0025, dist_radius: 0.0055, dist_half_h: 0.006,
};

struct ThumbSpec {
    /// MCP anchor on palm. Proximal-radial corner — anatomically this is
    /// where the thumb metacarpal's distal end sits when the metacarpal is
    /// embedded in the thenar eminence.
    palm_anchor_x: f64,
    palm_anchor_y: f64,
    /// Proximal phalanx (visible base segment of the thumb).
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    /// Distal phalanx (the tip segment).
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}

const THUMB: ThumbSpec = ThumbSpec {
    palm_anchor_x: 0.030,
    palm_anchor_y: 0.005,
    prox_mass: 0.012, prox_radius: 0.0095, prox_half_h: 0.018,
    dist_mass: 0.006, dist_radius: 0.0080, dist_half_h: 0.0125,
};

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
    parent: usize, child: usize,
    parent_distal_local: Vec3, child_proximal_local: Vec3,
    stiffness: f64, damping: f64,
) {
    let mut spring = PositionalSpring::new(parent, child, stiffness);
    spring.local_anchor_a = parent_distal_local;
    spring.local_anchor_b = child_proximal_local;
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(parent, child, damping);
    damp.local_anchor_a = parent_distal_local;
    damp.local_anchor_b = child_proximal_local;
    world.positional_dampers.push(damp);
}

fn add_finger_biomech(world: &mut XpbdWorld, parent: usize, child: usize, b: &JointBiomech) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        b.z_min, b.z_max, b.limit_stiffness, 2.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), b.damping));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), b.stiffness, b.neutral));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(), -0.10, 0.10, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), b.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), b.off_axis_c));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(), -0.10, 0.10, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), b.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), b.off_axis_c));
}

fn add_wrist_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(), WRIST_Z_MIN, WRIST_Z_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), WRIST_Z_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), WRIST_Z_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(), WRIST_X_MIN, WRIST_X_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), WRIST_X_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), WRIST_X_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(), -WRIST_Y_LIM, WRIST_Y_LIM, WRIST_Y_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), WRIST_Y_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), WRIST_Y_K, 0.0));
}

fn add_finger(
    world: &mut XpbdWorld,
    palm_idx: usize,
    spec: &FingerSpec,
    initial: (f64, f64, f64),
) -> (usize, usize, usize) {
    let (init_mcp, init_pip, init_dip) = initial;
    let mcp_anchor_palm = Vec3::new(spec.palm_offset_x, -PALM_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let mcp_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, init_mcp));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        Vec3::new(0.0, spec.prox_half_h, 0.0), mcp_rel);
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(spec.mid_mass,
        capsule_inertia(spec.mid_mass, spec.mid_radius, spec.mid_half_h));
    let pip_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, init_pip));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        Vec3::new(0.0, -spec.prox_half_h, 0.0),
        Vec3::new(0.0, spec.mid_half_h, 0.0),
        pip_rel);
    mid.position = pos;
    mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let dip_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, init_dip));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        Vec3::new(0.0, -spec.mid_half_h, 0.0),
        Vec3::new(0.0, spec.dist_half_h, 0.0),
        dip_rel);
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm,
        Vec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, &MCP);

    add_elastic_joint(world, prox_idx, mid_idx,
        Vec3::new(0.0, -spec.prox_half_h, 0.0),
        Vec3::new(0.0, spec.mid_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, mid_idx, &PIP);

    add_elastic_joint(world, mid_idx, dist_idx,
        Vec3::new(0.0, -spec.mid_half_h, 0.0),
        Vec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_MID_DIST);
    add_finger_biomech(world, mid_idx, dist_idx, &DIP);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,  axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION_FINGER,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    (prox_idx, mid_idx, dist_idx)
}

/// Build the 2-bone thumb. Returns (prox_idx, dist_idx).
fn add_thumb(
    world: &mut XpbdWorld,
    palm_idx: usize,
    spec: &ThumbSpec,
    initial: (f64, f64),
) -> (usize, usize) {
    let (init_mcp, init_ip) = initial;
    let mcp_anchor_palm = Vec3::new(spec.palm_anchor_x, spec.palm_anchor_y, 0.0);

    // Prox phalanx — rest pose extends radially with slight distal tilt.
    // Initial MCP angle is the rotation from palm's z-axis default, with the
    // neutral being a large angle (≈1.45 rad ≈ 83°) to put the prox in the
    // radial direction; flexion drives it CW from there toward the
    // across-palm direction.
    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let mcp_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, init_mcp));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        Vec3::new(0.0, spec.prox_half_h, 0.0),
        mcp_rel);
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let ip_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, init_ip));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        Vec3::new(0.0, -spec.prox_half_h, 0.0),
        Vec3::new(0.0, spec.dist_half_h, 0.0),
        ip_rel);
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm,
        Vec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, &THUMB_MCP);

    add_elastic_joint(world, prox_idx, dist_idx,
        Vec3::new(0.0, -spec.prox_half_h, 0.0),
        Vec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, dist_idx, &THUMB_IP);

    // Flexor pollicis longus, spans MCP and IP.
    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: dist_idx, axis_local_a: Vec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION_THUMB,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    (prox_idx, dist_idx)
}

fn main() {
    let mut world = XpbdWorld::new();

    // Forearm vertical.
    let mut forearm = RigidBody::new(0.0, capsule_inertia(1.5, FOREARM_RADIUS, FOREARM_HALF_H));
    forearm.position = FOREARM_REST_POS;
    forearm.orientation = UnitQuaternion::identity();
    let forearm_idx = world.add_body(forearm);

    // Palm with initial wrist deflection.
    let mut palm = RigidBody::new(PALM_MASS, capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H));
    let wrist_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_WRIST_Z))
                  * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_WRIST_X, 0.0, 0.0));
    let (pos, orient) = place_child(
        &world.bodies[forearm_idx],
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0,  PALM_HALF_H, 0.0),
        wrist_rel);
    palm.position = pos;
    palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(&mut world, forearm_idx, palm_idx,
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0,  PALM_HALF_H, 0.0),
        WRIST_JOINT_K, WRIST_JOINT_C);
    add_wrist_biomech(&mut world, forearm_idx, palm_idx);

    // Four flat fingers.
    let initial_curl = (INITIAL_MCP, INITIAL_PIP, INITIAL_DIP);
    let (i_prox, i_mid, i_dist) = add_finger(&mut world, palm_idx, &INDEX,  initial_curl);
    let (m_prox, m_mid, m_dist) = add_finger(&mut world, palm_idx, &MIDDLE, initial_curl);
    let (r_prox, r_mid, r_dist) = add_finger(&mut world, palm_idx, &RING,   initial_curl);
    let (l_prox, l_mid, l_dist) = add_finger(&mut world, palm_idx, &LITTLE, initial_curl);

    // Thumb at deep flex (across-palm grip pose).
    let initial_thumb = (INITIAL_THUMB_MCP, INITIAL_THUMB_IP);
    let (t_prox, t_dist) = add_thumb(&mut world, palm_idx, &THUMB, initial_thumb);

    println!("HAND FULL (4 fingers + 2-bone thumb)  ({:.1}s, dt={:.5}s)", SIM_DURATION, DT);
    println!("Bodies: forearm + palm + 4 fingers × 3 + thumb × 2");
    println!("Initial: wrist_z={}, wrist_x={}, fingers MCP/PIP/DIP={}/{}/{}, thumb MCP/IP={}/{}",
             INITIAL_WRIST_Z, INITIAL_WRIST_X,
             INITIAL_MCP, INITIAL_PIP, INITIAL_DIP,
             INITIAL_THUMB_MCP, INITIAL_THUMB_IP);
    println!();
    println!("{:<6} {:<8} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} | {:<8} {:<8}",
             "t", "wr_z",
             "i_mcp", "i_pip", "i_dip",
             "m_mcp", "m_pip", "m_dip",
             "r_mcp", "r_pip", "r_dip",
             "l_mcp", "l_pip", "l_dip",
             "t_mcp", "t_ip");
    println!("{}", "-".repeat(155));

    let f_idx = |a: usize, b: usize, world: &XpbdWorld| -> f64 {
        (world.bodies[b].orientation * world.bodies[a].orientation.inverse())
            .scaled_axis().z
    };

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let wrist_z = (world.bodies[palm_idx].orientation
                * world.bodies[forearm_idx].orientation.inverse()).scaled_axis().z;
            println!("{:<6.2} {:<+8.3} | {:<+8.3} {:<+8.3} {:<+8.3} | {:<+8.3} {:<+8.3} {:<+8.3} | {:<+8.3} {:<+8.3} {:<+8.3} | {:<+8.3} {:<+8.3} {:<+8.3} | {:<+8.3} {:<+8.3}",
                     t, wrist_z,
                     f_idx(palm_idx, i_prox, &world), f_idx(i_prox, i_mid, &world), f_idx(i_mid, i_dist, &world),
                     f_idx(palm_idx, m_prox, &world), f_idx(m_prox, m_mid, &world), f_idx(m_mid, m_dist, &world),
                     f_idx(palm_idx, r_prox, &world), f_idx(r_prox, r_mid, &world), f_idx(r_mid, r_dist, &world),
                     f_idx(palm_idx, l_prox, &world), f_idx(l_prox, l_mid, &world), f_idx(l_mid, l_dist, &world),
                     f_idx(palm_idx, t_prox, &world), f_idx(t_prox, t_dist, &world));
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    println!("Final equilibria:");
    let print_finger = |name: &str, p: usize, m: usize, d: usize, palm: usize, world: &XpbdWorld| {
        let mcp = f_idx(palm, p, world);
        let pip = f_idx(p, m, world);
        let dip = f_idx(m, d, world);
        println!("  {:<7}: {:+.4}, {:+.4}, {:+.4}  ({:+.1}°, {:+.1}°, {:+.1}°)",
                 name, mcp, pip, dip,
                 mcp.to_degrees(), pip.to_degrees(), dip.to_degrees());
    };
    print_finger("index",  i_prox, i_mid, i_dist, palm_idx, &world);
    print_finger("middle", m_prox, m_mid, m_dist, palm_idx, &world);
    print_finger("ring",   r_prox, r_mid, r_dist, palm_idx, &world);
    print_finger("little", l_prox, l_mid, l_dist, palm_idx, &world);
    let tmcp = f_idx(palm_idx, t_prox, &world);
    let tip = f_idx(t_prox, t_dist, &world);
    println!("  {:<7}: {:+.4}, {:+.4}  ({:+.1}°, {:+.1}°)  [MCP, IP]",
             "thumb", tmcp, tip,
             tmcp.to_degrees(), tip.to_degrees());
}
