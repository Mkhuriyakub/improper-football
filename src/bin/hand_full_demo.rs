// Visual demo of the full hand: wrist + four flat fingers + thumb.
//
// Same body plan as `hand_full.rs`. PD controllers drive the wrist through
// anatomical ROM and all five digits through curl/release/extend cycles.
// The thumb's curl uses a different mapping than the four flat fingers
// because its CMC's resting neutral is π/2 (radial extension) rather than
// a small angle, and its joint count is 3 (CMC + MCP + IP) instead of 3
// (MCP + PIP + DIP).

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::tendon::{Tendon, TendonSpan};
use physics::math::{Vec3 as NaVec3, Mat3};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

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

const WRIST_PD_KP: f64 = 50.0;
const WRIST_PD_KD: f64 =  1.0;

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
// 2-bone thumb (prox + dist phalanges, no metacarpal body). MCP rest pose
// extends prox radially (neutral ≈ 1.45 rad ≈ 83°); flexion drives the
// angle DOWN toward 0 to swing the prox CW toward across-palm direction.
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

const MCP_PD_KP: f64 = 2.0;
const MCP_PD_KD: f64 = 0.004;
const PIP_PD_KP: f64 = 0.5;
const PIP_PD_KD: f64 = 0.001;
const DIP_PD_KP: f64 = 0.2;
const DIP_PD_KD: f64 = 0.0004;

const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION_FINGER: f64 =
    TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);
const TENDON_REST_EXCURSION_THUMB: f64 =
    TENDON_MOMENT_ARM * (1.45 + (-0.10));

const FOREARM_REST_POS: NaVec3 = NaVec3::new(0.0, 1.0, 0.0);

const WRIST_SCHEDULE_PERIOD: f64 = 36.0;
const FINGER_PERIOD: f64 = 12.0;

const Z_FLEX_PEAK:  f64 =  1.20;
const Z_EXT_PEAK:   f64 = -1.00;
const X_ULNAR_PEAK: f64 =  0.55;
const X_RADIAL_PEAK:f64 = -0.30;
const CIRCLE_AMP_Z: f64 =  0.90;
const CIRCLE_AMP_X: f64 =  0.40;

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct ThumbSpec {
    palm_anchor_x: f64,
    palm_anchor_y: f64,
    /// Proximal phalanx (the visible base segment of the thumb).
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

struct FingerHandles {
    spec: FingerSpec,
    prox_idx: usize,
    mid_idx: usize,
    dist_idx: usize,
    mcp_pd: usize,
    pip_pd: usize,
    dip_pd: usize,
}

struct ThumbHandles {
    prox_idx: usize,
    dist_idx: usize,
    mcp_pd: usize,
    ip_pd: usize,
}

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
    parent_distal_local: NaVec3,
    child_proximal_local: NaVec3,
    rel_rot: UnitQuaternion<f64>,
) -> (NaVec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint_world = parent.position + parent.orientation * parent_distal_local;
    let child_pos = joint_world - child_orient * child_proximal_local;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    parent_distal_local: NaVec3, child_proximal_local: NaVec3,
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
        parent, child, NaVec3::z(),
        b.z_min, b.z_max, b.limit_stiffness, 2.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::z(), b.damping));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::z(), b.stiffness, b.neutral));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::x(), -0.10, 0.10, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::x(), b.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::x(), b.off_axis_c));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::y(), -0.10, 0.10, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::y(), b.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::y(), b.off_axis_c));
}

fn add_wrist_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::z(), WRIST_Z_MIN, WRIST_Z_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::z(), WRIST_Z_C));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::z(), WRIST_Z_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::x(), WRIST_X_MIN, WRIST_X_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::x(), WRIST_X_C));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::x(), WRIST_X_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::y(), -WRIST_Y_LIM, WRIST_Y_LIM, WRIST_Y_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::y(), WRIST_Y_C));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::y(), WRIST_Y_K, 0.0));
}

fn add_finger(
    world: &mut XpbdWorld,
    palm_idx: usize,
    spec: FingerSpec,
) -> FingerHandles {
    let mcp_anchor_palm = NaVec3::new(spec.palm_offset_x, -PALM_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, MCP.neutral)));
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(spec.mid_mass,
        capsule_inertia(spec.mid_mass, spec.mid_radius, spec.mid_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.mid_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, PIP.neutral)));
    mid.position = pos;
    mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, DIP.neutral)));
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, &MCP);

    add_elastic_joint(world, prox_idx, mid_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.mid_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, mid_idx, &PIP);

    add_elastic_joint(world, mid_idx, dist_idx,
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_MID_DIST);
    add_finger_biomech(world, mid_idx, dist_idx, &DIP);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,  axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION_FINGER,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    let mcp_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, NaVec3::z(), MCP_PD_KP, MCP_PD_KD)
            .with_target(MCP.neutral));
    let pip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, mid_idx, NaVec3::z(), PIP_PD_KP, PIP_PD_KD)
            .with_target(PIP.neutral));
    let dip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(mid_idx, dist_idx, NaVec3::z(), DIP_PD_KP, DIP_PD_KD)
            .with_target(DIP.neutral));

    FingerHandles { spec, prox_idx, mid_idx, dist_idx, mcp_pd, pip_pd, dip_pd }
}

fn add_thumb(
    world: &mut XpbdWorld,
    palm_idx: usize,
    spec: ThumbSpec,
) -> ThumbHandles {
    let mcp_anchor_palm = NaVec3::new(spec.palm_anchor_x, spec.palm_anchor_y, 0.0);

    // Prox phalanx — rest pose extends radially (initial MCP angle is the
    // big rotation; flexion drives that angle DOWN from there).
    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, THUMB_MCP.neutral)));
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, THUMB_IP.neutral)));
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, &THUMB_MCP);

    add_elastic_joint(world, prox_idx, dist_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, dist_idx, &THUMB_IP);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: dist_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION_THUMB,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    let mcp_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, NaVec3::z(), MCP_PD_KP, MCP_PD_KD)
            .with_target(THUMB_MCP.neutral));
    let ip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, dist_idx, NaVec3::z(), PIP_PD_KP, PIP_PD_KD)
            .with_target(THUMB_IP.neutral));

    ThumbHandles { prox_idx, dist_idx, mcp_pd, ip_pd }
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

fn wrist_targets_at(t: f64) -> (f64, f64, &'static str) {
    let t = t.rem_euclid(WRIST_SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, Z_FLEX_PEAK), 0.0, "palmar flexion")
    } else if t < 12.0 {
        (half_cosine(t - 6.0, 6.0, Z_EXT_PEAK), 0.0, "dorsi extension")
    } else if t < 18.0 {
        (0.0, half_cosine(t - 12.0, 6.0, X_ULNAR_PEAK), "ulnar deviation")
    } else if t < 24.0 {
        (0.0, half_cosine(t - 18.0, 6.0, X_RADIAL_PEAK), "radial deviation")
    } else {
        let local = (t - 24.0).clamp(0.0, 12.0);
        let envelope = (std::f64::consts::PI * local / 12.0).sin();
        let phase = std::f64::consts::TAU * 2.0 * local / 12.0;
        (CIRCLE_AMP_Z * envelope * phase.sin(),
         CIRCLE_AMP_X * envelope * phase.cos(),
         "circumduction")
    }
}

fn finger_curl(t: f64) -> f64 {
    let phase = (t / FINGER_PERIOD).rem_euclid(1.0);
    (std::f64::consts::TAU * phase).sin()
}

fn finger_targets(curl: f64) -> (f64, f64, f64) {
    if curl >= 0.0 {
        (MCP.neutral + curl * (1.40 - MCP.neutral),
         PIP.neutral + curl * (1.60 - PIP.neutral),
         DIP.neutral + curl * (1.10 - DIP.neutral))
    } else {
        (MCP.neutral + curl * MCP.neutral,
         PIP.neutral + curl * PIP.neutral,
         DIP.neutral + curl * DIP.neutral)
    }
}

/// Thumb curl→target mapping. The thumb's resting MCP is large (≈1.45 rad,
/// putting the prox phalanx in the radial direction). Flexion (curl > 0)
/// drives MCP and IP DOWN (CW about palm-z) toward the across-palm
/// direction; extension (curl < 0) raises both back up to the maximum
/// radial-out pose.
fn thumb_targets(curl: f64) -> (f64, f64) {
    let mcp = if curl >= 0.0 {
        // Flex: from 1.45 down to 0.50 (full across-palm).
        THUMB_MCP.neutral - curl * (THUMB_MCP.neutral - 0.50)
    } else {
        // Ext: from 1.45 up to 1.95 (further radial).
        THUMB_MCP.neutral + (-curl) * (1.95 - THUMB_MCP.neutral)
    };
    let ip = if curl >= 0.0 {
        // Flex: from -0.10 down to -1.20 (distal phalanx tucked).
        THUMB_IP.neutral - curl * (-0.10 - (-1.20))  // = -0.10 - curl * 1.10
    } else {
        // Ext: from -0.10 up to +0.05 (nearly straight).
        THUMB_IP.neutral + (-curl) * (0.05 - THUMB_IP.neutral)
    };
    (mcp, ip)
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    forearm_idx: usize,
    palm_idx: usize,
    fingers: Vec<FingerHandles>,
    thumb: ThumbHandles,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut forearm = RigidBody::new(0.0, capsule_inertia(1.5, FOREARM_RADIUS, FOREARM_HALF_H));
    forearm.position = FOREARM_REST_POS;
    forearm.orientation = UnitQuaternion::identity();
    let forearm_idx = world.add_body(forearm);

    let mut palm = RigidBody::new(PALM_MASS, capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[forearm_idx],
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0),
        UnitQuaternion::identity());
    palm.position = pos;
    palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(&mut world, forearm_idx, palm_idx,
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0),
        WRIST_JOINT_K, WRIST_JOINT_C);
    add_wrist_biomech(&mut world, forearm_idx, palm_idx);

    // Wrist PD controllers (indices 0, 1).
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, NaVec3::z(), WRIST_PD_KP, WRIST_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, NaVec3::x(), WRIST_PD_KP, WRIST_PD_KD)
            .with_target(0.0));

    let fingers = vec![
        add_finger(&mut world, palm_idx, INDEX),
        add_finger(&mut world, palm_idx, MIDDLE),
        add_finger(&mut world, palm_idx, RING),
        add_finger(&mut world, palm_idx, LITTLE),
    ];
    let thumb = add_thumb(&mut world, palm_idx, THUMB);

    let forearm_mesh = meshes.add(Capsule3d::new(FOREARM_RADIUS as f32, (2.0 * FOREARM_HALF_H) as f32));
    let palm_mesh    = meshes.add(Capsule3d::new(PALM_RADIUS    as f32, (2.0 * PALM_HALF_H)    as f32));

    let forearm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.58, 0.45), ..default() });
    let palm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.72, 0.58), ..default() });

    commands.spawn((PbrBundle {
        mesh: forearm_mesh, material: forearm_mat,
        transform: rigid_body_to_transform(&world.bodies[forearm_idx]),
        ..default()
    }, BodyRef(forearm_idx)));
    commands.spawn((PbrBundle {
        mesh: palm_mesh, material: palm_mat,
        transform: rigid_body_to_transform(&world.bodies[palm_idx]),
        ..default()
    }, BodyRef(palm_idx)));

    let finger_palettes: [[Color; 3]; 4] = [
        [Color::srgb(0.95, 0.75, 0.60), Color::srgb(0.97, 0.78, 0.62), Color::srgb(0.99, 0.81, 0.64)],
        [Color::srgb(0.93, 0.73, 0.60), Color::srgb(0.95, 0.76, 0.62), Color::srgb(0.97, 0.79, 0.64)],
        [Color::srgb(0.91, 0.71, 0.60), Color::srgb(0.93, 0.74, 0.62), Color::srgb(0.95, 0.77, 0.64)],
        [Color::srgb(0.88, 0.69, 0.60), Color::srgb(0.90, 0.72, 0.62), Color::srgb(0.92, 0.75, 0.64)],
    ];

    for (i, f) in fingers.iter().enumerate() {
        let prox_mesh = meshes.add(Capsule3d::new(f.spec.prox_radius as f32, (2.0 * f.spec.prox_half_h) as f32));
        let mid_mesh  = meshes.add(Capsule3d::new(f.spec.mid_radius  as f32, (2.0 * f.spec.mid_half_h)  as f32));
        let dist_mesh = meshes.add(Capsule3d::new(f.spec.dist_radius as f32, (2.0 * f.spec.dist_half_h) as f32));
        let prox_mat = materials.add(StandardMaterial { base_color: finger_palettes[i][0], ..default() });
        let mid_mat  = materials.add(StandardMaterial { base_color: finger_palettes[i][1], ..default() });
        let dist_mat = materials.add(StandardMaterial { base_color: finger_palettes[i][2], ..default() });
        commands.spawn((PbrBundle {
            mesh: prox_mesh, material: prox_mat,
            transform: rigid_body_to_transform(&world.bodies[f.prox_idx]),
            ..default()
        }, BodyRef(f.prox_idx)));
        commands.spawn((PbrBundle {
            mesh: mid_mesh, material: mid_mat,
            transform: rigid_body_to_transform(&world.bodies[f.mid_idx]),
            ..default()
        }, BodyRef(f.mid_idx)));
        commands.spawn((PbrBundle {
            mesh: dist_mesh, material: dist_mat,
            transform: rigid_body_to_transform(&world.bodies[f.dist_idx]),
            ..default()
        }, BodyRef(f.dist_idx)));
    }

    // Thumb in distinct slightly-warmer colors. Two segments only (no
    // separate metacarpal — its anatomical role is absorbed into the palm
    // capsule and the wider MCP range).
    let thumb_palette = [
        Color::srgb(1.00, 0.80, 0.62),
        Color::srgb(1.00, 0.83, 0.64),
    ];
    let t_prox_mesh = meshes.add(Capsule3d::new(THUMB.prox_radius as f32, (2.0 * THUMB.prox_half_h) as f32));
    let t_dist_mesh = meshes.add(Capsule3d::new(THUMB.dist_radius as f32, (2.0 * THUMB.dist_half_h) as f32));
    commands.spawn((PbrBundle {
        mesh: t_prox_mesh,
        material: materials.add(StandardMaterial { base_color: thumb_palette[0], ..default() }),
        transform: rigid_body_to_transform(&world.bodies[thumb.prox_idx]),
        ..default()
    }, BodyRef(thumb.prox_idx)));
    commands.spawn((PbrBundle {
        mesh: t_dist_mesh,
        material: materials.add(StandardMaterial { base_color: thumb_palette[1], ..default() }),
        transform: rigid_body_to_transform(&world.bodies[thumb.dist_idx]),
        ..default()
    }, BodyRef(thumb.dist_idx)));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(2.0, 2.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.85, 0.55)
            .looking_at(Vec3::new(0.0, 0.78, 0.0), Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: true, ..default() },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE, brightness: 220.0 });

    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 18.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0),
        ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, forearm_idx, palm_idx, fingers, thumb, sim_time: 0.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (z_t, x_t, _) = wrist_targets_at(phys.sim_time);
    let curl = finger_curl(phys.sim_time);
    let (mcp_t, pip_t, dip_t) = finger_targets(curl);
    let (t_mcp_t, t_ip_t) = thumb_targets(curl);

    phys.world.angular_pd_controllers[0].target_angle = z_t;
    phys.world.angular_pd_controllers[1].target_angle = x_t;
    let finger_handles: Vec<(usize, usize, usize)> = phys.fingers.iter()
        .map(|f| (f.mcp_pd, f.pip_pd, f.dip_pd))
        .collect();
    for (mcp, pip, dip) in finger_handles {
        phys.world.angular_pd_controllers[mcp].target_angle = mcp_t;
        phys.world.angular_pd_controllers[pip].target_angle = pip_t;
        phys.world.angular_pd_controllers[dip].target_angle = dip_t;
    }
    let t_mcp_pd = phys.thumb.mcp_pd;
    let t_ip_pd = phys.thumb.ip_pd;
    phys.world.angular_pd_controllers[t_mcp_pd].target_angle = t_mcp_t;
    phys.world.angular_pd_controllers[t_ip_pd].target_angle  = t_ip_t;

    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut query: Query<(&BodyRef, &mut Transform)>,
) {
    for (body_ref, mut transform) in &mut query {
        let body = &phys.world.bodies[body_ref.0];
        *transform = rigid_body_to_transform(body);
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let f  = &phys.world.bodies[phys.forearm_idx];
    let pl = &phys.world.bodies[phys.palm_idx];
    let wrist = (pl.orientation * f.orientation.inverse()).scaled_axis();
    let (z_t, x_t, phase) = wrist_targets_at(phys.sim_time);
    let curl = finger_curl(phys.sim_time);

    let mut digit_lines = String::new();
    for fh in &phys.fingers {
        let p = &phys.world.bodies[fh.prox_idx];
        let m = &phys.world.bodies[fh.mid_idx];
        let d = &phys.world.bodies[fh.dist_idx];
        let mcp = (p.orientation * pl.orientation.inverse()).scaled_axis().z;
        let pip = (m.orientation * p.orientation.inverse()).scaled_axis().z;
        let dip = (d.orientation * m.orientation.inverse()).scaled_axis().z;
        digit_lines.push_str(&format!(
            "{:>7}: MCP {:+.2}  PIP {:+.2}  DIP {:+.2}\n",
            fh.spec.name, mcp, pip, dip,
        ));
    }
    {
        let pr = &phys.world.bodies[phys.thumb.prox_idx];
        let di = &phys.world.bodies[phys.thumb.dist_idx];
        let tmcp = (pr.orientation * pl.orientation.inverse()).scaled_axis().z;
        let tip = (di.orientation * pr.orientation.inverse()).scaled_axis().z;
        digit_lines.push_str(&format!(
            "{:>7}: MCP {:+.2}   IP {:+.2}\n",
            "thumb", tmcp, tip,
        ));
    }

    for mut text in &mut query {
        text.sections[0].value = format!(
            "wrist phase: {}\nfinger curl: {:+.2}\nt: {:.2} s\n\nwrist Z (flex):  target {:+.2}  →  {:+.2}\nwrist X (dev):   target {:+.2}  →  {:+.2}\n\n{}",
            phase, curl, phys.sim_time,
            z_t, wrist.z, x_t, wrist.x,
            digit_lines,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Hand demo: full hand (4 fingers + thumb)".into(),
                resolution: (1024.0, 768.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, update_readout))
        .run();
}
