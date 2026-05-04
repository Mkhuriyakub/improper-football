// Full bilateral upper-limb demo: torso + both arms (right and left), each
// with a complete shoulder + elbow + wrist + hand chain. 36 bodies,
// 34 joints, 10 tendons, 40 PD controllers.
//
// Both arms are built from a single `add_arm(side)` helper. The Side enum
// applies a ±1 sign multiplier to:
//   - shoulder anchor x-position on the torso
//   - finger and thumb palm-local x-offsets
//   - thumb biomech neutral and z-range (mirror)
//   - schedule targets for axes that anatomically mirror (abd, rot, w_dev)
//
// Axes that don't mirror (flex/ext, elbow flexion, wrist palmar flex) use
// the same world axis for both arms, so the same target produces the same
// anatomical motion bilaterally.

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

// Geometry / mass --------------------------------------------------------

const TORSO_HALF_H: f64 = 0.20;
const TORSO_RADIUS: f64 = 0.15;

const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

const FOREARM_MASS:    f64 = 1.2;
const FOREARM_RADIUS:  f64 = 0.038;
const FOREARM_HALF_H:  f64 = 0.135;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

// Joint elastic anchors --------------------------------------------------

const SHOULDER_JOINT_K: f64 = 100_000.0;
const SHOULDER_JOINT_C: f64 =     200.0;
const ELBOW_JOINT_K:    f64 =  80_000.0;
const ELBOW_JOINT_C:    f64 =     150.0;
const WRIST_JOINT_K:    f64 =  50_000.0;
const WRIST_JOINT_C:    f64 =     100.0;
const FJOINT_K: f64 = 1_000.0;
const C_PALM_PROX:  f64 = 4.0;
const C_PROX_MID:   f64 = 2.5;
const C_MID_DIST:   f64 = 2.0;

// Shoulder biomech -------------------------------------------------------
const FLEX_K: f64 = 3.0;
const ABD_K:  f64 = 3.0;
const ROT_K:  f64 = 1.5;
const FLEX_C: f64 = 0.50;
const ABD_C:  f64 = 0.50;
const ROT_C:  f64 = 0.40;
const FLEX_MIN: f64 = -0.87;
const FLEX_MAX: f64 =  3.14;
const ABD_MIN:  f64 = -0.78;
const ABD_MAX:  f64 =  3.14;
const ROT_MIN:  f64 = -1.40;
const ROT_MAX:  f64 =  1.57;
const SHOULDER_LIMIT_K:   f64 = 100.0;
const SHOULDER_LIMIT_EXP: f64 =   2.0;

// Elbow biomech.
const ELBOW_K:        f64 = 1.5;
const ELBOW_C:        f64 = 0.05;
const ELBOW_NEUTRAL:  f64 = 0.20;
const ELBOW_MIN:      f64 = -0.40;
const ELBOW_MAX:      f64 =  2.20;
const ELBOW_LIMIT_K:  f64 = 500.0;
const ELBOW_OFF_AXIS_K:        f64 = 8.0;
const ELBOW_OFF_AXIS_C:        f64 = 0.60;
const ELBOW_OFF_AXIS_LIM:      f64 = 0.40;
const ELBOW_OFF_AXIS_LIMIT_K:  f64 = 30.0;

// Wrist biomech.
const WRIST_FLEX_K: f64 = 0.4;
const WRIST_DEV_K:  f64 = 0.6;
const WRIST_TWIST_K: f64 = 0.5;
const WRIST_FLEX_C: f64 = 0.05;
const WRIST_DEV_C:  f64 = 0.06;
const WRIST_TWIST_C: f64 = 0.015;
const WRIST_FLEX_MIN: f64 = -1.10;
const WRIST_FLEX_MAX: f64 =  1.30;
const WRIST_DEV_MIN:  f64 = -0.60;
const WRIST_DEV_MAX:  f64 =  0.35;
const WRIST_TWIST_LIM: f64 = 0.30;
const WRIST_LIMIT_K:   f64 = 200.0;
const WRIST_TWIST_LIMIT_K: f64 = 20.0;

// Finger biomech.
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
// Thumb biomech defined for the RIGHT arm. The Left arm uses sign-mirrored
// values built from these in `add_arm`.
const THUMB_MCP_R: JointBiomech = JointBiomech {
    z_min: -2.00, z_max: -0.40,
    damping: 0.0006, stiffness: 0.06, neutral: -1.45,
    limit_stiffness: 0.5,
    off_axis_k: 0.05, off_axis_c: 0.0004,
};
const THUMB_IP_R: JointBiomech = JointBiomech {
    z_min: -0.10, z_max:  1.30,
    damping: 0.00015, stiffness: 0.04, neutral: 0.10,
    limit_stiffness: 0.4,
    off_axis_k: 0.04, off_axis_c: 0.0002,
};

// PD gains.
const SHOULDER_PD_KP: f64 = 120.0;
const SHOULDER_PD_KD: f64 =  14.0;
const ELBOW_PD_KP:    f64 = 100.0;
const ELBOW_PD_KD:    f64 =   3.5;
const WRIST_PD_KP:    f64 =  50.0;
const WRIST_PD_KD:    f64 =   1.0;
const MCP_PD_KP: f64 = 2.0;
const MCP_PD_KD: f64 = 0.004;
const PIP_PD_KP: f64 = 0.5;
const PIP_PD_KD: f64 = 0.001;
const DIP_PD_KP: f64 = 0.2;
const DIP_PD_KD: f64 = 0.0004;

// Tendons.
const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION_FINGER: f64 =
    TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);
// Thumb rest excursion is computed per-arm in add_thumb (sign of neutral
// mirrors with side).

// Convergence factor for finger tilt.
const CONVERGENCE_FACTOR: f64 = 8.0;

// Anatomical positioning.
const TORSO_REST_POS: NaVec3 = NaVec3::new(0.0, 1.40, 0.0);
const SHOULDER_ANCHOR_TORSO_X: f64 = 0.15;
const SHOULDER_ANCHOR_TORSO_Y: f64 = 0.18;

const SCHEDULE_PERIOD: f64 = 96.0;

// Side -------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Side { Right, Left }
impl Side {
    fn sign(&self) -> f64 { match self { Side::Right => 1.0, Side::Left => -1.0 } }
    fn name(&self) -> &'static str { match self { Side::Right => "R", Side::Left => "L" } }
}

// Specs ------------------------------------------------------------------

#[derive(Clone, Copy)]
struct FingerSpec {
    name: &'static str,
    palm_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    mid_mass: f64,  mid_radius: f64,  mid_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
// Right-arm offsets — left arm mirrors via Side::sign().
const INDEX: FingerSpec = FingerSpec {
    name: "index",   palm_offset_x: -0.035,
    prox_mass: 0.008, prox_radius: 0.0080, prox_half_h: 0.022,
    mid_mass:  0.005, mid_radius:  0.0072, mid_half_h:  0.012,
    dist_mass: 0.003, dist_radius: 0.0065, dist_half_h: 0.0085,
};
const MIDDLE: FingerSpec = FingerSpec {
    name: "middle",  palm_offset_x: -0.012,
    prox_mass: 0.010, prox_radius: 0.0090, prox_half_h: 0.026,
    mid_mass:  0.007, mid_radius:  0.0080, mid_half_h:  0.014,
    dist_mass: 0.004, dist_radius: 0.0070, dist_half_h: 0.0085,
};
const RING: FingerSpec = FingerSpec {
    name: "ring",    palm_offset_x:  0.012,
    prox_mass: 0.009, prox_radius: 0.0085, prox_half_h: 0.023,
    mid_mass:  0.006, mid_radius:  0.0075, mid_half_h:  0.0135,
    dist_mass: 0.0035, dist_radius: 0.0067, dist_half_h: 0.0085,
};
const LITTLE: FingerSpec = FingerSpec {
    name: "little",  palm_offset_x:  0.035,
    prox_mass: 0.006, prox_radius: 0.0070, prox_half_h: 0.019,
    mid_mass:  0.004, mid_radius:  0.0062, mid_half_h:  0.011,
    dist_mass: 0.0025, dist_radius: 0.0055, dist_half_h: 0.006,
};
#[derive(Clone, Copy)]
struct ThumbSpec {
    palm_anchor_x: f64,
    palm_anchor_y: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
const THUMB: ThumbSpec = ThumbSpec {
    palm_anchor_x: -0.030, palm_anchor_y: 0.005,
    prox_mass: 0.012, prox_radius: 0.0095, prox_half_h: 0.018,
    dist_mass: 0.006, dist_radius: 0.0080, dist_half_h: 0.0125,
};

// Handles ----------------------------------------------------------------

struct FingerHandles {
    spec: FingerSpec,
    prox_idx: usize, mid_idx: usize, dist_idx: usize,
    mcp_pd: usize, pip_pd: usize, dip_pd: usize,
}
struct ThumbHandles {
    prox_idx: usize, dist_idx: usize,
    mcp_pd: usize, ip_pd: usize,
}
struct ArmHandles {
    side: Side,
    upper_idx: usize,
    forearm_idx: usize,
    palm_idx: usize,
    fingers: Vec<FingerHandles>,
    thumb: ThumbHandles,
    pd_s_flex: usize, pd_s_abd: usize, pd_s_rot: usize,
    pd_elbow: usize,
    pd_w_flex: usize, pd_w_dev: usize,
}

// Helpers ----------------------------------------------------------------

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
    parent_anchor: NaVec3, child_anchor: NaVec3,
    stiffness: f64, damping: f64,
) {
    let mut spring = PositionalSpring::new(parent, child, stiffness);
    spring.local_anchor_a = parent_anchor;
    spring.local_anchor_b = child_anchor;
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(parent, child, damping);
    damp.local_anchor_a = parent_anchor;
    damp.local_anchor_b = child_anchor;
    world.positional_dampers.push(damp);
}

fn add_axis(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    axis_local_a: NaVec3,
    min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64, neutral: f64,
    limit_stiffness: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a,
        min_angle, max_angle, limit_stiffness, SHOULDER_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn add_finger_biomech(
    world: &mut XpbdWorld, parent: usize, child: usize, b: &JointBiomech,
    curl_axis: NaVec3, off1: NaVec3, off2: NaVec3,
) {
    add_axis(world, parent, child, curl_axis,
        b.z_min, b.z_max, b.stiffness, b.damping, b.neutral, b.limit_stiffness);
    add_axis(world, parent, child, off1, -0.10, 0.10, b.off_axis_k, b.off_axis_c, 0.0, 0.3);
    add_axis(world, parent, child, off2, -0.10, 0.10, b.off_axis_k, b.off_axis_c, 0.0, 0.3);
}

fn add_finger(world: &mut XpbdWorld, palm_idx: usize, spec: FingerSpec) -> FingerHandles {
    let mcp_anchor_palm = NaVec3::new(spec.palm_offset_x, -PALM_HALF_H, 0.0);
    let tilt = spec.palm_offset_x * CONVERGENCE_FACTOR;
    let curl_axis = NaVec3::new(1.0, tilt, 0.0).normalize();
    let off_perp1 = NaVec3::new(-tilt, 1.0, 0.0).normalize();
    let off_z = NaVec3::new(0.0, 0.0, 1.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(curl_axis * MCP.neutral));
    prox.position = pos; prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(spec.mid_mass,
        capsule_inertia(spec.mid_mass, spec.mid_radius, spec.mid_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.mid_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(curl_axis * PIP.neutral));
    mid.position = pos; mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(curl_axis * DIP.neutral));
    dist.position = pos; dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm, NaVec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, &MCP, curl_axis, off_perp1, off_z);

    add_elastic_joint(world, prox_idx, mid_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.mid_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, mid_idx, &PIP, curl_axis, off_perp1, off_z);

    add_elastic_joint(world, mid_idx, dist_idx,
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_MID_DIST);
    add_finger_biomech(world, mid_idx, dist_idx, &DIP, curl_axis, off_perp1, off_z);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: curl_axis, moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,  axis_local_a: curl_axis, moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx, axis_local_a: curl_axis, moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION_FINGER,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    let mcp_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, curl_axis, MCP_PD_KP, MCP_PD_KD)
            .with_target(MCP.neutral));
    let pip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, mid_idx, curl_axis, PIP_PD_KP, PIP_PD_KD)
            .with_target(PIP.neutral));
    let dip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(mid_idx, dist_idx, curl_axis, DIP_PD_KP, DIP_PD_KD)
            .with_target(DIP.neutral));

    FingerHandles { spec, prox_idx, mid_idx, dist_idx, mcp_pd, pip_pd, dip_pd }
}

fn add_thumb(
    world: &mut XpbdWorld, palm_idx: usize,
    spec: ThumbSpec,
    mcp_biomech: &JointBiomech, ip_biomech: &JointBiomech,
) -> ThumbHandles {
    let mcp_anchor_palm = NaVec3::new(spec.palm_anchor_x, spec.palm_anchor_y, 0.0);
    let thumb_curl = NaVec3::z();
    let thumb_off1 = NaVec3::x();
    let thumb_off2 = NaVec3::y();

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[palm_idx], mcp_anchor_palm,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, mcp_biomech.neutral)));
    prox.position = pos; prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, ip_biomech.neutral)));
    dist.position = pos; dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, palm_idx, prox_idx,
        mcp_anchor_palm, NaVec3::new(0.0, spec.prox_half_h, 0.0),
        FJOINT_K, C_PALM_PROX);
    add_finger_biomech(world, palm_idx, prox_idx, mcp_biomech, thumb_curl, thumb_off1, thumb_off2);

    add_elastic_joint(world, prox_idx, dist_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0, spec.dist_half_h, 0.0),
        FJOINT_K, C_PROX_MID);
    add_finger_biomech(world, prox_idx, dist_idx, ip_biomech, thumb_curl, thumb_off1, thumb_off2);

    let rest_excursion = TENDON_MOMENT_ARM * (mcp_biomech.neutral + ip_biomech.neutral);
    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: palm_idx, body_b: prox_idx, axis_local_a: thumb_curl, moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: dist_idx, axis_local_a: thumb_curl, moment_arm: TENDON_MOMENT_ARM },
        ],
        rest_excursion,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    let mcp_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, thumb_curl, MCP_PD_KP, MCP_PD_KD)
            .with_target(mcp_biomech.neutral));
    let ip_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, dist_idx, thumb_curl, PIP_PD_KP, PIP_PD_KD)
            .with_target(ip_biomech.neutral));

    ThumbHandles { prox_idx, dist_idx, mcp_pd, ip_pd }
}

/// Build a complete arm (upper arm + forearm + palm + 4 fingers + thumb)
/// off the torso on the given side. Right arm goes at +x, left at -x; the
/// hand structures mirror about the y-axis so the thumb sits on the
/// medial (body-side) of each hand and the four fingers spread laterally.
fn add_arm(world: &mut XpbdWorld, torso_idx: usize, side: Side) -> ArmHandles {
    let s = side.sign();

    // 1. Upper arm
    let shoulder_anchor = NaVec3::new(s * SHOULDER_ANCHOR_TORSO_X, SHOULDER_ANCHOR_TORSO_Y, 0.0);
    let mut upper = RigidBody::new(UPPER_ARM_MASS,
        capsule_inertia(UPPER_ARM_MASS, UPPER_ARM_RADIUS, UPPER_ARM_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[torso_idx],
        shoulder_anchor,
        NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        UnitQuaternion::identity());
    upper.position = pos; upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(world, torso_idx, upper_idx,
        shoulder_anchor, NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);

    let s_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let s_abd_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let s_rot_axis  = NaVec3::new( 0.0, 1.0, 0.0);
    // Asymmetric ranges (e.g. abduction 180° vs adduction 45°) must mirror
    // for the left arm because its targets are sign-flipped in apply step.
    // Without this, the left arm's "lateral abduction" target = -3.00 would
    // hit the unmirrored ABD_MIN = -0.78 limit and stall at ~45° instead of
    // reaching 180°. Flex range stays the same (axis convention isn't
    // sign-flipped per arm — same anatomical direction works for both).
    let (abd_min_a, abd_max_a) = if s > 0.0 { (ABD_MIN, ABD_MAX) } else { (-ABD_MAX, -ABD_MIN) };
    let (rot_min_a, rot_max_a) = if s > 0.0 { (ROT_MIN, ROT_MAX) } else { (-ROT_MAX, -ROT_MIN) };
    add_axis(world, torso_idx, upper_idx, s_flex_axis, FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C, 0.0, SHOULDER_LIMIT_K);
    add_axis(world, torso_idx, upper_idx, s_abd_axis,  abd_min_a, abd_max_a, ABD_K, ABD_C, 0.0, SHOULDER_LIMIT_K);
    add_axis(world, torso_idx, upper_idx, s_rot_axis,  rot_min_a, rot_max_a, ROT_K, ROT_C, 0.0, SHOULDER_LIMIT_K);

    let pd_s_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, s_flex_axis, SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));
    let pd_s_abd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, s_abd_axis,  SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));
    let pd_s_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, s_rot_axis,  SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));

    // 2. Forearm
    let mut forearm = RigidBody::new(FOREARM_MASS,
        capsule_inertia(FOREARM_MASS, FOREARM_RADIUS, FOREARM_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[upper_idx],
        NaVec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        NaVec3::new(0.0, FOREARM_HALF_H, 0.0),
        UnitQuaternion::identity());
    forearm.position = pos; forearm.orientation = orient;
    let forearm_idx = world.add_body(forearm);

    add_elastic_joint(world, upper_idx, forearm_idx,
        NaVec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        NaVec3::new(0.0, FOREARM_HALF_H, 0.0),
        ELBOW_JOINT_K, ELBOW_JOINT_C);

    let elbow_axis = NaVec3::new(-1.0, 0.0, 0.0);
    add_axis(world, upper_idx, forearm_idx, elbow_axis,
        ELBOW_MIN, ELBOW_MAX, ELBOW_K, ELBOW_C, ELBOW_NEUTRAL, ELBOW_LIMIT_K);
    add_axis(world, upper_idx, forearm_idx, NaVec3::z(),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0, ELBOW_OFF_AXIS_LIMIT_K);
    add_axis(world, upper_idx, forearm_idx, NaVec3::y(),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0, ELBOW_OFF_AXIS_LIMIT_K);

    let pd_elbow = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(upper_idx, forearm_idx, elbow_axis, ELBOW_PD_KP, ELBOW_PD_KD).with_target(ELBOW_NEUTRAL));

    // 3. Palm
    let mut palm = RigidBody::new(PALM_MASS,
        capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[forearm_idx],
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0),
        UnitQuaternion::identity());
    palm.position = pos; palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(world, forearm_idx, palm_idx,
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0),
        WRIST_JOINT_K, WRIST_JOINT_C);

    let w_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let w_dev_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let w_twist_axis = NaVec3::new(0.0, 1.0, 0.0);
    // Wrist dev range mirrors per arm for the same reason as shoulder abd —
    // its target is sign-flipped on the left side. (Wrist flex doesn't flip,
    // so its range stays the same. Twist range is already symmetric.)
    let (dev_min_a, dev_max_a) = if s > 0.0 { (WRIST_DEV_MIN, WRIST_DEV_MAX) } else { (-WRIST_DEV_MAX, -WRIST_DEV_MIN) };
    add_axis(world, forearm_idx, palm_idx, w_flex_axis,
        WRIST_FLEX_MIN, WRIST_FLEX_MAX, WRIST_FLEX_K, WRIST_FLEX_C, 0.0, WRIST_LIMIT_K);
    add_axis(world, forearm_idx, palm_idx, w_dev_axis,
        dev_min_a, dev_max_a, WRIST_DEV_K, WRIST_DEV_C, 0.0, WRIST_LIMIT_K);
    add_axis(world, forearm_idx, palm_idx, w_twist_axis,
        -WRIST_TWIST_LIM, WRIST_TWIST_LIM, WRIST_TWIST_K, WRIST_TWIST_C, 0.0, WRIST_TWIST_LIMIT_K);

    let pd_w_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, w_flex_axis, WRIST_PD_KP, WRIST_PD_KD).with_target(0.0));
    let pd_w_dev = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, w_dev_axis, WRIST_PD_KP, WRIST_PD_KD).with_target(0.0));

    // 4. Fingers + thumb (mirrored offsets via side sign).
    let mirror_finger = |spec: FingerSpec| FingerSpec {
        palm_offset_x: s * spec.palm_offset_x,
        ..spec
    };
    let fingers = vec![
        add_finger(world, palm_idx, mirror_finger(INDEX)),
        add_finger(world, palm_idx, mirror_finger(MIDDLE)),
        add_finger(world, palm_idx, mirror_finger(RING)),
        add_finger(world, palm_idx, mirror_finger(LITTLE)),
    ];

    let thumb_spec = ThumbSpec {
        palm_anchor_x: s * THUMB.palm_anchor_x,
        ..THUMB
    };
    // Thumb biomech mirrors: neutral and z-range flip for the left side.
    // (For s = -1, the canonical right-arm range [-2.00, -0.40] becomes
    // [+0.40, +2.00].)
    let mcp_biomech = JointBiomech {
        z_min: if s > 0.0 { THUMB_MCP_R.z_min } else { -THUMB_MCP_R.z_max },
        z_max: if s > 0.0 { THUMB_MCP_R.z_max } else { -THUMB_MCP_R.z_min },
        neutral: s * THUMB_MCP_R.neutral,
        damping: THUMB_MCP_R.damping,
        stiffness: THUMB_MCP_R.stiffness,
        limit_stiffness: THUMB_MCP_R.limit_stiffness,
        off_axis_k: THUMB_MCP_R.off_axis_k,
        off_axis_c: THUMB_MCP_R.off_axis_c,
    };
    let ip_biomech = JointBiomech {
        z_min: if s > 0.0 { THUMB_IP_R.z_min } else { -THUMB_IP_R.z_max },
        z_max: if s > 0.0 { THUMB_IP_R.z_max } else { -THUMB_IP_R.z_min },
        neutral: s * THUMB_IP_R.neutral,
        damping: THUMB_IP_R.damping,
        stiffness: THUMB_IP_R.stiffness,
        limit_stiffness: THUMB_IP_R.limit_stiffness,
        off_axis_k: THUMB_IP_R.off_axis_k,
        off_axis_c: THUMB_IP_R.off_axis_c,
    };
    let thumb = add_thumb(world, palm_idx, thumb_spec, &mcp_biomech, &ip_biomech);

    ArmHandles {
        side, upper_idx, forearm_idx, palm_idx, fingers, thumb,
        pd_s_flex, pd_s_abd, pd_s_rot,
        pd_elbow,
        pd_w_flex, pd_w_dev,
    }
}

// Schedule ---------------------------------------------------------------

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

fn ramp(t: f64, duration: f64, a: f64, b: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    a + (b - a) * 0.5 * (1.0 - (std::f64::consts::PI * phase).cos())
}

/// Anatomical targets for ONE arm. Positive abd = lateral abduction in body
/// terms (mirrored across arms via per-arm sign multiplier in apply step).
struct Targets {
    s_flex: f64, s_abd: f64, s_rot: f64,
    elbow:  f64,
    w_flex: f64, w_dev: f64,
    finger_curl: f64,
    thumb_curl:  f64,
    /// If true, this is an asymmetric phase — the targets apply to the right
    /// arm, and the left arm uses `Targets_left` from a parallel call.
    phase: &'static str,
}

fn empty_targets() -> Targets {
    Targets {
        s_flex: 0.0, s_abd: 0.0, s_rot: 0.0,
        elbow: 0.0,
        w_flex: 0.0, w_dev: 0.0,
        finger_curl: 0.0,
        thumb_curl: 0.0,
        phase: "",
    }
}

/// Returns (right_arm_targets, left_arm_targets, phase_label). Both arms get
/// independent target structs so asymmetric phases can drive them separately.
fn schedule(t: f64) -> (Targets, Targets, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);

    // Phase A (0–48 s): bilateral joint isolation — both arms do same anatomical motion.
    if t < 6.0 {
        let v = half_cosine(t, 6.0, 3.00);
        let r = Targets { s_flex: v, phase: "bilateral forward flexion", ..empty_targets() };
        let l = Targets { s_flex: v, phase: "bilateral forward flexion", ..empty_targets() };
        return (r, l, "bilateral forward flexion");
    } else if t < 10.0 {
        let v = half_cosine(t - 6.0, 4.0, -0.80);
        let r = Targets { s_flex: v, phase: "bilateral extension", ..empty_targets() };
        let l = Targets { s_flex: v, phase: "bilateral extension", ..empty_targets() };
        return (r, l, "bilateral extension");
    } else if t < 16.0 {
        let v = half_cosine(t - 10.0, 6.0, 3.00);
        let r = Targets { s_abd: v, ..empty_targets() };
        let l = Targets { s_abd: v, ..empty_targets() };
        return (r, l, "bilateral lateral abduction");
    } else if t < 20.0 {
        // Cross-body adduction: each arm crosses toward midline. Right arm
        // adducts in the negative biomech direction (toward -x); left arm
        // adducts in positive (toward +x). Same anatomical command, opposite
        // signs — handled by the per-arm sign multiplier in apply().
        let v = half_cosine(t - 16.0, 4.0, -0.70);
        let r = Targets { s_abd: v, ..empty_targets() };
        let l = Targets { s_abd: v, ..empty_targets() };
        return (r, l, "bilateral cross-body adduction");
    } else if t < 28.0 {
        // Hold abducted + elbow flexed; rotation cycles.
        let local = t - 20.0;
        let pose_ramp  = (local / 1.5).clamp(0.0, 1.0);
        let pose_unramp = ((28.0 - t) / 1.5).clamp(0.0, 1.0);
        let pose_envelope = pose_ramp.min(pose_unramp);
        let abd = 1.57 * pose_envelope;
        let elbow_target = 1.57 * pose_envelope;
        let rot_t = local - 1.5;
        let rot_phase = (rot_t / 5.0).clamp(0.0, 1.0);
        let rot = if rot_phase < 0.33 {
            ramp(rot_phase / 0.33, 1.0, 0.0, -1.30)
        } else if rot_phase < 0.66 {
            ramp((rot_phase - 0.33) / 0.33, 1.0, -1.30, 1.40)
        } else {
            ramp((rot_phase - 0.66) / 0.34, 1.0, 1.40, 0.0)
        };
        let r = Targets { s_abd: abd, s_rot: rot * pose_envelope, elbow: elbow_target, ..empty_targets() };
        let l = Targets { s_abd: abd, s_rot: rot * pose_envelope, elbow: elbow_target, ..empty_targets() };
        return (r, l, "bilateral rotation @ 90° abd");
    } else if t < 34.0 {
        let v = half_cosine(t - 28.0, 6.0, 2.20);
        let r = Targets { elbow: v, ..empty_targets() };
        let l = Targets { elbow: v, ..empty_targets() };
        return (r, l, "bilateral elbow flexion");
    } else if t < 38.0 {
        let v = half_cosine(t - 34.0, 4.0, 1.20);
        let r = Targets { w_flex: v, ..empty_targets() };
        let l = Targets { w_flex: v, ..empty_targets() };
        return (r, l, "bilateral wrist flex/ext");
    } else if t < 42.0 {
        let v = half_cosine(t - 38.0, 4.0, 0.30);
        let r = Targets { w_dev: v, ..empty_targets() };
        let l = Targets { w_dev: v, ..empty_targets() };
        return (r, l, "bilateral wrist deviation");
    } else if t < 48.0 {
        let fc = half_cosine(t - 42.0, 6.0, 1.0);
        let r = Targets { finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        let l = Targets { finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        return (r, l, "bilateral hand curl");

    // Phase B (48–72 s): symmetric functional combos.
    } else if t < 56.0 {
        // Bilateral reach forward + grab + release.
        let local = t - 48.0;
        let s_flex = if local < 4.0 {
            ramp(local / 4.0, 1.0, 0.0, 1.57)
        } else {
            ramp((local - 4.0) / 4.0, 1.0, 1.57, 0.0)
        };
        let fc = if local < 1.5 { 0.0 }
            else if local < 3.0 { ramp((local - 1.5) / 1.5, 1.0, 0.0, -0.50) }
            else if local < 4.5 { ramp((local - 3.0) / 1.5, 1.0, -0.50, 0.80) }
            else if local < 6.5 { ramp((local - 4.5) / 2.0, 1.0, 0.80, 0.0) }
            else { 0.0 };
        let r = Targets { s_flex, finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        let l = Targets { s_flex, finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        return (r, l, "bilateral reach + grab");
    } else if t < 66.0 {
        // Bilateral throwing cock pose.
        let local = t - 56.0;
        let env = (local / 3.0).min(1.0).min((10.0 - local) / 3.0).max(0.0);
        let r = Targets {
            s_abd: 1.57 * env,
            s_rot: 1.40 * env,
            elbow: 1.57 * env,
            w_flex: -0.80 * env,
            finger_curl: -0.70 * env,
            thumb_curl: -0.70 * env,
            ..empty_targets()
        };
        let l = Targets {
            s_abd: 1.57 * env,
            s_rot: 1.40 * env,
            elbow: 1.57 * env,
            w_flex: -0.80 * env,
            finger_curl: -0.70 * env,
            thumb_curl: -0.70 * env,
            ..empty_targets()
        };
        return (r, l, "bilateral throwing cock pose");
    } else if t < 72.0 {
        // Bilateral throw release.
        let local = t - 66.0;
        let env = (local / 1.5).min(1.0);
        let unwind = ((local - 1.5) / 4.5).clamp(0.0, 1.0);
        let s_flex = ramp(unwind, 1.0, ramp(env, 1.0, 0.0, 1.30), 0.0);
        let elbow  = ramp(unwind, 1.0, ramp(env, 1.0, 0.0, 0.30), 0.0);
        let w_flex = ramp(unwind, 1.0, ramp(env, 1.0, 0.0, 0.80), 0.0);
        let fc     = ramp(unwind, 1.0, ramp(env, 1.0, -0.20, 0.80), 0.0);
        let r = Targets { s_flex, elbow, w_flex, finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        let l = Targets { s_flex, elbow, w_flex, finger_curl: fc, thumb_curl: fc, ..empty_targets() };
        return (r, l, "bilateral throw release");

    // Phase C (72–88 s): asymmetric coordination — arms doing different things.
    } else if t < 80.0 {
        // Opposite circumduction: right arm circles forward, left arm
        // circles backward (like crawl-stroke arm motion).
        let local = (t - 72.0).clamp(0.0, 8.0);
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let phase = std::f64::consts::TAU * local / 8.0;  // one lap each
        let r = Targets {
            s_flex: (1.20 + 1.00 * phase.sin()) * envelope,
            s_abd:  (0.60 + 0.60 * phase.cos().max(0.0)) * envelope,
            elbow:  (0.40 + 0.30 * phase.sin().abs()) * envelope,
            ..empty_targets()
        };
        // Left arm: opposite phase (sin shifted by π).
        let l = Targets {
            s_flex: (1.20 + 1.00 * (phase + std::f64::consts::PI).sin()) * envelope,
            s_abd:  (0.60 + 0.60 * (phase + std::f64::consts::PI).cos().max(0.0)) * envelope,
            elbow:  (0.40 + 0.30 * (phase + std::f64::consts::PI).sin().abs()) * envelope,
            ..empty_targets()
        };
        return (r, l, "asymmetric circumduction");
    } else if t < 88.0 {
        // Alternating reach: right reaches forward then left reaches forward,
        // like a running arm swing. Each arm does one full reach-and-back.
        let local = t - 80.0;  // 0–8 s
        let r_phase = (local / 4.0).clamp(0.0, 1.0);
        let l_phase = ((local - 2.0) / 4.0).clamp(0.0, 1.0);
        let r_flex = if r_phase < 0.5 {
            ramp(r_phase / 0.5, 1.0, 0.0, 1.20)
        } else {
            ramp((r_phase - 0.5) / 0.5, 1.0, 1.20, -0.50)  // back-swing
        };
        let l_flex = if l_phase < 0.5 {
            ramp(l_phase / 0.5, 1.0, 0.0, 1.20)
        } else {
            ramp((l_phase - 0.5) / 0.5, 1.0, 1.20, -0.50)
        };
        let r = Targets { s_flex: r_flex, elbow: 0.30 + 0.50 * r_flex.abs(), ..empty_targets() };
        let l = Targets { s_flex: l_flex, elbow: 0.30 + 0.50 * l_flex.abs(), ..empty_targets() };
        return (r, l, "alternating reach (running swing)");

    // Phase D (88–96 s): cool-down to neutral.
    } else {
        let local = t - 88.0;
        let unwind = (local / 8.0).clamp(0.0, 1.0);
        let zero = ramp(unwind, 1.0, 0.0, 0.0);
        let r = Targets { s_flex: zero, ..empty_targets() };
        let l = Targets { s_flex: zero, ..empty_targets() };
        return (r, l, "cooldown to neutral");
    }
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

/// Side-aware thumb target: signs flip across arms.
fn thumb_targets(curl: f64, side: Side) -> (f64, f64) {
    let s = side.sign();
    // Canonical right-arm thumb targets, then mirror for left.
    let neutral_mcp = -1.45 * s;
    let flex_mcp_delta = 0.95;
    let ext_mcp_delta  = 0.50;
    let mcp = if curl >= 0.0 {
        neutral_mcp + curl * flex_mcp_delta * s
    } else {
        neutral_mcp + curl * ext_mcp_delta * s
    };
    let neutral_ip = 0.10 * s;
    let flex_ip_delta = 1.10;
    let ext_ip_delta  = 0.20;
    let ip = if curl >= 0.0 {
        neutral_ip + curl * flex_ip_delta * s
    } else {
        neutral_ip + curl * ext_ip_delta * s
    };
    (mcp, ip)
}

fn apply_targets_to_arm(world: &mut XpbdWorld, t: &Targets, arm: &ArmHandles) {
    let s = arm.side.sign();
    world.angular_pd_controllers[arm.pd_s_flex].target_angle = t.s_flex;       // same axis both arms
    world.angular_pd_controllers[arm.pd_s_abd ].target_angle = t.s_abd * s;    // mirror sign
    world.angular_pd_controllers[arm.pd_s_rot ].target_angle = t.s_rot * s;    // mirror sign
    world.angular_pd_controllers[arm.pd_elbow ].target_angle = t.elbow + ELBOW_NEUTRAL;
    world.angular_pd_controllers[arm.pd_w_flex].target_angle = t.w_flex;       // same
    world.angular_pd_controllers[arm.pd_w_dev ].target_angle = t.w_dev * s;    // mirror sign

    let (mcp_t, pip_t, dip_t) = finger_targets(t.finger_curl);
    let finger_pds: Vec<(usize, usize, usize)> = arm.fingers.iter()
        .map(|f| (f.mcp_pd, f.pip_pd, f.dip_pd))
        .collect();
    for (mcp, pip, dip) in finger_pds {
        world.angular_pd_controllers[mcp].target_angle = mcp_t;
        world.angular_pd_controllers[pip].target_angle = pip_t;
        world.angular_pd_controllers[dip].target_angle = dip_t;
    }

    let (t_mcp_t, t_ip_t) = thumb_targets(t.thumb_curl, arm.side);
    world.angular_pd_controllers[arm.thumb.mcp_pd].target_angle = t_mcp_t;
    world.angular_pd_controllers[arm.thumb.ip_pd ].target_angle = t_ip_t;
}

// Bevy state -------------------------------------------------------------

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    torso_idx: usize,
    arms: Vec<ArmHandles>,  // [right, left]
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct Readout;

fn render_arm_bodies(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    world: &XpbdWorld,
    arm: &ArmHandles,
) {
    // Color palette per side: warm-orange right, cool-blue left.
    let (limb_a, limb_b, palm_c) = match arm.side {
        Side::Right => (
            Color::srgb(0.85, 0.65, 0.50),
            Color::srgb(0.88, 0.68, 0.53),
            Color::srgb(0.92, 0.72, 0.58),
        ),
        Side::Left => (
            Color::srgb(0.55, 0.65, 0.85),
            Color::srgb(0.58, 0.68, 0.88),
            Color::srgb(0.62, 0.72, 0.92),
        ),
    };
    let upper_mesh   = meshes.add(Capsule3d::new(UPPER_ARM_RADIUS as f32, (2.0 * UPPER_ARM_HALF_H) as f32));
    let forearm_mesh = meshes.add(Capsule3d::new(FOREARM_RADIUS as f32, (2.0 * FOREARM_HALF_H) as f32));
    let palm_mesh    = meshes.add(Capsule3d::new(PALM_RADIUS as f32, (2.0 * PALM_HALF_H) as f32));
    commands.spawn((PbrBundle {
        mesh: upper_mesh, material: materials.add(StandardMaterial { base_color: limb_a, ..default() }),
        transform: rigid_body_to_transform(&world.bodies[arm.upper_idx]),
        ..default()
    }, BodyRef(arm.upper_idx)));
    commands.spawn((PbrBundle {
        mesh: forearm_mesh, material: materials.add(StandardMaterial { base_color: limb_b, ..default() }),
        transform: rigid_body_to_transform(&world.bodies[arm.forearm_idx]),
        ..default()
    }, BodyRef(arm.forearm_idx)));
    commands.spawn((PbrBundle {
        mesh: palm_mesh, material: materials.add(StandardMaterial { base_color: palm_c, ..default() }),
        transform: rigid_body_to_transform(&world.bodies[arm.palm_idx]),
        ..default()
    }, BodyRef(arm.palm_idx)));

    // Finger palette — slightly lighter than palm.
    let finger_shades: [Color; 3] = match arm.side {
        Side::Right => [
            Color::srgb(0.95, 0.75, 0.60),
            Color::srgb(0.97, 0.78, 0.62),
            Color::srgb(0.99, 0.81, 0.64),
        ],
        Side::Left => [
            Color::srgb(0.65, 0.75, 0.95),
            Color::srgb(0.68, 0.78, 0.97),
            Color::srgb(0.71, 0.81, 0.99),
        ],
    };
    for f in &arm.fingers {
        let prox_mesh = meshes.add(Capsule3d::new(f.spec.prox_radius as f32, (2.0 * f.spec.prox_half_h) as f32));
        let mid_mesh  = meshes.add(Capsule3d::new(f.spec.mid_radius  as f32, (2.0 * f.spec.mid_half_h)  as f32));
        let dist_mesh = meshes.add(Capsule3d::new(f.spec.dist_radius as f32, (2.0 * f.spec.dist_half_h) as f32));
        commands.spawn((PbrBundle {
            mesh: prox_mesh,
            material: materials.add(StandardMaterial { base_color: finger_shades[0], ..default() }),
            transform: rigid_body_to_transform(&world.bodies[f.prox_idx]),
            ..default()
        }, BodyRef(f.prox_idx)));
        commands.spawn((PbrBundle {
            mesh: mid_mesh,
            material: materials.add(StandardMaterial { base_color: finger_shades[1], ..default() }),
            transform: rigid_body_to_transform(&world.bodies[f.mid_idx]),
            ..default()
        }, BodyRef(f.mid_idx)));
        commands.spawn((PbrBundle {
            mesh: dist_mesh,
            material: materials.add(StandardMaterial { base_color: finger_shades[2], ..default() }),
            transform: rigid_body_to_transform(&world.bodies[f.dist_idx]),
            ..default()
        }, BodyRef(f.dist_idx)));
    }
    let thumb_shade = match arm.side {
        Side::Right => [Color::srgb(1.00, 0.80, 0.62), Color::srgb(1.00, 0.83, 0.64)],
        Side::Left  => [Color::srgb(0.62, 0.80, 1.00), Color::srgb(0.64, 0.83, 1.00)],
    };
    let t_prox_mesh = meshes.add(Capsule3d::new(THUMB.prox_radius as f32, (2.0 * THUMB.prox_half_h) as f32));
    let t_dist_mesh = meshes.add(Capsule3d::new(THUMB.dist_radius as f32, (2.0 * THUMB.dist_half_h) as f32));
    commands.spawn((PbrBundle {
        mesh: t_prox_mesh,
        material: materials.add(StandardMaterial { base_color: thumb_shade[0], ..default() }),
        transform: rigid_body_to_transform(&world.bodies[arm.thumb.prox_idx]),
        ..default()
    }, BodyRef(arm.thumb.prox_idx)));
    commands.spawn((PbrBundle {
        mesh: t_dist_mesh,
        material: materials.add(StandardMaterial { base_color: thumb_shade[1], ..default() }),
        transform: rigid_body_to_transform(&world.bodies[arm.thumb.dist_idx]),
        ..default()
    }, BodyRef(arm.thumb.dist_idx)));
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    // Torso (pinned, center).
    let mut torso = RigidBody::new(0.0, capsule_inertia(50.0, TORSO_RADIUS, TORSO_HALF_H));
    torso.position = TORSO_REST_POS;
    torso.orientation = UnitQuaternion::identity();
    let torso_idx = world.add_body(torso);

    // Both arms.
    let arms = vec![
        add_arm(&mut world, torso_idx, Side::Right),
        add_arm(&mut world, torso_idx, Side::Left),
    ];

    // Visuals: torso plus both arms.
    let torso_mesh = meshes.add(Capsule3d::new(TORSO_RADIUS as f32, (2.0 * TORSO_HALF_H) as f32));
    let torso_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    commands.spawn((PbrBundle {
        mesh: torso_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[torso_idx]),
        ..default()
    }, BodyRef(torso_idx)));
    for arm in &arms {
        render_arm_bodies(&mut commands, &mut meshes, &mut materials, &world, arm);
    }

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: centered, pulled back enough to capture both arms at full
    // overhead lateral abduction (T-pose) and full forward reach.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 1.55, 2.60)
            .looking_at(Vec3::new(0.0, 1.30, 0.0), Vec3::Y),
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

    commands.insert_resource(PhysicsWorld { world, torso_idx, arms, sim_time: 0.0 });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (r_targets, l_targets, _phase) = schedule(phys.sim_time);
    // Need to detach arms from phys to mutate world while iterating.
    let arms_count = phys.arms.len();
    for i in 0..arms_count {
        let t = if phys.arms[i].side == Side::Right { &r_targets } else { &l_targets };
        // Borrow trick: read arm fields needed by apply_targets_to_arm
        // separately because both need mutable world access.
        let arm_clone_handles = ArmHandlesView::from(&phys.arms[i]);
        apply_targets_view(&mut phys.world, t, &arm_clone_handles);
    }
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

/// Lightweight view of an ArmHandles for target application — just the PD
/// indices, side, and finger/thumb PD lookups. Avoids borrowing whole arm
/// while mutating world.
struct ArmHandlesView {
    side: Side,
    pd_s_flex: usize, pd_s_abd: usize, pd_s_rot: usize,
    pd_elbow: usize,
    pd_w_flex: usize, pd_w_dev: usize,
    finger_pds: Vec<(usize, usize, usize)>,
    thumb_mcp_pd: usize, thumb_ip_pd: usize,
}
impl ArmHandlesView {
    fn from(arm: &ArmHandles) -> Self {
        Self {
            side: arm.side,
            pd_s_flex: arm.pd_s_flex, pd_s_abd: arm.pd_s_abd, pd_s_rot: arm.pd_s_rot,
            pd_elbow: arm.pd_elbow,
            pd_w_flex: arm.pd_w_flex, pd_w_dev: arm.pd_w_dev,
            finger_pds: arm.fingers.iter().map(|f| (f.mcp_pd, f.pip_pd, f.dip_pd)).collect(),
            thumb_mcp_pd: arm.thumb.mcp_pd, thumb_ip_pd: arm.thumb.ip_pd,
        }
    }
}

fn apply_targets_view(world: &mut XpbdWorld, t: &Targets, arm: &ArmHandlesView) {
    let s = arm.side.sign();
    world.angular_pd_controllers[arm.pd_s_flex].target_angle = t.s_flex;
    world.angular_pd_controllers[arm.pd_s_abd ].target_angle = t.s_abd * s;
    world.angular_pd_controllers[arm.pd_s_rot ].target_angle = t.s_rot * s;
    world.angular_pd_controllers[arm.pd_elbow ].target_angle = t.elbow + ELBOW_NEUTRAL;
    world.angular_pd_controllers[arm.pd_w_flex].target_angle = t.w_flex;
    world.angular_pd_controllers[arm.pd_w_dev ].target_angle = t.w_dev * s;
    let (mcp_t, pip_t, dip_t) = finger_targets(t.finger_curl);
    for &(mcp, pip, dip) in &arm.finger_pds {
        world.angular_pd_controllers[mcp].target_angle = mcp_t;
        world.angular_pd_controllers[pip].target_angle = pip_t;
        world.angular_pd_controllers[dip].target_angle = dip_t;
    }
    let (t_mcp_t, t_ip_t) = thumb_targets(t.thumb_curl, arm.side);
    world.angular_pd_controllers[arm.thumb_mcp_pd].target_angle = t_mcp_t;
    world.angular_pd_controllers[arm.thumb_ip_pd ].target_angle = t_ip_t;
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
    let (_, _, phase) = schedule(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\nbodies: {}\nPDs: {}",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            phys.world.bodies.len(),
            phys.world.angular_pd_controllers.len(),
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bilateral arm demo (full ROM)".into(),
                resolution: (1280.0, 800.0).into(),
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
