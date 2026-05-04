// Full leg assembly: pelvis (pinned) → hip (3-DOF) → thigh → knee (1-DOF)
// → shin → ankle (2-DOF) → foot → 5 toes (14 phalanges).
//
// Sequential ROM-only schedule. Each joint sweeps through its anatomical
// range independently — no coordinated multi-joint poses yet. The point of
// this demo is to verify the chain is wired correctly: parent → child
// orientations propagate, the foot's visual offset still pivots around the
// ankle (which now moves with the shin which moves with the thigh), and
// each PD drives its joint with the rest of the chain hanging passively.
//
// Schedule (48 s, looping):
//
//   0–6 s     hip flexion          0 → +1.50 → 0
//   6–10 s    hip extension        0 → -0.30 → 0
//   10–16 s   hip abd/add          -0.45 → +0.70 → 0
//   16–22 s   hip int/ext rot      -0.55 → +0.80 → 0
//   22–28 s   knee flex            0 → +2.20 → 0
//   28–32 s   ankle PF             0 → +0.85 → 0
//   32–36 s   ankle DF             0 → -0.40 → 0
//   36–42 s   ankle inv/ev         -0.25 → +0.50 → 0
//   42–48 s   toe curl/extend      ±0.85 cycle

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
use physics::math::{Vec3 as NaVec3, Mat3};
use improper_football::render_util::{rigid_body_to_transform, nalgebra_vec3_to_bevy, nalgebra_quat_to_bevy};

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ─── Pelvis (pinned) ──────────────────────────────────────────────────
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;
const PELVIS_REST_POS: NaVec3 = NaVec3::new(0.0, 1.00, 0.0);
const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

// ─── Thigh ────────────────────────────────────────────────────────────
const THIGH_MASS:   f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =     400.0;

// Hip biomech (matches hip_compliant.rs).
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K:   f64 = 120.0;
const HIP_LIMIT_EXP: f64 =   2.0;

// Hip PD — KP/KD bumped from the hip-only demo (500/30) to handle the
// full leg's ~12 kg + 2.8 kg·m² inertia at the hip. ω_n ≈ 17, ζ ≈ 0.85.
const HIP_PD_KP: f64 = 800.0;
const HIP_PD_KD: f64 =  80.0;

// ─── Shin (lower leg) ─────────────────────────────────────────────────
const SHIN_MASS:   f64 = 3.5;
const SHIN_RADIUS: f64 = 0.050;
const SHIN_HALF_H: f64 = 0.210;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;

const KNEE_K:        f64 =  5.0;
const KNEE_C:        f64 =  1.5;
const KNEE_NEUTRAL:  f64 =  0.10;
const KNEE_MIN:      f64 = -0.087;
const KNEE_MAX:      f64 =  2.44;
const KNEE_LIMIT_K:  f64 =  80.0;
const KNEE_LIMIT_EXP:f64 =   2.0;

const KNEE_OFF_AXIS_K:    f64 = 12.0;
const KNEE_OFF_AXIS_C:    f64 =  0.80;
const KNEE_OFF_AXIS_LIM:  f64 =  0.30;
const KNEE_OFF_AXIS_LIMIT_K: f64 = 50.0;

// Knee PD — drives shin+foot+toes (~4.5 kg, I ≈ 0.36). ω_n=28 with KP=300.
const KNEE_PD_KP: f64 = 300.0;
const KNEE_PD_KD: f64 =  20.0;

// ─── Foot ─────────────────────────────────────────────────────────────
// Sized so the rendered capsule (body + rounded ends) ≈ 25 cm long, 6.4 cm
// wide — matches an adult foot from heel to MTPJ, with toes adding the
// remaining ~5 cm.
const FOOT_MASS:   f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

const PF_K:    f64 = 4.0;   const PF_C:    f64 = 1.2;
const INV_K:   f64 = 3.5;   const INV_C:   f64 = 1.0;
const TWIST_K: f64 = 8.0;   const TWIST_C: f64 = 0.5;
const PF_MIN:    f64 = -0.44;  const PF_MAX:    f64 =  0.87;
const INV_MIN:   f64 = -0.26;  const INV_MAX:   f64 =  0.52;
const TWIST_LIM: f64 =  0.10;
const ANKLE_LIMIT_K:    f64 = 80.0;
const TWIST_LIMIT_K:    f64 = 200.0;
const ANKLE_LIMIT_EXP:  f64 =  2.0;

const ANKLE_PD_KP: f64 = 80.0;
const ANKLE_PD_KD: f64 =  6.0;

// ─── Toes (matches ankle_demo) ────────────────────────────────────────
const TJOINT_K: f64 = 800.0;
const C_FOOT_TOE: f64 = 1.5;
const C_TOE_TOE:  f64 = 0.8;

struct ToeJoint {
    flex_min: f64, flex_max: f64,
    spring: f64, damping: f64, neutral: f64,
    limit_k: f64,
    off_axis_k: f64, off_axis_c: f64,
}
const HALLUX_MTPJ: ToeJoint = ToeJoint {
    flex_min: -1.40, flex_max: 0.50, spring: 0.04, damping: 0.0008, neutral: 0.0,
    limit_k: 0.5, off_axis_k: 0.05, off_axis_c: 0.0005,
};
const HALLUX_IPJ: ToeJoint = ToeJoint {
    flex_min: -0.10, flex_max: 1.40, spring: 0.03, damping: 0.0004, neutral: 0.0,
    limit_k: 0.4, off_axis_k: 0.04, off_axis_c: 0.0003,
};
const TOE_MTPJ: ToeJoint = ToeJoint {
    flex_min: -0.70, flex_max: 0.40, spring: 0.025, damping: 0.0005, neutral: 0.0,
    limit_k: 0.4, off_axis_k: 0.04, off_axis_c: 0.0003,
};
const TOE_PIPJ: ToeJoint = ToeJoint {
    flex_min: -0.05, flex_max: 1.20, spring: 0.02, damping: 0.0003, neutral: 0.0,
    limit_k: 0.3, off_axis_k: 0.03, off_axis_c: 0.0002,
};
const TOE_DIPJ: ToeJoint = ToeJoint {
    flex_min: -0.05, flex_max: 1.00, spring: 0.015, damping: 0.0002, neutral: 0.0,
    limit_k: 0.25, off_axis_k: 0.025, off_axis_c: 0.0002,
};

const HALLUX_MTPJ_KP: f64 = 0.6;   const HALLUX_MTPJ_KD: f64 = 0.0;
const HALLUX_IPJ_KP:  f64 = 0.3;   const HALLUX_IPJ_KD:  f64 = 0.0;
const TOE_MTPJ_KP:    f64 = 0.25;  const TOE_MTPJ_KD:    f64 = 0.0;
const TOE_PIPJ_KP:    f64 = 0.10;  const TOE_PIPJ_KD:    f64 = 0.0;
const TOE_DIPJ_KP:    f64 = 0.04;  const TOE_DIPJ_KD:    f64 = 0.0;
const HALLUX_MTPJ_SCALE: f64 = 0.80;
const HALLUX_IPJ_SCALE:  f64 = 0.50;
const TOE_MTPJ_SCALE:    f64 = 0.50;
const TOE_PIPJ_SCALE:    f64 = 0.70;
const TOE_DIPJ_SCALE:    f64 = 0.40;

struct HalluxSpec {
    foot_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
// Right foot: medial side = -x (toward body midline), lateral = +x.
// Hallux is medial (most -x), toes 2–5 progressively lateral.
const HALLUX: HalluxSpec = HalluxSpec {
    foot_offset_x: -0.024,
    prox_mass: 0.020, prox_radius: 0.013, prox_half_h: 0.018,
    dist_mass: 0.012, dist_radius: 0.011, dist_half_h: 0.012,
};

struct ToeSpec {
    foot_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    mid_mass: f64,  mid_radius: f64,  mid_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
const TOE2: ToeSpec = ToeSpec {
    foot_offset_x: -0.010,
    prox_mass: 0.012, prox_radius: 0.011, prox_half_h: 0.014,
    mid_mass:  0.005, mid_radius:  0.0095, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.008, dist_half_h: 0.005,
};
const TOE3: ToeSpec = ToeSpec {
    foot_offset_x: 0.003,
    prox_mass: 0.011, prox_radius: 0.010, prox_half_h: 0.013,
    mid_mass:  0.005, mid_radius:  0.0090, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.0075, dist_half_h: 0.005,
};
const TOE4: ToeSpec = ToeSpec {
    foot_offset_x: 0.014,
    prox_mass: 0.009, prox_radius: 0.0095, prox_half_h: 0.011,
    mid_mass:  0.004, mid_radius:  0.0085, mid_half_h: 0.005,
    dist_mass: 0.002, dist_radius: 0.007,  dist_half_h: 0.004,
};
const TOE5: ToeSpec = ToeSpec {
    foot_offset_x: 0.024,
    prox_mass: 0.006, prox_radius: 0.008,  prox_half_h: 0.009,
    mid_mass:  0.003, mid_radius:  0.0075, mid_half_h: 0.004,
    dist_mass: 0.002, dist_radius: 0.0065, dist_half_h: 0.004,
};

const SCHEDULE_PERIOD: f64 = 48.0;

// ─── Helpers ──────────────────────────────────────────────────────────

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
    limit_stiffness: f64, limit_exp: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a, min_angle, max_angle, limit_stiffness, limit_exp));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn add_toe_joint(world: &mut XpbdWorld, parent: usize, child: usize, j: &ToeJoint) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::x(),
        j.flex_min, j.flex_max, j.limit_k, 2.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::x(), j.damping));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::x(), j.spring, j.neutral));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::y(), -0.10, 0.10, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::y(), j.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::y(), j.off_axis_c));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::z(), -0.15, 0.15, 0.3, 2.0));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::z(), j.off_axis_k, 0.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::z(), j.off_axis_c));
}

struct ToeChainEntry {
    body_idx: usize,
    parent_chain_idx: Option<usize>,  // None = parent is foot
    parent_distal_local: NaVec3,
    proximal_local: NaVec3,
}

fn add_hallux(
    world: &mut XpbdWorld, foot_idx: usize, spec: &HalluxSpec,
    toe_pds: &mut Vec<(usize, f64)>,
) -> [ToeChainEntry; 2] {
    let mtpj_anchor_foot = NaVec3::new(spec.foot_offset_x, -FOOT_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(&world.bodies[foot_idx], mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0), UnitQuaternion::identity());
    prox.position = pos; prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(&world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        UnitQuaternion::identity());
    dist.position = pos; dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, foot_idx, prox_idx, mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0), TJOINT_K, C_FOOT_TOE);
    add_toe_joint(world, foot_idx, prox_idx, &HALLUX_MTPJ);

    add_elastic_joint(world, prox_idx, dist_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0), TJOINT_K, C_TOE_TOE);
    add_toe_joint(world, prox_idx, dist_idx, &HALLUX_IPJ);

    let mtpj_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(foot_idx, prox_idx, NaVec3::x(),
            HALLUX_MTPJ_KP, HALLUX_MTPJ_KD).with_target(0.0));
    toe_pds.push((mtpj_pd, HALLUX_MTPJ_SCALE));

    let ipj_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, dist_idx, NaVec3::x(),
            HALLUX_IPJ_KP, HALLUX_IPJ_KD).with_target(0.0));
    toe_pds.push((ipj_pd, HALLUX_IPJ_SCALE));

    [
        ToeChainEntry { body_idx: prox_idx, parent_chain_idx: None,
            parent_distal_local: mtpj_anchor_foot,
            proximal_local: NaVec3::new(0.0, spec.prox_half_h, 0.0) },
        ToeChainEntry { body_idx: dist_idx, parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.prox_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.dist_half_h, 0.0) },
    ]
}

fn add_toe(
    world: &mut XpbdWorld, foot_idx: usize, spec: &ToeSpec,
    toe_pds: &mut Vec<(usize, f64)>,
) -> [ToeChainEntry; 3] {
    let mtpj_anchor_foot = NaVec3::new(spec.foot_offset_x, -FOOT_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(&world.bodies[foot_idx], mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0), UnitQuaternion::identity());
    prox.position = pos; prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(spec.mid_mass,
        capsule_inertia(spec.mid_mass, spec.mid_radius, spec.mid_half_h));
    let (pos, orient) = place_child(&world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.mid_half_h, 0.0),
        UnitQuaternion::identity());
    mid.position = pos; mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(&world.bodies[mid_idx],
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        UnitQuaternion::identity());
    dist.position = pos; dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, foot_idx, prox_idx, mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0), TJOINT_K, C_FOOT_TOE);
    add_toe_joint(world, foot_idx, prox_idx, &TOE_MTPJ);
    add_elastic_joint(world, prox_idx, mid_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.mid_half_h, 0.0), TJOINT_K, C_TOE_TOE);
    add_toe_joint(world, prox_idx, mid_idx, &TOE_PIPJ);
    add_elastic_joint(world, mid_idx, dist_idx,
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0), TJOINT_K, C_TOE_TOE);
    add_toe_joint(world, mid_idx, dist_idx, &TOE_DIPJ);

    let mtpj_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(foot_idx, prox_idx, NaVec3::x(),
            TOE_MTPJ_KP, TOE_MTPJ_KD).with_target(0.0));
    toe_pds.push((mtpj_pd, TOE_MTPJ_SCALE));
    let pipj_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, mid_idx, NaVec3::x(),
            TOE_PIPJ_KP, TOE_PIPJ_KD).with_target(0.0));
    toe_pds.push((pipj_pd, TOE_PIPJ_SCALE));
    let dipj_pd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(mid_idx, dist_idx, NaVec3::x(),
            TOE_DIPJ_KP, TOE_DIPJ_KD).with_target(0.0));
    toe_pds.push((dipj_pd, TOE_DIPJ_SCALE));

    [
        ToeChainEntry { body_idx: prox_idx, parent_chain_idx: None,
            parent_distal_local: mtpj_anchor_foot,
            proximal_local: NaVec3::new(0.0, spec.prox_half_h, 0.0) },
        ToeChainEntry { body_idx: mid_idx, parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.prox_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.mid_half_h, 0.0) },
        ToeChainEntry { body_idx: dist_idx, parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.mid_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.dist_half_h, 0.0) },
    ]
}

// ─── Schedule ─────────────────────────────────────────────────────────

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

struct LegTargets {
    hip_flex: f64, hip_abd: f64, hip_rot: f64,
    knee: f64,
    ankle_pf: f64, ankle_inv: f64,
    toe: f64,
    phase: &'static str,
}

fn leg_targets(t: f64) -> LegTargets {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    let zero = LegTargets {
        hip_flex: 0.0, hip_abd: 0.0, hip_rot: 0.0,
        knee: 0.0, ankle_pf: 0.0, ankle_inv: 0.0, toe: 0.0,
        phase: "",
    };
    if t < 6.0 {
        LegTargets { hip_flex: half_cosine(t, 6.0, 1.50), phase: "hip flexion (knee toward chest)", ..zero }
    } else if t < 10.0 {
        LegTargets { hip_flex: half_cosine(t - 6.0, 4.0, -0.30), phase: "hip extension", ..zero }
    } else if t < 16.0 {
        LegTargets { hip_abd: half_cosine(t - 10.0, 6.0, 0.70), phase: "hip abduction (lateral lift)", ..zero }
    } else if t < 22.0 {
        LegTargets { hip_rot: half_cosine(t - 16.0, 6.0, 0.80), phase: "hip external rotation", ..zero }
    } else if t < 28.0 {
        LegTargets { knee: half_cosine(t - 22.0, 6.0, 2.20), phase: "knee flexion", ..zero }
    } else if t < 32.0 {
        LegTargets { ankle_pf: half_cosine(t - 28.0, 4.0, 0.85), phase: "ankle plantarflex", ..zero }
    } else if t < 36.0 {
        LegTargets { ankle_pf: half_cosine(t - 32.0, 4.0, -0.40), phase: "ankle dorsiflex", ..zero }
    } else if t < 42.0 {
        LegTargets { ankle_inv: half_cosine(t - 36.0, 6.0, 0.50), phase: "ankle inversion", ..zero }
    } else {
        // Toe curl/extend cycle.
        let local = t - 42.0;
        let phase_v = std::f64::consts::TAU * local / 6.0;
        LegTargets { toe: 0.85 * phase_v.sin(), phase: "toe curl/extend", ..zero }
    }
}

// ─── ECS resources / setup ────────────────────────────────────────────

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    pelvis_idx: usize,
    thigh_idx: usize,
    shin_idx: usize,
    foot_idx: usize,
    sim_time: f64,
    pd_hip_flex: usize, pd_hip_abd: usize, pd_hip_rot: usize,
    pd_knee: usize,
    pd_ankle_pf: usize, pd_ankle_inv: usize,
    toe_pds: Vec<(usize, f64)>,
}

#[derive(Resource)]
struct FootChain {
    foot_idx: usize,
    foot_proximal_local: NaVec3,
    nodes: Vec<ToeChainEntry>,
}

#[derive(Component)] struct BodyRef(usize);
#[derive(Component)] struct FootBodyRef(usize);
#[derive(Component)] struct ToeBodyRef(usize);
#[derive(Component)] struct KneeMarker;
#[derive(Component)] struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    // Pelvis (pinned).
    let mut pelvis = RigidBody::new(0.0, capsule_inertia(15.0, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_REST_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Thigh — hangs from hip anchor.
    let hip_anchor_pelvis = NaVec3::new(HIP_ANCHOR_PELVIS_X, HIP_ANCHOR_PELVIS_Y, 0.0);
    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor_pelvis, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        UnitQuaternion::identity());
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        hip_anchor_pelvis, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);

    let hip_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let hip_abd_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let hip_rot_axis  = NaVec3::new( 0.0, 1.0, 0.0);
    add_axis(&mut world, pelvis_idx, thigh_idx, hip_flex_axis,
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP);
    add_axis(&mut world, pelvis_idx, thigh_idx, hip_abd_axis,
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP);
    add_axis(&mut world, pelvis_idx, thigh_idx, hip_rot_axis,
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_ROT_K, HIP_ROT_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP);

    let pd_hip_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_flex_axis,
            HIP_PD_KP, HIP_PD_KD).with_target(0.0));
    let pd_hip_abd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_abd_axis,
            HIP_PD_KP, HIP_PD_KD).with_target(0.0));
    let pd_hip_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_rot_axis,
            HIP_PD_KP, HIP_PD_KD).with_target(0.0));

    // Shin (lower leg).
    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        UnitQuaternion::identity());
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(&mut world, thigh_idx, shin_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);

    let knee_axis = NaVec3::new(1.0, 0.0, 0.0);
    add_axis(&mut world, thigh_idx, shin_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL,
        KNEE_LIMIT_K, KNEE_LIMIT_EXP);
    add_axis(&mut world, thigh_idx, shin_idx, NaVec3::z(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP);
    add_axis(&mut world, thigh_idx, shin_idx, NaVec3::y(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP);

    let pd_knee = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, shin_idx, knee_axis,
            KNEE_PD_KP, KNEE_PD_KD).with_target(KNEE_NEUTRAL));

    // Foot (vertical capsule per ankle_compliant convention).
    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        UnitQuaternion::identity());
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, shin_idx, foot_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    let pf_axis    = NaVec3::new(1.0, 0.0, 0.0);
    let inv_axis   = NaVec3::new(0.0, 0.0, 1.0);
    let twist_axis = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(&mut world, shin_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP);
    add_axis(&mut world, shin_idx, foot_idx, inv_axis,
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP);
    add_axis(&mut world, shin_idx, foot_idx, twist_axis,
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0,
        TWIST_LIMIT_K, ANKLE_LIMIT_EXP);

    let pd_ankle_pf = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, pf_axis,
            ANKLE_PD_KP, ANKLE_PD_KD).with_target(0.0));
    let pd_ankle_inv = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, inv_axis,
            ANKLE_PD_KP, ANKLE_PD_KD).with_target(0.0));

    // Toes.
    let mut chain_nodes: Vec<ToeChainEntry> = Vec::new();
    let mut toe_pds: Vec<(usize, f64)> = Vec::new();
    let push_chain = |nodes: &mut Vec<ToeChainEntry>, mut entries: Vec<ToeChainEntry>| {
        let base = nodes.len();
        for (i, e) in entries.iter_mut().enumerate() {
            if e.parent_chain_idx == Some(usize::MAX) {
                e.parent_chain_idx = Some(base + i - 1);
            }
        }
        nodes.extend(entries);
    };
    push_chain(&mut chain_nodes, add_hallux(&mut world, foot_idx, &HALLUX, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(&mut world, foot_idx, &TOE2, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(&mut world, foot_idx, &TOE3, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(&mut world, foot_idx, &TOE4, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(&mut world, foot_idx, &TOE5, &mut toe_pds).into_iter().collect());

    // Meshes.
    let pelvis_mesh = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let thigh_mesh  = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let shin_mesh   = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let foot_mesh   = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));

    let pelvis_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let thigh_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let shin_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.88, 0.68, 0.53), ..default() });
    let foot_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let toe_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.65, 0.45, 0.30), ..default() });

    commands.spawn((PbrBundle {
        mesh: pelvis_mesh, material: pelvis_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default()
    }, BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle {
        mesh: thigh_mesh, material: thigh_mat,
        transform: rigid_body_to_transform(&world.bodies[thigh_idx]), ..default()
    }, BodyRef(thigh_idx)));
    commands.spawn((PbrBundle {
        mesh: shin_mesh, material: shin_mat,
        transform: rigid_body_to_transform(&world.bodies[shin_idx]), ..default()
    }, BodyRef(shin_idx)));
    commands.spawn((PbrBundle {
        mesh: foot_mesh, material: foot_mat,
        transform: foot_render_transform(&world.bodies[foot_idx]), ..default()
    }, FootBodyRef(foot_idx)));

    // Knee marker (small lateral capsule on the thigh's distal end, makes
    // hip rotation visible).
    let marker_mesh = meshes.add(Capsule3d::new(0.014, 0.08));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.40, 0.65), ..default() });
    commands.spawn((PbrBundle {
        mesh: marker_mesh, material: marker_mat,
        transform: Transform::default(), ..default()
    }, KneeMarker));

    // Toe meshes.
    let toe_dims: Vec<(f32, f32)> = vec![
        (HALLUX.prox_radius as f32, HALLUX.prox_half_h as f32),
        (HALLUX.dist_radius as f32, HALLUX.dist_half_h as f32),
        (TOE2.prox_radius as f32, TOE2.prox_half_h as f32),
        (TOE2.mid_radius  as f32, TOE2.mid_half_h  as f32),
        (TOE2.dist_radius as f32, TOE2.dist_half_h as f32),
        (TOE3.prox_radius as f32, TOE3.prox_half_h as f32),
        (TOE3.mid_radius  as f32, TOE3.mid_half_h  as f32),
        (TOE3.dist_radius as f32, TOE3.dist_half_h as f32),
        (TOE4.prox_radius as f32, TOE4.prox_half_h as f32),
        (TOE4.mid_radius  as f32, TOE4.mid_half_h  as f32),
        (TOE4.dist_radius as f32, TOE4.dist_half_h as f32),
        (TOE5.prox_radius as f32, TOE5.prox_half_h as f32),
        (TOE5.mid_radius  as f32, TOE5.mid_half_h  as f32),
        (TOE5.dist_radius as f32, TOE5.dist_half_h as f32),
    ];
    for (i, (radius, half_h)) in toe_dims.iter().enumerate() {
        let mesh = meshes.add(Capsule3d::new(*radius, 2.0 * half_h));
        commands.spawn((PbrBundle {
            mesh, material: toe_mat.clone(),
            transform: Transform::IDENTITY, ..default()
        }, ToeBodyRef(i)));
    }

    let foot_chain = FootChain {
        foot_idx,
        foot_proximal_local: FOOT_ANCHOR_LOCAL,
        nodes: chain_nodes,
    };
    commands.insert_resource(foot_chain);

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera: 3/4 front-side view, framing pelvis-down.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.40, 0.85, 1.30)
            .looking_at(Vec3::new(0.10, 0.55, 0.0), Vec3::Y),
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
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, pelvis_idx, thigh_idx, shin_idx, foot_idx,
        sim_time: 0.0,
        pd_hip_flex, pd_hip_abd, pd_hip_rot,
        pd_knee, pd_ankle_pf, pd_ankle_inv,
        toe_pds,
    });
}

fn foot_visual_offset() -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0))
}

fn foot_render_transform(body: &RigidBody) -> Transform {
    let offset = foot_visual_offset();
    let render_orient = body.orientation * offset;
    let anchor_local = FOOT_ANCHOR_LOCAL;
    let ankle_world = body.position + body.orientation * anchor_local;
    let render_pos = ankle_world - render_orient * anchor_local;
    let q = render_orient.into_inner();
    Transform {
        translation: Vec3::new(render_pos.x as f32, render_pos.y as f32, render_pos.z as f32),
        rotation: Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32),
        scale: Vec3::ONE,
    }
}

fn chain_render(
    parent_render_orient: UnitQuaternion<f64>,
    parent_distal_world: NaVec3,
    parent_body: &RigidBody,
    child_body: &RigidBody,
    child_proximal_local: NaVec3,
) -> (UnitQuaternion<f64>, NaVec3) {
    let rel_rot = parent_body.orientation.inverse() * child_body.orientation;
    let render_orient = parent_render_orient * rel_rot;
    let render_pos = parent_distal_world - render_orient * child_proximal_local;
    (render_orient, render_pos)
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let tg = leg_targets(phys.sim_time);
    let pd_hip_flex = phys.pd_hip_flex;
    let pd_hip_abd  = phys.pd_hip_abd;
    let pd_hip_rot  = phys.pd_hip_rot;
    let pd_knee     = phys.pd_knee;
    let pd_ankle_pf = phys.pd_ankle_pf;
    let pd_ankle_inv = phys.pd_ankle_inv;
    phys.world.angular_pd_controllers[pd_hip_flex].target_angle = tg.hip_flex;
    phys.world.angular_pd_controllers[pd_hip_abd].target_angle = tg.hip_abd;
    phys.world.angular_pd_controllers[pd_hip_rot].target_angle = tg.hip_rot;
    phys.world.angular_pd_controllers[pd_knee].target_angle = KNEE_NEUTRAL + tg.knee;
    phys.world.angular_pd_controllers[pd_ankle_pf].target_angle = tg.ankle_pf;
    phys.world.angular_pd_controllers[pd_ankle_inv].target_angle = tg.ankle_inv;
    let toe_pds: Vec<(usize, f64)> = phys.toe_pds.clone();
    for (idx, scale) in toe_pds {
        phys.world.angular_pd_controllers[idx].target_angle = tg.toe * scale;
    }
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    chain: Res<FootChain>,
    mut q_body: Query<(&BodyRef, &mut Transform), (Without<FootBodyRef>, Without<ToeBodyRef>, Without<KneeMarker>)>,
    mut q_foot: Query<(&FootBodyRef, &mut Transform), (Without<BodyRef>, Without<ToeBodyRef>, Without<KneeMarker>)>,
    mut q_toe: Query<(&ToeBodyRef, &mut Transform), (Without<BodyRef>, Without<FootBodyRef>, Without<KneeMarker>)>,
    mut q_marker: Query<&mut Transform, (With<KneeMarker>, Without<BodyRef>, Without<FootBodyRef>, Without<ToeBodyRef>)>,
) {
    for (body_ref, mut transform) in &mut q_body {
        let body = &phys.world.bodies[body_ref.0];
        *transform = rigid_body_to_transform(body);
    }
    for (foot_ref, mut transform) in &mut q_foot {
        let body = &phys.world.bodies[foot_ref.0];
        *transform = foot_render_transform(body);
    }

    let foot_body = &phys.world.bodies[chain.foot_idx];
    let foot_offset = foot_visual_offset();
    let foot_render_orient = foot_body.orientation * foot_offset;
    let ankle_world = foot_body.position + foot_body.orientation * chain.foot_proximal_local;
    let foot_render_pos = ankle_world - foot_render_orient * chain.foot_proximal_local;

    let mut node_renders: Vec<(UnitQuaternion<f64>, NaVec3)> = Vec::with_capacity(chain.nodes.len());
    for node in &chain.nodes {
        let (parent_render_orient, parent_distal_world);
        match node.parent_chain_idx {
            None => {
                parent_render_orient = foot_render_orient;
                parent_distal_world = foot_render_pos
                    + foot_render_orient * node.parent_distal_local;
            }
            Some(pi) => {
                let (po, pp) = node_renders[pi];
                parent_render_orient = po;
                parent_distal_world = pp + po * node.parent_distal_local;
            }
        }
        let parent_body = match node.parent_chain_idx {
            None => foot_body,
            Some(pi) => &phys.world.bodies[chain.nodes[pi].body_idx],
        };
        let child_body = &phys.world.bodies[node.body_idx];
        let (orient, pos) = chain_render(
            parent_render_orient, parent_distal_world,
            parent_body, child_body, node.proximal_local);
        node_renders.push((orient, pos));
    }

    for (toe_ref, mut transform) in &mut q_toe {
        let (orient, pos) = node_renders[toe_ref.0];
        let q = orient.into_inner();
        *transform = Transform {
            translation: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
            rotation: Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32),
            scale: Vec3::ONE,
        };
    }

    // Knee marker sits perpendicular to the thigh at its distal end, makes
    // hip axial rotation visible.
    let thigh = &phys.world.bodies[phys.thigh_idx];
    let local_offset = NaVec3::new(0.08, -THIGH_HALF_H, 0.0);
    let world_pos = thigh.position + thigh.orientation * local_offset;
    let marker_rot = thigh.orientation
        * UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    for mut transform in &mut q_marker {
        *transform = Transform {
            translation: nalgebra_vec3_to_bevy(world_pos),
            rotation: nalgebra_quat_to_bevy(marker_rot),
            scale: Vec3::ONE,
        };
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let pelvis = &phys.world.bodies[phys.pelvis_idx];
    let thigh  = &phys.world.bodies[phys.thigh_idx];
    let shin   = &phys.world.bodies[phys.shin_idx];
    let foot   = &phys.world.bodies[phys.foot_idx];

    let hip_rel  = (thigh.orientation * pelvis.orientation.inverse()).scaled_axis();
    let knee_rel = (shin.orientation * thigh.orientation.inverse()).scaled_axis();
    let ankle_rel = (foot.orientation * shin.orientation.inverse()).scaled_axis();

    let hip_flex = hip_rel.dot(&NaVec3::new(-1.0, 0.0, 0.0));
    let hip_abd  = hip_rel.dot(&NaVec3::new( 0.0, 0.0, 1.0));
    let hip_rot  = hip_rel.dot(&NaVec3::new( 0.0, 1.0, 0.0));

    let tg = leg_targets(phys.sim_time);

    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  hip flex : tgt {:+.2}  →  {:+.2}\n  hip abd  : tgt {:+.2}  →  {:+.2}\n  hip rot  : tgt {:+.2}  →  {:+.2}\n  knee     : tgt {:+.2}  →  {:+.2}\n  ankle PF : tgt {:+.2}  →  {:+.2}\n  ankle INV: tgt {:+.2}  →  {:+.2}\n  toe drive: {:+.2}",
            tg.phase, phys.sim_time, SCHEDULE_PERIOD,
            tg.hip_flex, hip_flex,
            tg.hip_abd,  hip_abd,
            tg.hip_rot,  hip_rot,
            KNEE_NEUTRAL + tg.knee, knee_rel.x,
            tg.ankle_pf, ankle_rel.x,
            tg.ankle_inv, ankle_rel.z,
            tg.toe,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Leg full demo (pelvis → hip → knee → ankle → toes)".into(),
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
