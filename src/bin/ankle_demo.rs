// Visual demo of the ankle on its own (no leg yet).
//
// Pinned vertical lower leg + foot on a 2-DOF biomech joint (PF/DF, Inv/Ev,
// twist locked). Hallux (2 phalanges) and four lesser toes (3 phalanges
// each) hang off the foot's distal end on individual MTPJ/IPJ joints. PD
// controllers drive the foot through its anatomical ROM:
//
//   0–6 s     PF/DF sweep         x: 0 → +0.87 (50° plantarflex) → -0.44 (25° dorsi) → 0
//   6–10 s    Inv/Ev sweep        z: 0 → +0.52 (30° inversion) → -0.26 (15° eversion) → 0
//   10–18 s   walking-like cycle  small repeating PF push + slight roll
//   18–26 s   exploratory combo   smooth PF/INV figure
//
// PHYSICS NOTE: the foot is a vertical capsule (identity orientation).
// See ankle_compliant.rs for the bivector-artifact reasoning. The visual
// renderer applies an extra R_x(-π/2) on the foot+toe subtree (pivoting at
// the ankle) so it looks like a horizontal anatomical foot with toes
// pointing world +z.

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
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const LOWER_LEG_HALF_H: f64 = 0.210;
const LOWER_LEG_RADIUS: f64 = 0.050;

const FOOT_MASS:    f64 = 1.0;
const FOOT_HALF_H:  f64 = 0.125;
const FOOT_RADIUS:  f64 = 0.040;

const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

// Passive biomech (PD will dominate, but these keep stability).
const PF_K:        f64 = 4.0;
const INV_K:       f64 = 3.5;
const TWIST_K:     f64 = 8.0;
const PF_C:        f64 = 1.2;
const INV_C:       f64 = 1.0;
const TWIST_C:     f64 = 0.5;
const PF_MIN:      f64 = -0.44;
const PF_MAX:      f64 =  0.87;
const INV_MIN:     f64 = -0.26;
const INV_MAX:     f64 =  0.52;
const TWIST_LIM:   f64 =  0.10;
const ANKLE_LIMIT_K:    f64 = 80.0;
const TWIST_LIMIT_K:    f64 = 200.0;
const ANKLE_LIMIT_EXP:  f64 = 2.0;

// Active drive on the ankle.
const ANKLE_PD_KP: f64 = 80.0;
const ANKLE_PD_KD: f64 =  6.0;

// Active drive on the toe joints. KP scales with phalanx size; KD = 0
// because biomech dampers + PD KD on the smallest distal segment exceed
// the explicit-Euler stability bound (c·dt/I_red < 2). Damping comes from
// the per-joint biomech damper alone — settling is slightly underdamped
// but stable.
const HALLUX_MTPJ_KP: f64 = 0.6;   const HALLUX_MTPJ_KD: f64 = 0.0;
const HALLUX_IPJ_KP:  f64 = 0.3;   const HALLUX_IPJ_KD:  f64 = 0.0;
const TOE_MTPJ_KP:    f64 = 0.25;  const TOE_MTPJ_KD:    f64 = 0.0;
const TOE_PIPJ_KP:    f64 = 0.10;  const TOE_PIPJ_KD:    f64 = 0.0;
const TOE_DIPJ_KP:    f64 = 0.04;  const TOE_DIPJ_KD:    f64 = 0.0;

// Per-joint scale on the global toe_target value. Reflects how much each
// joint contributes to a curl: hallux MTPJ leads the toe-off pose; lesser
// toes' PIP flexes the most during a curl; DIP lags slightly.
const HALLUX_MTPJ_SCALE: f64 = 0.80;
const HALLUX_IPJ_SCALE:  f64 = 0.50;
const TOE_MTPJ_SCALE:    f64 = 0.50;
const TOE_PIPJ_SCALE:    f64 = 0.70;
const TOE_DIPJ_SCALE:    f64 = 0.40;

// Toe joint elastic anchor.
const TJOINT_K: f64 = 800.0;
const C_FOOT_TOE: f64 = 1.5;
const C_TOE_TOE:  f64 = 0.8;

// Toe biomech. Flex axis = body +x in foot's frame (matches PF axis).
// Positive rotation curls toes under the foot (toward heel in render);
// negative is dorsiflexion (toes raised — used for toe-off).
struct ToeJoint {
    flex_min: f64, flex_max: f64,
    spring: f64, damping: f64, neutral: f64,
    limit_k: f64,
    off_axis_k: f64, off_axis_c: f64,
}
const HALLUX_MTPJ: ToeJoint = ToeJoint {
    flex_min: -1.40, flex_max: 0.50,
    spring: 0.04, damping: 0.0008, neutral: 0.0,
    limit_k: 0.5,
    off_axis_k: 0.05, off_axis_c: 0.0005,
};
const HALLUX_IPJ: ToeJoint = ToeJoint {
    flex_min: -0.10, flex_max: 1.40,
    spring: 0.03, damping: 0.0004, neutral: 0.0,
    limit_k: 0.4,
    off_axis_k: 0.04, off_axis_c: 0.0003,
};
const TOE_MTPJ: ToeJoint = ToeJoint {
    flex_min: -0.70, flex_max: 0.40,
    spring: 0.025, damping: 0.0005, neutral: 0.0,
    limit_k: 0.4,
    off_axis_k: 0.04, off_axis_c: 0.0003,
};
const TOE_PIPJ: ToeJoint = ToeJoint {
    flex_min: -0.05, flex_max: 1.20,
    spring: 0.02, damping: 0.0003, neutral: 0.0,
    limit_k: 0.3,
    off_axis_k: 0.03, off_axis_c: 0.0002,
};
const TOE_DIPJ: ToeJoint = ToeJoint {
    flex_min: -0.05, flex_max: 1.00,
    spring: 0.015, damping: 0.0002, neutral: 0.0,
    limit_k: 0.25,
    off_axis_k: 0.025, off_axis_c: 0.0002,
};

// Toe placements on foot's distal end (body local y = -FOOT_HALF_H).
// +x is medial (right foot: hallux on +x, little toe on -x).
struct HalluxSpec {
    foot_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
const HALLUX: HalluxSpec = HalluxSpec {
    foot_offset_x: 0.030,
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
    foot_offset_x: 0.013,
    prox_mass: 0.012, prox_radius: 0.011, prox_half_h: 0.014,
    mid_mass:  0.005, mid_radius:  0.0095, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.008, dist_half_h: 0.005,
};
const TOE3: ToeSpec = ToeSpec {
    foot_offset_x: 0.000,
    prox_mass: 0.011, prox_radius: 0.010, prox_half_h: 0.013,
    mid_mass:  0.005, mid_radius:  0.0090, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.0075, dist_half_h: 0.005,
};
const TOE4: ToeSpec = ToeSpec {
    foot_offset_x: -0.013,
    prox_mass: 0.009, prox_radius: 0.0095, prox_half_h: 0.011,
    mid_mass:  0.004, mid_radius:  0.0085, mid_half_h: 0.005,
    dist_mass: 0.002, dist_radius: 0.007,  dist_half_h: 0.004,
};
const TOE5: ToeSpec = ToeSpec {
    foot_offset_x: -0.026,
    prox_mass: 0.006, prox_radius: 0.008,  prox_half_h: 0.009,
    mid_mass:  0.003, mid_radius:  0.0075, mid_half_h: 0.004,
    dist_mass: 0.002, dist_radius: 0.0065, dist_half_h: 0.004,
};

const LEG_REST_POS: NaVec3 = NaVec3::new(0.0, 1.0, 0.0);

const SCHEDULE_PERIOD: f64 = 26.0;

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
        min_angle, max_angle, limit_stiffness, ANKLE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn add_toe_joint(world: &mut XpbdWorld, parent: usize, child: usize, j: &ToeJoint) {
    // Flex axis: body +x (sagittal-plane curl/extend, matches foot PF axis).
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::x(),
        j.flex_min, j.flex_max, j.limit_k, 2.0));
    world.angular_dampers.push(AngularDamper::new(parent, child, NaVec3::x(), j.damping));
    world.angular_springs.push(AngularSpring::new(parent, child, NaVec3::x(), j.spring, j.neutral));

    // Off-axis: y (axial twist), z (abduction-adduction). Tight.
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
    parent_chain_idx: Option<usize>,  // index into FootChain.nodes; None = foot
    parent_distal_local: NaVec3,
    proximal_local: NaVec3,
}

#[derive(Resource)]
struct FootChain {
    foot_idx: usize,
    foot_proximal_local: NaVec3,  // ankle anchor on foot
    nodes: Vec<ToeChainEntry>,  // toes only (foot rendered separately)
}

fn add_hallux(
    world: &mut XpbdWorld,
    foot_idx: usize,
    spec: &HalluxSpec,
    toe_pds: &mut Vec<(usize, f64)>,
) -> [ToeChainEntry; 2] {
    let mtpj_anchor_foot = NaVec3::new(spec.foot_offset_x, -FOOT_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[foot_idx], mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::identity());
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        UnitQuaternion::identity());
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, foot_idx, prox_idx,
        mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        TJOINT_K, C_FOOT_TOE);
    add_toe_joint(world, foot_idx, prox_idx, &HALLUX_MTPJ);

    add_elastic_joint(world, prox_idx, dist_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        TJOINT_K, C_TOE_TOE);
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
        ToeChainEntry {
            body_idx: prox_idx,
            parent_chain_idx: None,  // parent is foot
            parent_distal_local: mtpj_anchor_foot,
            proximal_local: NaVec3::new(0.0, spec.prox_half_h, 0.0),
        },
        ToeChainEntry {
            body_idx: dist_idx,
            // Filled in after we know the prox's chain index.
            parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.prox_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.dist_half_h, 0.0),
        },
    ]
}

fn add_toe(
    world: &mut XpbdWorld,
    foot_idx: usize,
    spec: &ToeSpec,
    toe_pds: &mut Vec<(usize, f64)>,
) -> [ToeChainEntry; 3] {
    let mtpj_anchor_foot = NaVec3::new(spec.foot_offset_x, -FOOT_HALF_H, 0.0);

    let mut prox = RigidBody::new(spec.prox_mass,
        capsule_inertia(spec.prox_mass, spec.prox_radius, spec.prox_half_h));
    let (pos, orient) = place_child(
        &world.bodies[foot_idx], mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        UnitQuaternion::identity());
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(spec.mid_mass,
        capsule_inertia(spec.mid_mass, spec.mid_radius, spec.mid_half_h));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.mid_half_h, 0.0),
        UnitQuaternion::identity());
    mid.position = pos;
    mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(spec.dist_mass,
        capsule_inertia(spec.dist_mass, spec.dist_radius, spec.dist_half_h));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        UnitQuaternion::identity());
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(world, foot_idx, prox_idx,
        mtpj_anchor_foot,
        NaVec3::new(0.0, spec.prox_half_h, 0.0),
        TJOINT_K, C_FOOT_TOE);
    add_toe_joint(world, foot_idx, prox_idx, &TOE_MTPJ);

    add_elastic_joint(world, prox_idx, mid_idx,
        NaVec3::new(0.0, -spec.prox_half_h, 0.0),
        NaVec3::new(0.0,  spec.mid_half_h, 0.0),
        TJOINT_K, C_TOE_TOE);
    add_toe_joint(world, prox_idx, mid_idx, &TOE_PIPJ);

    add_elastic_joint(world, mid_idx, dist_idx,
        NaVec3::new(0.0, -spec.mid_half_h, 0.0),
        NaVec3::new(0.0,  spec.dist_half_h, 0.0),
        TJOINT_K, C_TOE_TOE);
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
        ToeChainEntry {
            body_idx: prox_idx,
            parent_chain_idx: None,
            parent_distal_local: mtpj_anchor_foot,
            proximal_local: NaVec3::new(0.0, spec.prox_half_h, 0.0),
        },
        ToeChainEntry {
            body_idx: mid_idx,
            parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.prox_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.mid_half_h, 0.0),
        },
        ToeChainEntry {
            body_idx: dist_idx,
            parent_chain_idx: Some(usize::MAX),
            parent_distal_local: NaVec3::new(0.0, -spec.mid_half_h, 0.0),
            proximal_local: NaVec3::new(0.0, spec.dist_half_h, 0.0),
        },
    ]
}

fn smooth_to(t: f64, t0: f64, t1: f64, v0: f64, v1: f64) -> f64 {
    let phase = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    let s = 0.5 - 0.5 * (std::f64::consts::PI * phase).cos();
    v0 + (v1 - v0) * s
}

/// Toe target value in [-1, +1]. Positive = curl (plantarflex toes under
/// foot); negative = extend (dorsiflex, toe-off pose). Each toe joint applies
/// its own per-joint scale on top of this.
fn toe_target(t: f64) -> f64 {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        // Mirror the foot motion: curl as foot plantarflexes, extend at dorsi.
        if t < 2.0 {
            smooth_to(t, 0.0, 2.0, 0.0, 0.85)
        } else if t < 4.0 {
            smooth_to(t, 2.0, 4.0, 0.85, -0.80)
        } else {
            smooth_to(t, 4.0, 6.0, -0.80, 0.0)
        }
    } else if t < 10.0 {
        // INV phase: a gentle squeeze (slight curl) — like grabbing the floor
        // for stability when the foot rolls.
        let local = t - 6.0;
        0.20 * (std::f64::consts::PI * local / 4.0).sin()
    } else if t < 18.0 {
        // Walking gait: dorsiflex at the end of stance (toe-off),
        // then settle through swing.
        let local = t - 10.0;
        let phase = std::f64::consts::TAU * local / 1.2;
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        // Negative half-cycle gives the toe-off lift; positive a small curl.
        -0.55 * phase.sin() * envelope
    } else {
        // Exploratory: independent toe motion (figure 8 in flex).
        let local = t - 18.0;
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        0.70 * (std::f64::consts::TAU * local / 3.0).sin() * envelope
    }
}

fn ankle_targets(t: f64) -> (f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        let pf = if t < 2.0 {
            smooth_to(t, 0.0, 2.0, 0.0, PF_MAX)
        } else if t < 4.0 {
            smooth_to(t, 2.0, 4.0, PF_MAX, PF_MIN)
        } else {
            smooth_to(t, 4.0, 6.0, PF_MIN, 0.0)
        };
        (pf, 0.0, "PF/DF sweep")
    } else if t < 10.0 {
        let local = t - 6.0;
        let inv = if local < 1.5 {
            smooth_to(local, 0.0, 1.5, 0.0, INV_MAX)
        } else if local < 3.0 {
            smooth_to(local, 1.5, 3.0, INV_MAX, INV_MIN)
        } else {
            smooth_to(local, 3.0, 4.0, INV_MIN, 0.0)
        };
        (0.0, inv, "Inv/Ev sweep")
    } else if t < 18.0 {
        let local = t - 10.0;
        let phase = std::f64::consts::TAU * local / 1.2;
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let pf = 0.45 * (0.5 - 0.5 * phase.cos()) * envelope;
        let inv = 0.10 * phase.sin() * envelope;
        (pf, inv, "walking cycle")
    } else {
        let local = t - 18.0;
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let pf  = 0.55 * (std::f64::consts::TAU * local / 4.0).sin() * envelope;
        let inv = 0.30 * (std::f64::consts::TAU * local / 5.0).sin() * envelope;
        (pf, inv, "exploratory combo")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    lower_idx: usize,
    foot_idx: usize,
    sim_time: f64,
    pd_pf_idx: usize,
    pd_inv_idx: usize,
    toe_pds: Vec<(usize, f64)>,  // (controller idx, scale on toe_target)
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct FootBodyRef(usize);

#[derive(Component)]
struct ToeBodyRef(usize);  // index into FootChain.nodes

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut lower = RigidBody::new(0.0, capsule_inertia(3.5, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    lower.position = LEG_REST_POS;
    lower.orientation = UnitQuaternion::identity();
    let lower_idx = world.add_body(lower);

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[lower_idx],
        NaVec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        UnitQuaternion::identity());
    foot.position = pos;
    foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, lower_idx, foot_idx,
        NaVec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    let pf_axis  = NaVec3::new(1.0, 0.0, 0.0);
    let inv_axis = NaVec3::new(0.0, 0.0, 1.0);
    let twist_axis = NaVec3::new(0.0, 1.0, 0.0);

    add_axis(&mut world, lower_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0, ANKLE_LIMIT_K);
    add_axis(&mut world, lower_idx, foot_idx, inv_axis,
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0, ANKLE_LIMIT_K);
    add_axis(&mut world, lower_idx, foot_idx, twist_axis,
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0, TWIST_LIMIT_K);

    let pd_pf_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(lower_idx, foot_idx, pf_axis, ANKLE_PD_KP, ANKLE_PD_KD)
            .with_target(0.0));
    let pd_inv_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(lower_idx, foot_idx, inv_axis, ANKLE_PD_KP, ANKLE_PD_KD)
            .with_target(0.0));

    // Build the toe chain (hallux + 4 lesser toes).
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

    // Meshes and materials.
    let lower_mesh = meshes.add(Capsule3d::new(LOWER_LEG_RADIUS as f32, (2.0 * LOWER_LEG_HALF_H) as f32));
    let foot_mesh  = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));

    let lower_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let foot_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let toe_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.45, 0.30), ..default() });

    commands.spawn((PbrBundle {
        mesh: lower_mesh, material: lower_mat,
        transform: rigid_body_to_transform(&world.bodies[lower_idx]),
        ..default()
    }, BodyRef(lower_idx)));

    commands.spawn((PbrBundle {
        mesh: foot_mesh, material: foot_mat,
        transform: foot_render_transform(&world.bodies[foot_idx]),
        ..default()
    }, FootBodyRef(foot_idx)));

    // Spawn one mesh per toe segment.
    for (i, node) in chain_nodes.iter().enumerate() {
        // Look up segment radius from the body's inertia is awkward — instead,
        // re-derive from spec lookups would be tedious. Use half_h to scale
        // the radius via a fixed slim ratio (toes are slimmer than the foot).
        // We actually stored the half_h on each entry; pull radius from a
        // small per-spec table by recomputing in the spawn-loop below.
        let _ = (i, node);
    }
    // Re-spawn with proper meshes (with both half_h and radius lookup).
    // We held off above so we could iterate by spec. Instead, we built the
    // chain in a known order: [hallux_prox, hallux_dist, toe2_prox, toe2_mid,
    // toe2_dist, ...]. Match that order here.
    let toe_meshes: Vec<(f32, f32, f32, f32)> = vec![
        // (radius, half_h) per chain node. Order must match push_chain calls above.
        (HALLUX.prox_radius as f32, HALLUX.prox_half_h as f32, 0.0, 0.0),
        (HALLUX.dist_radius as f32, HALLUX.dist_half_h as f32, 0.0, 0.0),
        (TOE2.prox_radius as f32, TOE2.prox_half_h as f32, 0.0, 0.0),
        (TOE2.mid_radius  as f32, TOE2.mid_half_h  as f32, 0.0, 0.0),
        (TOE2.dist_radius as f32, TOE2.dist_half_h as f32, 0.0, 0.0),
        (TOE3.prox_radius as f32, TOE3.prox_half_h as f32, 0.0, 0.0),
        (TOE3.mid_radius  as f32, TOE3.mid_half_h  as f32, 0.0, 0.0),
        (TOE3.dist_radius as f32, TOE3.dist_half_h as f32, 0.0, 0.0),
        (TOE4.prox_radius as f32, TOE4.prox_half_h as f32, 0.0, 0.0),
        (TOE4.mid_radius  as f32, TOE4.mid_half_h  as f32, 0.0, 0.0),
        (TOE4.dist_radius as f32, TOE4.dist_half_h as f32, 0.0, 0.0),
        (TOE5.prox_radius as f32, TOE5.prox_half_h as f32, 0.0, 0.0),
        (TOE5.mid_radius  as f32, TOE5.mid_half_h  as f32, 0.0, 0.0),
        (TOE5.dist_radius as f32, TOE5.dist_half_h as f32, 0.0, 0.0),
    ];
    for (i, (radius, half_h, _, _)) in toe_meshes.iter().enumerate() {
        let mesh = meshes.add(Capsule3d::new(*radius, 2.0 * half_h));
        commands.spawn((PbrBundle {
            mesh, material: toe_mat.clone(),
            transform: Transform::IDENTITY,  // first sync sets it
            ..default()
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
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.20, 0.85, 0.50)
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
        TextStyle { font_size: 20.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0),
        ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, lower_idx, foot_idx, sim_time: 0.0, pd_pf_idx, pd_inv_idx, toe_pds,
    });
}

fn foot_visual_offset() -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0))
}

/// Foot's render transform: applies the visual offset on the body's local
/// frame (Approach A — preserves anatomical motion under PF and INV) and
/// pivots around the foot's proximal anchor (the ankle), so the rendered
/// "top of foot" stays bolted to the physical joint.
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

/// Compute (orient, pos) for a child body whose proximal anchor must coincide
/// with `parent_distal_world` in the rendered scene, while preserving the
/// physics-relative orientation between parent_body and child_body.
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
    let (pf_t, inv_t, _) = ankle_targets(phys.sim_time);
    let toe_t = toe_target(phys.sim_time);
    let pd_pf_idx = phys.pd_pf_idx;
    let pd_inv_idx = phys.pd_inv_idx;
    phys.world.angular_pd_controllers[pd_pf_idx].target_angle = pf_t;
    phys.world.angular_pd_controllers[pd_inv_idx].target_angle = inv_t;
    let toe_pds: Vec<(usize, f64)> = phys.toe_pds.clone();
    for (idx, scale) in toe_pds {
        phys.world.angular_pd_controllers[idx].target_angle = toe_t * scale;
    }
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    chain: Res<FootChain>,
    mut q_body: Query<(&BodyRef, &mut Transform), (Without<FootBodyRef>, Without<ToeBodyRef>)>,
    mut q_foot: Query<(&FootBodyRef, &mut Transform), (Without<BodyRef>, Without<ToeBodyRef>)>,
    mut q_toe: Query<(&ToeBodyRef, &mut Transform), (Without<BodyRef>, Without<FootBodyRef>)>,
) {
    for (body_ref, mut transform) in &mut q_body {
        let body = &phys.world.bodies[body_ref.0];
        *transform = rigid_body_to_transform(body);
    }
    for (foot_ref, mut transform) in &mut q_foot {
        let body = &phys.world.bodies[foot_ref.0];
        *transform = foot_render_transform(body);
    }

    // Compute renders for every chain node in topological order (foot first,
    // then each toe segment from proximal to distal). Cache (orient, pos)
    // by chain index so children can read parent's render.
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
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let lo = &phys.world.bodies[phys.lower_idx];
    let fo = &phys.world.bodies[phys.foot_idx];
    let s = (fo.orientation * lo.orientation.inverse()).scaled_axis();
    let (pf_t, inv_t, phase) = ankle_targets(phys.sim_time);
    let toe_t = toe_target(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  PF:    target {:+.2}  →  {:+.2} ({:+.0}°)\n  INV:   target {:+.2}  →  {:+.2} ({:+.0}°)\n  twist: {:+.2} ({:+.0}°)\n\n  toe drive: {:+.2}   ({})",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            pf_t, s.x, s.x.to_degrees(),
            inv_t, s.z, s.z.to_degrees(),
            s.y, s.y.to_degrees(),
            toe_t,
            if toe_t > 0.05 { "curl" } else if toe_t < -0.05 { "extend" } else { "neutral" },
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Ankle demo (foot + 5 toes, 2-DOF biomech ROM)".into(),
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
