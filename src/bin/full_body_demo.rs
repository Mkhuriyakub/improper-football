// Full skeleton assembly: pelvis (pinned) + spine (lumbar + cervical) +
// head + bilateral arms (anchored to thoracic) + bilateral legs (anchored
// to pelvis). 71 bodies total: 3 spine + 17 per arm × 2 + 17 per leg × 2.
//
// This is the first time every Phase-1 subsystem is wired into a single
// connected body. Pelvis is still pinned (no ground contact yet); the
// rest of the skeleton hangs/stands off it.
//
// Schedule (60 s, looping) — kept very simple for this first integration:
//
//   0–4 s     rest                     verify everything is stable upright
//   4–12 s    lumbar trunk flex/ext    sagittal-plane spine motion
//   12–20 s   bilateral shoulder flex  both arms forward and down
//   20–28 s   bilateral hip flex       both legs forward (knees toward chest)
//   28–40 s   coordinated toe-touch    lumbar flex + arm forward + knee bend
//   40–60 s   relax to rest

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
use improper_football::render_util::{rigid_body_to_transform, nalgebra_vec3_to_bevy, nalgebra_quat_to_bevy};

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ═════════════════════════════════════════════════════════════════════
// SPINE constants
// ═════════════════════════════════════════════════════════════════════

const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;
const PELVIS_REST_POS: NaVec3 = NaVec3::new(0.0, 1.00, 0.0);

const THORACIC_MASS:    f64 = 25.0;
const THORACIC_RADIUS:  f64 = 0.130;
const THORACIC_HALF_H:  f64 = 0.180;

const HEAD_MASS:    f64 = 5.0;
const HEAD_RADIUS:  f64 = 0.080;
const HEAD_HALF_H:  f64 = 0.050;

const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =     600.0;
const L_FLEX_K: f64 = 150.0;  const L_FLEX_C: f64 = 25.0;
const L_BEND_K: f64 = 150.0;  const L_BEND_C: f64 = 25.0;
const L_ROT_K:  f64 =  50.0;  const L_ROT_C:  f64 =  5.0;
const L_FLEX_MIN: f64 = -0.44;   const L_FLEX_MAX: f64 = 1.05;
const L_BEND_MIN: f64 = -0.44;   const L_BEND_MAX: f64 = 0.44;
const L_ROT_MIN:  f64 = -0.087;  const L_ROT_MAX:  f64 = 0.087;
const LUMBAR_LIMIT_K:   f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 =   2.0;
const LUMBAR_PD_KP: f64 = 1500.0;
const LUMBAR_PD_KD: f64 =  100.0;

const CERV_JOINT_K: f64 = 80_000.0;
const CERV_JOINT_C: f64 =    200.0;
const C_FLEX_K: f64 = 5.0;  const C_FLEX_C: f64 = 0.60;
const C_BEND_K: f64 = 5.0;  const C_BEND_C: f64 = 0.60;
const C_ROT_K:  f64 = 3.0;  const C_ROT_C:  f64 = 0.50;
const C_FLEX_MIN: f64 = -1.05;   const C_FLEX_MAX: f64 = 0.87;
const C_BEND_MIN: f64 = -0.78;   const C_BEND_MAX: f64 = 0.78;
const C_ROT_MIN:  f64 = -1.40;   const C_ROT_MAX:  f64 = 1.40;
const CERV_LIMIT_K:   f64 = 80.0;
const CERV_LIMIT_EXP: f64 =  2.0;
const CERV_PD_KP: f64 = 60.0;
const CERV_PD_KD: f64 =  3.0;

// Shoulder anchor on the thoracic. +x for right side, -x for left. Y at
// the top of the thoracic body so the shoulders sit at the C7-T1 height.
const SHOULDER_ANCHOR_THORACIC_X: f64 = 0.15;
const SHOULDER_ANCHOR_THORACIC_Y: f64 = THORACIC_HALF_H;

// ═════════════════════════════════════════════════════════════════════
// ARM constants (from arms_full_demo)
// ═════════════════════════════════════════════════════════════════════

const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

const FOREARM_MASS:    f64 = 1.2;
const FOREARM_RADIUS:  f64 = 0.038;
const FOREARM_HALF_H:  f64 = 0.135;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

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

const FLEX_K: f64 = 3.0;  const FLEX_C: f64 = 0.50;
const ABD_K:  f64 = 3.0;  const ABD_C:  f64 = 0.50;
const ROT_K:  f64 = 1.5;  const ROT_C:  f64 = 0.40;
const FLEX_MIN: f64 = -0.87;  const FLEX_MAX: f64 = 3.14;
const ABD_MIN:  f64 = -0.78;  const ABD_MAX:  f64 = 3.14;
const ROT_MIN:  f64 = -1.40;  const ROT_MAX:  f64 = 1.57;
const SHOULDER_LIMIT_K:   f64 = 100.0;
const SHOULDER_LIMIT_EXP: f64 =   2.0;

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

#[derive(Clone, Copy)]
struct JointBiomech {
    z_min: f64, z_max: f64,
    damping: f64, stiffness: f64, neutral: f64,
    limit_stiffness: f64,
    off_axis_k: f64,
    off_axis_c: f64,
}
const MCP: JointBiomech = JointBiomech {
    z_min: -0.52, z_max: 1.57, damping: 0.0004, stiffness: 0.05, neutral: 0.30,
    limit_stiffness: 0.5, off_axis_k: 0.05, off_axis_c: 0.0004,
};
const PIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 2.09, damping: 0.00013, stiffness: 0.03, neutral: 0.40,
    limit_stiffness: 0.4, off_axis_k: 0.04, off_axis_c: 0.0002,
};
const DIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 1.40, damping: 0.00006, stiffness: 0.02, neutral: 0.30,
    limit_stiffness: 0.3, off_axis_k: 0.03, off_axis_c: 0.0001,
};
const THUMB_MCP_R: JointBiomech = JointBiomech {
    z_min: -2.00, z_max: -0.40, damping: 0.0006, stiffness: 0.06, neutral: -1.45,
    limit_stiffness: 0.5, off_axis_k: 0.05, off_axis_c: 0.0004,
};
const THUMB_IP_R: JointBiomech = JointBiomech {
    z_min: -0.10, z_max: 1.30, damping: 0.00015, stiffness: 0.04, neutral: 0.10,
    limit_stiffness: 0.4, off_axis_k: 0.04, off_axis_c: 0.0002,
};

const SHOULDER_PD_KP: f64 = 120.0;
const SHOULDER_PD_KD: f64 =  14.0;
const ELBOW_PD_KP:    f64 = 100.0;
const ELBOW_PD_KD:    f64 =   3.5;
const WRIST_PD_KP:    f64 =  50.0;
const WRIST_PD_KD:    f64 =   1.0;
const MCP_PD_KP: f64 = 2.0;     const MCP_PD_KD: f64 = 0.004;
const PIP_PD_KP: f64 = 0.5;     const PIP_PD_KD: f64 = 0.001;
const DIP_PD_KP: f64 = 0.2;     const DIP_PD_KD: f64 = 0.0004;

const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION_FINGER: f64 =
    TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);
const CONVERGENCE_FACTOR: f64 = 8.0;

#[derive(Clone, Copy)]
struct FingerSpec {
    palm_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    mid_mass: f64,  mid_radius: f64,  mid_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
const INDEX: FingerSpec = FingerSpec {
    palm_offset_x: -0.035,
    prox_mass: 0.008, prox_radius: 0.0080, prox_half_h: 0.022,
    mid_mass:  0.005, mid_radius:  0.0072, mid_half_h:  0.012,
    dist_mass: 0.003, dist_radius: 0.0065, dist_half_h: 0.0085,
};
const MIDDLE: FingerSpec = FingerSpec {
    palm_offset_x: -0.012,
    prox_mass: 0.010, prox_radius: 0.0090, prox_half_h: 0.026,
    mid_mass:  0.007, mid_radius:  0.0080, mid_half_h:  0.014,
    dist_mass: 0.004, dist_radius: 0.0070, dist_half_h: 0.0085,
};
const RING: FingerSpec = FingerSpec {
    palm_offset_x:  0.012,
    prox_mass: 0.009, prox_radius: 0.0085, prox_half_h: 0.023,
    mid_mass:  0.006, mid_radius:  0.0075, mid_half_h:  0.0135,
    dist_mass: 0.0035, dist_radius: 0.0067, dist_half_h: 0.0085,
};
const LITTLE: FingerSpec = FingerSpec {
    palm_offset_x:  0.035,
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

// ═════════════════════════════════════════════════════════════════════
// LEG constants (from bilateral_legs_demo)
// ═════════════════════════════════════════════════════════════════════

const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

const THIGH_MASS:   f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =     400.0;
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K:   f64 = 120.0;
const HIP_LIMIT_EXP: f64 =   2.0;
const HIP_PD_KP: f64 = 800.0;
const HIP_PD_KD: f64 =  80.0;

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
const KNEE_PD_KP: f64 = 300.0;
const KNEE_PD_KD: f64 =  20.0;

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
struct ToeSpec {
    foot_offset_x: f64,
    prox_mass: f64, prox_radius: f64, prox_half_h: f64,
    mid_mass: f64,  mid_radius: f64,  mid_half_h: f64,
    dist_mass: f64, dist_radius: f64, dist_half_h: f64,
}
const HALLUX_R: HalluxSpec = HalluxSpec {
    foot_offset_x: -0.024,
    prox_mass: 0.020, prox_radius: 0.013, prox_half_h: 0.018,
    dist_mass: 0.012, dist_radius: 0.011, dist_half_h: 0.012,
};
const TOE2_R: ToeSpec = ToeSpec {
    foot_offset_x: -0.010,
    prox_mass: 0.012, prox_radius: 0.011, prox_half_h: 0.014,
    mid_mass:  0.005, mid_radius:  0.0095, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.008, dist_half_h: 0.005,
};
const TOE3_R: ToeSpec = ToeSpec {
    foot_offset_x: 0.003,
    prox_mass: 0.011, prox_radius: 0.010, prox_half_h: 0.013,
    mid_mass:  0.005, mid_radius:  0.0090, mid_half_h: 0.005,
    dist_mass: 0.003, dist_radius: 0.0075, dist_half_h: 0.005,
};
const TOE4_R: ToeSpec = ToeSpec {
    foot_offset_x: 0.014,
    prox_mass: 0.009, prox_radius: 0.0095, prox_half_h: 0.011,
    mid_mass:  0.004, mid_radius:  0.0085, mid_half_h: 0.005,
    dist_mass: 0.002, dist_radius: 0.007,  dist_half_h: 0.004,
};
const TOE5_R: ToeSpec = ToeSpec {
    foot_offset_x: 0.024,
    prox_mass: 0.006, prox_radius: 0.008,  prox_half_h: 0.009,
    mid_mass:  0.003, mid_radius:  0.0075, mid_half_h: 0.004,
    dist_mass: 0.002, dist_radius: 0.0065, dist_half_h: 0.004,
};

const SCHEDULE_PERIOD: f64 = 84.0;

// ═════════════════════════════════════════════════════════════════════
// Side mirror
// ═════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq)]
enum Side { Right, Left }
impl Side {
    fn sign(self) -> f64 { match self { Side::Right => 1.0, Side::Left => -1.0 } }
}

fn mirror_hallux(side: Side, base: &HalluxSpec) -> HalluxSpec {
    HalluxSpec {
        foot_offset_x: if side == Side::Right { base.foot_offset_x } else { -base.foot_offset_x },
        prox_mass: base.prox_mass, prox_radius: base.prox_radius, prox_half_h: base.prox_half_h,
        dist_mass: base.dist_mass, dist_radius: base.dist_radius, dist_half_h: base.dist_half_h,
    }
}
fn mirror_toe(side: Side, base: &ToeSpec) -> ToeSpec {
    ToeSpec {
        foot_offset_x: if side == Side::Right { base.foot_offset_x } else { -base.foot_offset_x },
        prox_mass: base.prox_mass, prox_radius: base.prox_radius, prox_half_h: base.prox_half_h,
        mid_mass: base.mid_mass, mid_radius: base.mid_radius, mid_half_h: base.mid_half_h,
        dist_mass: base.dist_mass, dist_radius: base.dist_radius, dist_half_h: base.dist_half_h,
    }
}

// ═════════════════════════════════════════════════════════════════════
// Common helpers
// ═════════════════════════════════════════════════════════════════════

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
    world: &mut XpbdWorld, parent: usize, child: usize,
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
    world: &mut XpbdWorld, parent: usize, child: usize,
    axis_local_a: NaVec3, min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64, neutral: f64,
    limit_stiffness: f64, limit_exp: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a, min_angle, max_angle, limit_stiffness, limit_exp));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

// ═════════════════════════════════════════════════════════════════════
// Spine builders
// ═════════════════════════════════════════════════════════════════════

struct SpineHandles {
    thoracic_idx: usize,
    head_idx: usize,
    pd_l_flex: usize, pd_l_bend: usize, pd_l_rot: usize,
    pd_c_flex: usize, pd_c_bend: usize, pd_c_rot: usize,
}

fn add_spine(world: &mut XpbdWorld, pelvis_idx: usize) -> SpineHandles {
    let lumbar_anchor_pelvis = NaVec3::new(0.0, PELVIS_HALF_H, 0.0);
    let mut thoracic = RigidBody::new(THORACIC_MASS,
        capsule_inertia(THORACIC_MASS, THORACIC_RADIUS, THORACIC_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis,
        NaVec3::new(0.0, -THORACIC_HALF_H, 0.0),
        UnitQuaternion::identity());
    thoracic.position = pos; thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis, NaVec3::new(0.0, -THORACIC_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let l_flex_axis = NaVec3::new(1.0, 0.0, 0.0);
    let l_bend_axis = NaVec3::new(0.0, 0.0, -1.0);
    let l_rot_axis  = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(world, pelvis_idx, thoracic_idx, l_flex_axis,
        L_FLEX_MIN, L_FLEX_MAX, L_FLEX_K, L_FLEX_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);
    add_axis(world, pelvis_idx, thoracic_idx, l_bend_axis,
        L_BEND_MIN, L_BEND_MAX, L_BEND_K, L_BEND_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);
    add_axis(world, pelvis_idx, thoracic_idx, l_rot_axis,
        L_ROT_MIN, L_ROT_MAX, L_ROT_K, L_ROT_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);

    let pd_l_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_flex_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));
    let pd_l_bend = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_bend_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));
    let pd_l_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_rot_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));

    // Head + cervical joint.
    let cerv_anchor_thoracic = NaVec3::new(0.0, THORACIC_HALF_H, 0.0);
    let mut head = RigidBody::new(HEAD_MASS,
        capsule_inertia(HEAD_MASS, HEAD_RADIUS, HEAD_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        cerv_anchor_thoracic,
        NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        UnitQuaternion::identity());
    head.position = pos; head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(world, thoracic_idx, head_idx,
        cerv_anchor_thoracic, NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);
    let c_flex_axis = NaVec3::new(1.0, 0.0, 0.0);
    let c_bend_axis = NaVec3::new(0.0, 0.0, -1.0);
    let c_rot_axis  = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(world, thoracic_idx, head_idx, c_flex_axis,
        C_FLEX_MIN, C_FLEX_MAX, C_FLEX_K, C_FLEX_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);
    add_axis(world, thoracic_idx, head_idx, c_bend_axis,
        C_BEND_MIN, C_BEND_MAX, C_BEND_K, C_BEND_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);
    add_axis(world, thoracic_idx, head_idx, c_rot_axis,
        C_ROT_MIN, C_ROT_MAX, C_ROT_K, C_ROT_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);

    let pd_c_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_flex_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    let pd_c_bend = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_bend_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    let pd_c_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_rot_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));

    SpineHandles { thoracic_idx, head_idx,
        pd_l_flex, pd_l_bend, pd_l_rot,
        pd_c_flex, pd_c_bend, pd_c_rot }
}

// ═════════════════════════════════════════════════════════════════════
// Arm builders (from arms_full_demo)
// ═════════════════════════════════════════════════════════════════════

struct FingerHandles {
    prox_idx: usize, mid_idx: usize, dist_idx: usize,
}
struct ThumbHandles {
    prox_idx: usize, dist_idx: usize,
}
struct ArmHandles {
    upper_idx: usize, forearm_idx: usize, palm_idx: usize,
    fingers: Vec<FingerHandles>,
    thumb: ThumbHandles,
    pd_s_flex: usize, pd_s_abd: usize, pd_s_rot: usize,
    pd_elbow: usize,
    pd_w_flex: usize, pd_w_dev: usize,
}

fn add_finger_biomech(
    world: &mut XpbdWorld, parent: usize, child: usize, b: &JointBiomech,
    curl_axis: NaVec3, off1: NaVec3, off2: NaVec3,
) {
    add_axis(world, parent, child, curl_axis,
        b.z_min, b.z_max, b.stiffness, b.damping, b.neutral, b.limit_stiffness, 2.0);
    add_axis(world, parent, child, off1, -0.10, 0.10, b.off_axis_k, b.off_axis_c, 0.0, 0.3, 2.0);
    add_axis(world, parent, child, off2, -0.10, 0.10, b.off_axis_k, b.off_axis_c, 0.0, 0.3, 2.0);
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
        TENDON_REST_EXCURSION_FINGER, TENDON_STIFFNESS, TENDON_DAMPING));

    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, curl_axis, MCP_PD_KP, MCP_PD_KD)
            .with_target(MCP.neutral));
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, mid_idx, curl_axis, PIP_PD_KP, PIP_PD_KD)
            .with_target(PIP.neutral));
    world.angular_pd_controllers.push(
        AngularPDController::new(mid_idx, dist_idx, curl_axis, DIP_PD_KP, DIP_PD_KD)
            .with_target(DIP.neutral));

    FingerHandles { prox_idx, mid_idx, dist_idx }
}

fn add_thumb(
    world: &mut XpbdWorld, palm_idx: usize, spec: ThumbSpec,
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
        rest_excursion, TENDON_STIFFNESS, TENDON_DAMPING));

    world.angular_pd_controllers.push(
        AngularPDController::new(palm_idx, prox_idx, thumb_curl, MCP_PD_KP, MCP_PD_KD)
            .with_target(mcp_biomech.neutral));
    world.angular_pd_controllers.push(
        AngularPDController::new(prox_idx, dist_idx, thumb_curl, PIP_PD_KP, PIP_PD_KD)
            .with_target(ip_biomech.neutral));

    ThumbHandles { prox_idx, dist_idx }
}

fn add_arm(world: &mut XpbdWorld, thoracic_idx: usize, side: Side) -> ArmHandles {
    let s = side.sign();
    let shoulder_anchor = NaVec3::new(s * SHOULDER_ANCHOR_THORACIC_X, SHOULDER_ANCHOR_THORACIC_Y, 0.0);

    let mut upper = RigidBody::new(UPPER_ARM_MASS,
        capsule_inertia(UPPER_ARM_MASS, UPPER_ARM_RADIUS, UPPER_ARM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        shoulder_anchor, NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        UnitQuaternion::identity());
    upper.position = pos; upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(world, thoracic_idx, upper_idx,
        shoulder_anchor, NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);

    let s_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let s_abd_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let s_rot_axis  = NaVec3::new( 0.0, 1.0, 0.0);
    let (abd_min_a, abd_max_a) = if s > 0.0 { (ABD_MIN, ABD_MAX) } else { (-ABD_MAX, -ABD_MIN) };
    let (rot_min_a, rot_max_a) = if s > 0.0 { (ROT_MIN, ROT_MAX) } else { (-ROT_MAX, -ROT_MIN) };
    add_axis(world, thoracic_idx, upper_idx, s_flex_axis, FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP);
    add_axis(world, thoracic_idx, upper_idx, s_abd_axis,  abd_min_a, abd_max_a, ABD_K, ABD_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP);
    add_axis(world, thoracic_idx, upper_idx, s_rot_axis,  rot_min_a, rot_max_a, ROT_K, ROT_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP);

    let pd_s_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, upper_idx, s_flex_axis, SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));
    let pd_s_abd = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, upper_idx, s_abd_axis, SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));
    let pd_s_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, upper_idx, s_rot_axis, SHOULDER_PD_KP, SHOULDER_PD_KD).with_target(0.0));

    let mut forearm = RigidBody::new(FOREARM_MASS,
        capsule_inertia(FOREARM_MASS, FOREARM_RADIUS, FOREARM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[upper_idx],
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
        ELBOW_MIN, ELBOW_MAX, ELBOW_K, ELBOW_C, ELBOW_NEUTRAL, ELBOW_LIMIT_K, 2.0);
    add_axis(world, upper_idx, forearm_idx, NaVec3::z(),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0, ELBOW_OFF_AXIS_LIMIT_K, 2.0);
    add_axis(world, upper_idx, forearm_idx, NaVec3::y(),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0, ELBOW_OFF_AXIS_LIMIT_K, 2.0);
    let pd_elbow = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(upper_idx, forearm_idx, elbow_axis, ELBOW_PD_KP, ELBOW_PD_KD).with_target(ELBOW_NEUTRAL));

    let mut palm = RigidBody::new(PALM_MASS,
        capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[forearm_idx],
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
    let (dev_min_a, dev_max_a) = if s > 0.0 { (WRIST_DEV_MIN, WRIST_DEV_MAX) } else { (-WRIST_DEV_MAX, -WRIST_DEV_MIN) };
    add_axis(world, forearm_idx, palm_idx, w_flex_axis,
        WRIST_FLEX_MIN, WRIST_FLEX_MAX, WRIST_FLEX_K, WRIST_FLEX_C, 0.0, WRIST_LIMIT_K, 2.0);
    add_axis(world, forearm_idx, palm_idx, w_dev_axis,
        dev_min_a, dev_max_a, WRIST_DEV_K, WRIST_DEV_C, 0.0, WRIST_LIMIT_K, 2.0);
    add_axis(world, forearm_idx, palm_idx, w_twist_axis,
        -WRIST_TWIST_LIM, WRIST_TWIST_LIM, WRIST_TWIST_K, WRIST_TWIST_C, 0.0, WRIST_TWIST_LIMIT_K, 2.0);
    let pd_w_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, w_flex_axis, WRIST_PD_KP, WRIST_PD_KD).with_target(0.0));
    let pd_w_dev = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, w_dev_axis, WRIST_PD_KP, WRIST_PD_KD).with_target(0.0));

    let mirror_finger = |spec: FingerSpec| FingerSpec {
        palm_offset_x: s * spec.palm_offset_x, ..spec
    };
    let fingers = vec![
        add_finger(world, palm_idx, mirror_finger(INDEX)),
        add_finger(world, palm_idx, mirror_finger(MIDDLE)),
        add_finger(world, palm_idx, mirror_finger(RING)),
        add_finger(world, palm_idx, mirror_finger(LITTLE)),
    ];
    let thumb_spec = ThumbSpec { palm_anchor_x: s * THUMB.palm_anchor_x, ..THUMB };
    let mcp_biomech = JointBiomech {
        z_min: if s > 0.0 { THUMB_MCP_R.z_min } else { -THUMB_MCP_R.z_max },
        z_max: if s > 0.0 { THUMB_MCP_R.z_max } else { -THUMB_MCP_R.z_min },
        neutral: s * THUMB_MCP_R.neutral,
        ..THUMB_MCP_R
    };
    let ip_biomech = JointBiomech {
        z_min: if s > 0.0 { THUMB_IP_R.z_min } else { -THUMB_IP_R.z_max },
        z_max: if s > 0.0 { THUMB_IP_R.z_max } else { -THUMB_IP_R.z_min },
        neutral: s * THUMB_IP_R.neutral,
        ..THUMB_IP_R
    };
    let thumb = add_thumb(world, palm_idx, thumb_spec, &mcp_biomech, &ip_biomech);

    ArmHandles {
        upper_idx, forearm_idx, palm_idx, fingers, thumb,
        pd_s_flex, pd_s_abd, pd_s_rot,
        pd_elbow, pd_w_flex, pd_w_dev,
    }
}

// ═════════════════════════════════════════════════════════════════════
// Leg builders (from bilateral_legs_demo)
// ═════════════════════════════════════════════════════════════════════

fn add_toe_joint(world: &mut XpbdWorld, parent: usize, child: usize, j: &ToeJoint) {
    add_axis(world, parent, child, NaVec3::x(),
        j.flex_min, j.flex_max, j.spring, j.damping, j.neutral, j.limit_k, 2.0);
    add_axis(world, parent, child, NaVec3::y(),
        -0.10, 0.10, j.off_axis_k, j.off_axis_c, 0.0, 0.3, 2.0);
    add_axis(world, parent, child, NaVec3::z(),
        -0.15, 0.15, j.off_axis_k, j.off_axis_c, 0.0, 0.3, 2.0);
}

#[derive(Clone)]
struct ToeChainEntry {
    body_idx: usize,
    parent_chain_idx: Option<usize>,
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

struct LegHandles {
    thigh_idx: usize, shin_idx: usize, foot_idx: usize,
    pd_hip_flex: usize, pd_hip_abd: usize, pd_hip_rot: usize,
    pd_knee: usize,
    pd_ankle_pf: usize, pd_ankle_inv: usize,
    toe_pds: Vec<(usize, f64)>,
    chain_nodes: Vec<ToeChainEntry>,
}

fn add_leg(world: &mut XpbdWorld, pelvis_idx: usize, side: Side) -> LegHandles {
    let s = side.sign();
    let hip_anchor = NaVec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0);

    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        UnitQuaternion::identity());
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(world, pelvis_idx, thigh_idx,
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let hip_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let hip_abd_axis  = NaVec3::new( 0.0, 0.0,  s);
    let hip_rot_axis  = NaVec3::new( 0.0,  s, 0.0);
    add_axis(world, pelvis_idx, thigh_idx, hip_flex_axis,
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP);
    add_axis(world, pelvis_idx, thigh_idx, hip_abd_axis,
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP);
    add_axis(world, pelvis_idx, thigh_idx, hip_rot_axis,
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

    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        UnitQuaternion::identity());
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = NaVec3::new(1.0, 0.0, 0.0);
    add_axis(world, thigh_idx, shin_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL,
        KNEE_LIMIT_K, KNEE_LIMIT_EXP);
    add_axis(world, thigh_idx, shin_idx, NaVec3::z(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP);
    add_axis(world, thigh_idx, shin_idx, NaVec3::y(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP);
    let pd_knee = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, shin_idx, knee_axis,
            KNEE_PD_KP, KNEE_PD_KD).with_target(KNEE_NEUTRAL));

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        UnitQuaternion::identity());
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis    = NaVec3::new(1.0, 0.0, 0.0);
    let inv_axis   = NaVec3::new(0.0, 0.0,  s);
    let twist_axis = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(world, shin_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP);
    add_axis(world, shin_idx, foot_idx, inv_axis,
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP);
    add_axis(world, shin_idx, foot_idx, twist_axis,
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0, TWIST_LIMIT_K, ANKLE_LIMIT_EXP);
    let pd_ankle_pf = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, pf_axis,
            ANKLE_PD_KP, ANKLE_PD_KD).with_target(0.0));
    let pd_ankle_inv = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, inv_axis,
            ANKLE_PD_KP, ANKLE_PD_KD).with_target(0.0));

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
    let hallux = mirror_hallux(side, &HALLUX_R);
    let toe2 = mirror_toe(side, &TOE2_R);
    let toe3 = mirror_toe(side, &TOE3_R);
    let toe4 = mirror_toe(side, &TOE4_R);
    let toe5 = mirror_toe(side, &TOE5_R);
    push_chain(&mut chain_nodes, add_hallux(world, foot_idx, &hallux, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(world, foot_idx, &toe2, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(world, foot_idx, &toe3, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(world, foot_idx, &toe4, &mut toe_pds).into_iter().collect());
    push_chain(&mut chain_nodes, add_toe(world, foot_idx, &toe5, &mut toe_pds).into_iter().collect());

    LegHandles {
        thigh_idx, shin_idx, foot_idx,
        pd_hip_flex, pd_hip_abd, pd_hip_rot,
        pd_knee, pd_ankle_pf, pd_ankle_inv,
        toe_pds, chain_nodes,
    }
}

// ═════════════════════════════════════════════════════════════════════
// Schedule
// ═════════════════════════════════════════════════════════════════════

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

/// Smooth ramp-up to `peak`, hold, then ramp back to 0. Total duration =
/// `ramp + hold + ramp`. Velocity is zero at all four corners (start, end
/// of ramp-up, start of ramp-down, end). Useful for poses that need to be
/// visibly "held" for a moment before releasing.
fn ramp_hold_ramp(t: f64, ramp: f64, hold: f64, peak: f64) -> f64 {
    if t < 0.0 {
        0.0
    } else if t < ramp {
        peak * 0.5 * (1.0 - (std::f64::consts::PI * t / ramp).cos())
    } else if t < ramp + hold {
        peak
    } else if t < 2.0 * ramp + hold {
        let down = t - ramp - hold;
        peak * 0.5 * (1.0 + (std::f64::consts::PI * down / ramp).cos())
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Default)]
struct SpineTargets {
    l_flex: f64, l_bend: f64, l_rot: f64,
    c_flex: f64, c_bend: f64, c_rot: f64,
}
#[derive(Clone, Copy, Default)]
struct ArmTargets {
    s_flex: f64, s_abd: f64, s_rot: f64,
    elbow: f64,
    w_flex: f64, w_dev: f64,
}
#[derive(Clone, Copy, Default)]
struct LegTargets {
    hip_flex: f64, hip_abd: f64, hip_rot: f64,
    knee: f64,
    ankle_pf: f64, ankle_inv: f64,
    toe: f64,
}

struct AllTargets {
    spine: SpineTargets,
    arm_r: ArmTargets,
    arm_l: ArmTargets,
    leg_r: LegTargets,
    leg_l: LegTargets,
    phase: &'static str,
}

fn schedule(t: f64) -> AllTargets {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    let mut tg = AllTargets {
        spine: SpineTargets::default(),
        arm_r: ArmTargets::default(),
        arm_l: ArmTargets::default(),
        leg_r: LegTargets::default(),
        leg_l: LegTargets::default(),
        phase: "rest",
    };

    if t < 3.0 {
        tg.phase = "rest (verify stable upright)";
    } else if t < 9.0 {
        // Lumbar trunk flex (forward bow), return.
        let local = t - 3.0;
        tg.spine.l_flex = half_cosine(local, 6.0, 0.70);
        tg.phase = "spine: lumbar forward flexion";
    } else if t < 15.0 {
        // Lumbar lateral bend right, then left.
        let local = t - 9.0;
        let bend = if local < 3.0 {
            half_cosine(local, 3.0, 0.35)
        } else {
            half_cosine(local - 3.0, 3.0, -0.35)
        };
        tg.spine.l_bend = bend;
        tg.phase = "spine: lumbar lateral bend (R then L)";
    } else if t < 21.0 {
        // Cervical head turn — look right, then left.
        let local = t - 15.0;
        let rot = if local < 3.0 {
            half_cosine(local, 3.0, 1.00)
        } else {
            half_cosine(local - 3.0, 3.0, -1.00)
        };
        tg.spine.c_rot = rot;
        tg.phase = "neck: cervical rotation (look R then L)";
    } else if t < 27.0 {
        // Football TOUCHDOWN signal: both arms shot straight up via forward
        // flexion to ~3.00 rad (~172°, just under the FLEX_MAX soft limit
        // of 3.14), held for 2 s in that vertical pose, then released. The
        // ramp-hold-ramp envelope makes the held pose clearly visible.
        let local = t - 21.0;
        let v = ramp_hold_ramp(local, 2.0, 2.0, 3.00);
        tg.arm_r.s_flex = v;
        tg.arm_l.s_flex = v;
        tg.phase = "arms: TOUCHDOWN signal (bilateral overhead)";
    } else if t < 33.0 {
        // Both arms abducted laterally (jumping-jack style overhead).
        let local = t - 27.0;
        let v = half_cosine(local, 6.0, 2.50);
        tg.arm_r.s_abd = v;
        tg.arm_l.s_abd = v;
        tg.phase = "arms: bilateral abduction (overhead)";
    } else if t < 38.0 {
        // Both elbows flex (biceps curl pose).
        let local = t - 33.0;
        let v = half_cosine(local, 5.0, 1.80);
        tg.arm_r.elbow = v;
        tg.arm_l.elbow = v;
        tg.phase = "arms: bilateral elbow flexion (curl)";
    } else if t < 44.0 {
        // Both legs forward (knees toward chest).
        let local = t - 38.0;
        let v = half_cosine(local, 6.0, 1.20);
        tg.leg_r.hip_flex = v;
        tg.leg_l.hip_flex = v;
        tg.phase = "legs: bilateral hip flexion";
    } else if t < 49.0 {
        // Both legs abducted laterally (jumping-jack legs).
        let local = t - 44.0;
        let v = half_cosine(local, 5.0, 0.55);
        tg.leg_r.hip_abd = v;
        tg.leg_l.hip_abd = v;
        tg.phase = "legs: bilateral hip abduction";
    } else if t < 54.0 {
        // Both knees flex (with slight hip flex so the knees don't punch
        // through the floor — knee-only with extended hip would put the
        // foot above the ground but visually it's clearer with a small
        // hip lift).
        let local = t - 49.0;
        let v = half_cosine(local, 5.0, 1.80);
        let hip = half_cosine(local, 5.0, 0.30);
        tg.leg_r.knee = v;
        tg.leg_l.knee = v;
        tg.leg_r.hip_flex = hip;
        tg.leg_l.hip_flex = hip;
        tg.phase = "legs: bilateral knee flexion";
    } else if t < 60.0 {
        // Both feet plantarflex (rise to ball of foot).
        let local = t - 54.0;
        let v = half_cosine(local, 6.0, 0.65);
        tg.leg_r.ankle_pf = v;
        tg.leg_l.ankle_pf = v;
        tg.phase = "ankles: bilateral plantarflex (toe-rise)";
    } else if t < 72.0 {
        // Toe-touch (with pinned pelvis, the feet have to come UP to meet
        // the hands rather than the hands going down to the feet — since
        // the pelvis can't tilt forward, a real toe-touch isn't possible).
        // We get a "tucked" reach: deep hip + knee flex folds the legs so
        // the feet rise to roughly arm's-reach height, trunk bows forward
        // to its max, arms hang vertically from the bent trunk, head tucks.
        let local = t - 60.0;
        let env = half_cosine(local, 12.0, 1.0);
        let trunk_angle = 1.05 * env;
        tg.spine.l_flex = trunk_angle;
        tg.spine.c_flex = 0.55 * env;
        tg.arm_r.s_flex = trunk_angle;       // arm = trunk angle → hangs vertical in world
        tg.arm_l.s_flex = trunk_angle;
        tg.leg_r.hip_flex = 1.80 * env;      // thighs ~horizontal forward
        tg.leg_l.hip_flex = 1.80 * env;
        tg.leg_r.knee     = 2.20 * env;      // shins fold back against thighs
        tg.leg_l.knee     = 2.20 * env;
        tg.phase = "coordinated: tucked toe-touch (legs fold up to meet hands)";
    } else if t < 78.0 {
        // Coordinated reach overhead. Two safety adjustments vs. an
        // earlier version that destabilized the thoracic:
        //   1. Abduction target capped at 2.80 rad (~160°), well below
        //      the 3.14 soft limit — driving right at the limit pumped
        //      strong penalty torques back through the joint chain.
        //   2. Rotation target reduced to 0.80 rad (~46°) and timed to
        //      ramp in AFTER the abduction, not simultaneously. With
        //      both axes at extreme angles together, the angular
        //      bivector artifact (project_axis_angle_artifacts) feeds
        //      fake torques into the unused flex axis that the PD
        //      fights — destabilizing the chain.
        let local = t - 72.0;
        let env = half_cosine(local, 6.0, 1.0);
        tg.spine.l_flex = -0.15 * env;
        tg.arm_r.s_abd = 2.80 * env;
        tg.arm_l.s_abd = 2.80 * env;
        // Rotation envelope: rises later than abd so they don't peak
        // simultaneously. EXTERNAL rotation (positive) brings palms
        // together overhead per Codman's paradox.
        let rot_env = ((local - 1.5).max(0.0) / 4.5).clamp(0.0, 1.0);
        let rot_smooth = 0.5 - 0.5 * (std::f64::consts::TAU * rot_env).cos();
        tg.arm_r.s_rot = 0.80 * rot_smooth;
        tg.arm_l.s_rot = 0.80 * rot_smooth;
        tg.leg_r.ankle_pf = 0.40 * env;
        tg.leg_l.ankle_pf = 0.40 * env;
        tg.phase = "coordinated: reach overhead (palms inward)";
    } else {
        // Hold at zero (relax to rest before the loop restarts).
        tg.phase = "relax to rest";
    }

    tg
}

// ═════════════════════════════════════════════════════════════════════
// ECS resources / setup
// ═════════════════════════════════════════════════════════════════════

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    pelvis_idx: usize,
    spine: SpineHandles,
    arm_r: ArmHandles,
    arm_l: ArmHandles,
    leg_r: LegHandles,
    leg_l: LegHandles,
    sim_time: f64,
}

#[derive(Resource)]
struct FootChain {
    foot_idx: usize,
    nodes: Vec<ToeChainEntry>,
}
#[derive(Resource)]
struct FootChains { right: FootChain, left: FootChain }

#[derive(Component)] struct BodyRef(usize);
#[derive(Component)] struct FootBodyRef { idx: usize }
#[derive(Component)] struct ToeBodyRef { chain_idx: usize, side: Side }
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

    let spine = add_spine(&mut world, pelvis_idx);
    let arm_r = add_arm(&mut world, spine.thoracic_idx, Side::Right);
    let arm_l = add_arm(&mut world, spine.thoracic_idx, Side::Left);
    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);

    // ─── Meshes & materials ──────────────────────────────────────────
    let pelvis_mesh   = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let thoracic_mesh = meshes.add(Capsule3d::new(THORACIC_RADIUS as f32, (2.0 * THORACIC_HALF_H) as f32));
    let head_mesh     = meshes.add(Capsule3d::new(HEAD_RADIUS as f32, (2.0 * HEAD_HALF_H) as f32));
    let upper_mesh    = meshes.add(Capsule3d::new(UPPER_ARM_RADIUS as f32, (2.0 * UPPER_ARM_HALF_H) as f32));
    let forearm_mesh  = meshes.add(Capsule3d::new(FOREARM_RADIUS as f32, (2.0 * FOREARM_HALF_H) as f32));
    let palm_mesh     = meshes.add(Capsule3d::new(PALM_RADIUS as f32, (2.0 * PALM_HALF_H) as f32));
    let thigh_mesh    = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let shin_mesh     = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let foot_mesh     = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));

    let core_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let torso_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });
    let head_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.92, 0.78, 0.65), ..default() });
    let arm_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let palm_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.75, 0.55, 0.42), ..default() });
    let finger_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.58, 0.45), ..default() });
    let leg_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let foot_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let toe_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.65, 0.45, 0.30), ..default() });

    // Spine bodies.
    commands.spawn((PbrBundle {
        mesh: pelvis_mesh, material: core_mat.clone(),
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default()
    }, BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle {
        mesh: thoracic_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[spine.thoracic_idx]), ..default()
    }, BodyRef(spine.thoracic_idx)));
    commands.spawn((PbrBundle {
        mesh: head_mesh, material: head_mat,
        transform: rigid_body_to_transform(&world.bodies[spine.head_idx]), ..default()
    }, BodyRef(spine.head_idx)));

    // Arms (both sides).
    for arm in [&arm_r, &arm_l] {
        commands.spawn((PbrBundle {
            mesh: upper_mesh.clone(), material: arm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.upper_idx]), ..default()
        }, BodyRef(arm.upper_idx)));
        commands.spawn((PbrBundle {
            mesh: forearm_mesh.clone(), material: arm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.forearm_idx]), ..default()
        }, BodyRef(arm.forearm_idx)));
        commands.spawn((PbrBundle {
            mesh: palm_mesh.clone(), material: palm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.palm_idx]), ..default()
        }, BodyRef(arm.palm_idx)));
        // Fingers + thumb.
        for f in &arm.fingers {
            for &(idx, r, h) in &[
                (f.prox_idx, 0.008, 0.022),
                (f.mid_idx,  0.0072, 0.012),
                (f.dist_idx, 0.0065, 0.0085),
            ] {
                let m = meshes.add(Capsule3d::new(r, 2.0 * h));
                commands.spawn((PbrBundle {
                    mesh: m, material: finger_mat.clone(),
                    transform: rigid_body_to_transform(&world.bodies[idx]), ..default()
                }, BodyRef(idx)));
            }
        }
        for &(idx, r, h) in &[
            (arm.thumb.prox_idx, 0.0095, 0.018),
            (arm.thumb.dist_idx, 0.0080, 0.0125),
        ] {
            let m = meshes.add(Capsule3d::new(r, 2.0 * h));
            commands.spawn((PbrBundle {
                mesh: m, material: finger_mat.clone(),
                transform: rigid_body_to_transform(&world.bodies[idx]), ..default()
            }, BodyRef(idx)));
        }
    }

    // Legs (both sides).
    for leg in [&leg_r, &leg_l] {
        commands.spawn((PbrBundle {
            mesh: thigh_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.thigh_idx]), ..default()
        }, BodyRef(leg.thigh_idx)));
        commands.spawn((PbrBundle {
            mesh: shin_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.shin_idx]), ..default()
        }, BodyRef(leg.shin_idx)));
        commands.spawn((PbrBundle {
            mesh: foot_mesh.clone(), material: foot_mat.clone(),
            transform: foot_render_transform(&world.bodies[leg.foot_idx]), ..default()
        }, FootBodyRef { idx: leg.foot_idx }));
    }

    // Toe meshes per side. Order must match push_chain order in add_leg.
    fn spawn_toe_meshes(
        commands: &mut Commands, meshes: &mut ResMut<Assets<Mesh>>,
        toe_mat: &Handle<StandardMaterial>, side: Side,
        h: &HalluxSpec, t2: &ToeSpec, t3: &ToeSpec, t4: &ToeSpec, t5: &ToeSpec,
    ) {
        let dims: Vec<(f32, f32)> = vec![
            (h.prox_radius as f32, h.prox_half_h as f32),
            (h.dist_radius as f32, h.dist_half_h as f32),
            (t2.prox_radius as f32, t2.prox_half_h as f32),
            (t2.mid_radius  as f32, t2.mid_half_h  as f32),
            (t2.dist_radius as f32, t2.dist_half_h as f32),
            (t3.prox_radius as f32, t3.prox_half_h as f32),
            (t3.mid_radius  as f32, t3.mid_half_h  as f32),
            (t3.dist_radius as f32, t3.dist_half_h as f32),
            (t4.prox_radius as f32, t4.prox_half_h as f32),
            (t4.mid_radius  as f32, t4.mid_half_h  as f32),
            (t4.dist_radius as f32, t4.dist_half_h as f32),
            (t5.prox_radius as f32, t5.prox_half_h as f32),
            (t5.mid_radius  as f32, t5.mid_half_h  as f32),
            (t5.dist_radius as f32, t5.dist_half_h as f32),
        ];
        for (i, (radius, half_h)) in dims.iter().enumerate() {
            let mesh = meshes.add(Capsule3d::new(*radius, 2.0 * half_h));
            commands.spawn((PbrBundle {
                mesh, material: toe_mat.clone(),
                transform: Transform::IDENTITY, ..default()
            }, ToeBodyRef { chain_idx: i, side }));
        }
    }
    let h_r = mirror_hallux(Side::Right, &HALLUX_R);
    let t2r = mirror_toe(Side::Right, &TOE2_R);
    let t3r = mirror_toe(Side::Right, &TOE3_R);
    let t4r = mirror_toe(Side::Right, &TOE4_R);
    let t5r = mirror_toe(Side::Right, &TOE5_R);
    spawn_toe_meshes(&mut commands, &mut meshes, &toe_mat, Side::Right, &h_r, &t2r, &t3r, &t4r, &t5r);
    let h_l = mirror_hallux(Side::Left, &HALLUX_R);
    let t2l = mirror_toe(Side::Left, &TOE2_R);
    let t3l = mirror_toe(Side::Left, &TOE3_R);
    let t4l = mirror_toe(Side::Left, &TOE4_R);
    let t5l = mirror_toe(Side::Left, &TOE5_R);
    spawn_toe_meshes(&mut commands, &mut meshes, &toe_mat, Side::Left, &h_l, &t2l, &t3l, &t4l, &t5l);

    let foot_chains = FootChains {
        right: FootChain { foot_idx: leg_r.foot_idx, nodes: leg_r.chain_nodes.clone() },
        left:  FootChain { foot_idx: leg_l.foot_idx, nodes: leg_l.chain_nodes.clone()  },
    };
    commands.insert_resource(foot_chains);

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera: original oblique 3/4 view from upper-right-front.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.0, 1.20, 2.0)
            .looking_at(Vec3::new(0.0, 1.00, 0.0), Vec3::Y),
        ..default()
    });

    // Shadows kept off for now — at 71 bodies the shadow pass can stall
    // the renderer. Re-enable later once perf headroom is confirmed.
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: false, ..default() },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });

    // HUD readout — current time within the schedule cycle and phase label.
    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 22.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));
    commands.insert_resource(AmbientLight {
        color: Color::WHITE, brightness: 220.0 });

    commands.insert_resource(PhysicsWorld {
        world, pelvis_idx, spine, arm_r, arm_l, leg_r, leg_l, sim_time: 0.0,
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

fn render_foot_subtree(world: &XpbdWorld, chain: &FootChain) -> Vec<(UnitQuaternion<f64>, NaVec3)> {
    let foot_body = &world.bodies[chain.foot_idx];
    let foot_offset = foot_visual_offset();
    let foot_render_orient = foot_body.orientation * foot_offset;
    let ankle_world = foot_body.position + foot_body.orientation * FOOT_ANCHOR_LOCAL;
    let foot_render_pos = ankle_world - foot_render_orient * FOOT_ANCHOR_LOCAL;

    let mut renders: Vec<(UnitQuaternion<f64>, NaVec3)> = Vec::with_capacity(chain.nodes.len());
    for node in &chain.nodes {
        let (parent_render_orient, parent_distal_world);
        match node.parent_chain_idx {
            None => {
                parent_render_orient = foot_render_orient;
                parent_distal_world = foot_render_pos
                    + foot_render_orient * node.parent_distal_local;
            }
            Some(pi) => {
                let (po, pp) = renders[pi];
                parent_render_orient = po;
                parent_distal_world = pp + po * node.parent_distal_local;
            }
        }
        let parent_body = match node.parent_chain_idx {
            None => foot_body,
            Some(pi) => &world.bodies[chain.nodes[pi].body_idx],
        };
        let child_body = &world.bodies[node.body_idx];
        let (orient, pos) = chain_render(
            parent_render_orient, parent_distal_world,
            parent_body, child_body, node.proximal_local);
        renders.push((orient, pos));
    }
    renders
}

fn apply_arm(world: &mut XpbdWorld, arm: &ArmHandles, side: Side, t: &ArmTargets) {
    let s = side.sign();
    world.angular_pd_controllers[arm.pd_s_flex].target_angle = t.s_flex;
    // Abd and rot mirror per side (stored in world axes, but biomech ranges
    // mirror — see add_arm). Targets here are in anatomical convention
    // (positive abd = lateral away from body), so apply side-sign.
    world.angular_pd_controllers[arm.pd_s_abd].target_angle = t.s_abd * s;
    world.angular_pd_controllers[arm.pd_s_rot].target_angle = t.s_rot * s;
    world.angular_pd_controllers[arm.pd_elbow].target_angle = ELBOW_NEUTRAL + t.elbow;
    world.angular_pd_controllers[arm.pd_w_flex].target_angle = t.w_flex;
    world.angular_pd_controllers[arm.pd_w_dev].target_angle = t.w_dev * s;
}

fn apply_leg(world: &mut XpbdWorld, leg: &LegHandles, t: &LegTargets) {
    world.angular_pd_controllers[leg.pd_hip_flex].target_angle = t.hip_flex;
    world.angular_pd_controllers[leg.pd_hip_abd].target_angle  = t.hip_abd;
    world.angular_pd_controllers[leg.pd_hip_rot].target_angle  = t.hip_rot;
    world.angular_pd_controllers[leg.pd_knee].target_angle     = KNEE_NEUTRAL + t.knee;
    world.angular_pd_controllers[leg.pd_ankle_pf].target_angle = t.ankle_pf;
    world.angular_pd_controllers[leg.pd_ankle_inv].target_angle = t.ankle_inv;
    for (idx, scale) in &leg.toe_pds {
        world.angular_pd_controllers[*idx].target_angle = t.toe * scale;
    }
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let tg = schedule(phys.sim_time);

    // Spine PDs.
    let spine = std::mem::replace(&mut phys.spine, SpineHandles {
        thoracic_idx: 0, head_idx: 0,
        pd_l_flex: 0, pd_l_bend: 0, pd_l_rot: 0,
        pd_c_flex: 0, pd_c_bend: 0, pd_c_rot: 0,
    });
    phys.world.angular_pd_controllers[spine.pd_l_flex].target_angle = tg.spine.l_flex;
    phys.world.angular_pd_controllers[spine.pd_l_bend].target_angle = tg.spine.l_bend;
    phys.world.angular_pd_controllers[spine.pd_l_rot].target_angle  = tg.spine.l_rot;
    phys.world.angular_pd_controllers[spine.pd_c_flex].target_angle = tg.spine.c_flex;
    phys.world.angular_pd_controllers[spine.pd_c_bend].target_angle = tg.spine.c_bend;
    phys.world.angular_pd_controllers[spine.pd_c_rot].target_angle  = tg.spine.c_rot;
    phys.spine = spine;

    // Arm PDs.
    let arm_r = std::mem::replace(&mut phys.arm_r, empty_arm());
    let arm_l = std::mem::replace(&mut phys.arm_l, empty_arm());
    apply_arm(&mut phys.world, &arm_r, Side::Right, &tg.arm_r);
    apply_arm(&mut phys.world, &arm_l, Side::Left,  &tg.arm_l);
    phys.arm_r = arm_r;
    phys.arm_l = arm_l;

    // Leg PDs.
    let leg_r = std::mem::replace(&mut phys.leg_r, empty_leg());
    let leg_l = std::mem::replace(&mut phys.leg_l, empty_leg());
    apply_leg(&mut phys.world, &leg_r, &tg.leg_r);
    apply_leg(&mut phys.world, &leg_l, &tg.leg_l);
    phys.leg_r = leg_r;
    phys.leg_l = leg_l;

    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn empty_arm() -> ArmHandles {
    ArmHandles {
        upper_idx: 0, forearm_idx: 0, palm_idx: 0,
        fingers: vec![], thumb: ThumbHandles { prox_idx: 0, dist_idx: 0 },
        pd_s_flex: 0, pd_s_abd: 0, pd_s_rot: 0,
        pd_elbow: 0, pd_w_flex: 0, pd_w_dev: 0,
    }
}
fn empty_leg() -> LegHandles {
    LegHandles {
        thigh_idx: 0, shin_idx: 0, foot_idx: 0,
        pd_hip_flex: 0, pd_hip_abd: 0, pd_hip_rot: 0,
        pd_knee: 0, pd_ankle_pf: 0, pd_ankle_inv: 0,
        toe_pds: vec![], chain_nodes: vec![],
    }
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    chains: Res<FootChains>,
    mut q_body: Query<(&BodyRef, &mut Transform), (Without<FootBodyRef>, Without<ToeBodyRef>)>,
    mut q_foot: Query<(&FootBodyRef, &mut Transform), (Without<BodyRef>, Without<ToeBodyRef>)>,
    mut q_toe: Query<(&ToeBodyRef, &mut Transform), (Without<BodyRef>, Without<FootBodyRef>)>,
) {
    for (body_ref, mut transform) in &mut q_body {
        *transform = rigid_body_to_transform(&phys.world.bodies[body_ref.0]);
    }
    for (foot_ref, mut transform) in &mut q_foot {
        *transform = foot_render_transform(&phys.world.bodies[foot_ref.idx]);
    }
    let r_renders = render_foot_subtree(&phys.world, &chains.right);
    let l_renders = render_foot_subtree(&phys.world, &chains.left);
    for (toe_ref, mut transform) in &mut q_toe {
        let (orient, pos) = match toe_ref.side {
            Side::Right => r_renders[toe_ref.chain_idx],
            Side::Left  => l_renders[toe_ref.chain_idx],
        };
        let q = orient.into_inner();
        *transform = Transform {
            translation: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
            rotation: Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32),
            scale: Vec3::ONE,
        };
    }
    let _ = nalgebra_vec3_to_bevy;  // suppress unused warning
    let _ = nalgebra_quat_to_bevy;
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let tg = schedule(phys.sim_time);
    let cycle_t = phys.sim_time.rem_euclid(SCHEDULE_PERIOD);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "t: {:>5.1} s  /  cycle {:.0} s  ({:>5.1} s into cycle)\n\nphase: {}",
            phys.sim_time, SCHEDULE_PERIOD, cycle_t, tg.phase,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Full skeleton (pelvis + spine + bilateral arms + bilateral legs)".into(),
                resolution: (1100.0, 850.0).into(),
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
