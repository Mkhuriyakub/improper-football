// Phase 2 step 5: full unpinned skeleton standing on the ground.
//
// Setup: pelvis (unpinned) + spine (thoracic + head) + bilateral arms
// (upper + forearm + palm; no fingers — they don't change standing
// dynamics) + bilateral legs (thigh + shin + foot; no toes for the same
// reason). 17 bodies total.
//
// Initial pose: standing upright with both feet on the ground. Feet at
// horizontal anatomical pose (baseline R_x(-π/2)). Whole skeleton
// stationary, all PD targets at neutral.
//
// Ground contact on every body: feet use default plane_y = 0 (rounded
// capsule shape matches anatomy reasonably); all other bodies use
// plane_y = -radius so the unmodeled rounded capsule end below the body
// axis doesn't engage contact unless the body's anatomical bottom
// crosses the ground.
//
// Expected: the system is in *unstable* equilibrium initially (no
// active balance control). Numerical noise will push it off equilibrium,
// the body will start to topple in some direction, and as it falls
// various bodies will land on the ground in sequence. The chain should
// stay stable through the fall — no NaNs, no exploding bodies, contact
// behaving sensibly even as bodies tilt past horizontal.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.10;

// ─── Body constants ───────────────────────────────────────────────────

const PELVIS_MASS:    f64 = 8.0;
const PELVIS_HALF_H:  f64 = 0.10;
const PELVIS_RADIUS:  f64 = 0.18;
const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;
const SHOULDER_ANCHOR_THORACIC_X: f64 = 0.15;

const THORACIC_MASS:    f64 = 25.0;
const THORACIC_RADIUS:  f64 = 0.130;
const THORACIC_HALF_H:  f64 = 0.180;

const HEAD_MASS:    f64 = 5.0;
const HEAD_RADIUS:  f64 = 0.080;
const HEAD_HALF_H:  f64 = 0.050;

const THIGH_MASS:   f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const SHIN_MASS:   f64 = 3.5;
const SHIN_RADIUS: f64 = 0.050;
const SHIN_HALF_H: f64 = 0.210;

const FOOT_MASS:   f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

const FOREARM_MASS:    f64 = 1.2;
const FOREARM_RADIUS:  f64 = 0.038;
const FOREARM_HALF_H:  f64 = 0.135;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

// ─── Joint constants ──────────────────────────────────────────────────

const HIP_JOINT_K: f64 = 200_000.0;  const HIP_JOINT_C: f64 = 400.0;
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K:   f64 = 120.0;
const HIP_LIMIT_EXP: f64 =   2.0;

const KNEE_JOINT_K: f64 = 80_000.0;  const KNEE_JOINT_C: f64 = 200.0;
const KNEE_K: f64 = 5.0;  const KNEE_C: f64 = 1.5;  const KNEE_NEUTRAL: f64 = 0.10;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;  const KNEE_LIMIT_EXP: f64 = 2.0;
const KNEE_OFF_AXIS_K: f64 = 12.0;  const KNEE_OFF_AXIS_C: f64 = 0.80;
const KNEE_OFF_AXIS_LIM: f64 = 0.30;  const KNEE_OFF_AXIS_LIMIT_K: f64 = 50.0;

const ANKLE_JOINT_K: f64 = 60_000.0;  const ANKLE_JOINT_C: f64 = 150.0;
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const INV_K: f64 = 3.5;  const INV_C: f64 = 1.0;
const TWIST_K: f64 = 8.0;  const TWIST_C: f64 = 0.5;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const INV_MIN: f64 = -0.26;  const INV_MAX: f64 = 0.52;
const TWIST_LIM: f64 = 0.10;
const ANKLE_LIMIT_K: f64 = 80.0;  const TWIST_LIMIT_K: f64 = 200.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const LUMBAR_JOINT_K: f64 = 300_000.0;  const LUMBAR_JOINT_C: f64 = 600.0;
const L_FLEX_K: f64 = 150.0;  const L_FLEX_C: f64 = 25.0;
const L_BEND_K: f64 = 150.0;  const L_BEND_C: f64 = 25.0;
const L_ROT_K:  f64 =  50.0;  const L_ROT_C:  f64 =  5.0;
const L_FLEX_MIN: f64 = -0.44;  const L_FLEX_MAX: f64 = 1.05;
const L_BEND_MIN: f64 = -0.44;  const L_BEND_MAX: f64 = 0.44;
const L_ROT_MIN:  f64 = -0.087;  const L_ROT_MAX:  f64 = 0.087;
const LUMBAR_LIMIT_K: f64 = 200.0;  const LUMBAR_LIMIT_EXP: f64 = 2.0;

const CERV_JOINT_K: f64 = 80_000.0;  const CERV_JOINT_C: f64 = 200.0;
const C_FLEX_K: f64 = 5.0;  const C_FLEX_C: f64 = 0.60;
const C_BEND_K: f64 = 5.0;  const C_BEND_C: f64 = 0.60;
const C_ROT_K:  f64 = 3.0;  const C_ROT_C:  f64 = 0.50;
const C_FLEX_MIN: f64 = -1.05;  const C_FLEX_MAX: f64 = 0.87;
const C_BEND_MIN: f64 = -0.78;  const C_BEND_MAX: f64 = 0.78;
const C_ROT_MIN:  f64 = -1.40;  const C_ROT_MAX:  f64 = 1.40;
const CERV_LIMIT_K: f64 = 80.0;  const CERV_LIMIT_EXP: f64 = 2.0;

const SHOULDER_JOINT_K: f64 = 100_000.0;  const SHOULDER_JOINT_C: f64 = 200.0;
const FLEX_K: f64 = 3.0;  const FLEX_C: f64 = 0.50;
const ABD_K: f64 = 3.0;  const ABD_C: f64 = 0.50;
const ROT_K: f64 = 1.5;  const ROT_C: f64 = 0.40;
const FLEX_MIN: f64 = -0.87;  const FLEX_MAX: f64 = 3.14;
const ABD_MIN: f64 = -0.78;  const ABD_MAX: f64 = 3.14;
const ROT_MIN: f64 = -1.40;  const ROT_MAX: f64 = 1.57;
const SHOULDER_LIMIT_K: f64 = 100.0;  const SHOULDER_LIMIT_EXP: f64 = 2.0;

const ELBOW_JOINT_K: f64 = 80_000.0;  const ELBOW_JOINT_C: f64 = 150.0;
const ELBOW_K: f64 = 1.5;  const ELBOW_C: f64 = 0.05;  const ELBOW_NEUTRAL: f64 = 0.20;
const ELBOW_MIN: f64 = -0.40;  const ELBOW_MAX: f64 = 2.20;
const ELBOW_LIMIT_K: f64 = 500.0;
const ELBOW_OFF_AXIS_K: f64 = 8.0;  const ELBOW_OFF_AXIS_C: f64 = 0.60;
const ELBOW_OFF_AXIS_LIM: f64 = 0.40;  const ELBOW_OFF_AXIS_LIMIT_K: f64 = 30.0;

const WRIST_JOINT_K: f64 = 50_000.0;  const WRIST_JOINT_C: f64 = 100.0;
const WRIST_FLEX_K: f64 = 0.4;  const WRIST_DEV_K: f64 = 0.6;  const WRIST_TWIST_K: f64 = 0.5;
const WRIST_FLEX_C: f64 = 0.05;  const WRIST_DEV_C: f64 = 0.06;  const WRIST_TWIST_C: f64 = 0.015;
const WRIST_FLEX_MIN: f64 = -1.10;  const WRIST_FLEX_MAX: f64 = 1.30;
const WRIST_DEV_MIN: f64 = -0.60;  const WRIST_DEV_MAX: f64 = 0.35;
const WRIST_TWIST_LIM: f64 = 0.30;
const WRIST_LIMIT_K: f64 = 200.0;  const WRIST_TWIST_LIMIT_K: f64 = 20.0;

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
    parent: &RigidBody, parent_distal: Vec3, child_proximal: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: Vec3, ca: Vec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = pa; s.local_anchor_b = ca;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = pa; d.local_anchor_b = ca;
    world.positional_dampers.push(d);
}

fn add_axis(
    world: &mut XpbdWorld, parent: usize, child: usize,
    axis: Vec3, min_a: f64, max_a: f64,
    k: f64, c: f64, neutral: f64,
    limit_k: f64, limit_exp: f64,
    baseline: UnitQuaternion<f64>,
) {
    world.angular_soft_limits.push(
        AngularSoftLimit::new(parent, child, axis, min_a, max_a, limit_k, limit_exp)
            .with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis, c));
    world.angular_springs.push(
        AngularSpring::new(parent, child, axis, k, neutral).with_baseline(baseline));
}

#[derive(Clone, Copy, PartialEq)]
enum Side { Right, Left }
impl Side {
    fn sign(self) -> f64 { match self { Side::Right => 1.0, Side::Left => -1.0 } }
}

// ─── Builders ─────────────────────────────────────────────────────────

struct LegHandles { thigh_idx: usize, shin_idx: usize, foot_idx: usize }
struct ArmHandles { upper_idx: usize, forearm_idx: usize, palm_idx: usize }

fn add_leg(world: &mut XpbdWorld, pelvis_idx: usize, side: Side) -> LegHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Thigh.
    let hip_anchor = Vec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0);
    let mut thigh = RigidBody::new(THIGH_MASS, capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor, Vec3::new(0.0, THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(world, pelvis_idx, thigh_idx,
        hip_anchor, Vec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    add_axis(world, pelvis_idx, thigh_idx, Vec3::new(-1.0, 0.0, 0.0),
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, Vec3::new(0.0, 0.0, s),
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, Vec3::new(0.0, s, 0.0),
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_ROT_K, HIP_ROT_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);

    // Shin.
    let mut shin = RigidBody::new(SHIN_MASS, capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    add_axis(world, thigh_idx, shin_idx, Vec3::new(1.0, 0.0, 0.0),
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL,
        KNEE_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, Vec3::new(0.0, 0.0, 1.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, Vec3::new(0.0, 1.0, 0.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);

    // Foot.
    let mut foot = RigidBody::new(FOOT_MASS, capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    add_axis(world, shin_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0),
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, Vec3::new(0.0, 0.0, s),
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, Vec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0,
        TWIST_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);

    LegHandles { thigh_idx, shin_idx, foot_idx }
}

fn add_arm(world: &mut XpbdWorld, thoracic_idx: usize, side: Side) -> ArmHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let shoulder_anchor = Vec3::new(s * SHOULDER_ANCHOR_THORACIC_X, THORACIC_HALF_H, 0.0);

    // Upper arm.
    let mut upper = RigidBody::new(UPPER_ARM_MASS, capsule_inertia(UPPER_ARM_MASS, UPPER_ARM_RADIUS, UPPER_ARM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        shoulder_anchor, Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0), id);
    upper.position = pos; upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(world, thoracic_idx, upper_idx,
        shoulder_anchor, Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);
    let (abd_min, abd_max) = if s > 0.0 { (ABD_MIN, ABD_MAX) } else { (-ABD_MAX, -ABD_MIN) };
    let (rot_min, rot_max) = if s > 0.0 { (ROT_MIN, ROT_MAX) } else { (-ROT_MAX, -ROT_MIN) };
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(-1.0, 0.0, 0.0),
        FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(0.0, 0.0, 1.0),
        abd_min, abd_max, ABD_K, ABD_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(0.0, 1.0, 0.0),
        rot_min, rot_max, ROT_K, ROT_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);

    // Forearm.
    let mut forearm = RigidBody::new(FOREARM_MASS, capsule_inertia(FOREARM_MASS, FOREARM_RADIUS, FOREARM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[upper_idx],
        Vec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        Vec3::new(0.0, FOREARM_HALF_H, 0.0), id);
    forearm.position = pos; forearm.orientation = orient;
    let forearm_idx = world.add_body(forearm);

    add_elastic_joint(world, upper_idx, forearm_idx,
        Vec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        Vec3::new(0.0, FOREARM_HALF_H, 0.0),
        ELBOW_JOINT_K, ELBOW_JOINT_C);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(-1.0, 0.0, 0.0),
        ELBOW_MIN, ELBOW_MAX, ELBOW_K, ELBOW_C, ELBOW_NEUTRAL,
        ELBOW_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(0.0, 0.0, 1.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(0.0, 1.0, 0.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);

    // Palm (no fingers in this validation).
    let mut palm = RigidBody::new(PALM_MASS, capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H));
    let (pos, orient) = place_child(&world.bodies[forearm_idx],
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0, PALM_HALF_H, 0.0), id);
    palm.position = pos; palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(world, forearm_idx, palm_idx,
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0, PALM_HALF_H, 0.0),
        WRIST_JOINT_K, WRIST_JOINT_C);
    let (dev_min, dev_max) = if s > 0.0 { (WRIST_DEV_MIN, WRIST_DEV_MAX) } else { (-WRIST_DEV_MAX, -WRIST_DEV_MIN) };
    add_axis(world, forearm_idx, palm_idx, Vec3::new(-1.0, 0.0, 0.0),
        WRIST_FLEX_MIN, WRIST_FLEX_MAX, WRIST_FLEX_K, WRIST_FLEX_C, 0.0,
        WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, Vec3::new(0.0, 0.0, 1.0),
        dev_min, dev_max, WRIST_DEV_K, WRIST_DEV_C, 0.0,
        WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, Vec3::new(0.0, 1.0, 0.0),
        -WRIST_TWIST_LIM, WRIST_TWIST_LIM, WRIST_TWIST_K, WRIST_TWIST_C, 0.0,
        WRIST_TWIST_LIMIT_K, 2.0, id);

    ArmHandles { upper_idx, forearm_idx, palm_idx }
}

// ─── Main ─────────────────────────────────────────────────────────────

fn main() {
    let mut world = XpbdWorld::new();

    // Pelvis position chosen so feet sit exactly on ground at rest:
    //   foot center y = FOOT_RADIUS = 0.032 (foot horizontal, on ground)
    //   ankle y = 0.032 (= foot heel anchor world)
    //   shin center y = 0.032 + SHIN_HALF_H = 0.242
    //   knee y = 0.452
    //   thigh center y = 0.677
    //   hip y = 0.902
    //   pelvis center y = 0.902 + 0.05 = 0.952  (HIP_ANCHOR_PELVIS_Y = -0.05)
    let mut pelvis = RigidBody::new(PELVIS_MASS,
        capsule_inertia(PELVIS_MASS, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = Vec3::new(0.0, 0.952, 0.0);
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Spine: thoracic above pelvis (lumbar joint), head above thoracic (cervical joint).
    let id = UnitQuaternion::identity();
    let lumbar_anchor_pelvis = Vec3::new(0.0, PELVIS_HALF_H, 0.0);
    let mut thoracic = RigidBody::new(THORACIC_MASS,
        capsule_inertia(THORACIC_MASS, THORACIC_RADIUS, THORACIC_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis, Vec3::new(0.0, -THORACIC_HALF_H, 0.0), id);
    thoracic.position = pos; thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(&mut world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis, Vec3::new(0.0, -THORACIC_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    add_axis(&mut world, pelvis_idx, thoracic_idx, Vec3::new(1.0, 0.0, 0.0),
        L_FLEX_MIN, L_FLEX_MAX, L_FLEX_K, L_FLEX_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, Vec3::new(0.0, 0.0, -1.0),
        L_BEND_MIN, L_BEND_MAX, L_BEND_K, L_BEND_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, Vec3::new(0.0, 1.0, 0.0),
        L_ROT_MIN, L_ROT_MAX, L_ROT_K, L_ROT_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);

    let cerv_anchor_thoracic = Vec3::new(0.0, THORACIC_HALF_H, 0.0);
    let mut head = RigidBody::new(HEAD_MASS,
        capsule_inertia(HEAD_MASS, HEAD_RADIUS, HEAD_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        cerv_anchor_thoracic, Vec3::new(0.0, -HEAD_HALF_H, 0.0), id);
    head.position = pos; head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic, Vec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(1.0, 0.0, 0.0),
        C_FLEX_MIN, C_FLEX_MAX, C_FLEX_K, C_FLEX_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(0.0, 0.0, -1.0),
        C_BEND_MIN, C_BEND_MAX, C_BEND_K, C_BEND_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(0.0, 1.0, 0.0),
        C_ROT_MIN, C_ROT_MAX, C_ROT_K, C_ROT_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);

    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);
    let arm_r = add_arm(&mut world, thoracic_idx, Side::Right);
    let arm_l = add_arm(&mut world, thoracic_idx, Side::Left);

    // Ground contacts on all 17 bodies.
    // - Feet: default plane_y = 0 (capsule shape matches anatomy).
    // - All other bodies: plane_y = -radius (anatomical bottom = body axis
    //   endpoint, capsule rounded end below is unmodeled artifact).
    world.ground_contacts.push(GroundContact::new(pelvis_idx, PELVIS_HALF_H, PELVIS_RADIUS).with_plane_y(-PELVIS_RADIUS));
    world.ground_contacts.push(GroundContact::new(thoracic_idx, THORACIC_HALF_H, THORACIC_RADIUS).with_plane_y(-THORACIC_RADIUS));
    world.ground_contacts.push(GroundContact::new(head_idx, HEAD_HALF_H, HEAD_RADIUS).with_plane_y(-HEAD_RADIUS));
    for leg in [&leg_r, &leg_l] {
        world.ground_contacts.push(GroundContact::new(leg.thigh_idx, THIGH_HALF_H, THIGH_RADIUS).with_plane_y(-THIGH_RADIUS));
        world.ground_contacts.push(GroundContact::new(leg.shin_idx, SHIN_HALF_H, SHIN_RADIUS).with_plane_y(-SHIN_RADIUS));
        world.ground_contacts.push(GroundContact::new(leg.foot_idx, FOOT_HALF_H, FOOT_RADIUS));
    }
    for arm in [&arm_r, &arm_l] {
        world.ground_contacts.push(GroundContact::new(arm.upper_idx, UPPER_ARM_HALF_H, UPPER_ARM_RADIUS).with_plane_y(-UPPER_ARM_RADIUS));
        world.ground_contacts.push(GroundContact::new(arm.forearm_idx, FOREARM_HALF_H, FOREARM_RADIUS).with_plane_y(-FOREARM_RADIUS));
        world.ground_contacts.push(GroundContact::new(arm.palm_idx, PALM_HALF_H, PALM_RADIUS).with_plane_y(-PALM_RADIUS));
    }

    let head_y_init = world.bodies[head_idx].position.y;

    println!("FULL SKELETON STANDING (Phase 2 step 5)");
    println!("17 bodies: pelvis (UNPINNED, 8 kg) + spine + 2 arms + 2 legs.");
    println!("Initial pose: standing upright, both feet on ground (y=0.032).");
    println!("Pelvis center y = {:.3}, head top y ≈ {:.3}.",
             world.bodies[pelvis_idx].position.y,
             head_y_init + HEAD_HALF_H + HEAD_RADIUS);
    println!("All PD targets at neutral. No active balance — system will fall.");
    println!();
    println!("{:<6} {:<11} {:<11} {:<11} {:<11} {:<11}",
             "t", "pelvis.y", "head.y", "Rfoot.y", "Lfoot.y", "head.z");
    println!("{}", "-".repeat(64));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_pelvis_drop = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        let p_y = world.bodies[pelvis_idx].position.y;
        let drop = 0.952 - p_y;
        if drop > max_pelvis_drop { max_pelvis_drop = drop; }

        if t >= next_print {
            let p = &world.bodies[pelvis_idx];
            let h = &world.bodies[head_idx];
            let rf = &world.bodies[leg_r.foot_idx];
            let lf = &world.bodies[leg_l.foot_idx];
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, p.position.y, h.position.y,
                     rf.position.y, lf.position.y, h.position.z);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let p = &world.bodies[pelvis_idx];
    let h = &world.bodies[head_idx];
    let rf = &world.bodies[leg_r.foot_idx];
    let lf = &world.bodies[leg_l.foot_idx];
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  pelvis = ({:+.3}, {:+.3}, {:+.3})", p.position.x, p.position.y, p.position.z);
    println!("  head   = ({:+.3}, {:+.3}, {:+.3})", h.position.x, h.position.y, h.position.z);
    println!("  R foot = ({:+.3}, {:+.3}, {:+.3})", rf.position.x, rf.position.y, rf.position.z);
    println!("  L foot = ({:+.3}, {:+.3}, {:+.3})", lf.position.x, lf.position.y, lf.position.z);
    println!("  pelvis dropped {:.3} m from start (started at y=0.952)", max_pelvis_drop);

    let all_finite = p.position.y.is_finite()
        && h.position.y.is_finite()
        && rf.position.y.is_finite() && lf.position.y.is_finite();
    if all_finite {
        println!();
        println!("All bodies finite — chain remained stable through the fall.");
    } else {
        println!();
        println!("WARNING: NaN detected — chain went unstable.");
    }
}
