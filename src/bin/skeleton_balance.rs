// Phase 3 step 1: active ankle balance.
//
// Same skeleton as `skeleton_standing.rs` (17 bodies — pelvis, spine,
// bilateral arms, bilateral legs without toes/fingers). Ground contact
// on every body. The difference: ankle plantarflexion is now actively
// driven by a per-frame balance controller that senses body lean and
// applies a counter-torque at the ankle.
//
// Controller (ankle strategy, sagittal plane only for v0):
//   lean       = head.position.z - 0.5*(R_foot.z + L_foot.z)
//   lean_rate  = head.linear_velocity.z - 0.5*(R_foot.vz + L_foot.vz)
//   ankle_pf_target = K_p * lean + K_d * lean_rate
//
// Both ankles get the same PF target. When the body leans forward
// (lean > 0), the controller commands positive ankle PF (the joint
// rotates the foot relative to shin) — which torques the shin (and
// everything above it) backward at the ankle, righting the body.
//
// Without active balance, this skeleton fell over in ~1 s
// (skeleton_standing.rs). With balance working, the body should stay
// upright for the full 5 s with bounded lean.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::Vec3;
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;
const PRINT_INTERVAL: f64 = 0.10;

// Balance controller gains.
const BAL_KP: f64 = 5.0;     // ankle PF target (rad) per metre of forward lean
const BAL_KD: f64 = 1.0;     // ankle PF target (rad) per (m/s) of forward lean rate

// Ankle PD gains (the feedback loop converting target error → ankle torque).
// Higher than the ankle_demo's 80/6 because the ankle now has to torque
// the entire body's mass, not just the foot.
const ANKLE_PD_KP: f64 = 250.0;
const ANKLE_PD_KD: f64 =  20.0;

// Postural PD gains for the major joints. Real human balance involves
// constant low-level muscle tone holding the knees, hips, and trunk in
// the standing pose, with the ankle modulating on top. Without these
// the body collapses through the knees (knee buckle was the actual
// failure mode in the first attempt at this validation, NOT the ankle).
const KNEE_PD_KP:  f64 = 600.0;  const KNEE_PD_KD:  f64 = 35.0;
const HIP_PD_KP:   f64 = 600.0;  const HIP_PD_KD:   f64 = 40.0;
const LUMBAR_PD_KP: f64 = 1500.0; const LUMBAR_PD_KD: f64 = 80.0;

// ─── Body / joint constants (same as skeleton_standing.rs) ────────────

const PELVIS_MASS: f64 = 8.0;
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;
const HIP_ANCHOR_PELVIS_X: f64 = 0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;
const SHOULDER_ANCHOR_THORACIC_X: f64 = 0.15;

const THORACIC_MASS: f64 = 25.0;
const THORACIC_RADIUS: f64 = 0.130;
const THORACIC_HALF_H: f64 = 0.180;

const HEAD_MASS: f64 = 5.0;
const HEAD_RADIUS: f64 = 0.080;
const HEAD_HALF_H: f64 = 0.050;

const THIGH_MASS: f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const SHIN_MASS: f64 = 3.5;
const SHIN_RADIUS: f64 = 0.050;
const SHIN_HALF_H: f64 = 0.210;

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const UPPER_ARM_MASS: f64 = 2.0;
const UPPER_ARM_RADIUS: f64 = 0.045;
const UPPER_ARM_HALF_H: f64 = 0.150;

const FOREARM_MASS: f64 = 1.2;
const FOREARM_RADIUS: f64 = 0.038;
const FOREARM_HALF_H: f64 = 0.135;

const PALM_MASS: f64 = 0.30;
const PALM_HALF_H: f64 = 0.040;
const PALM_RADIUS: f64 = 0.030;

const HIP_JOINT_K: f64 = 200_000.0; const HIP_JOINT_C: f64 = 400.0;
const HIP_FLEX_K: f64 = 10.0; const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K: f64 = 10.0; const HIP_ABD_C: f64 = 7.0;
const HIP_ROT_K: f64 = 3.0; const HIP_ROT_C: f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35; const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN: f64 = -0.52; const HIP_ABD_MAX: f64 = 0.78;
const HIP_ROT_MIN: f64 = -0.61; const HIP_ROT_MAX: f64 = 0.87;
const HIP_LIMIT_K: f64 = 120.0; const HIP_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0; const KNEE_JOINT_C: f64 = 200.0;
const KNEE_K: f64 = 5.0; const KNEE_C: f64 = 1.5; const KNEE_NEUTRAL: f64 = 0.10;
const KNEE_MIN: f64 = -0.087; const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0; const KNEE_LIMIT_EXP: f64 = 2.0;
const KNEE_OFF_AXIS_K: f64 = 12.0; const KNEE_OFF_AXIS_C: f64 = 0.80;
const KNEE_OFF_AXIS_LIM: f64 = 0.30; const KNEE_OFF_AXIS_LIMIT_K: f64 = 50.0;

const ANKLE_JOINT_K: f64 = 60_000.0; const ANKLE_JOINT_C: f64 = 150.0;
const PF_K: f64 = 4.0; const PF_C: f64 = 1.2;
const INV_K: f64 = 3.5; const INV_C: f64 = 1.0;
const TWIST_K: f64 = 8.0; const TWIST_C: f64 = 0.5;
const PF_MIN: f64 = -0.44; const PF_MAX: f64 = 0.87;
const INV_MIN: f64 = -0.26; const INV_MAX: f64 = 0.52;
const TWIST_LIM: f64 = 0.10;
const ANKLE_LIMIT_K: f64 = 80.0; const TWIST_LIMIT_K: f64 = 200.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const LUMBAR_JOINT_K: f64 = 300_000.0; const LUMBAR_JOINT_C: f64 = 600.0;
const L_FLEX_K: f64 = 150.0; const L_FLEX_C: f64 = 25.0;
const L_BEND_K: f64 = 150.0; const L_BEND_C: f64 = 25.0;
const L_ROT_K: f64 = 50.0; const L_ROT_C: f64 = 5.0;
const L_FLEX_MIN: f64 = -0.44; const L_FLEX_MAX: f64 = 1.05;
const L_BEND_MIN: f64 = -0.44; const L_BEND_MAX: f64 = 0.44;
const L_ROT_MIN: f64 = -0.087; const L_ROT_MAX: f64 = 0.087;
const LUMBAR_LIMIT_K: f64 = 200.0; const LUMBAR_LIMIT_EXP: f64 = 2.0;

const CERV_JOINT_K: f64 = 80_000.0; const CERV_JOINT_C: f64 = 200.0;
const C_FLEX_K: f64 = 5.0; const C_FLEX_C: f64 = 0.60;
const C_BEND_K: f64 = 5.0; const C_BEND_C: f64 = 0.60;
const C_ROT_K: f64 = 3.0; const C_ROT_C: f64 = 0.50;
const C_FLEX_MIN: f64 = -1.05; const C_FLEX_MAX: f64 = 0.87;
const C_BEND_MIN: f64 = -0.78; const C_BEND_MAX: f64 = 0.78;
const C_ROT_MIN: f64 = -1.40; const C_ROT_MAX: f64 = 1.40;
const CERV_LIMIT_K: f64 = 80.0; const CERV_LIMIT_EXP: f64 = 2.0;

const SHOULDER_JOINT_K: f64 = 100_000.0; const SHOULDER_JOINT_C: f64 = 200.0;
const FLEX_K: f64 = 3.0; const FLEX_C: f64 = 0.50;
const ABD_K: f64 = 3.0; const ABD_C: f64 = 0.50;
const ROT_K: f64 = 1.5; const ROT_C: f64 = 0.40;
const FLEX_MIN: f64 = -0.87; const FLEX_MAX: f64 = 3.14;
const ABD_MIN: f64 = -0.78; const ABD_MAX: f64 = 3.14;
const ROT_MIN: f64 = -1.40; const ROT_MAX: f64 = 1.57;
const SHOULDER_LIMIT_K: f64 = 100.0; const SHOULDER_LIMIT_EXP: f64 = 2.0;

const ELBOW_JOINT_K: f64 = 80_000.0; const ELBOW_JOINT_C: f64 = 150.0;
const ELBOW_K: f64 = 1.5; const ELBOW_C: f64 = 0.05; const ELBOW_NEUTRAL: f64 = 0.20;
const ELBOW_MIN: f64 = -0.40; const ELBOW_MAX: f64 = 2.20;
const ELBOW_LIMIT_K: f64 = 500.0;
const ELBOW_OFF_AXIS_K: f64 = 8.0; const ELBOW_OFF_AXIS_C: f64 = 0.60;
const ELBOW_OFF_AXIS_LIM: f64 = 0.40; const ELBOW_OFF_AXIS_LIMIT_K: f64 = 30.0;

const WRIST_JOINT_K: f64 = 50_000.0; const WRIST_JOINT_C: f64 = 100.0;
const WRIST_FLEX_K: f64 = 0.4; const WRIST_DEV_K: f64 = 0.6; const WRIST_TWIST_K: f64 = 0.5;
const WRIST_FLEX_C: f64 = 0.05; const WRIST_DEV_C: f64 = 0.06; const WRIST_TWIST_C: f64 = 0.015;
const WRIST_FLEX_MIN: f64 = -1.10; const WRIST_FLEX_MAX: f64 = 1.30;
const WRIST_DEV_MIN: f64 = -0.60; const WRIST_DEV_MAX: f64 = 0.35;
const WRIST_TWIST_LIM: f64 = 0.30;
const WRIST_LIMIT_K: f64 = 200.0; const WRIST_TWIST_LIMIT_K: f64 = 20.0;

// ─── Helpers ──────────────────────────────────────────────────────────

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

#[allow(dead_code)]
struct LegHandles {
    thigh_idx: usize, shin_idx: usize, foot_idx: usize,
    ankle_pf_pd_idx: usize,
    hip_flex_pd_idx: usize,
    knee_pd_idx: usize,
}
#[allow(dead_code)]
struct ArmHandles { upper_idx: usize, forearm_idx: usize, palm_idx: usize }

fn add_leg(world: &mut XpbdWorld, pelvis_idx: usize, side: Side) -> LegHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let hip_anchor = Vec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0);
    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor, Vec3::new(0.0, THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(world, pelvis_idx, thigh_idx,
        hip_anchor, Vec3::new(0.0, THIGH_HALF_H, 0.0), HIP_JOINT_K, HIP_JOINT_C);
    let hip_flex_axis = Vec3::new(-1.0, 0.0, 0.0);
    add_axis(world, pelvis_idx, thigh_idx, hip_flex_axis,
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0, HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, Vec3::new(0.0, 0.0, s),
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0, HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, Vec3::new(0.0, s, 0.0),
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_ROT_K, HIP_ROT_C, 0.0, HIP_LIMIT_K, HIP_LIMIT_EXP, id);

    // Postural PD: hold hip flex at 0 (extended).
    let hip_flex_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_flex_axis, HIP_PD_KP, HIP_PD_KD)
            .with_target(0.0));

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0, SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0, SHIN_HALF_H, 0.0), KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    add_axis(world, thigh_idx, shin_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL, KNEE_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, Vec3::new(0.0, 0.0, 1.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, Vec3::new(0.0, 1.0, 0.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);

    // Postural PD: hold knee at neutral extension.
    let knee_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, shin_idx, knee_axis, KNEE_PD_KP, KNEE_PD_KD)
            .with_target(KNEE_NEUTRAL));

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL, ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    add_axis(world, shin_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, Vec3::new(0.0, 0.0, s),
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, Vec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0, TWIST_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);

    // Ankle PF PD controller — driven per-frame by the balance loop.
    let ankle_pf_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, pf_axis, ANKLE_PD_KP, ANKLE_PD_KD)
            .with_target(0.0).with_baseline(foot_baseline));

    LegHandles { thigh_idx, shin_idx, foot_idx, ankle_pf_pd_idx, hip_flex_pd_idx, knee_pd_idx }
}

fn add_arm(world: &mut XpbdWorld, thoracic_idx: usize, side: Side) -> ArmHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let shoulder_anchor = Vec3::new(s * SHOULDER_ANCHOR_THORACIC_X, THORACIC_HALF_H, 0.0);

    let mut upper = RigidBody::new_capsule(UPPER_ARM_MASS, UPPER_ARM_HALF_H, UPPER_ARM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        shoulder_anchor, Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0), id);
    upper.position = pos; upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(world, thoracic_idx, upper_idx,
        shoulder_anchor, Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0), SHOULDER_JOINT_K, SHOULDER_JOINT_C);
    let (abd_min, abd_max) = if s > 0.0 { (ABD_MIN, ABD_MAX) } else { (-ABD_MAX, -ABD_MIN) };
    let (rot_min, rot_max) = if s > 0.0 { (ROT_MIN, ROT_MAX) } else { (-ROT_MAX, -ROT_MIN) };
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(-1.0, 0.0, 0.0),
        FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(0.0, 0.0, 1.0),
        abd_min, abd_max, ABD_K, ABD_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, Vec3::new(0.0, 1.0, 0.0),
        rot_min, rot_max, ROT_K, ROT_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);

    let mut forearm = RigidBody::new_capsule(FOREARM_MASS, FOREARM_HALF_H, FOREARM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[upper_idx],
        Vec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        Vec3::new(0.0, FOREARM_HALF_H, 0.0), id);
    forearm.position = pos; forearm.orientation = orient;
    let forearm_idx = world.add_body(forearm);

    add_elastic_joint(world, upper_idx, forearm_idx,
        Vec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        Vec3::new(0.0, FOREARM_HALF_H, 0.0), ELBOW_JOINT_K, ELBOW_JOINT_C);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(-1.0, 0.0, 0.0),
        ELBOW_MIN, ELBOW_MAX, ELBOW_K, ELBOW_C, ELBOW_NEUTRAL, ELBOW_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(0.0, 0.0, 1.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, Vec3::new(0.0, 1.0, 0.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);

    let mut palm = RigidBody::new_capsule(PALM_MASS, PALM_HALF_H, PALM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[forearm_idx],
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0, PALM_HALF_H, 0.0), id);
    palm.position = pos; palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(world, forearm_idx, palm_idx,
        Vec3::new(0.0, -FOREARM_HALF_H, 0.0),
        Vec3::new(0.0, PALM_HALF_H, 0.0), WRIST_JOINT_K, WRIST_JOINT_C);
    let (dev_min, dev_max) = if s > 0.0 { (WRIST_DEV_MIN, WRIST_DEV_MAX) } else { (-WRIST_DEV_MAX, -WRIST_DEV_MIN) };
    add_axis(world, forearm_idx, palm_idx, Vec3::new(-1.0, 0.0, 0.0),
        WRIST_FLEX_MIN, WRIST_FLEX_MAX, WRIST_FLEX_K, WRIST_FLEX_C, 0.0, WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, Vec3::new(0.0, 0.0, 1.0),
        dev_min, dev_max, WRIST_DEV_K, WRIST_DEV_C, 0.0, WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, Vec3::new(0.0, 1.0, 0.0),
        -WRIST_TWIST_LIM, WRIST_TWIST_LIM, WRIST_TWIST_K, WRIST_TWIST_C, 0.0,
        WRIST_TWIST_LIMIT_K, 2.0, id);

    ArmHandles { upper_idx, forearm_idx, palm_idx }
}

fn main() {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    pelvis.position = Vec3::new(0.0, 0.952, 0.0);
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    let lumbar_anchor_pelvis = Vec3::new(0.0, PELVIS_HALF_H, 0.0);
    let mut thoracic = RigidBody::new_capsule(THORACIC_MASS, THORACIC_HALF_H, THORACIC_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis, Vec3::new(0.0, -THORACIC_HALF_H, 0.0), id);
    thoracic.position = pos; thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(&mut world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis, Vec3::new(0.0, -THORACIC_HALF_H, 0.0), LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let lumbar_flex_axis = Vec3::new(1.0, 0.0, 0.0);
    add_axis(&mut world, pelvis_idx, thoracic_idx, lumbar_flex_axis,
        L_FLEX_MIN, L_FLEX_MAX, L_FLEX_K, L_FLEX_C, 0.0, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, Vec3::new(0.0, 0.0, -1.0),
        L_BEND_MIN, L_BEND_MAX, L_BEND_K, L_BEND_C, 0.0, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, Vec3::new(0.0, 1.0, 0.0),
        L_ROT_MIN, L_ROT_MAX, L_ROT_K, L_ROT_C, 0.0, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);

    // Postural PD: hold lumbar flex at 0 (upright trunk).
    let lumbar_flex_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, lumbar_flex_axis, LUMBAR_PD_KP, LUMBAR_PD_KD)
            .with_target(0.0));

    let cerv_anchor_thoracic = Vec3::new(0.0, THORACIC_HALF_H, 0.0);
    let mut head = RigidBody::new_capsule(HEAD_MASS, HEAD_HALF_H, HEAD_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        cerv_anchor_thoracic, Vec3::new(0.0, -HEAD_HALF_H, 0.0), id);
    head.position = pos; head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic, Vec3::new(0.0, -HEAD_HALF_H, 0.0), CERV_JOINT_K, CERV_JOINT_C);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(1.0, 0.0, 0.0),
        C_FLEX_MIN, C_FLEX_MAX, C_FLEX_K, C_FLEX_C, 0.0, CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(0.0, 0.0, -1.0),
        C_BEND_MIN, C_BEND_MAX, C_BEND_K, C_BEND_C, 0.0, CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, Vec3::new(0.0, 1.0, 0.0),
        C_ROT_MIN, C_ROT_MAX, C_ROT_K, C_ROT_C, 0.0, CERV_LIMIT_K, CERV_LIMIT_EXP, id);

    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);
    let _arm_r = add_arm(&mut world, thoracic_idx, Side::Right);
    let _arm_l = add_arm(&mut world, thoracic_idx, Side::Left);

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. At neutral pose only the feet touch; other
    // segments collide if the skeleton falls. Pre-migration this binary
    // disabled per-body contact for non-feet via plane_y = -radius hacks.

    println!("SKELETON BALANCE (Phase 3 step 1: ankle-strategy active balance)");
    println!();
    println!("Sagittal-plane balance only (ankle PF feedback). Side-to-side");
    println!("lean isn't actively countered — initial pose is already laterally");
    println!("balanced so this works for v0; full balance comes later.");
    println!();
    println!("Controller: ankle_pf_target = {:.1}*lean_z + {:.1}*lean_rate_z",
             BAL_KP, BAL_KD);
    println!("Ankle PD: KP={:.0}, KD={:.0}", ANKLE_PD_KP, ANKLE_PD_KD);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "t", "head.z", "lean", "ankle_tgt", "head.y", "pelvis.y");
    println!("{}", "-".repeat(60));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_lean = 0.0_f64;

    while t < SIM_DURATION {
        // ─── Balance controller: sense → compute → set targets ───────
        // Sense lean against the ANKLE position (the joint, where the
        // support reference is), not the foot body's center. The foot's
        // center is FOOT_HALF_H forward of the ankle, so using the foot
        // center directly gives lean = -0.09 at neutral standing — which
        // would make the controller fight a phantom backward lean.
        let head = &world.bodies[head_idx];
        let foot_r = &world.bodies[leg_r.foot_idx];
        let foot_l = &world.bodies[leg_l.foot_idx];
        let ankle_r_world = foot_r.position + foot_r.orientation * FOOT_ANCHOR_LOCAL;
        let ankle_l_world = foot_l.position + foot_l.orientation * FOOT_ANCHOR_LOCAL;
        let support_z = 0.5 * (ankle_r_world.z + ankle_l_world.z);
        let lean = head.position.z - support_z;
        let lean_rate = head.linear_velocity.z;  // ankles ~stationary while standing

        let ankle_pf_target = BAL_KP * lean + BAL_KD * lean_rate;

        // Clamp inside anatomical ROM. Wider margin than before so the
        // controller has more usable range before it pins against limits.
        let ankle_pf_target = ankle_pf_target.clamp(-0.30, 0.60);

        world.angular_pd_controllers[leg_r.ankle_pf_pd_idx].target_angle = ankle_pf_target;
        world.angular_pd_controllers[leg_l.ankle_pf_pd_idx].target_angle = ankle_pf_target;
        // Postural PDs are configured at construction with constant
        // targets (knee→neutral, hip→0, lumbar→0); no per-frame update.
        let _ = lumbar_flex_pd_idx;
        let _ = leg_r.hip_flex_pd_idx;
        let _ = leg_r.knee_pd_idx;
        let _ = leg_l.hip_flex_pd_idx;
        let _ = leg_l.knee_pd_idx;

        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if lean.abs() > max_lean { max_lean = lean.abs(); }

        if t >= next_print {
            let head = &world.bodies[head_idx];
            let pelvis = &world.bodies[pelvis_idx];
            println!("{:<6.2} {:<+10.5} {:<+10.5} {:<+10.5} {:<+10.5} {:<+10.5}",
                     t, head.position.z, lean, ankle_pf_target,
                     head.position.y, pelvis.position.y);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let head = &world.bodies[head_idx];
    let pelvis = &world.bodies[pelvis_idx];
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  pelvis y      = {:+.4} (started at 0.952)", pelvis.position.y);
    println!("  head y        = {:+.4} (started near 1.46)", head.position.y);
    println!("  head z (lean) = {:+.4} m", head.position.z);
    println!("  max |lean|    = {:.4} m", max_lean);
    println!();
    let upright = head.position.y > 1.0;
    if upright {
        println!("✓ Skeleton stayed upright (head still above 1 m).");
    } else {
        println!("✗ Skeleton fell — head y dropped below 1 m. Balance failed.");
    }
}
