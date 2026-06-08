// Visual demo of the Phase 2 milestone: full unpinned skeleton stands on
// the ground and topples over without active balance.
//
// Same setup as `skeleton_standing.rs` (17 bodies — pelvis + spine +
// bilateral arms + bilateral legs, no toes/fingers, ground contact on
// every body). Renders the skeleton in 3D with Bevy so you can see the
// topple visually instead of reading position numbers from a log.
//
// Foot meshes use the same horizontal-physics rendering as the rest of
// the bodies — the swing-twist refactor means `rigid_body_to_transform`
// gives the correct foot orientation directly.

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::Vec3 as NaVec3;
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

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
const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, FOOT_HALF_H, 0.0);

const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

const FOREARM_MASS:    f64 = 1.2;
const FOREARM_RADIUS:  f64 = 0.038;
const FOREARM_HALF_H:  f64 = 0.135;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

// ─── Joint constants (same as skeleton_standing.rs) ───────────────────

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

fn place_child(
    parent: &RigidBody, parent_distal: NaVec3, child_proximal: NaVec3,
    rel_rot: UnitQuaternion<f64>,
) -> (NaVec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: NaVec3, ca: NaVec3, k: f64, c: f64,
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
    axis: NaVec3, min_a: f64, max_a: f64,
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

struct LegHandles { thigh_idx: usize, shin_idx: usize, foot_idx: usize }
struct ArmHandles { upper_idx: usize, forearm_idx: usize, palm_idx: usize }

fn add_leg(world: &mut XpbdWorld, pelvis_idx: usize, side: Side) -> LegHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let hip_anchor = NaVec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0);
    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(world, pelvis_idx, thigh_idx,
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    add_axis(world, pelvis_idx, thigh_idx, NaVec3::new(-1.0, 0.0, 0.0),
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, NaVec3::new(0.0, 0.0, s),
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(world, pelvis_idx, thigh_idx, NaVec3::new(0.0, s, 0.0),
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_ROT_K, HIP_ROT_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    add_axis(world, thigh_idx, shin_idx, NaVec3::new(1.0, 0.0, 0.0),
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL,
        KNEE_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, NaVec3::new(0.0, 0.0, 1.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(world, thigh_idx, shin_idx, NaVec3::new(0.0, 1.0, 0.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    add_axis(world, shin_idx, foot_idx, NaVec3::new(1.0, 0.0, 0.0),
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, NaVec3::new(0.0, 0.0, s),
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(world, shin_idx, foot_idx, NaVec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0,
        TWIST_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);

    LegHandles { thigh_idx, shin_idx, foot_idx }
}

fn add_arm(world: &mut XpbdWorld, thoracic_idx: usize, side: Side) -> ArmHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let shoulder_anchor = NaVec3::new(s * SHOULDER_ANCHOR_THORACIC_X, THORACIC_HALF_H, 0.0);

    let mut upper = RigidBody::new_capsule(UPPER_ARM_MASS, UPPER_ARM_HALF_H, UPPER_ARM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        shoulder_anchor, NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0), id);
    upper.position = pos; upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(world, thoracic_idx, upper_idx,
        shoulder_anchor, NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);
    let (abd_min, abd_max) = if s > 0.0 { (ABD_MIN, ABD_MAX) } else { (-ABD_MAX, -ABD_MIN) };
    let (rot_min, rot_max) = if s > 0.0 { (ROT_MIN, ROT_MAX) } else { (-ROT_MAX, -ROT_MIN) };
    add_axis(world, thoracic_idx, upper_idx, NaVec3::new(-1.0, 0.0, 0.0),
        FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, NaVec3::new(0.0, 0.0, 1.0),
        abd_min, abd_max, ABD_K, ABD_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);
    add_axis(world, thoracic_idx, upper_idx, NaVec3::new(0.0, 1.0, 0.0),
        rot_min, rot_max, ROT_K, ROT_C, 0.0, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP, id);

    let mut forearm = RigidBody::new_capsule(FOREARM_MASS, FOREARM_HALF_H, FOREARM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[upper_idx],
        NaVec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        NaVec3::new(0.0, FOREARM_HALF_H, 0.0), id);
    forearm.position = pos; forearm.orientation = orient;
    let forearm_idx = world.add_body(forearm);

    add_elastic_joint(world, upper_idx, forearm_idx,
        NaVec3::new(0.0, -UPPER_ARM_HALF_H, 0.0),
        NaVec3::new(0.0, FOREARM_HALF_H, 0.0),
        ELBOW_JOINT_K, ELBOW_JOINT_C);
    add_axis(world, upper_idx, forearm_idx, NaVec3::new(-1.0, 0.0, 0.0),
        ELBOW_MIN, ELBOW_MAX, ELBOW_K, ELBOW_C, ELBOW_NEUTRAL,
        ELBOW_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, NaVec3::new(0.0, 0.0, 1.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);
    add_axis(world, upper_idx, forearm_idx, NaVec3::new(0.0, 1.0, 0.0),
        -ELBOW_OFF_AXIS_LIM, ELBOW_OFF_AXIS_LIM,
        ELBOW_OFF_AXIS_K, ELBOW_OFF_AXIS_C, 0.0,
        ELBOW_OFF_AXIS_LIMIT_K, 2.0, id);

    let mut palm = RigidBody::new_capsule(PALM_MASS, PALM_HALF_H, PALM_RADIUS);
    let (pos, orient) = place_child(&world.bodies[forearm_idx],
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0), id);
    palm.position = pos; palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    add_elastic_joint(world, forearm_idx, palm_idx,
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0, PALM_HALF_H, 0.0),
        WRIST_JOINT_K, WRIST_JOINT_C);
    let (dev_min, dev_max) = if s > 0.0 { (WRIST_DEV_MIN, WRIST_DEV_MAX) } else { (-WRIST_DEV_MAX, -WRIST_DEV_MIN) };
    add_axis(world, forearm_idx, palm_idx, NaVec3::new(-1.0, 0.0, 0.0),
        WRIST_FLEX_MIN, WRIST_FLEX_MAX, WRIST_FLEX_K, WRIST_FLEX_C, 0.0,
        WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, NaVec3::new(0.0, 0.0, 1.0),
        dev_min, dev_max, WRIST_DEV_K, WRIST_DEV_C, 0.0,
        WRIST_LIMIT_K, 2.0, id);
    add_axis(world, forearm_idx, palm_idx, NaVec3::new(0.0, 1.0, 0.0),
        -WRIST_TWIST_LIM, WRIST_TWIST_LIM, WRIST_TWIST_K, WRIST_TWIST_C, 0.0,
        WRIST_TWIST_LIMIT_K, 2.0, id);

    ArmHandles { upper_idx, forearm_idx, palm_idx }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    sim_time: f64,
    pelvis_idx: usize,
    head_idx: usize,
    bodies_to_render: Vec<usize>,
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
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    // Pelvis (UNPINNED) at the height that puts feet on ground.
    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    pelvis.position = NaVec3::new(0.0, 0.952, 0.0);
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Spine.
    let lumbar_anchor_pelvis = NaVec3::new(0.0, PELVIS_HALF_H, 0.0);
    let mut thoracic = RigidBody::new_capsule(THORACIC_MASS, THORACIC_HALF_H, THORACIC_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis, NaVec3::new(0.0, -THORACIC_HALF_H, 0.0), id);
    thoracic.position = pos; thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(&mut world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis, NaVec3::new(0.0, -THORACIC_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    add_axis(&mut world, pelvis_idx, thoracic_idx, NaVec3::new(1.0, 0.0, 0.0),
        L_FLEX_MIN, L_FLEX_MAX, L_FLEX_K, L_FLEX_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, NaVec3::new(0.0, 0.0, -1.0),
        L_BEND_MIN, L_BEND_MAX, L_BEND_K, L_BEND_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thoracic_idx, NaVec3::new(0.0, 1.0, 0.0),
        L_ROT_MIN, L_ROT_MAX, L_ROT_K, L_ROT_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP, id);

    let cerv_anchor_thoracic = NaVec3::new(0.0, THORACIC_HALF_H, 0.0);
    let mut head = RigidBody::new_capsule(HEAD_MASS, HEAD_HALF_H, HEAD_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        cerv_anchor_thoracic, NaVec3::new(0.0, -HEAD_HALF_H, 0.0), id);
    head.position = pos; head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic, NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);
    add_axis(&mut world, thoracic_idx, head_idx, NaVec3::new(1.0, 0.0, 0.0),
        C_FLEX_MIN, C_FLEX_MAX, C_FLEX_K, C_FLEX_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, NaVec3::new(0.0, 0.0, -1.0),
        C_BEND_MIN, C_BEND_MAX, C_BEND_K, C_BEND_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);
    add_axis(&mut world, thoracic_idx, head_idx, NaVec3::new(0.0, 1.0, 0.0),
        C_ROT_MIN, C_ROT_MAX, C_ROT_K, C_ROT_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP, id);

    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);
    let arm_r = add_arm(&mut world, thoracic_idx, Side::Right);
    let arm_l = add_arm(&mut world, thoracic_idx, Side::Left);

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. Pre-migration this binary disabled contact
    // for non-feet bodies via plane_y = -radius hacks.

    // ─── Meshes & materials ────────────────────────────────────────────
    let core_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let torso_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });
    let head_mat_h = materials.add(StandardMaterial { base_color: Color::srgb(0.92, 0.78, 0.65), ..default() });
    let arm_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let palm_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.75, 0.55, 0.42), ..default() });
    let leg_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let foot_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });

    let pelvis_mesh   = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let thoracic_mesh = meshes.add(Capsule3d::new(THORACIC_RADIUS as f32, (2.0 * THORACIC_HALF_H) as f32));
    let head_mesh     = meshes.add(Capsule3d::new(HEAD_RADIUS as f32, (2.0 * HEAD_HALF_H) as f32));
    let upper_mesh    = meshes.add(Capsule3d::new(UPPER_ARM_RADIUS as f32, (2.0 * UPPER_ARM_HALF_H) as f32));
    let forearm_mesh  = meshes.add(Capsule3d::new(FOREARM_RADIUS as f32, (2.0 * FOREARM_HALF_H) as f32));
    let palm_mesh     = meshes.add(Capsule3d::new(PALM_RADIUS as f32, (2.0 * PALM_HALF_H) as f32));
    let thigh_mesh    = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let shin_mesh     = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let foot_mesh     = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));

    let mut bodies_to_render = Vec::new();

    // Spine.
    commands.spawn((PbrBundle { mesh: pelvis_mesh, material: core_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default() },
        BodyRef(pelvis_idx)));
    bodies_to_render.push(pelvis_idx);
    commands.spawn((PbrBundle { mesh: thoracic_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[thoracic_idx]), ..default() },
        BodyRef(thoracic_idx)));
    bodies_to_render.push(thoracic_idx);
    commands.spawn((PbrBundle { mesh: head_mesh, material: head_mat_h,
        transform: rigid_body_to_transform(&world.bodies[head_idx]), ..default() },
        BodyRef(head_idx)));
    bodies_to_render.push(head_idx);

    for leg in [&leg_r, &leg_l] {
        commands.spawn((PbrBundle { mesh: thigh_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.thigh_idx]), ..default() },
            BodyRef(leg.thigh_idx)));
        bodies_to_render.push(leg.thigh_idx);
        commands.spawn((PbrBundle { mesh: shin_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.shin_idx]), ..default() },
            BodyRef(leg.shin_idx)));
        bodies_to_render.push(leg.shin_idx);
        commands.spawn((PbrBundle { mesh: foot_mesh.clone(), material: foot_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.foot_idx]), ..default() },
            BodyRef(leg.foot_idx)));
        bodies_to_render.push(leg.foot_idx);
    }

    for arm in [&arm_r, &arm_l] {
        commands.spawn((PbrBundle { mesh: upper_mesh.clone(), material: arm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.upper_idx]), ..default() },
            BodyRef(arm.upper_idx)));
        bodies_to_render.push(arm.upper_idx);
        commands.spawn((PbrBundle { mesh: forearm_mesh.clone(), material: arm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.forearm_idx]), ..default() },
            BodyRef(arm.forearm_idx)));
        bodies_to_render.push(arm.forearm_idx);
        commands.spawn((PbrBundle { mesh: palm_mesh.clone(), material: palm_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[arm.palm_idx]), ..default() },
            BodyRef(arm.palm_idx)));
        bodies_to_render.push(arm.palm_idx);
    }

    // Ground.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera: side-front view, framed for full-body topple.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.5, 1.20, 2.5)
            .looking_at(Vec3::new(0.0, 0.7, 0.0), Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: false, ..default() },
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
        world, sim_time: 0.0, pelvis_idx, head_idx, bodies_to_render,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut q: Query<(&BodyRef, &mut Transform)>,
) {
    for (body_ref, mut t) in &mut q {
        *t = rigid_body_to_transform(&phys.world.bodies[body_ref.0]);
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut q: Query<&mut Text, With<Readout>>,
) {
    let pelvis = &phys.world.bodies[phys.pelvis_idx];
    let head = &phys.world.bodies[phys.head_idx];
    for mut text in &mut q {
        text.sections[0].value = format!(
            "t: {:.2} s\n\nFull skeleton standing on the ground (no active balance).\n\npelvis y: {:+.3} m   head y: {:+.3} m\nhead z:   {:+.3} m   pelvis z: {:+.3} m\n\nThe skeleton starts upright and topples under gravity.\nIt's caught by ground contact as it lands.",
            phys.sim_time,
            pelvis.position.y, head.position.y,
            head.position.z, pelvis.position.z,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Skeleton standing (Phase 2 milestone — falls without active balance)".into(),
                resolution: (1100.0, 800.0).into(),
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
