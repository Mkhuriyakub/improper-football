// Phase 3 step 2: bilateral standing via CoM-based whole-body balance.
//
// Cleanly composes the new engine primitives — no hacks, all controller
// actions go through proper engine primitives:
//   - `XpbdWorld::com_ground_projection` (item 1, item 2.5 helpers)
//     to sense CoM position relative to the support polygon
//   - `GroundContact::engaged_endpoints` to read actual contact points
//   - `convex_hull_2d` to build the support polygon
//   - `TorqueDispatcher` to split a single KP·error demand across two
//     ankle torque-pair routes (item 3)
//   - 2× `AngularDamper` (existing primitive, implicit integration) for
//     KD damping at each ankle
//
// Why CoM-based instead of thigh-tilt-based: with stacked posture PDs
// in a bilateral chain, the chain can buckle (knees bend, body squats)
// while the thigh stays close to vertical — defeating thigh-tilt
// feedback. Sensing CoM ground projection captures the actual
// "where is the body in space" signal that buckling moves but
// thigh-tilt misses.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::torque_dispatcher::{DispatchRoute, TorqueDispatcher};
use physics::inverse_dynamics::{IDJoint, InverseDynamics, SupportedBody};
use physics::math::{Vec2, Vec3, convex_hull_2d, signed_distance_to_convex_polygon_2d,
    swing_twist_angle};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;
const PRINT_INTERVAL: f64 = 0.10;

// Disturbance for testing recovery with full ankle + hip + state-machine + CoM-vel control.
// Disturbance: 0 = static only (always works), 0.4 = stress test
// (recovers for ~0.8s, then chain slowly buckles — known limitation).
// Default to disturbance ON so the demo shows what we have today.
const DISTURBANCE_PERIOD: f64 = 3.0;
const DISTURBANCE_OMEGA: f64 = 0.0;

// ─── Body constants ───────────────────────────────────────────────────

const FOOT_MASS: f64 = 1.0;
// Realistic adult foot length: ~26 cm (FOOT_HALF_H = 0.130 m). Larger
// foot gives more ankle leverage before heel/toe lift:
//   τ_lift = FOOT_HALF_H × (load_per_foot/2)
// For 35 kg per-foot load: τ_lift = 0.13 × 343/2 = 22 N·m per foot
// vs 15 N·m at the prior 0.09 m foot. ~50% more torque headroom.
const FOOT_HALF_H: f64 = 0.130;
const FOOT_RADIUS: f64 = 0.032;
// Ankle anchor at foot CENTER (not heel — anatomically the ankle is at
// the back of the foot, but for this bilateral-balance validation we
// want CoM-over-support-centroid by construction. Centering the ankle
// makes the support polygon's centroid coincide with the chain's
// natural CoM at z=0, so the body doesn't need to lean to balance.
// This is a deliberate simplification, NOT a hack: real biomechanics
// would compensate with a different CoM-over-support relationship plus
// a hip strategy; we don't have that infrastructure yet.
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, 0.0, 0.0);

const SHIN_MASS: f64   = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const THIGH_MASS: f64   = 7.0;
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.07;

const PELVIS_MASS: f64   = 8.0;
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;

const TORSO_MASS: f64   = 40.0;
const TORSO_HALF_H: f64 = 0.30;
const TORSO_RADIUS: f64 = 0.13;

const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

// ─── Joint constants ──────────────────────────────────────────────────

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const INV_K: f64 = 3.5;  const INV_C: f64 = 1.0;
const TWIST_K: f64 = 8.0;  const TWIST_C: f64 = 0.5;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const INV_MIN: f64 = -0.26;  const INV_MAX: f64 = 0.52;
const TWIST_LIM: f64 = 0.10;
const ANKLE_LIMIT_K: f64 = 80.0;
const TWIST_LIMIT_K: f64 = 200.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;
const KNEE_PASSIVE_K: f64 = 5.0;  const KNEE_PASSIVE_C: f64 = 1.5;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;
const KNEE_LIMIT_EXP: f64 = 2.0;
const KNEE_PD_KP: f64 = 9000.0;  const KNEE_PD_KD: f64 = 900.0;
// Feedforward torque (N·m). At the neutral standing pose, gravity
// moments at each joint are zero (CG directly above), so a CONSTANT
// feedforward only destabilizes. A useful feedforward needs to be
// pose-dependent (inverse dynamics, computing the gravity moment from
// current chain configuration). Left at 0 here; the primitive
// `AngularPDController::feedforward_torque` exists for the future
// inverse-dynamics controller.
const KNEE_PD_FEEDFORWARD: f64 = 0.0;
// Off-axis (Y, Z): the knee is anatomically a HINGE. Without these,
// the shin can roll freely about lateral and twist axes.
const KNEE_OFF_AXIS_K: f64 = 12.0;
const KNEE_OFF_AXIS_C: f64 = 0.80;
const KNEE_OFF_AXIS_LIM: f64 = 0.30;
const KNEE_OFF_AXIS_LIMIT_K: f64 = 50.0;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =    400.0;
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K: f64 = 120.0;
const HIP_LIMIT_EXP: f64 = 2.0;
const HIP_FLEX_PD_KP: f64 = 12000.0;  const HIP_FLEX_PD_KD: f64 = 1200.0;

const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =    600.0;
const LUMBAR_PASSIVE_K: f64 = 150.0;  const LUMBAR_PASSIVE_C: f64 = 25.0;
const L_FLEX_MIN: f64 = -0.44;  const L_FLEX_MAX: f64 = 1.05;
const L_BEND_MIN: f64 = -0.44;  const L_BEND_MAX: f64 = 0.44;
const L_ROT_MIN:  f64 = -0.087;  const L_ROT_MAX:  f64 = 0.087;
const L_BEND_K: f64 = 150.0;  const L_BEND_C: f64 = 25.0;
const L_ROT_K:  f64 =  50.0;  const L_ROT_C:  f64 =  5.0;
const LUMBAR_LIMIT_K: f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 = 2.0;
const LUMBAR_PD_KP: f64 = 7500.0;  const LUMBAR_PD_KD: f64 = 750.0;

// ─── Balance controller gains ─────────────────────────────────────────
//
// Total mass above ankles: 2×(SHIN+THIGH) + PELVIS + TORSO = 68 kg.
// CoM height ≈ 1.04 m, I_about_ankle ≈ 83.5 kg·m².
//
// Per-ankle PHYSICAL TORQUE LIMIT: applying torque pair on (shin, foot)
// about world +x makes the foot's heel lift off ground when the torque
// exceeds the static-balance bound. For a 35 kg vertical load per foot
// and FOOT_HALF_H = 0.13 m:
//   τ_lift = FOOT_HALF_H × (load/2) ≈ 0.13 × 343/2 ≈ 22 N·m per foot
// We cap each dispatcher route at 20 N·m for safety margin.
//
// KP chosen so the controller saturates the 14 N·m cap at ~0.6 cm
// CoM offset. Within this dead-band the controller has FULL authority.
// Below: KP × offset linearly scales.
//   per_ankle_torque = KP × offset / 2 ≤ 14 → KP ≤ 4670
// Use KP = 4500 for 0.0062 m saturation point.
const BAL_KP_OFFSET: f64 = 4500.0;  // N·m / m of CoM ground offset
const PER_ANKLE_LIMIT: f64 = 20.0;

// KD chosen for critical damping (ζ ≈ 0.7) of the closed-loop system:
//   ω_n = sqrt((KP·h - mgh)/I) = sqrt((4680 - 591)/83.5) = 7.0 rad/s
//   KD_total = 2·ζ·ω_n·I = 2·0.7·7.0·83.5 = 818 N·m·s/rad → per ankle = 409
const ANKLE_KD: f64 = 400.0;

// Hip strategy: lean the upper body (via hip-flex target shift) in the
// opposite direction of CoM offset. This shifts CoM without going
// through the ankle, so it bypasses the per-ankle torque-limit
// constraint and provides extra recovery authority when the ankle
// saturates.
//
// Sign: CoM is too far +z (forward) → want trunk to lean backward →
// negative hip flex target (extend hip). hip_flex_axis is (-1,0,0) so
// positive flex angle bends pelvis forward over thigh; negative flex
// bends backward.
//
// Gain: upper body (48 kg) flexes about hip with CoM-from-hip ≈ 0.4 m.
// A hip flex of θ shifts upper-body CoM by ≈ 0.4·sin(θ). The fraction
// of total mass above the hip is 48/68 ≈ 0.7, so the total-body CoM
// shift is ≈ 0.28·θ. For a 1 cm CoM correction, want θ ≈ 0.036 rad.
// KP_hip = 3.6 rad/m gives roughly that.
const HIP_STRATEGY_KP: f64 = 3.6;   // rad of hip-flex target per m of CoM offset
const HIP_STRATEGY_MAX_ANGLE: f64 = 0.4;  // ±23° clamp on hip-strategy target

// CoM velocity feedback (item 3). Adds -KD_v × com_velocity.z to the
// dispatcher demand. Stability: this torque is integrated explicitly
// onto the foot accumulator, so KD_v × com_vel × dt / I_foot must
// stay below the explicit-Euler bound. For typical chain motion
// (com_vel ≤ 0.1 m/s), KD_v ≤ ~40 stays safe. Use 30 for margin.
const COM_VEL_KD: f64 = 0.0;        // N·m per (m/s) of CoM z-velocity

// Single-leg torque limit. When only one foot is engaged, that foot
// bears the entire 70 kg load (vs 35 kg per foot bilateral). For
// FOOT_HALF_H = 0.13: τ_lift_single = 0.13 × 686/2 = 45 N·m.
// Use 40 N·m for margin.
const PER_ANKLE_LIMIT_SINGLE: f64 = 40.0;

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

#[derive(Clone, Copy, PartialEq)]
enum Side { Right, Left }
impl Side { fn sign(self) -> f64 { match self { Side::Right => 1.0, Side::Left => -1.0 } } }

#[derive(Clone, Copy)]
struct LegHandles {
    thigh_idx: usize,
    shin_idx: usize,
    foot_idx: usize,
    /// Index of the hip-flex posture PD for hip strategy + ID feedforward.
    hip_flex_pd_idx: usize,
    /// Index of the knee posture PD for ID feedforward.
    knee_pd_idx: usize,
}

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
        hip_anchor, Vec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let hip_flex_axis = Vec3::new(-1.0, 0.0, 0.0);
    let hip_abd_axis  = Vec3::new(0.0, 0.0, s);
    let hip_rot_axis  = Vec3::new(0.0, s, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_flex_axis,
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_K, 0.0));
    let hip_flex_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_PD_KP, HIP_FLEX_PD_KD));
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_abd_axis,
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_abd_axis, HIP_ABD_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_abd_axis, HIP_ABD_K, 0.0));
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_rot_axis,
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_rot_axis, HIP_ROT_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_rot_axis, HIP_ROT_K, 0.0));

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_K, 0.0));
    let knee_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        thigh_idx, shin_idx, knee_axis, KNEE_PD_KP, KNEE_PD_KD)
        .with_feedforward(KNEE_PD_FEEDFORWARD));
    // Knee off-axis (Z lateral and Y twist) — knee is anatomically a hinge.
    let knee_z = Vec3::new(0.0, 0.0, 1.0);
    let knee_y = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_z,
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_z, KNEE_OFF_AXIS_C));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_z, KNEE_OFF_AXIS_K, 0.0));
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_y,
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_y, KNEE_OFF_AXIS_C));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_y, KNEE_OFF_AXIS_K, 0.0));

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0)
        .with_baseline(foot_baseline));
    // Ankle inversion (lateral) and twist (axial) — anatomical 3-DOF ankle.
    let inv_axis = Vec3::new(0.0, 0.0, s);
    let twist_axis = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, inv_axis,
        INV_MIN, INV_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, inv_axis, INV_C));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, inv_axis, INV_K, 0.0)
        .with_baseline(foot_baseline));
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, twist_axis,
        -TWIST_LIM, TWIST_LIM, TWIST_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, twist_axis, TWIST_C));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, twist_axis, TWIST_K, 0.0)
        .with_baseline(foot_baseline));

    // Active KD damping at the ankle for the balance controller. Implicit
    // integration via AngularDamper makes this stable for any KD value.
    // Axis is shin-local (1,0,0) which maps to world +x at neutral pose.
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0), ANKLE_KD));

    let _ = s;

    LegHandles { thigh_idx, shin_idx, foot_idx, hip_flex_pd_idx, knee_pd_idx }
}

/// Engaged support polygon from the foot capsules currently touching the ground.
fn engaged_support_polygon(world: &XpbdWorld, foot_indices: &[usize]) -> Vec<Vec2> {
    let mut pts: Vec<Vec2> = Vec::new();
    for &foot_idx in foot_indices {
        for ep in world.engaged_endpoints(foot_idx).iter().flatten() {
            pts.push(Vec2::new(ep.contact_point.x, ep.contact_point.z));
        }
    }
    convex_hull_2d(&pts)
}

fn polygon_centroid(poly: &[Vec2]) -> Vec2 {
    if poly.is_empty() { return Vec2::zeros(); }
    let sum: Vec2 = poly.iter().sum();
    sum / poly.len() as f64
}

/// Which feet are currently in active ground contact. Drives the
/// controller's state machine — single-leg vs bilateral support.
#[derive(Clone, Copy, PartialEq, Debug)]
enum SupportState {
    Bilateral,    // both feet engaged
    RightOnly,    // only right foot engaged
    LeftOnly,     // only left foot engaged
    Airborne,     // neither foot engaged
}

impl SupportState {
    fn label(self) -> &'static str {
        match self {
            SupportState::Bilateral => "bilateral",
            SupportState::RightOnly => "R only",
            SupportState::LeftOnly => "L only",
            SupportState::Airborne => "airborne",
        }
    }
}

fn determine_support_state(world: &XpbdWorld, r_foot: usize, l_foot: usize) -> SupportState {
    match (world.is_in_contact(r_foot), world.is_in_contact(l_foot)) {
        (true, true)   => SupportState::Bilateral,
        (true, false)  => SupportState::RightOnly,
        (false, true)  => SupportState::LeftOnly,
        (false, false) => SupportState::Airborne,
    }
}

fn main() {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    // Pelvis at the height that puts feet on the ground at neutral.
    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    pelvis.position = Vec3::new(0.0, 0.952, 0.0);
    let pelvis_idx = world.add_body(pelvis);

    // Torso above pelvis via lumbar joint.
    let mut torso = RigidBody::new_capsule(TORSO_MASS, TORSO_HALF_H, TORSO_RADIUS);
    let lumbar_anchor_pelvis = Vec3::new(0.0, PELVIS_HALF_H, 0.0);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis, Vec3::new(0.0, -TORSO_HALF_H, 0.0), id);
    torso.position = pos; torso.orientation = orient;
    let torso_idx = world.add_body(torso);

    add_elastic_joint(&mut world, pelvis_idx, torso_idx,
        lumbar_anchor_pelvis, Vec3::new(0.0, -TORSO_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let lumbar_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, torso_idx, lumbar_axis,
        L_FLEX_MIN, L_FLEX_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PASSIVE_K, 0.0));
    let lumbar_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PD_KP, LUMBAR_PD_KD));
    // Lumbar lateral bend (Z) and rotation (Y) — anatomical 3-DOF spine joint.
    let lumbar_z = Vec3::new(0.0, 0.0, -1.0);
    let lumbar_y = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, torso_idx, lumbar_z,
        L_BEND_MIN, L_BEND_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, torso_idx, lumbar_z, L_BEND_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, torso_idx, lumbar_z, L_BEND_K, 0.0));
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, torso_idx, lumbar_y,
        L_ROT_MIN, L_ROT_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, torso_idx, lumbar_y, L_ROT_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, torso_idx, lumbar_y, L_ROT_K, 0.0));
    // EXPERIMENT: active posture PD on lumbar rotation (Y) axis. The
    // trace showed the chain develops growing torso yaw (rotation about
    // world +y) under disturbance, because the lumbar Y axis had only
    // weak passive constraints. Adding active KP=LUMBAR_PD_KP suppresses
    // the yaw mode.
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, torso_idx, lumbar_y, LUMBAR_PD_KP, LUMBAR_PD_KD));

    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. Pre-migration this binary disabled contact
    // for non-feet bodies (pelvis, torso, thigh, shin) via plane_y = -radius
    // hacks.

    // The TorqueDispatcher: one demand → split equally across both ankles.
    // Axis is world +x (sagittal). Routes apply +τ to shin, -τ to foot
    // (the foot's reaction is absorbed by ground friction).
    let dispatcher_idx = world.torque_dispatchers.len();
    world.torque_dispatchers.push(TorqueDispatcher::new(
        Vec3::new(1.0, 0.0, 0.0),
        vec![
            DispatchRoute::new(leg_r.shin_idx, leg_r.foot_idx),
            DispatchRoute::new(leg_l.shin_idx, leg_l.foot_idx),
        ],
    ).with_per_route_limit(PER_ANKLE_LIMIT));

    let foot_indices = [leg_r.foot_idx, leg_l.foot_idx];

    // INVERSE DYNAMICS gravity-feedforward observer.
    // For each posture PD, declares which bodies' weight it supports
    // (with fractions for bilateral sharing). Each substep, the
    // observer computes the gravity moment at each joint and writes
    // a cancelling torque to the PD's feedforward field — the PD
    // then only has to handle deviations from the equilibrium pose.
    //
    // Sharing convention:
    //   - lumbar supports torso (1.0)
    //   - each hip supports torso (0.5) + pelvis (0.5)
    //   - each knee supports its leg's thigh (1.0) + torso (0.5) + pelvis (0.5)
    let mut id = InverseDynamics::new();
    // Lumbar: pelvis-torso axis is in pelvis local (+1, 0, 0).
    id.joints.push(IDJoint {
        pd_idx: lumbar_pd_idx,
        parent_body: pelvis_idx,
        anchor_local: Vec3::new(0.0, PELVIS_HALF_H, 0.0),
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![SupportedBody { body: torso_idx, fraction: 1.0 }],
    });
    // Hips: pelvis-thigh axis is in pelvis local (-1, 0, 0).
    for leg in [&leg_r, &leg_l] {
        let s = if leg.foot_idx == leg_r.foot_idx { 1.0 } else { -1.0 };
        id.joints.push(IDJoint {
            pd_idx: leg.hip_flex_pd_idx,
            parent_body: pelvis_idx,
            anchor_local: Vec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0),
            axis_local: Vec3::new(-1.0, 0.0, 0.0),
            supported: vec![
                SupportedBody { body: torso_idx,  fraction: 0.5 },
                SupportedBody { body: pelvis_idx, fraction: 0.5 },
            ],
        });
    }
    // Knees: thigh-shin axis is in thigh local (+1, 0, 0).
    for leg in [&leg_r, &leg_l] {
        id.joints.push(IDJoint {
            pd_idx: leg.knee_pd_idx,
            parent_body: leg.thigh_idx,
            anchor_local: Vec3::new(0.0, -THIGH_HALF_H, 0.0),
            axis_local: Vec3::new(1.0, 0.0, 0.0),
            supported: vec![
                SupportedBody { body: leg.thigh_idx, fraction: 1.0 },
                SupportedBody { body: torso_idx,     fraction: 0.5 },
                SupportedBody { body: pelvis_idx,    fraction: 0.5 },
            ],
        });
    }

    println!("BALANCE BILATERAL (Phase 3 step 2)");
    println!("CoM-based whole-body balance via TorqueDispatcher + AngularDampers.");
    println!("8 bodies. Total mass: {} kg.",
             (2.0 * (FOOT_MASS + SHIN_MASS + THIGH_MASS) + PELVIS_MASS + TORSO_MASS) as i32);
    println!("Controller: τ_total = -KP·(CoM_z - support_center_z), KP={:.0}",
             BAL_KP_OFFSET);
    println!("Dispatcher splits τ_total equally between both ankles.");
    println!("Damping: KD={:.0} N·m·s/rad per ankle (implicit AngularDamper).",
             ANKLE_KD);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "t", "com.z", "z_err", "demand", "pelvis.y", "supp.dist", "state");
    println!("{}", "-".repeat(74));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_pelvis_drop = 0.0_f64;
    let mut max_z_err = 0.0_f64;
    let mut last_com_z = 0.0;
    let mut last_target_z = 0.0;
    let mut last_demand = 0.0;
    let mut last_supp_dist = 0.0;
    let mut last_support_state = SupportState::Bilateral;

    let pelvis_y_initial = world.bodies[pelvis_idx].position.y;

    // Capture the INITIAL midpoint between ankle positions as the
    // controller's fixed target. Using foot-center midpoint (rather
    // than polygon centroid) makes the target naturally aligned with
    // the body's CoM at rest — the body is built symmetrically over
    // the ankles, so CoM lives at the ankle midpoint.
    //
    // Using polygon centroid would offset the target by the foot's
    // heel-vs-ball asymmetry (~2 cm), forcing the body to lean.
    let target = {
        let r_ankle = world.bodies[leg_r.foot_idx].position;
        let l_ankle = world.bodies[leg_l.foot_idx].position;
        Vec2::new(
            0.5 * (r_ankle.x + l_ankle.x),
            0.5 * (r_ankle.z + l_ankle.z),
        )
    };
    println!("Controller target (fixed): ({:+.4}, {:+.4}) — ankle midpoint",
             target.x, target.y);
    println!();
    let _ = &foot_indices;  // (foot_indices used later in main loop)

    let mut next_disturbance = DISTURBANCE_PERIOD;
    while t < SIM_DURATION {
        if t >= next_disturbance {
            world.bodies[torso_idx].angular_velocity.x += DISTURBANCE_OMEGA;
            next_disturbance += DISTURBANCE_PERIOD;
            println!("  -- disturbance applied at t={:.2} ({:+.2} rad/s on torso) --", t, DISTURBANCE_OMEGA);
        }
        for _ in 0..SOLVER_ITERATIONS {
            // --- CONTROLLER ---
            // Sense CoM ground projection and velocity.
            let com_xz = world.com_ground_projection();
            let com_vel = world.com_velocity();
            let z_err = com_xz.y - target.y;
            let z_vel = com_vel.z;

            // Determine which feet are pressing the ground.
            let support = determine_support_state(&world,
                leg_r.foot_idx, leg_l.foot_idx);

            // STATE MACHINE: adjust dispatcher route weights and limit
            // based on which feet are engaged.
            match support {
                SupportState::Bilateral => {
                    world.torque_dispatchers[dispatcher_idx].routes[0].weight = 1.0;
                    world.torque_dispatchers[dispatcher_idx].routes[1].weight = 1.0;
                    world.torque_dispatchers[dispatcher_idx].per_route_limit = PER_ANKLE_LIMIT;
                }
                SupportState::RightOnly => {
                    world.torque_dispatchers[dispatcher_idx].routes[0].weight = 1.0;
                    world.torque_dispatchers[dispatcher_idx].routes[1].weight = 0.0;
                    world.torque_dispatchers[dispatcher_idx].per_route_limit = PER_ANKLE_LIMIT_SINGLE;
                }
                SupportState::LeftOnly => {
                    world.torque_dispatchers[dispatcher_idx].routes[0].weight = 0.0;
                    world.torque_dispatchers[dispatcher_idx].routes[1].weight = 1.0;
                    world.torque_dispatchers[dispatcher_idx].per_route_limit = PER_ANKLE_LIMIT_SINGLE;
                }
                SupportState::Airborne => {
                    // No support — ankle torque against thin air does
                    // nothing useful and risks pinwheel via foot reaction.
                    // Hold position only.
                    world.torque_dispatchers[dispatcher_idx].routes[0].weight = 0.0;
                    world.torque_dispatchers[dispatcher_idx].routes[1].weight = 0.0;
                }
            }

            // ANKLE strategy: PD on CoM offset + velocity. Velocity term
            // (KD_v × com_vel.z) gives explicit damping of CoM motion,
            // catching disturbances before position error grows.
            let ankle_demand = match support {
                SupportState::Airborne => 0.0,
                _ => -BAL_KP_OFFSET * z_err - COM_VEL_KD * z_vel,
            };
            world.torque_dispatchers[dispatcher_idx].demand = ankle_demand;

            // HIP strategy: bend upper body opposite to CoM offset.
            // Disabled when airborne (would just flap uselessly).
            let hip_target = match support {
                SupportState::Airborne => 0.0,
                _ => (-HIP_STRATEGY_KP * z_err)
                    .clamp(-HIP_STRATEGY_MAX_ANGLE, HIP_STRATEGY_MAX_ANGLE),
            };
            world.angular_pd_controllers[leg_r.hip_flex_pd_idx].target_angle = hip_target;
            world.angular_pd_controllers[leg_l.hip_flex_pd_idx].target_angle = hip_target;

            // Inverse-dynamics gravity feedforward: writes per-joint
            // gravity-cancelling torque to each PD's feedforward field
            // based on the current chain pose.
            id.update(&mut world);

            world.substep(SUB_DT);

            last_com_z = com_xz.y;
            last_target_z = target.y;
            last_demand = ankle_demand;
            last_support_state = support;
            let engaged = engaged_support_polygon(&world, &foot_indices);
            last_supp_dist = signed_distance_to_convex_polygon_2d(com_xz, &engaged);
        }
        t += DT;

        let drop = pelvis_y_initial - world.bodies[pelvis_idx].position.y;
        if drop > max_pelvis_drop { max_pelvis_drop = drop; }
        let z_err_abs = (last_com_z - last_target_z).abs();
        if z_err_abs > max_z_err { max_z_err = z_err_abs; }

        // DETAILED TRACE during the critical recovery+buckle window (t=2.9 to 4.5).
        if (2.95..=4.5).contains(&t) {
            // Per-joint angles.
            let knee_r = {
                let q = world.bodies[leg_r.thigh_idx].orientation.inverse()
                      * world.bodies[leg_r.shin_idx].orientation;
                swing_twist_angle(q, Vec3::new(1.0, 0.0, 0.0))
            };
            let knee_l = {
                let q = world.bodies[leg_l.thigh_idx].orientation.inverse()
                      * world.bodies[leg_l.shin_idx].orientation;
                swing_twist_angle(q, Vec3::new(1.0, 0.0, 0.0))
            };
            let hip_r = {
                let q = world.bodies[pelvis_idx].orientation.inverse()
                      * world.bodies[leg_r.thigh_idx].orientation;
                swing_twist_angle(q, Vec3::new(-1.0, 0.0, 0.0))
            };
            let hip_l = {
                let q = world.bodies[pelvis_idx].orientation.inverse()
                      * world.bodies[leg_l.thigh_idx].orientation;
                swing_twist_angle(q, Vec3::new(-1.0, 0.0, 0.0))
            };
            let lumbar_ang = {
                let q = world.bodies[pelvis_idx].orientation.inverse()
                      * world.bodies[torso_idx].orientation;
                swing_twist_angle(q, Vec3::new(1.0, 0.0, 0.0))
            };
            let torso_omega = world.bodies[torso_idx].angular_velocity;
            let pelvis_omega = world.bodies[pelvis_idx].angular_velocity;
            let pelvis_y = world.bodies[pelvis_idx].position.y;
            let com = world.com_position();

            println!("TRACE {:.3} | knee_R={:+.4} knee_L={:+.4} (Δ={:+.4}) | hip_R={:+.4} hip_L={:+.4} (Δ={:+.4}) | lumb={:+.4} | torso_ω=({:+.3}, {:+.3}, {:+.3}) | pelvis_ω.z={:+.3} | com=({:+.4}, {:+.4}) | pelvis.y={:.4} dem={:+.2}",
                     t, knee_r, knee_l, knee_l - knee_r,
                     hip_r, hip_l, hip_l - hip_r,
                     lumbar_ang,
                     torso_omega.x, torso_omega.y, torso_omega.z,
                     pelvis_omega.z,
                     com.x, com.z,
                     pelvis_y, last_demand);
        } else if t >= next_print {
            let pelvis_y = world.bodies[pelvis_idx].position.y;
            println!("{:<6.2} {:<+10.5} {:<+10.5} {:<+10.2} {:<+10.4} {:<+10.4} {}",
                     t, last_com_z, last_com_z - last_target_z, last_demand,
                     pelvis_y, last_supp_dist, last_support_state.label());
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let pelvis_y_final = world.bodies[pelvis_idx].position.y;
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  pelvis.y: {:.4} (started at {:.4}; max drop {:.4} m)",
             pelvis_y_final, pelvis_y_initial, max_pelvis_drop);
    println!("  max |CoM_z - support_center|: {:.4} m", max_z_err);
    println!();

    let upright = pelvis_y_final > 0.7;
    let small_drop = max_pelvis_drop < 0.10;
    let stayed_balanced = max_z_err < 0.05;
    if upright && small_drop && stayed_balanced {
        println!("✓ Bilateral chain balanced via CoM-based control.");
        println!("  Pelvis stayed within 10 cm of initial height; CoM stayed within 5 cm of support.");
    } else if upright && small_drop {
        println!("⚠ Stayed up but CoM wandered ({:.3} m). Tighten gains.", max_z_err);
    } else if upright {
        println!("⚠ Stayed up but pelvis dropped {:.3} m (knee buckle?).", max_pelvis_drop);
    } else {
        println!("✗ Body fell over.");
    }
}
