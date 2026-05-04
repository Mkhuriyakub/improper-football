// Shoulder validation: pinned torso stub + upper arm hanging on a 3-DOF
// glenohumeral joint. First joint in the body where every rotational axis
// is a primary anatomical DOF, not a stay-near-zero off-axis lock:
//
//   Z (palm-local z): flexion / extension     — anatomical range  -50° to +180°
//   X (palm-local x): abduction / adduction   — anatomical range  -45° to +180°
//   Y (palm-local y): internal / external rot — anatomical range  -80° to  +90°
//
// All three axes have their own elastic spring + damper and a soft anatomical
// limit. There is no axis to "lock tight" — every direction is real motion.
//
// Caveat (per project_axis_angle_artifacts.md): our angular primitives use
// `scaled_axis().dot(axis)` which mixes a bivector cross-term into the third
// axis when two axes are simultaneously rotated. At the shoulder all three
// axes are active, so combined poses (e.g. flexed + abducted) will read
// non-zero on the Y projection even with no real internal rotation. The
// springs then drive REAL internal rotation in response. Equilibria for
// pure single-axis motions are unaffected; combined-pose equilibria show
// some axis cross-coupling. Tracking this as a known limitation; the proper
// fix is swing-twist Euler decomposition in the angular primitives.
//
// Validation: release upper arm from a deflected initial pose (Z = +0.6,
// X = +0.4, Y = +0.3). Expect settle to (0, 0, ?) — Z and X driven to
// zero by gravity + spring, Y settles to whatever the residual artifact
// drives it to (small but nonzero).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, Mat3};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 4.0;
const PRINT_INTERVAL: f64 = 0.10;

// Torso: pinned, vertical capsule. Mass = 0 so it doesn't move; the inertia
// only matters for visual rendering.
const TORSO_HALF_H: f64 = 0.20;
const TORSO_RADIUS: f64 = 0.15;

// Upper arm.
const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

// Elastic shoulder anchor.
const SHOULDER_JOINT_K: f64 = 100_000.0;
const SHOULDER_JOINT_C: f64 =     200.0;

// Per-axis biomechanics.
//   Z (flex/ext): primary forward/back motion. Spring restores to neutral 0.
//   X (abd/add):  out-to-side motion.
//   Y (int/ext):  axial rotation of humerus. Lower stiffness — in real life
//                 there's less restorative tone here than in flex/abd.
const SHOULDER_Z_K:       f64 = 3.0;
const SHOULDER_X_K:       f64 = 3.0;
const SHOULDER_Y_K:       f64 = 1.5;
const SHOULDER_Z_C:       f64 = 0.30;
const SHOULDER_X_C:       f64 = 0.30;
const SHOULDER_Y_C:       f64 = 0.10;

// Anatomical ranges (asymmetric).
const SHOULDER_Z_MIN: f64 = -0.87;   //  -50° extension
const SHOULDER_Z_MAX: f64 =  3.14;   // +180° full flexion overhead
const SHOULDER_X_MIN: f64 = -0.78;   //  -45° adduction (across body)
const SHOULDER_X_MAX: f64 =  3.14;   // +180° full abduction overhead
const SHOULDER_Y_MIN: f64 = -1.40;   //  -80° internal rotation
const SHOULDER_Y_MAX: f64 =  1.57;   //  +90° external rotation
const SHOULDER_LIMIT_K: f64 = 100.0;
const SHOULDER_LIMIT_EXP: f64 = 2.0;

const INITIAL_Z: f64 = 0.60;
const INITIAL_X: f64 = 0.40;
const INITIAL_Y: f64 = 0.30;

const TORSO_REST_POS: Vec3 = Vec3::new(0.0, 1.40, 0.0);
// Shoulder anchor on the torso, in torso-local coords. +x = right side,
// +y = above torso center (toward the head).
const SHOULDER_ANCHOR_TORSO_X: f64 = 0.15;
const SHOULDER_ANCHOR_TORSO_Y: f64 = 0.18;

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
    parent_anchor_local: Vec3, child_anchor_local: Vec3,
    stiffness: f64, damping: f64,
) {
    let mut spring = PositionalSpring::new(parent, child, stiffness);
    spring.local_anchor_a = parent_anchor_local;
    spring.local_anchor_b = child_anchor_local;
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(parent, child, damping);
    damp.local_anchor_a = parent_anchor_local;
    damp.local_anchor_b = child_anchor_local;
    world.positional_dampers.push(damp);
}

fn add_shoulder_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    // Z axis: flex/ext.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        SHOULDER_Z_MIN, SHOULDER_Z_MAX, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), SHOULDER_Z_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), SHOULDER_Z_K, 0.0));

    // X axis: abd/add.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(),
        SHOULDER_X_MIN, SHOULDER_X_MAX, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), SHOULDER_X_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), SHOULDER_X_K, 0.0));

    // Y axis: int/ext rotation.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(),
        SHOULDER_Y_MIN, SHOULDER_Y_MAX, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), SHOULDER_Y_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), SHOULDER_Y_K, 0.0));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Torso: pinned vertical capsule, identity orientation.
    let mut torso = RigidBody::new(0.0, capsule_inertia(50.0, TORSO_RADIUS, TORSO_HALF_H));
    torso.position = TORSO_REST_POS;
    torso.orientation = UnitQuaternion::identity();
    let torso_idx = world.add_body(torso);

    // Upper arm: hangs straight down from the shoulder anchor in initial pose.
    // We apply the initial deflection as a relative rotation (Z, X, Y all
    // composed in that order).
    let initial_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_Z))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_X, 0.0, 0.0))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(0.0, INITIAL_Y, 0.0));
    let mut upper = RigidBody::new(UPPER_ARM_MASS,
        capsule_inertia(UPPER_ARM_MASS, UPPER_ARM_RADIUS, UPPER_ARM_HALF_H));
    let shoulder_anchor_torso = Vec3::new(SHOULDER_ANCHOR_TORSO_X, SHOULDER_ANCHOR_TORSO_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[torso_idx],
        shoulder_anchor_torso,
        Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        initial_rel);
    upper.position = pos;
    upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(&mut world, torso_idx, upper_idx,
        shoulder_anchor_torso,
        Vec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);
    add_shoulder_biomech(&mut world, torso_idx, upper_idx);

    println!("SHOULDER COMPLIANT (3-DOF ball + full biomech)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("Initial: Z={}, X={}, Y={}", INITIAL_Z, INITIAL_X, INITIAL_Y);
    println!();
    println!("{:<6} {:<11} {:<11} {:<11}",
             "t", "Z (flex)", "X (abd)", "Y (rot)");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let torso = &world.bodies[torso_idx];
            let upper = &world.bodies[upper_idx];
            let rel = (upper.orientation * torso.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.z, rel.x, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let torso = &world.bodies[torso_idx];
    let upper = &world.bodies[upper_idx];
    let rel = (upper.orientation * torso.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: Z = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       X = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       Y = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());
    println!();
    println!("Predicted: with no gravity-induced lever (arm hanging straight),");
    println!("equilibrium should be (0, 0, ~0). Any residual Y is the bivector");
    println!("artifact from the relaxation path through nonzero Z and X.");
}
