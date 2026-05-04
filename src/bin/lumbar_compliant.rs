// Lumbar validation: pinned pelvis + thoracic stub sitting upright on a
// 3-DOF lumbar joint. Direct analog of the hip ball-and-socket — same
// joint primitives, same axis conventions — but the child body sits ABOVE
// the parent rather than hanging below, so gravity is *destabilizing*
// rather than restoring (any tilt makes it tip further). The biomech
// spring stiffness has to exceed the gravity destabilization for the
// upright pose to be stable:
//
//   K > m·g·half_h  ≈ 25·9.81·0.18 ≈ 44 N·m/rad
//
// Joint axes (pelvis-local frame = world at rest):
//   Z (forward/back): flexion / extension. Positive = forward flex.
//   X (sideways):     lateral bend (right = +x for a frontal observer).
//   Y (vertical):     axial rotation. Anatomically very limited (±5°).
//
// Anatomical ROM:
//   Z (flex/ext):  -0.44 (-25° ext) to +1.05 (+60° flex)
//   X (bend):      ±0.44 (±25°)
//   Y (rot):       ±0.087 (±5°) — very tight
//
// Validation: release thoracic from a deflected initial pose
// (Z = +0.30, X = +0.20, Y = +0.05). Expect settle to (0, 0, ~0) with
// any residual Y from the bivector artifact (project_axis_angle_artifacts).

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

// Pelvis: pinned, vertical capsule. Same geometry as hip_compliant.
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;

// Thoracic stub — chest + abdominal organs above lumbar. Heavier than
// the thigh because it carries everything above the waist (sans head and
// arms which we model separately later).
const THORACIC_MASS:    f64 = 25.0;
const THORACIC_RADIUS:  f64 = 0.130;
const THORACIC_HALF_H:  f64 = 0.180;

// Elastic lumbar anchor — very stiff (intervertebral disc + ligaments
// hold pelvis-to-thoracic alignment under heavy load).
const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =     600.0;

// Per-axis biomech. Stiffness must exceed the gravity destabilization
// (K_grav ≈ 44 for our thoracic) for the upright pose to be stable.
//   Z (flex/ext): K=150, ζ ≈ 0.94 against (K - K_grav) = 106 effective.
//   X (bend):     same.
//   Y (rot):      no gravity coupling (axis is along body length); K=50
//                 with ζ ≈ 0.77 against I_axial only.
const LUMBAR_FLEX_K: f64 = 150.0;
const LUMBAR_BEND_K: f64 = 150.0;
const LUMBAR_ROT_K:  f64 =  50.0;
const LUMBAR_FLEX_C: f64 =  25.0;
const LUMBAR_BEND_C: f64 =  25.0;
const LUMBAR_ROT_C:  f64 =   5.0;

const LUMBAR_FLEX_MIN: f64 = -0.44;   //  -25° extension
const LUMBAR_FLEX_MAX: f64 =  1.05;   //  +60° flexion
const LUMBAR_BEND_MIN: f64 = -0.44;   //  -25° bend
const LUMBAR_BEND_MAX: f64 =  0.44;   //  +25° bend
const LUMBAR_ROT_MIN:  f64 = -0.087;  //  -5°  rotation
const LUMBAR_ROT_MAX:  f64 =  0.087;  //  +5°  rotation
const LUMBAR_LIMIT_K:    f64 = 200.0;
const LUMBAR_LIMIT_EXP:  f64 =   2.0;

const INITIAL_Z: f64 = 0.30;
const INITIAL_X: f64 = 0.20;
const INITIAL_Y: f64 = 0.05;

const PELVIS_REST_POS: Vec3 = Vec3::new(0.0, 1.00, 0.0);
// Lumbar anchor on the pelvis, in pelvis-local coords. +y end (top of
// pelvis = sacroiliac region).
const LUMBAR_ANCHOR_PELVIS_Y: f64 = PELVIS_HALF_H;

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

fn add_lumbar_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    // Z axis: flex/ext.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        LUMBAR_FLEX_MIN, LUMBAR_FLEX_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), LUMBAR_FLEX_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), LUMBAR_FLEX_K, 0.0));

    // X axis: lateral bend.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(),
        LUMBAR_BEND_MIN, LUMBAR_BEND_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), LUMBAR_BEND_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), LUMBAR_BEND_K, 0.0));

    // Y axis: axial rotation.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(),
        LUMBAR_ROT_MIN, LUMBAR_ROT_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), LUMBAR_ROT_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), LUMBAR_ROT_K, 0.0));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Pelvis: pinned, identity orientation.
    let mut pelvis = RigidBody::new(0.0, capsule_inertia(15.0, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_REST_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Thoracic: deflected initial pose. Composition order matches hip
    // for parity (Z, then X, then Y) so relaxation paths are comparable.
    let initial_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_Z))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_X, 0.0, 0.0))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(0.0, INITIAL_Y, 0.0));
    let mut thoracic = RigidBody::new(THORACIC_MASS,
        capsule_inertia(THORACIC_MASS, THORACIC_RADIUS, THORACIC_HALF_H));
    let lumbar_anchor_pelvis = Vec3::new(0.0, LUMBAR_ANCHOR_PELVIS_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[pelvis_idx],
        lumbar_anchor_pelvis,
        Vec3::new(0.0, -THORACIC_HALF_H, 0.0),
        initial_rel);
    thoracic.position = pos;
    thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(&mut world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis,
        Vec3::new(0.0, -THORACIC_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    add_lumbar_biomech(&mut world, pelvis_idx, thoracic_idx);

    println!("LUMBAR COMPLIANT (3-DOF + full biomech)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("Initial: Z={}, X={}, Y={}", INITIAL_Z, INITIAL_X, INITIAL_Y);
    println!("K_gravity ≈ {:.1} N·m/rad (m·g·half_h, destabilizing)",
             THORACIC_MASS * 9.81 * THORACIC_HALF_H);
    println!("K_spring  = {:.1} (flex/bend),  net stable since spring > gravity",
             LUMBAR_FLEX_K);
    println!();
    println!("{:<6} {:<11} {:<11} {:<11}",
             "t", "Z (flex)", "X (bend)", "Y (rot)");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let pelvis = &world.bodies[pelvis_idx];
            let thoracic = &world.bodies[thoracic_idx];
            let rel = (thoracic.orientation * pelvis.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.z, rel.x, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let pelvis = &world.bodies[pelvis_idx];
    let thoracic = &world.bodies[thoracic_idx];
    let rel = (thoracic.orientation * pelvis.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: Z = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       X = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       Y = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());
    println!();
    println!("Predicted: thoracic settles to ~upright (0, 0, ~0). Any residual");
    println!("Y is the bivector artifact from the relaxation path.");
}
