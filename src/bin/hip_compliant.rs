// Hip validation: pinned pelvis stub + thigh hanging on a 3-DOF
// ball-and-socket joint. Direct analog of the shoulder, but at leg scale
// (m_thigh ≈ 7 kg vs upper arm ~2 kg) with a much smaller flexion ROM
// (+120° vs the shoulder's +180°) — you can't pull your knee past your
// chest the way you can swing your arm overhead.
//
// Joint axes (pelvis-local frame = world at rest):
//   Z (forward/back): flexion / extension. Positive = flex (knee toward chest).
//   X (sideways):     abduction / adduction. Positive = abduction (lateral lift).
//   Y (vertical):     external / internal rotation. Positive = external rotation.
//
// Anatomical ROM:
//   Z (flex/ext):  -0.35 (-20° extension) to +2.09 (+120° flexion)
//   X (abd/add):   -0.52 (-30° adduction) to +0.78 (+45° abduction)
//   Y (rot):       -0.61 (-35° internal)  to +0.87 (+50° external)
//
// Bivector-artifact caveat (per project_axis_angle_artifacts.md): same as
// shoulder — combined poses produce small cross-coupled readings on the
// third axis even when no real rotation is present there. Single-axis
// equilibria are unaffected.
//
// Validation: release thigh from a deflected initial pose (Z = +0.6,
// X = +0.4, Y = +0.3). Expect settle to roughly (0, 0, ~0) under gravity.

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

// Pelvis: pinned, vertical capsule (simplification; real pelvis is more
// of a flat ring). Mass = 0; inertia only matters for visual rendering.
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;

// Thigh (femur + soft tissue), single capsule. Same dimensions as the
// knee_compliant thigh.
const THIGH_MASS:    f64 = 7.0;
const THIGH_RADIUS:  f64 = 0.070;
const THIGH_HALF_H:  f64 = 0.225;

// Elastic hip anchor — the hip is one of the strongest joints in the body
// (full weight-bearing), so anchor stiffness is roughly 2× shoulder.
const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =     400.0;

// Per-axis biomech. Damping targets ζ ≈ 0.9 against the combined
// (spring + gravity) stiffness — the thigh is heavy (~7 kg at 0.225 m
// lever) so gravity contributes ~15 N·m/rad of restoring stiffness,
// which dominates the biomech spring. With K_spring=10 and gravity
// added: K_total ≈ 25, I ≈ 0.48 → c_crit ≈ 7.
//   Z (flex/ext): primary sagittal-plane motion. Stiffer than shoulder
//                 (heavier limb, deeper ligament corset).
//   X (abd/add):  frontal-plane motion. Similar stiffness.
//   Y (rot):      axial rotation around femoral shaft. Lower stiffness
//                 — less gravity coupling, so damping stays small.
const HIP_Z_K:       f64 = 10.0;
const HIP_X_K:       f64 = 10.0;
const HIP_Y_K:       f64 =  3.0;
const HIP_Z_C:       f64 =  7.0;
const HIP_X_C:       f64 =  7.0;
const HIP_Y_C:       f64 =  0.40;

const HIP_Z_MIN: f64 = -0.35;   //  -20° extension
const HIP_Z_MAX: f64 =  2.09;   // +120° full flexion (knee-to-chest)
const HIP_X_MIN: f64 = -0.52;   //  -30° adduction (cross-body)
const HIP_X_MAX: f64 =  0.78;   //  +45° abduction (lateral lift)
const HIP_Y_MIN: f64 = -0.61;   //  -35° internal rotation
const HIP_Y_MAX: f64 =  0.87;   //  +50° external rotation
const HIP_LIMIT_K:    f64 = 120.0;
const HIP_LIMIT_EXP:  f64 =   2.0;

const INITIAL_Z: f64 = 0.60;
const INITIAL_X: f64 = 0.40;
const INITIAL_Y: f64 = 0.30;

const PELVIS_REST_POS: Vec3 = Vec3::new(0.0, 1.00, 0.0);
// Hip anchor on the pelvis, in pelvis-local coords. +x = lateral (right),
// -y = below pelvis center (at femoral head).
const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

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

fn add_hip_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    // Z axis: flex/ext.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        HIP_Z_MIN, HIP_Z_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), HIP_Z_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), HIP_Z_K, 0.0));

    // X axis: abd/add.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(),
        HIP_X_MIN, HIP_X_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), HIP_X_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), HIP_X_K, 0.0));

    // Y axis: int/ext rotation.
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(),
        HIP_Y_MIN, HIP_Y_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), HIP_Y_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), HIP_Y_K, 0.0));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Pelvis: pinned vertical capsule, identity orientation.
    let mut pelvis = RigidBody::new(0.0, capsule_inertia(15.0, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_REST_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Thigh: deflected initial pose. Composition order matches shoulder
    // for parity (Z, then X, then Y) so the relaxation path is comparable.
    let initial_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_Z))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_X, 0.0, 0.0))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(0.0, INITIAL_Y, 0.0));
    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let hip_anchor_pelvis = Vec3::new(HIP_ANCHOR_PELVIS_X, HIP_ANCHOR_PELVIS_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[pelvis_idx],
        hip_anchor_pelvis,
        Vec3::new(0.0, THIGH_HALF_H, 0.0),
        initial_rel);
    thigh.position = pos;
    thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        hip_anchor_pelvis,
        Vec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    add_hip_biomech(&mut world, pelvis_idx, thigh_idx);

    println!("HIP COMPLIANT (3-DOF ball + full biomech)  ({:.1}s, dt={:.5}s)",
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
            let pelvis = &world.bodies[pelvis_idx];
            let thigh = &world.bodies[thigh_idx];
            let rel = (thigh.orientation * pelvis.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.z, rel.x, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let pelvis = &world.bodies[pelvis_idx];
    let thigh = &world.bodies[thigh_idx];
    let rel = (thigh.orientation * pelvis.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: Z = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       X = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       Y = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());
    println!();
    println!("Predicted: with thigh hanging straight down (no lever past hip),");
    println!("equilibrium should be (0, 0, ~0). Any residual Y is the bivector");
    println!("artifact from the relaxation path through nonzero Z and X.");
}
