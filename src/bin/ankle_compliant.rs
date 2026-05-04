// Ankle/foot validation: pinned vertical lower leg + foot capsule on a
// 2-DOF ankle joint.
//
// IMPORTANT geometry note: the foot is built with IDENTITY orientation
// (vertical capsule hanging from the ankle, foot tip pointing world -y),
// not the anatomical "horizontal, toes forward" pose. Reason: a non-
// identity baseline (e.g. R_x(π/2) for horizontal) causes the bivector
// cross-coupling in `scaled_axis().dot(axis)` to produce huge artifact
// readings on whichever axis the baseline rotation aligns with. With PF
// baseline = π/2 and any INV deflection, the twist axis reads ½·sin(PF)·INV
// which feeds into the limit constraints and the system goes chaotic.
// Documented in project_axis_angle_artifacts.md. Proper fix is swing-twist
// Euler decomposition — engine work for later. For now, foot starts
// vertical; the visual demo can rotate the rendered mesh independently.
//
// Joint axes in lower-leg-local frame (= world at rest):
//   +x  plantar/dorsi flexion. Positive = forward tilt of foot tip.
//   +z  inversion/eversion. Positive = inversion (sole inward).
//   +y  axial twist — locked tight.
//
// Anatomical ROM:
//   PF:  −0.44 (dorsi 25°) to +0.87 (plantar 50°)
//   INV: −0.26 (eversion 15°) to +0.52 (inversion 30°)
//
// Validation: release foot from PF=+0.40, INV=+0.20. Watch settle near
// neutral (gravity is conservative around the vertical rest pose).

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

const LOWER_LEG_HALF_H: f64 = 0.210;
const LOWER_LEG_RADIUS: f64 = 0.050;

// Foot — single capsule, horizontal at neutral.
const FOOT_MASS:    f64 = 1.0;
const FOOT_HALF_H:  f64 = 0.125;
const FOOT_RADIUS:  f64 = 0.040;

// Foot at identity orientation: capsule axis = body +y = world +y. Body
// local +y points UP toward ankle, body -y points DOWN (foot tip). Anchor
// for ankle joint at body-local +y direction (top of foot capsule).
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

// Per-axis biomech. With foot at identity, all three axes read 0 at rest
// — no bivector contamination, anatomical ranges apply directly.
const PF_K:        f64 = 4.0;
const INV_K:       f64 = 3.5;
const TWIST_K:     f64 = 8.0;
const PF_C:        f64 = 1.2;
const INV_C:       f64 = 1.0;
const TWIST_C:     f64 = 0.5;
const PF_NEUTRAL:  f64 = 0.0;
const INV_NEUTRAL: f64 = 0.0;
const PF_MIN:      f64 = -0.44;
const PF_MAX:      f64 =  0.87;
const INV_MIN:     f64 = -0.26;
const INV_MAX:     f64 =  0.52;
const TWIST_LIM:   f64 =  0.10;
const ANKLE_LIMIT_K:    f64 = 80.0;
const TWIST_LIMIT_K:    f64 = 200.0;
const ANKLE_LIMIT_EXP:  f64 = 2.0;

const INITIAL_PF: f64 = 0.40;   // ≈ 23° plantarflex
const INITIAL_INV: f64 = 0.20;  // ≈ 11° inversion

const LEG_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);

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
    parent_anchor: Vec3, child_anchor: Vec3,
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
    axis_local_a: Vec3,
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

fn main() {
    let mut world = XpbdWorld::new();

    // Lower leg pinned vertically.
    let mut lower = RigidBody::new(0.0, capsule_inertia(3.5, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    lower.position = LEG_REST_POS;
    lower.orientation = UnitQuaternion::identity();
    let lower_idx = world.add_body(lower);

    // Foot at identity rest pose (vertical capsule). Apply the initial PF
    // and INV deflections about the ankle biomech axes (lower-leg-local
    // frame = world at rest).
    let initial_pf  = UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_PF, 0.0, 0.0));
    let initial_inv = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_INV));
    let foot_rel = initial_pf * initial_inv;

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[lower_idx],
        Vec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        foot_rel);
    foot.position = pos;
    foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, lower_idx, foot_idx,
        Vec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    // PF/DF axis = lower-leg-local +x.
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0),
        PF_MIN, PF_MAX, PF_K, PF_C, PF_NEUTRAL, ANKLE_LIMIT_K);
    // Inv/Ev axis = lower-leg-local +z.
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(0.0, 0.0, 1.0),
        INV_MIN, INV_MAX, INV_K, INV_C, INV_NEUTRAL, ANKLE_LIMIT_K);
    // Twist axis (locked).
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0, TWIST_LIMIT_K);

    println!("ANKLE COMPLIANT (2-DOF + tight twist)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("Foot at identity rest pose. PF/INV/twist read directly from");
    println!("scaled_axis projection (no baseline subtraction needed).");
    println!("Initial: PF = {:.2}, INV = {:.2}", INITIAL_PF, INITIAL_INV);
    println!();
    println!("{:<6} {:<11} {:<11} {:<11}",
             "t", "PF", "INV", "twist");
    println!("{}", "-".repeat(48));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let lo = &world.bodies[lower_idx];
            let fo = &world.bodies[foot_idx];
            let s = (fo.orientation * lo.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, s.x, s.z, s.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let lo = &world.bodies[lower_idx];
    let fo = &world.bodies[foot_idx];
    let s = (fo.orientation * lo.orientation.inverse()).scaled_axis();
    println!();
    println!("Final:");
    println!("  PF    = {:+.5} rad   ({:+.2}°)", s.x, s.x.to_degrees());
    println!("  INV   = {:+.5} rad   ({:+.2}°)", s.z, s.z.to_degrees());
    println!("  twist = {:+.5} rad   ({:+.2}°)", s.y, s.y.to_degrees());
    println!();
    println!("Predicted: foot relaxes toward (PF, INV, twist) = (0, 0, 0).");
}
