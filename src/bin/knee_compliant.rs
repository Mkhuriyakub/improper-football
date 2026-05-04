// Knee validation: pinned vertical thigh + lower leg hanging on a 1-DOF
// hinge. Direct analog of the elbow setup but at leg scale (m_lower ≈ 3.5 kg
// vs elbow's lower arm ~1.2 kg, full flexion to ~140°, slight hyperextension
// allowed at ~-5°).
//
// Axes are inverse of the elbow's: flexion bends the shin POSTERIORLY
// (toward -z, the back of the body), so the flex axis is thigh-local +x
// with positive biomech = anatomical flex. (Compare to elbow flex which
// uses -x because elbow flex is anteriorly directed.)
//
// Validation: release lower leg from a deep flex (Z = +1.20 rad ≈ 69°) and
// watch it settle to a small gravity equilibrium near the rest neutral.
// Expected equilibrium ≈ +0.10 rad (spring neutral) plus a slight gravity
// bias since the shin's lever past the knee has both -y and -z components
// when slightly flexed.

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

// Thigh (femur) — pinned for this validation. Anthropometric upper-leg
// values for a 70 kg adult.
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.070;

// Lower leg (tibia + fibula + soft tissue, simplified to a single capsule).
const LOWER_LEG_MASS:    f64 = 3.5;
const LOWER_LEG_RADIUS:  f64 = 0.050;
const LOWER_LEG_HALF_H:  f64 = 0.210;

// Knee elastic anchor.
const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;

// Knee biomech. Damping ≈ critical (ζ ≈ 1.45) for fastest settling.
// Higher ζ over-damps the slow pole and SLOWS the approach — this is the
// counterintuitive overdamping pitfall: |slow_pole| ≈ ω_n / (2ζ) for
// ζ >> 1, so doubling damping doubles the settling time. Keep ζ near 1.
const KNEE_K:        f64 = 5.0;
const KNEE_C:        f64 = 1.5;
const KNEE_NEUTRAL:  f64 = 0.10;
const KNEE_MIN:      f64 = -0.087;   //  -5° (mild physiological hyperextension)
const KNEE_MAX:      f64 =  2.44;    // 140° (full flexion)
const KNEE_LIMIT_K:  f64 =  80.0;
const KNEE_LIMIT_EXP:f64 =   2.0;

// Off-axis: knee is essentially a hinge — keep it tight.
const KNEE_OFF_AXIS_K:        f64 = 12.0;
const KNEE_OFF_AXIS_C:        f64 = 0.80;
const KNEE_OFF_AXIS_LIM:      f64 = 0.30;
const KNEE_OFF_AXIS_LIMIT_K:  f64 = 50.0;

const INITIAL_KNEE_FLEX: f64 = 1.20;  // ≈ 69°, deep flex initial pose

const THIGH_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);

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
        min_angle, max_angle, limit_stiffness, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Thigh — pinned vertical.
    let mut thigh = RigidBody::new(0.0, capsule_inertia(7.0, THIGH_RADIUS, THIGH_HALF_H));
    thigh.position = THIGH_REST_POS;
    thigh.orientation = UnitQuaternion::identity();
    let thigh_idx = world.add_body(thigh);

    // Lower leg — initial deep flex. Knee axis = thigh-local +x. Positive
    // rotation about +x by INITIAL_KNEE_FLEX takes shin direction from -y
    // toward -z (posterior, anatomical flex direction).
    let mut lower = RigidBody::new(LOWER_LEG_MASS,
        capsule_inertia(LOWER_LEG_MASS, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    let knee_rel = UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_KNEE_FLEX, 0.0, 0.0));
    let (pos, orient) = place_child(
        &world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0, LOWER_LEG_HALF_H, 0.0),
        knee_rel);
    lower.position = pos;
    lower.orientation = orient;
    let lower_idx = world.add_body(lower);

    add_elastic_joint(&mut world, thigh_idx, lower_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0, LOWER_LEG_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);

    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    add_axis(&mut world, thigh_idx, lower_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL, KNEE_LIMIT_K);
    add_axis(&mut world, thigh_idx, lower_idx, Vec3::z(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0, KNEE_OFF_AXIS_LIMIT_K);
    add_axis(&mut world, thigh_idx, lower_idx, Vec3::y(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0, KNEE_OFF_AXIS_LIMIT_K);

    println!("KNEE COMPLIANT (1-DOF hinge + full biomech)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("Initial: knee_flex = {:.2} rad ({:.1}°)",
             INITIAL_KNEE_FLEX, INITIAL_KNEE_FLEX.to_degrees());
    println!();
    println!("{:<6} {:<11} {:<11} {:<11}",
             "t", "knee_flex", "off_z", "off_y");
    println!("{}", "-".repeat(46));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let th = &world.bodies[thigh_idx];
            let lo = &world.bodies[lower_idx];
            let rel = (lo.orientation * th.orientation.inverse()).scaled_axis();
            // knee_flex = projection on +x axis (= biomech axis).
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.x, rel.z, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let th = &world.bodies[thigh_idx];
    let lo = &world.bodies[lower_idx];
    let rel = (lo.orientation * th.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: knee_flex = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       off_z   = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       off_y   = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());

    // Analytical equilibrium on the flex axis under gravity. Linearize
    // around small angle: gravity torque ≈ -m·g·lever·θ (extending), spring
    // torque = -K·(θ - neutral). Setting sum to zero:
    //   θ_eq = K·neutral / (K + m·g·lever)
    // The denominator > K means gravity pulls the equilibrium down toward 0
    // from the spring's preferred neutral.
    let predicted_flex =
        KNEE_K * KNEE_NEUTRAL / (KNEE_K + LOWER_LEG_MASS * 9.81 * LOWER_LEG_HALF_H);
    println!();
    println!("Predicted equilibrium ≈ {:+.5} rad ({:+.2}°)",
             predicted_flex, predicted_flex.to_degrees());
}
