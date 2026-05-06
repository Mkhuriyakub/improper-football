// Demonstration of the swing-twist refactor: ankle/foot validation with the
// foot at its ANATOMICALLY HORIZONTAL pose in physics.
//
// Compare to `ankle_compliant.rs`: that file builds the foot at identity
// orientation (vertical capsule hanging from the ankle) as a workaround for
// the bivector cross-coupling artifact in `scaled_axis().dot(axis)`. The
// workaround required the visual demo to rotate the rendered mesh
// independently from the physics body, which broke down as soon as we
// added ground contact (Phase 2).
//
// With the swing-twist refactor (math::swing_twist_angle plus the new
// `baseline` field on each angular primitive), the foot can sit at its
// real horizontal pose — `body.orient = R_x(-π/2)` so body +y points
// world -z (heel side), body -y points world +z (toes forward). The
// ankle joint primitives all carry `with_baseline(R_x(-π/2))`, which
// makes anatomical neutral read as angle = 0 on each axis. The same PF,
// INV, twist axes apply the same way as in the vertical-foot version.
//
// Validation: release foot from a deflected pose (PF=+0.40, INV=+0.20)
// applied ON TOP of the horizontal baseline. Expect settle to (0, 0, ~0)
// with no bivector contamination on the locked twist axis.

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

const FOOT_MASS:    f64 = 1.0;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;

// Foot body's proximal anchor (at body +y end of the capsule). With
// horizontal baseline R_x(-π/2), this anchor points to world -z (back of
// foot, the heel-side connection to the ankle joint).
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

const PF_K:        f64 = 4.0;
const INV_K:       f64 = 3.5;
const TWIST_K:     f64 = 8.0;
const PF_C:        f64 = 1.2;
const INV_C:       f64 = 1.0;
const TWIST_C:     f64 = 0.5;
const PF_MIN:      f64 = -0.44;
const PF_MAX:      f64 =  0.87;
const INV_MIN:     f64 = -0.26;
const INV_MAX:     f64 =  0.52;
const TWIST_LIM:   f64 =  0.10;
const ANKLE_LIMIT_K:    f64 = 80.0;
const TWIST_LIMIT_K:    f64 = 200.0;
const ANKLE_LIMIT_EXP:  f64 = 2.0;

const INITIAL_PF: f64 = 0.40;
const INITIAL_INV: f64 = 0.20;

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
    baseline: UnitQuaternion<f64>,
) {
    world.angular_soft_limits.push(
        AngularSoftLimit::new(
            parent, child, axis_local_a,
            min_angle, max_angle, limit_stiffness, ANKLE_LIMIT_EXP)
        .with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(
        AngularSpring::new(parent, child, axis_local_a, stiffness, neutral)
        .with_baseline(baseline));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Lower leg pinned vertical.
    let mut lower = RigidBody::new(0.0, capsule_inertia(3.5, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    lower.position = LEG_REST_POS;
    lower.orientation = UnitQuaternion::identity();
    let lower_idx = world.add_body(lower);

    // Foot at HORIZONTAL rest pose: baseline rotation R_x(-π/2) takes body
    // +y to world -z (heel) and body -y to world +z (toes forward). The
    // initial deflection (PF=+0.40 about +x, INV=+0.20 about +z) is
    // applied ON TOP of the baseline.
    let baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let initial_pf  = UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_PF, 0.0, 0.0));
    let initial_inv = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_INV));
    let foot_rel = initial_pf * initial_inv * baseline;

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

    // Joint axes are in the SHIN's local frame (= world at rest, since shin
    // is identity). Anatomical mapping is unchanged from the vertical-foot
    // version: the baseline only shifts the rest pose, not the axis meanings.
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0),
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0, ANKLE_LIMIT_K, baseline);
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(0.0, 0.0, 1.0),
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0, ANKLE_LIMIT_K, baseline);
    add_axis(&mut world, lower_idx, foot_idx, Vec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0, TWIST_LIMIT_K, baseline);

    println!("ANKLE HORIZONTAL COMPLIANT (post-refactor)");
    println!("Foot at anatomical horizontal pose in physics: baseline = R_x(-π/2)");
    println!("Initial deflection: PF = {:.2}, INV = {:.2}", INITIAL_PF, INITIAL_INV);
    println!("({:.1}s sim, dt={:.5}s)", SIM_DURATION, DT);
    println!();
    println!("Per-axis angles measured via swing-twist factorization, relative");
    println!("to the horizontal baseline. Anatomical neutral reads as 0.");
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
            // Read each axis the same way the joint primitives do — via
            // swing-twist on `q_from_rest = q_b_in_a · baseline.inverse()`.
            let lo = &world.bodies[lower_idx];
            let fo = &world.bodies[foot_idx];
            let q_b_in_a = lo.orientation.inverse() * fo.orientation;
            let q_from_rest = q_b_in_a * baseline.inverse();
            let pf    = physics::math::swing_twist_angle(q_from_rest, Vec3::new(1.0, 0.0, 0.0));
            let inv   = physics::math::swing_twist_angle(q_from_rest, Vec3::new(0.0, 0.0, 1.0));
            let twist = physics::math::swing_twist_angle(q_from_rest, Vec3::new(0.0, 1.0, 0.0));
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, pf, inv, twist);
            next_print += PRINT_INTERVAL;
        }
    }

    let lo = &world.bodies[lower_idx];
    let fo = &world.bodies[foot_idx];
    let q_b_in_a = lo.orientation.inverse() * fo.orientation;
    let q_from_rest = q_b_in_a * baseline.inverse();
    let pf    = physics::math::swing_twist_angle(q_from_rest, Vec3::new(1.0, 0.0, 0.0));
    let inv   = physics::math::swing_twist_angle(q_from_rest, Vec3::new(0.0, 0.0, 1.0));
    let twist = physics::math::swing_twist_angle(q_from_rest, Vec3::new(0.0, 1.0, 0.0));
    println!();
    println!("Final (relative to horizontal baseline):");
    println!("  PF    = {:+.5} rad   ({:+.2}°)", pf, pf.to_degrees());
    println!("  INV   = {:+.5} rad   ({:+.2}°)", inv, inv.to_degrees());
    println!("  twist = {:+.5} rad   ({:+.2}°)", twist, twist.to_degrees());
    println!();
    println!("Foot world orientation (final):");
    println!("  scaled_axis = ({:+.4}, {:+.4}, {:+.4})",
             fo.orientation.scaled_axis().x,
             fo.orientation.scaled_axis().y,
             fo.orientation.scaled_axis().z);
    println!("  Compare to baseline R_x(-π/2) = (-1.5708, 0, 0): the foot");
    println!("  is at its horizontal anatomical pose, no visual offset hack.");
    println!();
    println!("Compare with `cargo run --release --bin ankle_compliant` — same");
    println!("dynamics (PF/INV settle, twist reads ~0) but that one has the");
    println!("foot vertical in physics with the visual offset patched in the");
    println!("renderer. This version has the foot horizontal in physics.");
}
