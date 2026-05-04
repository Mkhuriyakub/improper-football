// Cervical validation: pinned thoracic stub + head sitting upright on a
// 3-DOF cervical joint. Same structural pattern as the lumbar — child
// body sits ABOVE parent — but with a much lighter load (head ≈ 5 kg vs
// thoracic ≈ 25 kg) and far wider rotational range (±80° vs ±5°).
//
// The cervical region in vivo is several joints in series (atlanto-
// occipital + atlanto-axial + C2–C7 motion segments). We collapse them
// into one 3-DOF ball joint at the C7–T1 boundary.
//
// Joint axes (thoracic-local frame = world at rest):
//   Z (forward/back): flexion / extension. Positive = chin-to-chest.
//   X (sideways):     lateral bend (ear-to-shoulder). Positive = right.
//   Y (vertical):     axial rotation. Positive = look over right shoulder.
//
// Anatomical ROM:
//   Z (flex/ext):  -1.05 (-60° "look up") to +0.87 (+50° flexion)
//   X (bend):      ±0.78 (±45°)
//   Y (rot):       ±1.40 (±80°) — much wider than lumbar
//
// Stability note: head's gravity destabilization is tiny
//   K_grav = m·g·half_h ≈ 5·9.81·0.05 ≈ 2.45 N·m/rad
// so even a low passive stiffness (we use 5) keeps the head upright. In
// real anatomy this barely-sufficient passive support is supplemented by
// active muscle tone (which is why you nod when falling asleep upright).
//
// Validation: release head from (Z=+0.30, X=+0.20, Y=+0.10). Expect
// settle to (0, 0, ~0).

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

// Thoracic stub: pinned. Same as in lumbar but acts as parent here.
const THORACIC_HALF_H: f64 = 0.180;
const THORACIC_RADIUS: f64 = 0.130;

// Head: roughly spherical (~14 cm wide, ~17 cm tall). Modeled as a short
// vertical capsule. Mass ~5 kg.
const HEAD_MASS:    f64 = 5.0;
const HEAD_RADIUS:  f64 = 0.080;
const HEAD_HALF_H:  f64 = 0.050;

// Elastic cervical anchor — moderately stiff. Lower than lumbar/hip
// because the joint is smaller and more delicate.
const CERV_JOINT_K: f64 = 80_000.0;
const CERV_JOINT_C: f64 =    200.0;

// Per-axis biomech.
//   Z (flex/ext): K=5 vs gravity 2.45 → net stable.
//   X (bend):     same.
//   Y (rot):      no gravity coupling (axis is along head's vertical).
const CERV_FLEX_K: f64 = 5.0;
const CERV_BEND_K: f64 = 5.0;
const CERV_ROT_K:  f64 = 3.0;
const CERV_FLEX_C: f64 = 0.60;
const CERV_BEND_C: f64 = 0.60;
const CERV_ROT_C:  f64 = 0.50;

const CERV_FLEX_MIN: f64 = -1.05;   //  -60° "look up" (extension)
const CERV_FLEX_MAX: f64 =  0.87;   //  +50° "chin to chest"
const CERV_BEND_MIN: f64 = -0.78;   //  -45° left bend
const CERV_BEND_MAX: f64 =  0.78;   //  +45° right bend
const CERV_ROT_MIN:  f64 = -1.40;   //  -80° left rotation
const CERV_ROT_MAX:  f64 =  1.40;   //  +80° right rotation
const CERV_LIMIT_K:   f64 = 80.0;
const CERV_LIMIT_EXP: f64 =  2.0;

const INITIAL_Z: f64 = 0.30;
const INITIAL_X: f64 = 0.20;
const INITIAL_Y: f64 = 0.10;

const THORACIC_REST_POS: Vec3 = Vec3::new(0.0, 1.28, 0.0);
// Cervical anchor on the thoracic, in thoracic-local coords. +y end (top
// of thoracic = C7–T1 boundary).
const CERV_ANCHOR_THORACIC_Y: f64 = THORACIC_HALF_H;

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

fn add_cervical_biomech(world: &mut XpbdWorld, parent: usize, child: usize) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::z(),
        CERV_FLEX_MIN, CERV_FLEX_MAX, CERV_LIMIT_K, CERV_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::z(), CERV_FLEX_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::z(), CERV_FLEX_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::x(),
        CERV_BEND_MIN, CERV_BEND_MAX, CERV_LIMIT_K, CERV_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::x(), CERV_BEND_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::x(), CERV_BEND_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, Vec3::y(),
        CERV_ROT_MIN, CERV_ROT_MAX, CERV_LIMIT_K, CERV_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, Vec3::y(), CERV_ROT_C));
    world.angular_springs.push(AngularSpring::new(parent, child, Vec3::y(), CERV_ROT_K, 0.0));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Thoracic stub: pinned. Phantom mass for inertia tensor (so the
    // damper's I_inv contribution is well-conditioned for body_a).
    let mut thoracic = RigidBody::new(0.0,
        capsule_inertia(25.0, THORACIC_RADIUS, THORACIC_HALF_H));
    thoracic.position = THORACIC_REST_POS;
    thoracic.orientation = UnitQuaternion::identity();
    let thoracic_idx = world.add_body(thoracic);

    // Head: deflected initial pose (Z, then X, then Y for parity with hip
    // and lumbar).
    let initial_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_Z))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(INITIAL_X, 0.0, 0.0))
                    * UnitQuaternion::from_scaled_axis(Vec3::new(0.0, INITIAL_Y, 0.0));
    let mut head = RigidBody::new(HEAD_MASS,
        capsule_inertia(HEAD_MASS, HEAD_RADIUS, HEAD_HALF_H));
    let cerv_anchor_thoracic = Vec3::new(0.0, CERV_ANCHOR_THORACIC_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[thoracic_idx],
        cerv_anchor_thoracic,
        Vec3::new(0.0, -HEAD_HALF_H, 0.0),
        initial_rel);
    head.position = pos;
    head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic,
        Vec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);
    add_cervical_biomech(&mut world, thoracic_idx, head_idx);

    println!("CERVICAL COMPLIANT (3-DOF + full biomech)  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("Initial: Z={}, X={}, Y={}", INITIAL_Z, INITIAL_X, INITIAL_Y);
    println!("K_gravity ≈ {:.2} N·m/rad (m·g·half_h, destabilizing)",
             HEAD_MASS * 9.81 * HEAD_HALF_H);
    println!("K_spring  = {:.1} (flex/bend), {:.1} (rot)",
             CERV_FLEX_K, CERV_ROT_K);
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
            let thoracic = &world.bodies[thoracic_idx];
            let head = &world.bodies[head_idx];
            let rel = (head.orientation * thoracic.orientation.inverse()).scaled_axis();
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, rel.z, rel.x, rel.y);
            next_print += PRINT_INTERVAL;
        }
    }

    let thoracic = &world.bodies[thoracic_idx];
    let head = &world.bodies[head_idx];
    let rel = (head.orientation * thoracic.orientation.inverse()).scaled_axis();
    println!();
    println!("Final: Z = {:+.5} rad ({:+.2}°)", rel.z, rel.z.to_degrees());
    println!("       X = {:+.5} rad ({:+.2}°)", rel.x, rel.x.to_degrees());
    println!("       Y = {:+.5} rad ({:+.2}°)", rel.y, rel.y.to_degrees());
    println!();
    println!("Predicted: head settles to ~upright (0, 0, ~0). Any residual");
    println!("Y is the bivector artifact from the relaxation path.");
}
