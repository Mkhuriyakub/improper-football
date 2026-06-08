// Bisection step 3: single leg (foot + shin + thigh) on FootContact,
// with knee posture PD (implicit damping). No second leg, no pelvis,
// no controller. Tests whether the kinematic chain itself destabilizes
// when paired with FootContact.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::foot_contact::FootContact;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Mat3, Vec3};
use nalgebra::UnitQuaternion;

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.130;
const FOOT_RADIUS: f64 = 0.032;
const SHIN_MASS: f64 = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;
const THIGH_MASS: f64 = 7.0;
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.07;
// Pin a heavy "anchor" at the top of the thigh to represent the body weight.
const LOAD_MASS: f64 = 56.0;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(i_trans, 0.0, 0.0, 0.0, i_axial, 0.0, 0.0, 0.0, i_trans)
}

fn place_child(
    parent: &RigidBody, parent_distal: Vec3, child_proximal: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn main() {
    let mut world = XpbdWorld::new();
    let id = UnitQuaternion::identity();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Build chain top-down.
    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    thigh.position = Vec3::new(0.0, FOOT_RADIUS + 2.0 * SHIN_HALF_H + THIGH_HALF_H, 0.0);
    let thigh_idx = world.add_body(thigh);

    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        Vec3::new(0.0, 0.0, 0.0), foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    // A "load" pin at top of thigh.
    let mut load = RigidBody::new(LOAD_MASS,
        capsule_inertia(LOAD_MASS, 0.13, 0.5));
    load.position = Vec3::new(0.0,
        world.bodies[thigh_idx].position.y + THIGH_HALF_H + 0.50, 0.0);
    let load_idx = world.add_body(load);

    // KNEE joint with implicit PD.
    let mut s = PositionalSpring::new(thigh_idx, shin_idx, 80_000.0);
    s.local_anchor_a = Vec3::new(0.0, -THIGH_HALF_H, 0.0);
    s.local_anchor_b = Vec3::new(0.0,  SHIN_HALF_H, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(thigh_idx, shin_idx, 200.0);
    d.local_anchor_a = Vec3::new(0.0, -THIGH_HALF_H, 0.0);
    d.local_anchor_b = Vec3::new(0.0,  SHIN_HALF_H, 0.0);
    world.positional_dampers.push(d);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
        -0.087, 2.44, 80.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_axis, 1.5));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_axis, 5.0, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        thigh_idx, shin_idx, knee_axis, 3000.0, 300.0));
    // Knee off-axis (Y, Z): the knee is anatomically a HINGE. Without
    // these, FootContact's lateral roll amplifies through an
    // unconstrained chain.
    let knee_z = Vec3::new(0.0, 0.0, 1.0);
    let knee_y = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_z,
        -0.30, 0.30, 50.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_z, 0.80));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_z, 12.0, 0.0));
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_y,
        -0.30, 0.30, 50.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_y, 0.80));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_y, 12.0, 0.0));

    // ANKLE joint (foot center = ankle anchor).
    let mut s = PositionalSpring::new(shin_idx, foot_idx, 60_000.0);
    s.local_anchor_a = Vec3::new(0.0, -SHIN_HALF_H, 0.0);
    s.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(shin_idx, foot_idx, 150.0);
    d.local_anchor_a = Vec3::new(0.0, -SHIN_HALF_H, 0.0);
    d.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_dampers.push(d);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
        -0.44, 0.87, 80.0, 2.0).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, 1.2));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, pf_axis, 4.0, 0.0)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0), 400.0));

    // LOAD-TO-THIGH "hip" — a simple positional spring at rest length
    // (rigid linkage; no rotation control).
    let mut s = PositionalSpring::new(load_idx, thigh_idx, 200_000.0);
    s.local_anchor_a = Vec3::new(0.0, -0.50, 0.0);
    s.local_anchor_b = Vec3::new(0.0,  THIGH_HALF_H, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(load_idx, thigh_idx, 400.0);
    d.local_anchor_a = Vec3::new(0.0, -0.50, 0.0);
    d.local_anchor_b = Vec3::new(0.0,  THIGH_HALF_H, 0.0);
    world.positional_dampers.push(d);

    // FootContact.
    world.foot_contacts.push(FootContact::tripod(foot_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    println!("Single-leg + FootContact bisection: foot+shin+thigh+load");
    println!("Load mass: {} kg. Knee posture PD active. No second leg.", LOAD_MASS as i32);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10}",
             "t", "load.y", "thigh.y", "shin.y", "foot.y");
    println!("{}", "-".repeat(58));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < 3.0 {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        t += dt;

        if t >= next_print {
            let l_y = world.bodies[load_idx].position.y;
            let th_y = world.bodies[thigh_idx].position.y;
            let sh_y = world.bodies[shin_idx].position.y;
            let f_y = world.bodies[foot_idx].position.y;
            println!("{:<6.2} {:<+10.4} {:<+10.4} {:<+10.4} {:<+10.4}",
                     t, l_y, th_y, sh_y, f_y);
            next_print += 0.10;
        }
    }

    let load_y_final = world.bodies[load_idx].position.y;
    println!();
    if load_y_final.is_finite() && load_y_final > 0.5 {
        println!("✓ Stable — chain remained upright.");
    } else {
        println!("✗ Unstable — chain collapsed (load_y={:.3}).", load_y_final);
    }
}
