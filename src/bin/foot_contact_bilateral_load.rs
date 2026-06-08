// Bisection step 2: TWO feet sharing a single load via two ankle joints.
// This mimics the bilateral chain dynamics WITHOUT the kinematic chain
// of knee/hip/lumbar joints. If this fails, the bilateral-two-feet
// configuration is the issue (independent of the kinematic chain).
// If it succeeds, the issue is in the kinematic chain above.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::foot_contact::FootContact;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Mat3, Vec3};
use nalgebra::UnitQuaternion;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(i_trans, 0.0, 0.0, 0.0, i_axial, 0.0, 0.0, 0.0, i_trans)
}

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.130;
const FOOT_RADIUS: f64 = 0.032;
const HIP_LATERAL: f64 = 0.10;  // each foot offset 10 cm laterally

fn main() {
    let mut world = XpbdWorld::new();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Two feet, offset laterally.
    let foot_y = FOOT_RADIUS + 0.005;

    let mut r_foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    r_foot.position = Vec3::new(HIP_LATERAL, foot_y, 0.0);
    r_foot.orientation = foot_baseline;
    let r_foot_idx = world.add_body(r_foot);

    let mut l_foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    l_foot.position = Vec3::new(-HIP_LATERAL, foot_y, 0.0);
    l_foot.orientation = foot_baseline;
    let l_foot_idx = world.add_body(l_foot);

    // Single shared "load body" (the body that the chain represents).
    // Placed centered above between the two feet, at height where the
    // ankle springs are at rest length (load bottom at foot top, ~0.10 m
    // above foot center).
    const LOAD_MASS: f64 = 70.0;  // total chain mass
    let mut load = RigidBody::new(LOAD_MASS,
        capsule_inertia(LOAD_MASS, 0.13, 0.5));
    load.position = Vec3::new(0.0, foot_y + 0.10 + 0.50, 0.0);
    // Note: load is connected to each foot via a separate ankle joint.
    // Each joint has anchor (HIP_LATERAL × side, -0.50 - 0.10, 0) on load
    // (lateral + downward).
    let load_idx = world.add_body(load);

    // Two ankle joints — load to each foot.
    for (foot_idx, side_sign) in [(r_foot_idx, 1.0_f64), (l_foot_idx, -1.0_f64)] {
        let load_anchor = Vec3::new(side_sign * HIP_LATERAL, -0.60, 0.0);
        let foot_anchor = Vec3::new(0.0, 0.0, 0.0);

        let mut s = PositionalSpring::new(load_idx, foot_idx, 60_000.0);
        s.local_anchor_a = load_anchor;
        s.local_anchor_b = foot_anchor;
        world.positional_springs.push(s);

        let mut d = PositionalDamper::new(load_idx, foot_idx, 150.0);
        d.local_anchor_a = load_anchor;
        d.local_anchor_b = foot_anchor;
        world.positional_dampers.push(d);

        world.angular_dampers.push(AngularDamper::new(
            load_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0), 400.0));

        world.foot_contacts.push(FootContact::tripod(
            foot_idx, FOOT_HALF_H, FOOT_RADIUS, side_sign));
    }

    println!("Bilateral FootContact bisection: 2 feet + 1 load via 2 ankle joints");
    println!("Load mass: {} kg. No kinematic chain above. No controller.", LOAD_MASS as i32);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10}",
             "t", "load.y", "r_foot.y", "l_foot.y", "r_tilt");
    println!("{}", "-".repeat(58));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;
    let baseline_axis = foot_baseline.scaled_axis();

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_foot_drop = 0.0_f64;

    while t < 3.0 {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        t += dt;

        let drop = foot_y - world.bodies[r_foot_idx].position.y;
        if drop > max_foot_drop { max_foot_drop = drop; }

        if t >= next_print {
            let load_y = world.bodies[load_idx].position.y;
            let r_foot_y = world.bodies[r_foot_idx].position.y;
            let l_foot_y = world.bodies[l_foot_idx].position.y;
            let r_tilt_world = world.bodies[r_foot_idx].orientation.scaled_axis();
            let r_tilt = (r_tilt_world - baseline_axis).norm();
            println!("{:<6.2} {:<+10.4} {:<+10.4} {:<+10.4} {:<+10.4}",
                     t, load_y, r_foot_y, l_foot_y, r_tilt);
            next_print += 0.10;
        }
    }

    let load_y_final = world.bodies[load_idx].position.y;
    let r_foot_y_final = world.bodies[r_foot_idx].position.y;
    println!();
    println!("Final state:");
    println!("  load.y = {:.4} (expected ≈ {:.4} after settle)",
             load_y_final, foot_y + 0.10 + 0.50 - 0.001);
    println!("  r_foot.y = {:.4} (started at {:.4})",
             r_foot_y_final, foot_y);
    if load_y_final.is_finite() && load_y_final > foot_y + 0.55 {
        println!("✓ Stable — load remained supported.");
    } else {
        println!("✗ Unstable.");
    }
}
