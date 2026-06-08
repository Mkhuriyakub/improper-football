// Isolates the bilateral-chain instability: tests whether a FootContact
// tripod is stable when loaded with the body weight it bears in the
// bilateral chain (~35 kg per foot), connected via the same ankle
// joint setup (positional spring at K=60000).
//
// If this test PASSES (foot stable under load), the bilateral
// instability is in the higher chain. If it FAILS, FootContact has
// a load-related instability.

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

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

fn run_one(load_mass: f64, frames: usize) -> (f64, f64, f64) {
    let mut world = XpbdWorld::new();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Foot at horizontal pose, just above ground.
    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let foot_y = FOOT_RADIUS + 0.005;
    foot.position = Vec3::new(0.0, foot_y, 0.0);
    foot.orientation = foot_baseline;
    let foot_idx = world.add_body(foot);

    // Above-foot "load" body — a single point mass standing in for the
    // chain. Inertia approximated as a small rigid body.
    let mut load = RigidBody::new(load_mass,
        capsule_inertia(load_mass, 0.05, 0.10));
    // Load placed so the ankle joint spring is at rest length:
    // load.position + (0, -0.10, 0) = foot.position + (0, 0, 0)
    // → load.position = foot.position + (0, 0.10, 0).
    load.position = Vec3::new(0.0, foot_y + 0.10, 0.0);
    let load_idx = world.add_body(load);

    // Ankle joint: positional spring + damper between foot's local (0,0,0)
    // and load's bottom (0, -0.10, 0).
    let mut s = PositionalSpring::new(load_idx, foot_idx, ANKLE_JOINT_K);
    s.local_anchor_a = Vec3::new(0.0, -0.10, 0.0);
    s.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(load_idx, foot_idx, ANKLE_JOINT_C);
    d.local_anchor_a = Vec3::new(0.0, -0.10, 0.0);
    d.local_anchor_b = Vec3::new(0.0, 0.0, 0.0);
    world.positional_dampers.push(d);

    // Active angular damper at the ankle (matching bilateral setup).
    world.angular_dampers.push(AngularDamper::new(
        load_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0), 400.0));

    // FootContact tripod, right foot.
    world.foot_contacts.push(FootContact::tripod(foot_idx, FOOT_HALF_H, FOOT_RADIUS, 1.0));

    let dt = 1.0 / 120.0;
    let sub_iters = 60;
    let sub_dt = dt / sub_iters as f64;

    let mut max_foot_tilt = 0.0_f64;
    let mut max_load_drop = 0.0_f64;
    let load_y_initial = world.bodies[load_idx].position.y;

    for _ in 0..frames {
        for _ in 0..sub_iters { world.substep(sub_dt); }
        let foot_tilt = world.bodies[foot_idx].orientation.scaled_axis().x;
        // The foot's orientation includes the baseline rotation; we want
        // tilt RELATIVE to baseline. Quick approximation: |orientation|
        // squared distance from baseline.
        let baseline_axis = foot_baseline.scaled_axis();
        let delta = world.bodies[foot_idx].orientation.scaled_axis() - baseline_axis;
        let foot_tilt_from_baseline = delta.norm();
        if foot_tilt_from_baseline.abs() > max_foot_tilt {
            max_foot_tilt = foot_tilt_from_baseline.abs();
        }
        let load_y = world.bodies[load_idx].position.y;
        let load_drop = load_y_initial - load_y;
        if load_drop > max_load_drop { max_load_drop = load_drop; }
        let _ = foot_tilt;
    }
    let final_y = world.bodies[load_idx].position.y;
    let total_force = world.foot_contacts[0].total_normal_force(&world.bodies);
    (final_y, max_foot_tilt, total_force)
}

fn main() {
    println!("FootContact under load — isolation test\n");

    for load_mass in [1.0, 5.0, 15.0, 35.0, 50.0] {
        let frames = 240; // 2 seconds
        let (final_y, max_tilt, force) = run_one(load_mass, frames);
        let expected_force = (FOOT_MASS + load_mass) * 9.81;
        let force_ok = (force - expected_force).abs() < 1.0;
        let stable = final_y.is_finite() && max_tilt < 0.3;  // tilt < 17°
        let status = if stable && force_ok { "✓ stable" } else { "✗ unstable" };
        println!("  load {:>5.1} kg: final load_y = {:+.4}, max foot tilt = {:.4} rad ({:.1}°), force = {:.2} N (expected {:.2}) — {}",
                 load_mass, final_y,
                 max_tilt, max_tilt.to_degrees(),
                 force, expected_force, status);
    }
}
