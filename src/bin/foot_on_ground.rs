// Phase 2 step 3: foot supported by ground contact through the ankle joint.
//
// Setup:
//   - Pinned vertical shin at world (0, 0.30, 0)
//   - Horizontal foot connected to shin via the 2-DOF ankle (PF + INV +
//     locked twist), with the foot's body in its anatomical horizontal
//     pose (baseline = R_x(-π/2)) — this is the configuration that
//     became possible after the swing-twist refactor.
//   - Ground contact (penalty spring + Coulomb friction) attached to
//     the foot. Ground plane at y = 0.
//
// Initial state: foot at anatomical neutral. Its bottom sits at
//   y_foot_bottom = ankle_y - foot_radius = (0.30 - 0.21) - 0.032 = 0.058 m
// — i.e. ~5.8 cm above the ground. Gravity will pull the foot down (+
// also tip its toes downward via the gravitational torque from the
// forward-offset center of mass), and the ankle joint's elastic anchor
// + biomech springs will resist. The foot lands, contacts the ground,
// and settles.
//
// Expected behavior:
//   - Foot drops, hits ground around t ≈ 0.11 s
//   - Settles with foot lying flat on ground
//   - Slight PF tilt (the foot wants to droop forward under gravity,
//     but the ground contact under the toe end pushes back, so the
//     tilt is smaller than the unsupported equilibrium of ~0.22 rad)
//   - INV ≈ 0 (no lateral asymmetry)
//   - Twist ≈ 0 (clean — no bivector contamination)
//   - Foot center y settles at exactly the radius (0.032 m) once the
//     foot is horizontal on the ground

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, swing_twist_angle};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.05;

const SHIN_HALF_H: f64 = 0.210;
const SHIN_RADIUS: f64 = 0.050;

const FOOT_MASS:    f64 = 1.0;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

const PF_K:    f64 = 4.0;   const PF_C:    f64 = 1.2;
const INV_K:   f64 = 3.5;   const INV_C:   f64 = 1.0;
const TWIST_K: f64 = 8.0;   const TWIST_C: f64 = 0.5;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const INV_MIN: f64 = -0.26; const INV_MAX: f64 = 0.52;
const TWIST_LIM: f64 = 0.10;
const ANKLE_LIMIT_K: f64 = 80.0;
const TWIST_LIMIT_K: f64 = 200.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

// Shin pinned so the ankle (shin's distal end) sits at exactly y =
// FOOT_RADIUS = 0.032 m. With the foot horizontal and the heel/toe end
// spheres on either side of the ankle, both end-spheres start with their
// bottoms at y = 0 (just touching the ground). Gravity will then tip the
// foot forward at the ankle (the foot's center sits ~9 cm forward of the
// ankle), the toe will penetrate, the ground will push back, and the
// system finds a static equilibrium with both ground contact and ankle
// joint forces in balance — which is the actual scenario we want to
// validate. Earlier `SHIN_REST_POS.y = 0.30` left the foot dangling 1.9 cm
// above the ground after its gravitational droop and never contacted.
const SHIN_REST_POS: Vec3 = Vec3::new(0.0, 0.21 + FOOT_RADIUS, 0.0);

fn place_child(
    parent: &RigidBody, parent_distal: Vec3, child_proximal: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: Vec3, ca: Vec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = pa; s.local_anchor_b = ca;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = pa; d.local_anchor_b = ca;
    world.positional_dampers.push(d);
}

fn main() {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);

    // Pinned shin.
    let mut shin = RigidBody::new_capsule(0.0, SHIN_HALF_H, SHIN_RADIUS);
    shin.position = SHIN_REST_POS;
    shin.orientation = UnitQuaternion::identity();
    let shin_idx = world.add_body(shin);

    // Foot at horizontal anatomical pose: baseline R_x(-π/2).
    let baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(
        &world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        baseline);
    foot.position = pos;
    foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    // Ankle biomech, all axes carry the horizontal baseline.
    let pf_axis    = Vec3::new(1.0, 0.0, 0.0);
    let inv_axis   = Vec3::new(0.0, 0.0, 1.0);
    let twist_axis = Vec3::new(0.0, 1.0, 0.0);

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
            PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0).with_baseline(baseline));

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, inv_axis,
            INV_MIN, INV_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, inv_axis, INV_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, inv_axis, INV_K, 0.0).with_baseline(baseline));

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, twist_axis,
            -TWIST_LIM, TWIST_LIM, TWIST_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, twist_axis, TWIST_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, twist_axis, TWIST_K, 0.0).with_baseline(baseline));

    // Ground contact applies to all bodies with capsule_shape (default μ=0.6).
    // Shin is pinned so it's skipped automatically; foot collides with the floor.
    let initial_foot_bottom = world.bodies[foot_idx].position.y - FOOT_RADIUS;
    println!("FOOT ON GROUND (post-refactor)");
    println!("Pinned shin at y={:.2}; foot starts horizontal, bottom at y={:.4}",
             SHIN_REST_POS.y, initial_foot_bottom);
    println!("Ground at y=0. Free-fall distance to first contact: {:.4} m",
             initial_foot_bottom);
    println!("Expected first contact: t ≈ {:.3} s",
             (2.0 * initial_foot_bottom / 9.81).sqrt());
    println!();
    println!("{:<6} {:<10} {:<10} {:<11} {:<11} {:<11}",
             "t", "foot.y", "vy", "PF", "INV", "twist");
    println!("{}", "-".repeat(64));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let shin = &world.bodies[shin_idx];
            let foot = &world.bodies[foot_idx];
            let q_b_in_a = shin.orientation.inverse() * foot.orientation;
            let q_from_rest = q_b_in_a * baseline.inverse();
            let pf    = swing_twist_angle(q_from_rest, pf_axis);
            let inv   = swing_twist_angle(q_from_rest, inv_axis);
            let twist = swing_twist_angle(q_from_rest, twist_axis);
            println!("{:<6.2} {:<+10.5} {:<+10.4} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, foot.position.y, foot.linear_velocity.y, pf, inv, twist);
            next_print += PRINT_INTERVAL;
        }
    }

    let shin = &world.bodies[shin_idx];
    let foot = &world.bodies[foot_idx];
    let q_b_in_a = shin.orientation.inverse() * foot.orientation;
    let q_from_rest = q_b_in_a * baseline.inverse();
    let pf    = swing_twist_angle(q_from_rest, pf_axis);
    let inv   = swing_twist_angle(q_from_rest, inv_axis);
    let twist = swing_twist_angle(q_from_rest, twist_axis);
    println!();
    println!("Final state:");
    println!("  foot center y = {:+.5} m  (expected ≈ {:.4} = FOOT_RADIUS)",
             foot.position.y, FOOT_RADIUS);
    println!("  foot vy       = {:+.6} m/s  (expected ≈ 0)", foot.linear_velocity.y);
    println!("  PF            = {:+.5} rad ({:+.2}°)", pf, pf.to_degrees());
    println!("  INV           = {:+.5} rad ({:+.2}°)", inv, inv.to_degrees());
    println!("  twist         = {:+.5} rad ({:+.2}°)", twist, twist.to_degrees());
}
