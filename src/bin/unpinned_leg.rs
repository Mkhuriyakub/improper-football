// Phase 2 step 4: unpinned single leg standing on the ground.
//
// Setup: pelvis stub (UNPINNED, mass 8 kg) + thigh + shin + foot, all
// in a vertical chain with the foot on the ground. Hip 3-DOF, knee
// 1-DOF (with off-axis locks), ankle 2-DOF + tight twist. Foot at
// horizontal anatomical pose (baseline R_x(-π/2)). Ground contact on
// the foot.
//
// Initial position: pelvis directly above the foot (pelvis CG at
// world x = +0.10 to align with the hip-anchor offset, putting the
// hip directly over the foot center). Whole body initially stationary.
//
// What we expect:
//   - The system is in an *unstable* equilibrium initially. Even tiny
//     numerical perturbations will start the pelvis tipping. There is
//     no active balance control, so the leg can't catch itself.
//   - For the first ~500 ms or so the pelvis stays close to upright,
//     supported by the foot's ground contact and the leg's elastic
//     joint anchors.
//   - Then the pelvis starts to fall in some direction (whichever
//     numerical noise pushes it toward), the knee buckles, the ankle
//     bends past its passive-spring resistance, and the leg collapses.
//   - The foot may or may not stay in contact during the fall, but
//     the chain should remain stable throughout (no NaNs, no exploding
//     bodies, contacts behaving sensibly even as the geometry rotates).
//
// This is the smallest test that puts a complete unpinned-pelvis leg
// chain on the ground. The "does it stand" question is no — that
// requires Phase 3 active balance — but "does the chain interact
// stably with ground contact during the fall" is yes/no, and is what
// we're validating here.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3, Mat3, swing_twist_angle};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.10;

// Pelvis stub: just enough mass to represent torso weight bearing down
// on the leg. Real pelvis + torso ≈ 35 kg; here we use 8 kg as a
// conservative load for this first integration test.
const PELVIS_MASS:    f64 = 8.0;
const PELVIS_HALF_H:  f64 = 0.10;
const PELVIS_RADIUS:  f64 = 0.18;
const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

const THIGH_MASS:   f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =     400.0;
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K:   f64 = 120.0;
const HIP_LIMIT_EXP: f64 =   2.0;

const SHIN_MASS:   f64 = 3.5;
const SHIN_RADIUS: f64 = 0.050;
const SHIN_HALF_H: f64 = 0.210;
const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;
const KNEE_K:        f64 =  5.0;
const KNEE_C:        f64 =  1.5;
const KNEE_NEUTRAL:  f64 =  0.10;
const KNEE_MIN:      f64 = -0.087;
const KNEE_MAX:      f64 =  2.44;
const KNEE_LIMIT_K:  f64 =  80.0;
const KNEE_LIMIT_EXP:f64 =   2.0;
const KNEE_OFF_AXIS_K:    f64 = 12.0;
const KNEE_OFF_AXIS_C:    f64 =  0.80;
const KNEE_OFF_AXIS_LIM:  f64 =  0.30;
const KNEE_OFF_AXIS_LIMIT_K: f64 = 50.0;

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

// Geometry: foot center at y = FOOT_RADIUS (= 0.032). Ankle at y = 0.032.
// Knee at y = 0.032 + 2*SHIN_HALF_H = 0.452. Hip at y = 0.452 + 2*THIGH_HALF_H
// = 0.902. Pelvis center at y = 0.902 - HIP_ANCHOR_PELVIS_Y = 0.952.
// Pelvis CG at x = HIP_ANCHOR_PELVIS_X = 0.10 (directly over the foot).
const PELVIS_INITIAL_POS: Vec3 = Vec3::new(0.10, 0.952, 0.0);

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

fn add_axis(
    world: &mut XpbdWorld, parent: usize, child: usize,
    axis: Vec3, min_a: f64, max_a: f64,
    k: f64, c: f64, neutral: f64,
    limit_k: f64, limit_exp: f64,
    baseline: UnitQuaternion<f64>,
) {
    world.angular_soft_limits.push(
        AngularSoftLimit::new(parent, child, axis, min_a, max_a, limit_k, limit_exp)
            .with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis, c));
    world.angular_springs.push(
        AngularSpring::new(parent, child, axis, k, neutral).with_baseline(baseline));
}

fn main() {
    let mut world = XpbdWorld::new();

    // Pelvis (UNPINNED).
    let mut pelvis = RigidBody::new(PELVIS_MASS,
        capsule_inertia(PELVIS_MASS, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_INITIAL_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Thigh.
    let hip_anchor_pelvis = Vec3::new(HIP_ANCHOR_PELVIS_X, HIP_ANCHOR_PELVIS_Y, 0.0);
    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor_pelvis, Vec3::new(0.0, THIGH_HALF_H, 0.0),
        UnitQuaternion::identity());
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        hip_anchor_pelvis, Vec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let id = UnitQuaternion::identity();
    add_axis(&mut world, pelvis_idx, thigh_idx, Vec3::new(-1.0, 0.0, 0.0),
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_FLEX_K, HIP_FLEX_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thigh_idx, Vec3::new(0.0, 0.0, 1.0),
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_ABD_K, HIP_ABD_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);
    add_axis(&mut world, pelvis_idx, thigh_idx, Vec3::new(0.0, 1.0, 0.0),
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_ROT_K, HIP_ROT_C, 0.0,
        HIP_LIMIT_K, HIP_LIMIT_EXP, id);

    // Shin.
    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0),
        UnitQuaternion::identity());
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(&mut world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    add_axis(&mut world, thigh_idx, shin_idx, Vec3::new(1.0, 0.0, 0.0),
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL,
        KNEE_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(&mut world, thigh_idx, shin_idx, Vec3::new(0.0, 0.0, 1.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);
    add_axis(&mut world, thigh_idx, shin_idx, Vec3::new(0.0, 1.0, 0.0),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0,
        KNEE_OFF_AXIS_LIMIT_K, KNEE_LIMIT_EXP, id);

    // Foot at horizontal anatomical pose: baseline R_x(-π/2).
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    add_axis(&mut world, shin_idx, foot_idx, Vec3::new(1.0, 0.0, 0.0),
        PF_MIN, PF_MAX, PF_K, PF_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(&mut world, shin_idx, foot_idx, Vec3::new(0.0, 0.0, 1.0),
        INV_MIN, INV_MAX, INV_K, INV_C, 0.0,
        ANKLE_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);
    add_axis(&mut world, shin_idx, foot_idx, Vec3::new(0.0, 1.0, 0.0),
        -TWIST_LIM, TWIST_LIM, TWIST_K, TWIST_C, 0.0,
        TWIST_LIMIT_K, ANKLE_LIMIT_EXP, foot_baseline);

    // Ground contacts on every body. The foot uses the default plane_y=0
    // (its capsule shape — heel and toe end-spheres at ankle height —
    // matches anatomy reasonably). Non-foot bodies use plane_y = -radius
    // so contact engages when the body's axis endpoint crosses the
    // ground, not when the unmodeled rounded capsule end below the axis
    // does. Without this offset the shin's lower rounded end would
    // already be 1.8 cm below ground at rest, jolting the leg upward
    // at t=0. With this offset, every body sits cleanly above ground at
    // start and contact engages as the leg buckles and parts approach
    // ground level.
    world.ground_contacts.push(
        GroundContact::new(foot_idx, FOOT_HALF_H, FOOT_RADIUS));
    world.ground_contacts.push(
        GroundContact::new(shin_idx, SHIN_HALF_H, SHIN_RADIUS)
            .with_plane_y(-SHIN_RADIUS));
    world.ground_contacts.push(
        GroundContact::new(thigh_idx, THIGH_HALF_H, THIGH_RADIUS)
            .with_plane_y(-THIGH_RADIUS));
    world.ground_contacts.push(
        GroundContact::new(pelvis_idx, PELVIS_HALF_H, PELVIS_RADIUS)
            .with_plane_y(-PELVIS_RADIUS));

    println!("UNPINNED SINGLE LEG (Phase 2 step 4)");
    println!("Pelvis (8 kg, unpinned) + thigh (7 kg) + shin (3.5 kg) + foot (1 kg)");
    println!("Initial: pelvis CG at ({:.2}, {:.3}, 0), directly above foot",
             PELVIS_INITIAL_POS.x, PELVIS_INITIAL_POS.y);
    println!("No active balance — system is in unstable equilibrium and will fall.");
    println!();
    println!("{:<6} {:<11} {:<11} {:<11} {:<11} {:<11}",
             "t", "pelvis.y", "pelvis.z", "hip_flex", "knee", "foot.y");
    println!("{}", "-".repeat(64));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_pelvis_y_drop = 0.0_f64;
    let mut foot_lifted_at: Option<f64> = None;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        let pelvis = &world.bodies[pelvis_idx];
        let drop = PELVIS_INITIAL_POS.y - pelvis.position.y;
        if drop > max_pelvis_y_drop {
            max_pelvis_y_drop = drop;
        }
        let foot = &world.bodies[foot_idx];
        if foot_lifted_at.is_none() && foot.position.y > FOOT_RADIUS + 0.01 {
            foot_lifted_at = Some(t);
        }

        if t >= next_print {
            let p = &world.bodies[pelvis_idx];
            let th = &world.bodies[thigh_idx];
            let sh = &world.bodies[shin_idx];
            let fo = &world.bodies[foot_idx];
            let hip_q_b_in_a = p.orientation.inverse() * th.orientation;
            let hip_flex = swing_twist_angle(hip_q_b_in_a, Vec3::new(-1.0, 0.0, 0.0));
            let knee_q_b_in_a = th.orientation.inverse() * sh.orientation;
            let knee = swing_twist_angle(knee_q_b_in_a, Vec3::new(1.0, 0.0, 0.0));
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, p.position.y, p.position.z, hip_flex, knee, fo.position.y);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let pelvis = &world.bodies[pelvis_idx];
    let foot = &world.bodies[foot_idx];
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  pelvis position = ({:+.3}, {:+.3}, {:+.3})",
             pelvis.position.x, pelvis.position.y, pelvis.position.z);
    println!("  pelvis dropped {:.3} m from start", max_pelvis_y_drop);
    println!("  foot position   = ({:+.3}, {:+.3}, {:+.3})",
             foot.position.x, foot.position.y, foot.position.z);
    if let Some(tl) = foot_lifted_at {
        println!("  foot lifted off ground at t = {:.2} s", tl);
    } else {
        println!("  foot stayed on / near ground throughout");
    }
    if pelvis.position.y.is_finite() && foot.position.y.is_finite() {
        println!();
        println!("All bodies finite — the chain remained stable through the fall.");
    } else {
        println!();
        println!("WARNING: NaN detected — the chain went unstable.");
    }
}
