// Phase 3 step 1b: balance an articulated two-segment pendulum (foot +
// shin + thigh-lump) using the same ankle absolute-tilt strategy from
// step 1a, plus a posture PD locking the knee at neutral.
//
// Why this step exists:
//   Step 1a balanced a single rigid stick over a foot. That worked, but
//   it didn't tell us whether the ankle strategy still works when the
//   chain above the ankle is articulated (i.e., when the knee can buckle
//   under gravity). Step 1b adds exactly one articulation: the knee.
//   The ankle balance code is unchanged. The new question is: can a
//   knee posture PD make the knee behave rigidly enough that the chain
//   stays upright?
//
// Geometry: total length above ankle and total mass above ankle are kept
// the same as step 1a's 50 kg / 1.0 m stick. We split them anatomically:
//   shin       3 kg, half_h 0.21 m   (length 0.42)
//   thigh-lump 47 kg, half_h 0.29 m  (length 0.58)
// Sum of half_h*2 = 1.0 m. Sum of mass = 50 kg. So the inverted-pendulum
// tip dynamics should match step 1a closely IF the knee stays locked.
//
// Sensing: thigh's absolute tilt about world +x. With knee locked,
// thigh tilt equals whole-chain tilt; that's the right pendulum angle.
// Acting: torque pair between shin and foot about world +x (ankle-axis
// torque expressed in world frame, valid at small angles since at
// neutral pose the ankle PF axis IS world +x).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, swing_twist_angle};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;
const PRINT_INTERVAL: f64 = 0.10;

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const SHIN_MASS: f64   = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

// "Thigh lump" — represents thigh + everything-above in one rigid body.
// Total chain above ankle: 0.42 + 0.58 = 1.0 m, 50 kg. Matches step 1a.
const THIGH_MASS: f64   = 47.0;
const THIGH_HALF_H: f64 = 0.29;
const THIGH_RADIUS: f64 = 0.07;

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

// Passive ankle (PF only — single DOF for this v0).
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const PF_MIN: f64 = -0.44;
const PF_MAX: f64 =  0.87;
const ANKLE_LIMIT_K: f64 = 80.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;

// Passive knee — anatomical limits, very weak passive spring.
// (The posture PD below is what actually holds the knee straight.)
const KNEE_PASSIVE_K: f64 = 5.0;  const KNEE_PASSIVE_C: f64 = 1.5;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;
const KNEE_LIMIT_EXP: f64 = 2.0;

// Posture PD locking the knee at neutral. This is the new mechanism in
// step 1b. Sized to easily resist the gravity-driven knee-buckle moment:
// a 47 kg thigh-lump with CG 0.29 m above the knee produces ~134 N·m/rad
// of buckle torque at small angles. KP=2000 gives ~15× headroom, so the
// knee should track ~0 closely; KD critically damps it.
const KNEE_PD_KP: f64 = 2000.0;
const KNEE_PD_KD: f64 =  200.0;

// Ankle balance gains. Stick analysis (step 1a):
//   I_chain ≈ 16.7 kg·m², m·g·h = 245 N·m/rad, KP=600, KD=60 worked.
// Same setup, same gains here.
const BAL_KP: f64 = 600.0;
const BAL_KD: f64 =  60.0;

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
    let id = UnitQuaternion::identity();

    // Build chain top-down (so we can place each child at the parent's
    // distal anchor). Then compute foot position from shin's distal end.
    //
    // Coordinate system: world +y is up. Chain at neutral pose:
    //   foot center at y = FOOT_RADIUS = 0.032
    //   ankle (foot's heel anchor in world) at y = 0.032
    //   shin center at y = 0.032 + SHIN_HALF_H = 0.242
    //   knee at y = 0.032 + 2*SHIN_HALF_H = 0.452
    //   thigh-lump center at y = 0.452 + THIGH_HALF_H = 0.742
    //   top of thigh-lump at y = 0.452 + 2*THIGH_HALF_H = 1.032
    //
    // Equivalently: top of chain is 1.0 m above ankle (matches step 1a).

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    thigh.position = Vec3::new(0.0, 0.452 + THIGH_HALF_H, 0.0);
    thigh.orientation = id;
    let thigh_idx = world.add_body(thigh);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    // Knee joint (thigh ↔ shin). Elastic positional + passive 1-DOF
    // (sagittal flex/extend) + posture PD locking at neutral.
    add_elastic_joint(&mut world, thigh_idx, shin_idx,
        Vec3::new(0.0, -THIGH_HALF_H, 0.0),
        Vec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
            KNEE_MIN, KNEE_MAX, KNEE_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, shin_idx, knee_axis,
            KNEE_PD_KP, KNEE_PD_KD));

    // Ankle joint (shin ↔ foot) — same as step 1a.
    add_elastic_joint(&mut world, shin_idx, foot_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
            PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP)
            .with_baseline(foot_baseline));
    world.angular_dampers.push(
        AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0)
            .with_baseline(foot_baseline));

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. At neutral pose only the foot touches; shin
    // and thigh collide if the chain falls. (Pre-migration this binary
    // used per-body disable hacks for shin/thigh; bodies now respect the
    // ground unless pinned.)

    println!("BALANCE TWO-SEGMENT (Phase 3 step 1b)");
    println!("foot ({} kg) + shin ({} kg) + thigh-lump ({} kg)",
             FOOT_MASS as i32, SHIN_MASS as i32, THIGH_MASS as i32);
    println!("Chain above ankle: 1.0 m, 50 kg total (matches step 1a stick).");
    println!("Knee locked at neutral by posture PD (KP={:.0}, KD={:.0}).",
             KNEE_PD_KP, KNEE_PD_KD);
    println!("Ankle balance: torque = {:.0}*thigh_tilt_x + {:.0}*omega_x",
             BAL_KP, BAL_KD);
    println!("(applied between shin and foot about world +x)");
    println!();
    println!("{:<6} {:<11} {:<11} {:<11} {:<11} {:<11}",
             "t", "thigh_tilt", "knee_bend", "torque", "thigh.y", "thigh_top.y");
    println!("{}", "-".repeat(66));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_thigh_tilt = 0.0_f64;
    let mut max_knee_bend = 0.0_f64;
    let mut last_thigh_tilt = 0.0;
    let mut last_tau = 0.0;

    while t < SIM_DURATION {
        // Controller fires INSIDE the substep loop — XpbdWorld clears
        // torque accumulators every substep (see project memory).
        for _ in 0..SOLVER_ITERATIONS {
            let thigh = &world.bodies[thigh_idx];
            let thigh_tilt_x  = thigh.orientation.scaled_axis().x;
            let thigh_omega_x = thigh.angular_velocity.x;

            // Ankle balance torque pair (shin + foot reaction).
            let tau_x = -BAL_KP * thigh_tilt_x - BAL_KD * thigh_omega_x;
            let tau_world = Vec3::new(tau_x, 0.0, 0.0);
            world.bodies[shin_idx].apply_torque(tau_world);
            world.bodies[foot_idx].apply_torque(-tau_world);

            world.substep(SUB_DT);

            last_thigh_tilt = thigh_tilt_x;
            last_tau = tau_x;
        }
        t += DT;

        // Knee bend: signed angle of shin about thigh's local +x.
        let q_shin_in_thigh = world.bodies[thigh_idx].orientation.inverse()
                            * world.bodies[shin_idx].orientation;
        let knee_bend = swing_twist_angle(q_shin_in_thigh, Vec3::new(1.0, 0.0, 0.0));

        if last_thigh_tilt.abs() > max_thigh_tilt { max_thigh_tilt = last_thigh_tilt.abs(); }
        if knee_bend.abs() > max_knee_bend { max_knee_bend = knee_bend.abs(); }

        if t >= next_print {
            let thigh = &world.bodies[thigh_idx];
            let thigh_top_y = thigh.position.y + (thigh.orientation
                * Vec3::new(0.0, THIGH_HALF_H, 0.0)).y;
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, last_thigh_tilt, knee_bend, last_tau,
                     thigh.position.y, thigh_top_y);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let thigh = &world.bodies[thigh_idx];
    let thigh_top = thigh.position + thigh.orientation * Vec3::new(0.0, THIGH_HALF_H, 0.0);
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  thigh CG:     ({:+.3}, {:+.3}, {:+.3})",
             thigh.position.x, thigh.position.y, thigh.position.z);
    println!("  thigh top:    ({:+.3}, {:+.3}, {:+.3})",
             thigh_top.x, thigh_top.y, thigh_top.z);
    println!("  max |thigh tilt|: {:.4} rad ({:.2}°)",
             max_thigh_tilt, max_thigh_tilt.to_degrees());
    println!("  max |knee bend|:  {:.4} rad ({:.2}°)",
             max_knee_bend, max_knee_bend.to_degrees());
    println!();

    let upright = thigh.position.y > 0.5;
    let small_lean = max_thigh_tilt < 0.10;       // < ~5.7°
    let knee_held = max_knee_bend < 0.10;          // < ~5.7°
    if upright && small_lean && knee_held {
        println!("✓ Two-segment pendulum balanced.");
        println!("  Thigh stayed within 5.7° of vertical, knee stayed within 5.7° of locked.");
    } else if upright && small_lean {
        println!("⚠ Balanced overall but knee bent {:.2}° (>{}°).",
                 max_knee_bend.to_degrees(), 5.7);
        println!("  Increase KNEE_PD_KP to lock the knee more firmly.");
    } else if upright {
        println!("⚠ Stayed up but lean got large ({:.2}°). Tighten ankle gains.",
                 max_thigh_tilt.to_degrees());
    } else {
        println!("✗ Pendulum fell over.");
    }
}
