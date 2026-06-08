// Phase 3 step 1a: balance the simplest meaningful inverted pendulum.
//
// Setup: ONE foot (horizontal, on ground) + ONE rigid stick body above
// it, connected by a single 1-DOF ankle (just PF for now). The stick
// represents the whole upper body lumped into one rigid mass — no
// knees, no hips, no lumbar to confound the dynamics.
//
// Active ankle PD driven by stick lean. If the controller can hold this
// pendulum upright, then adding articulation back (knee, hip, lumbar)
// is a tuning problem. If the controller can't hold even THIS upright,
// the architecture is wrong and we redesign.
//
// What "upright" means here: stick.position.z stays close to the ankle
// position (< 5 cm of horizontal lean) over a 5 s simulation. Without
// active control, the natural inverted-pendulum tipping time for a
// 50 kg stick of half-height 0.5 m is ~0.6 s.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::Vec3;
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

const STICK_MASS:   f64 = 50.0;       // lumped upper body
const STICK_HALF_H: f64 = 0.50;       // 1 m total length — represents leg + torso
const STICK_RADIUS: f64 = 0.06;

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

// Passive ankle biomech (anatomical ROM). Same numbers as the validated
// ankle, just placed on the foot/stick joint here.
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const PF_MIN: f64 = -0.44;
const PF_MAX: f64 =  0.87;
const ANKLE_LIMIT_K: f64 = 80.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

// Active balance: torque applied DIRECTLY to the stick (about world +x),
// proportional to the stick's tilt and tilt-rate. The reaction on the
// foot is absorbed by the ground.
//
// Why not use a joint-angle PD here: with a PF-target PD, the foot can
// rotate (toes up = dorsiflexion is unconstrained) to satisfy the joint
// target without ever uprighting the stick. The PD reaches its target by
// tilting the foot, body keeps falling. Driving torque against the body's
// ABSOLUTE TILT bypasses that — torque is applied to the stick and the
// foot's reaction goes into the ground via contact.
const BAL_KP: f64 = 600.0;            // N·m torque per rad of stick tilt
const BAL_KD: f64 =  60.0;            // N·m·s torque per (rad/s) of tilt rate

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

    // Foot at horizontal anatomical pose. Body_a = stick (the parent in
    // this single-pendulum setup), body_b = foot. Ankle joint axes are in
    // stick-local frame; with the stick at identity at rest, axes match
    // world axes.
    //
    // Wait — actually we want body_a = STICK (the parent). The foot is
    // the "ground end" of the chain, not the parent. Let me reverse: in
    // a real ankle, the foot is the distal segment. The shin is the
    // parent (proximal). Following the rest of the codebase, body_a =
    // shin/stick, body_b = foot.
    let stick_baseline = UnitQuaternion::identity();
    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    stick.position = Vec3::new(0.0, FOOT_RADIUS + STICK_HALF_H, 0.0);
    stick.orientation = stick_baseline;
    let stick_idx = world.add_body(stick);

    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[stick_idx],
        Vec3::new(0.0, -STICK_HALF_H, 0.0),  // bottom of stick = ankle
        FOOT_ANCHOR_LOCAL,                    // foot's heel anchor
        foot_baseline);
    foot.position = pos;
    foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, stick_idx, foot_idx,
        Vec3::new(0.0, -STICK_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    // Passive ankle (PF only — single DOF for this v0).
    let pf_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(stick_idx, foot_idx, pf_axis,
            PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP)
            .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(stick_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(
        AngularSpring::new(stick_idx, foot_idx, pf_axis, PF_K, 0.0)
            .with_baseline(foot_baseline));

    // (No active joint-angle PD — balance is applied as a direct torque
    // pair on stick/foot in the per-frame loop below.)

    // Ground contact via the world-level system: every body with
    // capsule_shape collides with the ground at y=0. While balanced, only
    // the foot touches; the stick collides if the pendulum tips far enough.
    println!("BALANCE PENDULUM (Phase 3 step 1a)");
    println!("Single-segment inverted pendulum: 1 kg foot + {} kg stick", STICK_MASS as i32);
    println!("(stick half-height {} m, total length {} m)",
             STICK_HALF_H, STICK_HALF_H * 2.0);
    println!();
    println!("Without active balance, natural tip time ≈ {:.2} s",
             (STICK_HALF_H / 9.81).sqrt());
    println!("Controller: torque = {:.0}*stick_tilt_x + {:.0}*omega_x",
             BAL_KP, BAL_KD);
    println!("(applied directly to stick about world +x; reaction on foot)");
    println!();
    println!("{:<6} {:<11} {:<11} {:<11} {:<11} {:<11}",
             "t", "stick.z", "tilt_x", "torque", "stick.y", "stick_top.z");
    println!("{}", "-".repeat(66));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_lean = 0.0_f64;

    let mut last_tilt_x = 0.0;
    let mut last_tau = 0.0;

    while t < SIM_DURATION {
        // Controller fires INSIDE the substep loop. XpbdWorld clears the
        // torque accumulator at the end of every substep, so applying once
        // per frame would only fire on 1 of 60 substeps (1/60th authority).
        for _ in 0..SOLVER_ITERATIONS {
            let stick = &world.bodies[stick_idx];
            // For small angles, scaled_axis projection gives the tilt
            // about each world axis directly.
            let stick_tilt_x = stick.orientation.scaled_axis().x;
            let stick_omega_x = stick.angular_velocity.x;

            // Torque to push back to vertical: counter-rotate about +x.
            let tau_x = -BAL_KP * stick_tilt_x - BAL_KD * stick_omega_x;
            let tau_world = Vec3::new(tau_x, 0.0, 0.0);

            // Apply to stick, equal-and-opposite reaction on foot. (The
            // foot's reaction is absorbed by ground contact; that's where
            // the controller gets its leverage.)
            world.bodies[stick_idx].apply_torque(tau_world);
            world.bodies[foot_idx].apply_torque(-tau_world);

            world.substep(SUB_DT);

            last_tilt_x = stick_tilt_x;
            last_tau = tau_x;
        }
        t += DT;

        let stick = &world.bodies[stick_idx];
        let foot = &world.bodies[foot_idx];
        let stick_top = stick.position + stick.orientation * Vec3::new(0.0, STICK_HALF_H, 0.0);
        let ankle_world = foot.position + foot.orientation * FOOT_ANCHOR_LOCAL;
        let lean = stick_top.z - ankle_world.z;
        if lean.abs() > max_lean { max_lean = lean.abs(); }

        if t >= next_print {
            println!("{:<6.2} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5} {:<+11.5}",
                     t, stick.position.z,
                     last_tilt_x,
                     last_tau,
                     stick.position.y, stick_top.z);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let stick = &world.bodies[stick_idx];
    let stick_top = stick.position + stick.orientation * Vec3::new(0.0, STICK_HALF_H, 0.0);
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  stick CG:  ({:+.3}, {:+.3}, {:+.3})",
             stick.position.x, stick.position.y, stick.position.z);
    println!("  stick top: ({:+.3}, {:+.3}, {:+.3})",
             stick_top.x, stick_top.y, stick_top.z);
    println!("  max |lean|: {:.4} m", max_lean);
    println!();
    let upright = stick.position.y > 0.5;
    if upright && max_lean < 0.10 {
        println!("✓ Pendulum balanced — stayed upright with |lean| < 10 cm.");
    } else if upright {
        println!("⚠ Pendulum stayed up but lean got large ({:.3} m). Tighten gains.", max_lean);
    } else {
        println!("✗ Pendulum fell over.");
    }
}
