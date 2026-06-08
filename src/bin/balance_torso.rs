// Phase 3 step 1c: balance a 4-articulation chain (foot + shin + thigh
// + pelvis + torso-lump) using the same ankle absolute-tilt strategy,
// with posture PDs locking knee, hip, and lumbar at neutral.
//
// Why this step exists:
//   Step 1b validated that a single posture PD (knee) keeps a 2-body
//   chain rigid above the ankle. Step 1c is the next question: do
//   stacked posture PDs (knee + hip + lumbar) compose without
//   destabilizing each other? And does the ankle strategy still work
//   when the chain is taller, heavier, and segmented at three places
//   above the ankle?
//
// Geometry — anatomically realistic now (no longer using a single
// "thigh-lump" to lump the upper body):
//   shin    3 kg,  half_h 0.21 m  (length 0.42)
//   thigh   7 kg,  half_h 0.225 m (length 0.45)
//   pelvis  8 kg,  half_h 0.10 m  (length 0.20)
//   torso  40 kg,  half_h 0.30 m  (length 0.60) — lumps thoracic + head + arms
//
// Total above ankle: 0.42 + 0.45 + 0.20 + 0.60 = 1.67 m, 58 kg.
// Top of chain at y = 0.032 + 1.67 = 1.70 m.
//
// Inverted-pendulum dynamics (chain treated as rigid rod above ankle):
//   m·g·h ≈ 58·9.81·0.835 ≈ 475 N·m/rad
//   I_about_ankle ≈ (1/3)·m·L² ≈ 54 kg·m²
//   tip time ≈ sqrt(L/(2g)) ≈ 0.29 s
//
// Ankle gains scaled to keep ω_n similar to 1a/1b:
//   1a/1b: ω_n² = (600-245)/16.7 = 21 → ω_n ≈ 4.6 rad/s
//   1c:    KP = m·g·h + 21·I = 475 + 1130 = 1600 N·m/rad
//          KD = 2·ζ·ω_n·I  with ζ ≈ 0.7  → KD ≈ 350 N·m·s/rad

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

// ─── Body constants ───────────────────────────────────────────────────

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: Vec3 = Vec3::new(0.0, FOOT_HALF_H, 0.0);

const SHIN_MASS: f64   = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const THIGH_MASS: f64   = 7.0;
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.07;

const PELVIS_MASS: f64   = 8.0;
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.13;

const TORSO_MASS: f64   = 40.0;  // thoracic + head + arms lumped
const TORSO_HALF_H: f64 = 0.30;
const TORSO_RADIUS: f64 = 0.13;

// ─── Joint constants ──────────────────────────────────────────────────

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const ANKLE_LIMIT_K: f64 = 80.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;
const KNEE_PASSIVE_K: f64 = 5.0;  const KNEE_PASSIVE_C: f64 = 1.5;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;
const KNEE_LIMIT_EXP: f64 = 2.0;
const KNEE_PD_KP: f64 = 3000.0;  const KNEE_PD_KD: f64 = 300.0;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =    400.0;
const HIP_PASSIVE_K: f64 = 10.0;  const HIP_PASSIVE_C: f64 = 7.0;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_LIMIT_K: f64 = 120.0;
const HIP_LIMIT_EXP: f64 = 2.0;
// Hip carries pelvis+torso = 48 kg above it; gravity buckle ≈ 188 N·m/rad.
// During ankle-driven recovery the inertial moment from accelerating the
// upper body adds substantially. KP=4000 gives ~20× gravity headroom.
const HIP_PD_KP: f64 = 4000.0;  const HIP_PD_KD: f64 = 400.0;

const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =    600.0;
const LUMBAR_PASSIVE_K: f64 = 150.0;  const LUMBAR_PASSIVE_C: f64 = 25.0;
const L_FLEX_MIN: f64 = -0.44;  const L_FLEX_MAX: f64 = 1.05;
const LUMBAR_LIMIT_K: f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 = 2.0;
// Lumbar carries torso-lump only (40 kg, lever 0.3 m → 118 N·m/rad).
const LUMBAR_PD_KP: f64 = 2500.0;  const LUMBAR_PD_KD: f64 = 250.0;

// Ankle balance. With the chain only "approximately rigid" via posture
// PDs, sensing torso tilt and pushing too hard amplifies upper-body
// flex modes. Using gains barely above stability threshold and sensing
// THIGH tilt (closest rigid segment to the actuator) avoids exciting
// those modes. Stick analysis (chain treated as rigid rod above ankle):
//   m·g·h ≈ 475 N·m/rad → need KP > 475 for restoring force
//   ω_n² = (KP - mgh)/I, with KP=1000, I=54 → ω_n ≈ 3.1 rad/s
//   KD = 2·ζ·ω_n·I with ζ=0.7 → KD ≈ 235 N·m·s/rad
const BAL_KP: f64 = 1000.0;
const BAL_KD: f64 =  235.0;

// ─── Helpers ──────────────────────────────────────────────────────────

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

    // Build chain top-down. Top of torso at y = 1.702.
    //   torso center at y = 1.402  (top - HALF_H)
    //   lumbar joint at y = 1.102
    //   pelvis center at y = 1.002
    //   hip joint at y = 0.902
    //   thigh center at y = 0.677
    //   knee joint at y = 0.452
    //   shin center at y = 0.242
    //   ankle at y = 0.032
    //   foot center at y = 0.032

    let mut torso = RigidBody::new_capsule(TORSO_MASS, TORSO_HALF_H, TORSO_RADIUS);
    torso.position = Vec3::new(0.0, 1.402, 0.0);
    torso.orientation = id;
    let torso_idx = world.add_body(torso);

    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    let (pos, orient) = place_child(&world.bodies[torso_idx],
        Vec3::new(0.0, -TORSO_HALF_H, 0.0),
        Vec3::new(0.0,  PELVIS_HALF_H, 0.0), id);
    pelvis.position = pos; pelvis.orientation = orient;
    let pelvis_idx = world.add_body(pelvis);

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        Vec3::new(0.0, -PELVIS_HALF_H, 0.0),
        Vec3::new(0.0,  THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
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

    // ─── Joints ────────────────────────────────────────────────────────

    // Lumbar (torso ↔ pelvis).
    add_elastic_joint(&mut world, torso_idx, pelvis_idx,
        Vec3::new(0.0, -TORSO_HALF_H, 0.0),
        Vec3::new(0.0,  PELVIS_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let lumbar_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(torso_idx, pelvis_idx, lumbar_axis,
            L_FLEX_MIN, L_FLEX_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(torso_idx, pelvis_idx, lumbar_axis, LUMBAR_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(torso_idx, pelvis_idx, lumbar_axis, LUMBAR_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, pelvis_idx, lumbar_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD));

    // Hip (pelvis ↔ thigh).
    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        Vec3::new(0.0, -PELVIS_HALF_H, 0.0),
        Vec3::new(0.0,  THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let hip_axis = Vec3::new(-1.0, 0.0, 0.0);  // matches skeleton_standing convention
    world.angular_soft_limits.push(
        AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_axis,
            HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(pelvis_idx, thigh_idx, hip_axis, HIP_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(pelvis_idx, thigh_idx, hip_axis, HIP_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_axis,
            HIP_PD_KP, HIP_PD_KD));

    // Knee (thigh ↔ shin).
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

    // Ankle (shin ↔ foot) — same passive 1-DOF as 1a/1b.
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
    // with the ground at y=0. At neutral pose only the foot touches; other
    // segments collide if the chain falls.

    println!("BALANCE TORSO CHAIN (Phase 3 step 1c)");
    println!("foot ({} kg) + shin ({} kg) + thigh ({} kg) + pelvis ({} kg) + torso ({} kg)",
             FOOT_MASS as i32, SHIN_MASS as i32, THIGH_MASS as i32,
             PELVIS_MASS as i32, TORSO_MASS as i32);
    println!("Chain above ankle: {:.2} m, {:.0} kg total.",
             1.67, SHIN_MASS + THIGH_MASS + PELVIS_MASS + TORSO_MASS);
    println!("Posture PDs: knee KP={:.0}, hip KP={:.0}, lumbar KP={:.0}.",
             KNEE_PD_KP, HIP_PD_KP, LUMBAR_PD_KP);
    println!("Ankle balance: torque = {:.0}*thigh_tilt + {:.0}*omega_x",
             BAL_KP, BAL_KD);
    println!();
    println!("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "t", "thigh_tlt", "knee_bnd", "hip_bnd", "lumb_bnd", "torque", "torso.y");
    println!("{}", "-".repeat(74));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut max_torso_tilt = 0.0_f64;
    let mut max_thigh_tilt = 0.0_f64;
    let mut max_knee_bend = 0.0_f64;
    let mut max_hip_bend = 0.0_f64;
    let mut max_lumb_bend = 0.0_f64;
    let mut last_thigh_tilt = 0.0;
    let mut last_tau = 0.0;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            // Sense the THIGH (closest rigid segment to the actuator) —
            // not the torso. With finite-stiffness posture PDs, the
            // upper body lags the actuator; sensing the lagged segment
            // excites flex-mode resonance through the controller. The
            // thigh tracks the foot/shin closely via the (near-rigid)
            // knee, so its tilt is the cleanest pendulum-angle proxy.
            let thigh = &world.bodies[thigh_idx];
            let thigh_tilt_x  = thigh.orientation.scaled_axis().x;
            let thigh_omega_x = thigh.angular_velocity.x;

            let tau_x = -BAL_KP * thigh_tilt_x - BAL_KD * thigh_omega_x;
            let tau_world = Vec3::new(tau_x, 0.0, 0.0);
            world.bodies[shin_idx].apply_torque(tau_world);
            world.bodies[foot_idx].apply_torque(-tau_world);

            world.substep(SUB_DT);

            last_thigh_tilt = thigh_tilt_x;
            last_tau = tau_x;
        }
        t += DT;

        // Joint angle diagnostics.
        let q_shin_in_thigh = world.bodies[thigh_idx].orientation.inverse()
                            * world.bodies[shin_idx].orientation;
        let knee_bend = swing_twist_angle(q_shin_in_thigh, Vec3::new(1.0, 0.0, 0.0));

        let q_thigh_in_pelvis = world.bodies[pelvis_idx].orientation.inverse()
                              * world.bodies[thigh_idx].orientation;
        let hip_bend = swing_twist_angle(q_thigh_in_pelvis, Vec3::new(-1.0, 0.0, 0.0));

        let q_pelvis_in_torso = world.bodies[torso_idx].orientation.inverse()
                              * world.bodies[pelvis_idx].orientation;
        let lumb_bend = swing_twist_angle(q_pelvis_in_torso, Vec3::new(1.0, 0.0, 0.0));

        let torso_tilt = world.bodies[torso_idx].orientation.scaled_axis().x;
        if last_thigh_tilt.abs() > max_thigh_tilt { max_thigh_tilt = last_thigh_tilt.abs(); }
        if torso_tilt.abs() > max_torso_tilt { max_torso_tilt = torso_tilt.abs(); }
        if knee_bend.abs() > max_knee_bend { max_knee_bend = knee_bend.abs(); }
        if hip_bend.abs() > max_hip_bend { max_hip_bend = hip_bend.abs(); }
        if lumb_bend.abs() > max_lumb_bend { max_lumb_bend = lumb_bend.abs(); }

        if t >= next_print {
            let torso = &world.bodies[torso_idx];
            println!("{:<6.2} {:<+10.5} {:<+10.5} {:<+10.5} {:<+10.5} {:<+10.3} {:<+10.4}",
                     t, last_thigh_tilt, knee_bend, hip_bend, lumb_bend,
                     last_tau, torso.position.y);
            next_print += PRINT_INTERVAL;
        }
    }

    println!();
    let torso = &world.bodies[torso_idx];
    println!("Final state (after {:.1} s):", SIM_DURATION);
    println!("  torso CG:   ({:+.3}, {:+.3}, {:+.3})",
             torso.position.x, torso.position.y, torso.position.z);
    println!("  max |thigh tilt|:  {:.4} rad ({:.2}°)",
             max_thigh_tilt, max_thigh_tilt.to_degrees());
    println!("  max |torso tilt|:  {:.4} rad ({:.2}°)",
             max_torso_tilt, max_torso_tilt.to_degrees());
    println!("  max |knee bend|:   {:.4} rad ({:.2}°)",
             max_knee_bend, max_knee_bend.to_degrees());
    println!("  max |hip bend|:    {:.4} rad ({:.2}°)",
             max_hip_bend, max_hip_bend.to_degrees());
    println!("  max |lumbar bend|: {:.4} rad ({:.2}°)",
             max_lumb_bend, max_lumb_bend.to_degrees());
    println!();

    let upright = torso.position.y > 1.0;
    let small_lean = max_thigh_tilt < 0.10 && max_torso_tilt < 0.10;
    let joints_held = max_knee_bend < 0.10 && max_hip_bend < 0.10 && max_lumb_bend < 0.10;
    if upright && small_lean && joints_held {
        println!("✓ Four-articulation chain balanced.");
        println!("  Torso stayed within 5.7° of vertical, all locked joints stayed within 5.7°.");
    } else if upright && small_lean {
        println!("⚠ Balanced overall but at least one joint bent significantly.");
        println!("  Increase the corresponding posture PD KP.");
    } else if upright {
        println!("⚠ Stayed up but lean got large ({:.2}°). Tighten ankle gains.",
                 max_torso_tilt.to_degrees());
    } else {
        println!("✗ Chain fell over.");
    }
}
