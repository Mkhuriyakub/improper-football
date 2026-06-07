// Phase 1 anatomical ankle — text validation binary.
//
// Builds the multi-body ankle assembly (talocrural + subtalar + MTP)
// per the spec at docs/superpowers/specs/2026-05-18-anatomical-ankle-phase1-design.md
// and runs 9 PASS/FAIL test scenarios.
//
// All masses, lengths, ROMs, and gains come from the spec.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3, Mat3, swing_twist_angle};
use nalgebra::UnitQuaternion;

// ─── Simulation timing ────────────────────────────────────────────────

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ─── Body parameters (from spec §3) ───────────────────────────────────

const SHIN_MASS:    f64 = 3.00;
const SHIN_HALF_H:  f64 = 0.21;
const SHIN_RADIUS:  f64 = 0.05;

const TALUS_MASS:   f64 = 0.05;
const TALUS_HALF_H: f64 = 0.025;
const TALUS_RADIUS: f64 = 0.020;

const FOOT_MASS:    f64 = 0.85;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;

const FOREFOOT_MASS:    f64 = 0.10;
const FOREFOOT_HALF_H:  f64 = 0.035;
const FOREFOOT_RADIUS:  f64 = 0.020;

const STICK_MASS:    f64 = 7.00;  // load above shin; sized so total above-ankle ≈ 10 kg
const STICK_HALF_H:  f64 = 0.40;
const STICK_RADIUS:  f64 = 0.08;

// ─── Joint parameters (from spec §4) ──────────────────────────────────

// Talocrural (shin ↔ talus)
const TALOCRURAL_JOINT_K: f64 = 60_000.0;
const TALOCRURAL_JOINT_C: f64 =    150.0;
const TALOCRURAL_PRIMARY_KP: f64 = 80.0;
const TALOCRURAL_PRIMARY_KD: f64 =  1.0;
const TALOCRURAL_PASSIVE_K:  f64 =  4.0;
const TALOCRURAL_PASSIVE_C:  f64 =  1.2;
const TALOCRURAL_ROM_MIN: f64 = -0.35;
const TALOCRURAL_ROM_MAX: f64 = 0.70;
const TALOCRURAL_LIMIT_K: f64 = 10_000.0;
const TALOCRURAL_LIMIT_EXP: f64 = 2.0;
const TALOCRURAL_OFF_AXIS_K:    f64 = 200.0;
const TALOCRURAL_OFF_AXIS_LIM:  f64 = 0.05;
const TALOCRURAL_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

// Subtalar (talus ↔ foot)
const SUBTALAR_JOINT_K: f64 = 60_000.0;
const SUBTALAR_JOINT_C: f64 =    150.0;
const SUBTALAR_PRIMARY_KP: f64 = 50.0;
const SUBTALAR_PRIMARY_KD: f64 =  2.0;
const SUBTALAR_PASSIVE_K:  f64 =  3.5;
const SUBTALAR_PASSIVE_C:  f64 =  1.0;
const SUBTALAR_ROM_MIN: f64 = -0.26;
const SUBTALAR_ROM_MAX: f64 = 0.52;
const SUBTALAR_LIMIT_K: f64 = 8_000.0;
const SUBTALAR_LIMIT_EXP: f64 = 2.0;
const SUBTALAR_TWIST_K: f64 = 8.0;
const SUBTALAR_TWIST_C: f64 = 0.5;
const SUBTALAR_TWIST_LIM: f64 = 0.10;
const SUBTALAR_TWIST_LIMIT_K: f64 = 10_000.0;
const SUBTALAR_OFF_AXIS_K:    f64 = 200.0;
const SUBTALAR_OFF_AXIS_LIM:  f64 = 0.05;
const SUBTALAR_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

// MTP (foot ↔ forefoot)
const MTP_JOINT_K: f64 = 60_000.0;
const MTP_JOINT_C: f64 =    150.0;
const MTP_PRIMARY_KP: f64 = 30.0;
const MTP_PRIMARY_KD: f64 =  0.5;
const MTP_PASSIVE_K:  f64 =  5.0;
const MTP_PASSIVE_C:  f64 =  1.5;
const MTP_ROM_MIN: f64 = -0.35;
const MTP_ROM_MAX: f64 = 0.87;
const MTP_LIMIT_K: f64 = 5_000.0;
const MTP_LIMIT_EXP: f64 = 2.0;
const MTP_OFF_AXIS_K:    f64 = 100.0;
const MTP_OFF_AXIS_LIM:  f64 = 0.05;
const MTP_OFF_AXIS_LIMIT_K: f64 = 5_000.0;

// Stick weld to shin (stiff rigid attachment via springs)
const WELD_POS_K: f64 = 1_000_000.0;
const WELD_POS_C: f64 =     2_000.0;
const WELD_ANG_K: f64 =    50_000.0;
const WELD_ANG_C: f64 =       500.0;

// ─── Helpers ──────────────────────────────────────────────────────────

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(
        i_trans, 0.0, 0.0,
        0.0, i_axial, 0.0,
        0.0, 0.0, i_trans,
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

/// Welds two bodies together via stiff positional + 3-axis angular springs.
/// Use to attach the load stick rigidly to the shin top.
fn weld_bodies(
    world: &mut XpbdWorld, body_a: usize, body_b: usize,
    anchor_a: Vec3, anchor_b: Vec3,
) {
    // Positional weld
    add_elastic_joint(world, body_a, body_b, anchor_a, anchor_b, WELD_POS_K, WELD_POS_C);
    // 3-axis angular weld: K and 0-baseline springs on x, y, z to lock orientation.
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_springs.push(
            AngularSpring::new(body_a, body_b, axis, WELD_ANG_K, 0.0));
        world.angular_dampers.push(
            AngularDamper::new(body_a, body_b, axis, WELD_ANG_C));
    }
}

/// Read the signed angle of body_b relative to body_a about an axis in
/// body_a's local frame. Same convention as AngularPDController.
fn joint_angle(
    world: &XpbdWorld, body_a: usize, body_b: usize, axis_local_a: Vec3,
) -> f64 {
    let q_b_in_a = world.bodies[body_a].orientation.inverse()
                 * world.bodies[body_b].orientation;
    swing_twist_angle(q_b_in_a, axis_local_a)
}

fn main() {
    println!("PHASE 1 ANATOMICAL ANKLE VALIDATION");
    println!("(scaffold — tests not yet implemented)");
}
