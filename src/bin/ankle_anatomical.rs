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

#[derive(Clone, Copy)]
struct AnkleHandles {
    shin_idx: usize,
    talus_idx: usize,
    foot_idx: usize,
    forefoot_idx: usize,
    stick_idx: usize,
    talocrural_pd_idx: usize,
    subtalar_pd_idx: usize,
    mtp_pd_idx: usize,
}

fn build_world() -> (XpbdWorld, AnkleHandles) {
    let mut world = XpbdWorld::new();
    let id = UnitQuaternion::identity();

    // Foot baseline: R_x(-π/2). Local +Y maps to world -Z (heel direction),
    // local -Y to world +Z (toe direction).
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Foot: center at world (0, 0.032, +0.09); heel at world (0, 0.032, 0).
    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    foot.position = Vec3::new(0.0, FOOT_RADIUS, 0.09);
    foot.orientation = foot_baseline;
    let foot_idx = world.add_body(foot);

    // Forefoot: attached to foot's toe end (world (0,0.032,+0.18)) at its
    // own heel end (local +Y). Carries foot baseline.
    let mut forefoot = RigidBody::new(FOREFOOT_MASS,
        capsule_inertia(FOREFOOT_MASS, FOREFOOT_RADIUS, FOREFOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[foot_idx],
        Vec3::new(0.0, -FOOT_HALF_H, 0.0),         // foot toe end (foot local)
        Vec3::new(0.0, FOREFOOT_HALF_H, 0.0),      // forefoot heel end (forefoot local)
        id);
    forefoot.position = pos;
    forefoot.orientation = orient;
    let forefoot_idx = world.add_body(forefoot);

    // Talus: above foot's heel (subtalar joint at world (0,0.032,0)).
    // Talus is at identity orientation.
    let mut talus = RigidBody::new(TALUS_MASS,
        capsule_inertia(TALUS_MASS, TALUS_RADIUS, TALUS_HALF_H));
    talus.position = Vec3::new(0.0, FOOT_RADIUS + TALUS_HALF_H, 0.0);  // (0, 0.057, 0)
    talus.orientation = id;
    let talus_idx = world.add_body(talus);

    // Shin: above talus (talocrural joint at world (0, 0.082, 0)).
    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let talocrural_world_y = FOOT_RADIUS + 2.0 * TALUS_HALF_H;  // 0.082
    shin.position = Vec3::new(0.0, talocrural_world_y + SHIN_HALF_H, 0.0);  // (0, 0.292, 0)
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // ── Talocrural joint (shin ↔ talus) ─────────────────────────────
    // Positional joint at shin bottom / talus top.
    add_elastic_joint(&mut world, shin_idx, talus_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        Vec3::new(0.0, TALUS_HALF_H, 0.0),
        TALOCRURAL_JOINT_K, TALOCRURAL_JOINT_C);

    // Primary axis: world +x at neutral, in shin local frame.
    let tc_primary = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_ROM_MIN, TALOCRURAL_ROM_MAX,
        TALOCRURAL_LIMIT_K, TALOCRURAL_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_K, 0.0));
    let talocrural_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_PRIMARY_KP, TALOCRURAL_PRIMARY_KD));

    // Off-axes (Y, Z) locked: strong passive + soft-limit at narrow ROM.
    for off_axis in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            shin_idx, talus_idx, off_axis,
            -TALOCRURAL_OFF_AXIS_LIM, TALOCRURAL_OFF_AXIS_LIM,
            TALOCRURAL_OFF_AXIS_LIMIT_K, TALOCRURAL_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_OFF_AXIS_K, 0.0));
    }

    // ── Subtalar joint (talus ↔ foot) ───────────────────────────────
    // The foot baseline R_x(-π/2) is applied at this joint: rel_rot in
    // place_child was foot_baseline. The positional joint uses the
    // anatomical anchor points.
    add_elastic_joint(&mut world, talus_idx, foot_idx,
        Vec3::new(0.0, -TALUS_HALF_H, 0.0),
        Vec3::new(0.0, FOOT_HALF_H, 0.0),  // foot heel anchor in foot local
        SUBTALAR_JOINT_K, SUBTALAR_JOINT_C);

    // Primary axis: world +z at neutral (inversion/eversion).
    // In talus local frame (talus is at identity), axis = (0, 0, 1).
    let st_primary = Vec3::new(0.0, 0.0, 1.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_ROM_MIN, SUBTALAR_ROM_MAX,
        SUBTALAR_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_K, 0.0)
        .with_baseline(foot_baseline));
    let subtalar_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_PRIMARY_KP, SUBTALAR_PRIMARY_KD)
        .with_baseline(foot_baseline));

    // Twist axis (Y in talus local): very limited passive ROM.
    let st_twist = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_twist,
        -SUBTALAR_TWIST_LIM, SUBTALAR_TWIST_LIM,
        SUBTALAR_TWIST_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_K, 0.0)
        .with_baseline(foot_baseline));

    // Other off-axis (X in talus local): locked, prevents PF/DF leaking
    // into the subtalar joint (that's the talocrural's job).
    let st_locked = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_locked,
        -SUBTALAR_OFF_AXIS_LIM, SUBTALAR_OFF_AXIS_LIM,
        SUBTALAR_OFF_AXIS_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_OFF_AXIS_K, 0.0)
        .with_baseline(foot_baseline));

    // ── MTP joint (foot ↔ forefoot) ─────────────────────────────────
    add_elastic_joint(&mut world, foot_idx, forefoot_idx,
        Vec3::new(0.0, -FOOT_HALF_H, 0.0),
        Vec3::new(0.0, FOREFOOT_HALF_H, 0.0),
        MTP_JOINT_K, MTP_JOINT_C);

    // Primary axis: world +x at neutral. In foot local (after baseline),
    // axis (1, 0, 0) still maps to world +x. The MTP PD's baseline is
    // identity (no offset between foot and forefoot at rest).
    let mtp_primary = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_ROM_MIN, MTP_ROM_MAX,
        MTP_LIMIT_K, MTP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_K, 0.0));
    let mtp_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_PRIMARY_KP, MTP_PRIMARY_KD));

    // Off-axes (Y, Z) locked.
    for off_axis in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            foot_idx, forefoot_idx, off_axis,
            -MTP_OFF_AXIS_LIM, MTP_OFF_AXIS_LIM,
            MTP_OFF_AXIS_LIMIT_K, MTP_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            foot_idx, forefoot_idx, off_axis, MTP_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            foot_idx, forefoot_idx, off_axis, MTP_OFF_AXIS_K, 0.0));
    }

    // Stick (load): above shin, welded to shin top.
    let mut stick = RigidBody::new(STICK_MASS,
        capsule_inertia(STICK_MASS, STICK_RADIUS, STICK_HALF_H));
    let shin_top_y = talocrural_world_y + 2.0 * SHIN_HALF_H;  // 0.502
    stick.position = Vec3::new(0.0, shin_top_y + STICK_HALF_H, 0.0);  // (0, 0.902, 0)
    stick.orientation = id;
    let stick_idx = world.add_body(stick);

    // ── Stick welded to shin top ────────────────────────────────────
    weld_bodies(&mut world, shin_idx, stick_idx,
        Vec3::new(0.0, SHIN_HALF_H, 0.0),
        Vec3::new(0.0, -STICK_HALF_H, 0.0));

    // ── Ground contact on foot only ─────────────────────────────────
    // Foot uses single-capsule GroundContact (multi-region foot is
    // Phase 2). Other bodies' contact (forefoot when toes are up,
    // etc.) is intentionally not modeled in Phase 1.
    world.ground_contacts.push(
        GroundContact::new(foot_idx, FOOT_HALF_H, FOOT_RADIUS));

    (world, AnkleHandles {
        shin_idx, talus_idx, foot_idx, forefoot_idx, stick_idx,
        talocrural_pd_idx, subtalar_pd_idx, mtp_pd_idx,
    })
}

struct TestResult {
    name: &'static str,
    passed: bool,
    message: String,
}

impl TestResult {
    fn pass(name: &'static str) -> Self {
        Self { name, passed: true, message: String::from("PASS") }
    }
    fn fail(name: &'static str, message: String) -> Self {
        Self { name, passed: false, message }
    }
}

/// Run the simulation for `duration` seconds, optionally calling
/// `per_frame` once per frame (after each `SOLVER_ITERATIONS` substeps).
/// Returns the world afterward.
fn simulate(
    world: &mut XpbdWorld,
    duration: f64,
    mut per_frame: impl FnMut(&mut XpbdWorld, f64),
) {
    let mut t = 0.0;
    while t < duration {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;
        per_frame(world, t);
    }
}

/// Check that all body positions are finite (no NaN/inf).
fn all_positions_finite(world: &XpbdWorld) -> bool {
    world.bodies.iter().all(|b|
        b.position.x.is_finite() && b.position.y.is_finite() && b.position.z.is_finite())
}

fn main() {
    let (world, h) = build_world();
    println!("Body positions at neutral:");
    println!("  foot     y={:.4}, z={:+.4}", world.bodies[h.foot_idx].position.y,    world.bodies[h.foot_idx].position.z);
    println!("  forefoot y={:.4}, z={:+.4}", world.bodies[h.forefoot_idx].position.y, world.bodies[h.forefoot_idx].position.z);
    println!("  talus    y={:.4}, z={:+.4}", world.bodies[h.talus_idx].position.y,   world.bodies[h.talus_idx].position.z);
    println!("  shin     y={:.4}, z={:+.4}", world.bodies[h.shin_idx].position.y,    world.bodies[h.shin_idx].position.z);
    println!("  stick    y={:.4}, z={:+.4}", world.bodies[h.stick_idx].position.y,   world.bodies[h.stick_idx].position.z);
}
