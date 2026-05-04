// Step 7 of joint-by-joint physics validation.
//
// Sweep over substep counts to characterize how the constraint solver's
// energy injection scales with sub_dt. Same elbow-pendulum rig, no anchor
// damping (so the leak is unobstructed), 5-second runs at each substep
// count, reporting leak rate (J/s) and the observed scaling exponent.
//
// Hypothesis: leak per second ≈ K · sub_dt^n for some K and n.
// If n ≈ 2, halving sub_dt (doubling substep count) reduces leak by 4x.
// Linear scaling (n = 1) would be disappointing but still useful.
// Sub-linear or saturating would mean substepping isn't a viable path
// and we need a different constraint scheme.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::math::{Vec3, Mat3};
use physics::xpbd::GRAVITY;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SIM_DURATION: f64 = 5.0;

const INITIAL_FLEX: f64 = 0.30;
const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

fn world_inertia(body: &RigidBody) -> Mat3 {
    let r = body.orientation.to_rotation_matrix();
    r.matrix() * body.inertia_tensor * r.matrix().transpose()
}

fn body_kinetic_energy(body: &RigidBody) -> f64 {
    let ke_lin = 0.5 * body.mass * body.linear_velocity.norm_squared();
    let i_world = world_inertia(body);
    let ke_rot = 0.5 * body.angular_velocity.dot(&(i_world * body.angular_velocity));
    ke_lin + ke_rot
}

fn body_potential_energy(body: &RigidBody) -> f64 {
    -body.mass * GRAVITY.dot(&body.position)
}

fn run_one(substeps: usize) -> (f64, f64, f64, f64) {
    let sub_dt = DT / substeps as f64;

    let mut world = XpbdWorld::new();

    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(upper_def.mass, upper_def.inertia_tensor());
    upper.position = Vec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.orientation = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_FLEX));
    let rotated_anchor = lower.orientation * Vec3::new(0.0, 0.13, 0.0);
    lower.position = Vec3::new(0.0, ELBOW_WORLD_Y, 0.0) - rotated_anchor;
    let lower_idx = world.add_body(lower);

    let mut elbow = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow);

    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::z(), -1.57, 2.20));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let total_energy = |w: &XpbdWorld| {
        w.bodies.iter().map(|b| body_kinetic_energy(b) + body_potential_energy(b)).sum::<f64>()
    };

    let e_initial = total_energy(&world);

    let mut t = 0.0;
    let mut w_anchor: f64 = 0.0;
    let mut max_abs_elbow: f64 = 0.0;

    while t < SIM_DURATION {
        for _ in 0..substeps {
            let body = &mut world.bodies[upper_idx];
            let pos_err = body.position - upper_rest_pos;
            let v       = body.linear_velocity;
            let omega   = body.angular_velocity;
            let force   = -ANCHOR_KP_LIN * pos_err;
            let orient_err = body.orientation.scaled_axis();
            let torque  = -ANCHOR_KP_ANG * orient_err;
            body.apply_force(force);
            body.apply_torque(torque);

            w_anchor += (force.dot(&v) + torque.dot(&omega)) * sub_dt;

            world.substep(sub_dt);

            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            if elbow_z.abs() > max_abs_elbow {
                max_abs_elbow = elbow_z.abs();
            }
        }
        t += DT;
    }

    let e_final = total_energy(&world);
    let de = e_final - e_initial;
    let e_residual = de - w_anchor;
    let leak_rate = e_residual / SIM_DURATION;
    (sub_dt, leak_rate, max_abs_elbow, w_anchor)
}

fn main() {
    let counts = [30usize, 100, 300, 1000, 3000];

    println!("ELBOW PENDULUM SUBSTEP SWEEP  (sim {:.1}s, dt={:.5}s, no anchor damping)",
             SIM_DURATION, DT);
    println!("Leak per second is the constraint solver's energy injection rate.");
    println!();
    println!("{:<10} {:<14} {:<18} {:<12} {:<14}",
             "substeps", "sub_dt (s)", "leak_rate (J/s)", "max|θ|", "ratio_vs_30");
    println!("{}", "-".repeat(72));

    let baseline = run_one(counts[0]);

    for &n in &counts {
        let (sub_dt, leak_rate, max_amp, _w) = run_one(n);
        let ratio = leak_rate / baseline.1;
        println!("{:<10} {:<14.6e} {:<18.6e} {:<12.5} {:<14.4e}",
                 n, sub_dt, leak_rate, max_amp, ratio);
    }

    println!();
    println!("Scaling exponent: leak ∝ sub_dt^n. Check log-log slope between rows.");
    println!("If n ≈ 2, doubling substep count → leak drops 4x.");
    println!("If n ≈ 1, doubling substep count → leak drops 2x.");
}
