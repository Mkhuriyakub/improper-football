// Step 3 of joint-by-joint physics validation.
//
// Same pendulum rig as elbow_pendulum.rs, instrumented to track total mechanical
// energy and the work done by the upper-arm anchor. Lets us split the observed
// 1%-per-cycle decay between (a) anchor dissipation (a real sink, since the
// anchor applies a damping torque/force) and (b) numerical dissipation in the
// constraint solver / integrator (the unaccounted residual).
//
// Bookkeeping per substep:
//   E_total(t)     = sum over bodies of  ½·m·|v|² + ½·ω·I_world·ω + m·g·y
//   W_anchor(t)    = cumulative ∫ (F_anchor · v + τ_anchor · ω) dt
//   E_residual(t)  = E_total(t) - E_total(0) - W_anchor(t)
//
// E_residual is the energy unaccounted-for by either the conservative dynamics
// or the anchor's work. A non-zero E_residual is numerical dissipation
// (negative) or accretion (positive).

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::math::{Vec3, Mat3};
use physics::xpbd::GRAVITY;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;
const PRINT_INTERVAL: f64 = 0.25;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ANCHOR_KD_ANG: f64 =    100.0;

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
    // PE = -m · g · pos  (g is the gravity vector, here (0, -9.81, 0))
    -body.mass * GRAVITY.dot(&body.position)
}

fn main() {
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

    println!("ELBOW PENDULUM ENERGY DIAGNOSTIC  ({:.1}s, dt={:.5}s)", SIM_DURATION, DT);
    println!("E_initial = {:.6} J", e_initial);
    println!();
    println!("{:<7} {:<12} {:<12} {:<12} {:<12} {:<14}",
             "t", "E_total", "ΔE", "W_anchor", "E_residual", "loss_rate(J/s)");
    println!("{}", "-".repeat(80));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut w_anchor: f64 = 0.0;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            // Compute and apply anchor force/torque, accumulating its work using the
            // pre-substep velocity (a first-order approximation to ∫ F·v dt).
            let body = &mut world.bodies[upper_idx];
            let pos_err = body.position - upper_rest_pos;
            let v       = body.linear_velocity;
            let omega   = body.angular_velocity;
            let force   = -ANCHOR_KP_LIN * pos_err - ANCHOR_KD_LIN * v;
            let orient_err = body.orientation.scaled_axis();
            let torque  = -ANCHOR_KP_ANG * orient_err - ANCHOR_KD_ANG * omega;
            body.apply_force(force);
            body.apply_torque(torque);

            w_anchor += (force.dot(&v) + torque.dot(&omega)) * SUB_DT;

            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let e_total    = total_energy(&world);
            let de         = e_total - e_initial;
            let e_residual = de - w_anchor;
            let loss_rate  = if t > 0.0 { -e_residual / t } else { 0.0 };
            println!("{:<7.3} {:<12.6} {:<12.6} {:<12.6} {:<12.6} {:<14.6}",
                     t, e_total, de, w_anchor, e_residual, loss_rate);
            next_print += PRINT_INTERVAL;
        }
    }
}
