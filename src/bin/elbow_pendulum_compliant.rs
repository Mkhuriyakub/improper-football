// Step 6 of joint-by-joint physics validation.
//
// Same long-run energy diagnostic as elbow_pendulum_long.rs, but with non-zero
// compliance on the elbow positional constraint and the three angular limits.
// Compliance turns the rigid constraint into a stiff spring whose elastic PE
// is bookkept by XPBD's alpha-lambda formulation. The expected effect is to
// reduce the constraint solver's velocity-feedback energy injection, since
// each correction is now smaller (proportional to 1 / (w + alpha) rather than
// 1 / w) and the constraint better tolerates small position errors.

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
const SIM_DURATION: f64 = 30.0;
const PRINT_INTERVAL: f64 = 1.0;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KD_LIN: f64 =      0.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ANCHOR_KD_ANG: f64 =      0.0;

const ELBOW_COMPLIANCE: f64 = 1.0e-7;

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

    let mut elbow = PositionalConstraint::new(upper_idx, lower_idx, 0.0, ELBOW_COMPLIANCE);
    elbow.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow);

    let mut z_lim = AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::z(), -1.57, 2.20);
    let mut x_lim = AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05);
    let mut y_lim = AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05);
    z_lim.compliance = ELBOW_COMPLIANCE;
    x_lim.compliance = ELBOW_COMPLIANCE;
    y_lim.compliance = ELBOW_COMPLIANCE;
    world.angular_constraints.push(z_lim);
    world.angular_constraints.push(x_lim);
    world.angular_constraints.push(y_lim);

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let total_energy = |w: &XpbdWorld| {
        w.bodies.iter().map(|b| body_kinetic_energy(b) + body_potential_energy(b)).sum::<f64>()
    };

    let e_initial = total_energy(&world);

    println!("ELBOW PENDULUM, COMPLIANCE = {:e}  ({:.1}s, dt={:.5}s)",
             ELBOW_COMPLIANCE, SIM_DURATION, DT);
    println!("E_initial = {:.6} J", e_initial);
    println!();
    println!("{:<5} {:<11} {:<11} {:<11} {:<10} {:<10}",
             "t", "ΔE_mech", "W_anchor", "E_resid", "elbow_z", "max|θ|");
    println!("{}", "-".repeat(65));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut w_anchor: f64 = 0.0;
    let mut max_abs_elbow: f64 = 0.0;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
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

            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            if elbow_z.abs() > max_abs_elbow {
                max_abs_elbow = elbow_z.abs();
            }
        }
        t += DT;

        if t >= next_print {
            let e_total    = total_energy(&world);
            let de         = e_total - e_initial;
            let e_residual = de - w_anchor;
            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            println!("{:<5.1} {:<11.5} {:<11.5} {:<11.5} {:<10.5} {:<10.5}",
                     t, de, w_anchor, e_residual, elbow_z, max_abs_elbow);
            next_print += PRINT_INTERVAL;
        }
    }
}
