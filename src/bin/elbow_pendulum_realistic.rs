// Step 8 of joint-by-joint physics validation.
//
// First test of a biomechanically-realistic elbow joint. Same pendulum geometry
// as before, but the elbow now includes a passive viscoelastic damper as part
// of the joint primitive — modeling the resistance of joint capsule, ligaments,
// tendons, and unactivated muscle. Damping is active across the entire range
// of motion, not only at limits.
//
// Damping coefficient c = 0.05 N·m·s/rad is in the literature range for the
// human elbow (typical estimates: 0.02–0.10 N·m·s/rad in the mid-range).
//
// Upper-arm anchor has spring stiffness only — no damping there — so the
// elbow's joint damper is the sole dissipation source. With small-amplitude
// pendulum dynamics, ζ ≈ c / (2·√(I_pivot · m·g·L_cm)) ≈ 0.10 (lightly damped),
// and the amplitude envelope should decay as exp(−ζ·ω₀·t) with time constant
// 1/(ζ·ω₀) ≈ 1.3 s. Predicted amplitude after 5 s: ≈ 0.30·exp(−0.762·5) ≈ 0.007 rad.
//
// Validation: the lower arm should oscillate visibly for the first few cycles,
// then decay smoothly to rest. No perpetual motion. No accumulating drift.
// The joint should look like a real elbow let go from 17° flex.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::math::{Vec3, Mat3};
use physics::xpbd::GRAVITY;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 10.0;
const PRINT_INTERVAL: f64 = 0.25;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ELBOW_DAMPING: f64 = 0.05;  // N·m·s/rad — biomechanical estimate

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

    // Biomechanical addition: passive damper on the elbow's hinge axis.
    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, Vec3::z(), ELBOW_DAMPING));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let total_energy = |w: &XpbdWorld| {
        w.bodies.iter().map(|b| body_kinetic_energy(b) + body_potential_energy(b)).sum::<f64>()
    };

    let e_initial = total_energy(&world);

    // Predicted decay parameters for sanity reference.
    let r = lower_def.radius;
    let h = 2.0 * lower_def.half_height;
    let i_com_z = (1.0 / 12.0) * lower_def.mass * (3.0 * r * r + h * h);
    let l_cm    = lower_def.half_height;
    let i_pivot = i_com_z + lower_def.mass * l_cm * l_cm;
    let omega_0 = (lower_def.mass * 9.81 * l_cm / i_pivot).sqrt();
    let critical_c = 2.0 * (i_pivot * lower_def.mass * 9.81 * l_cm).sqrt();
    let zeta = ELBOW_DAMPING / critical_c;
    let decay_time_constant = 1.0 / (zeta * omega_0);

    println!("ELBOW PENDULUM, BIOMECHANICAL ELBOW  ({:.1}s, dt={:.5}s)",
             SIM_DURATION, DT);
    println!("damping c = {} N·m·s/rad   ζ = {:.3}   τ_decay = {:.2} s",
             ELBOW_DAMPING, zeta, decay_time_constant);
    println!("predicted amplitude: t=0 → 0.300, t=5s → {:.3}, t=10s → {:.4}",
             0.30 * (-decay_time_constant.recip() * 5.0).exp(),
             0.30 * (-decay_time_constant.recip() * 10.0).exp());
    println!();
    println!("{:<6} {:<10} {:<10} {:<11} {:<11} {:<11}",
             "t", "elbow_z", "max|θ|", "ΔE_mech", "W_anchor", "W_damper");
    println!("{}", "-".repeat(70));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;
    let mut w_anchor: f64 = 0.0;
    let mut w_damper: f64 = 0.0;
    let mut max_abs_elbow_window: f64 = 0.0;
    let mut max_abs_elbow_total: f64 = 0.0;
    let mut last_window_reset_time = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            // Upper-arm anchor: spring only, no damping (so elbow damper is the sole sink).
            let body = &mut world.bodies[upper_idx];
            let pos_err = body.position - upper_rest_pos;
            let v       = body.linear_velocity;
            let omega   = body.angular_velocity;
            let force   = -ANCHOR_KP_LIN * pos_err;
            let orient_err = body.orientation.scaled_axis();
            let torque  = -ANCHOR_KP_ANG * orient_err;
            body.apply_force(force);
            body.apply_torque(torque);

            w_anchor += (force.dot(&v) + torque.dot(&omega)) * SUB_DT;

            // Damper power dissipated this substep, accumulated as work done BY the damper.
            // power() returns ≤ 0, so w_damper accumulates monotonically negative.
            w_damper += world.angular_dampers[0].power(&world.bodies) * SUB_DT;

            world.substep(SUB_DT);

            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            if elbow_z.abs() > max_abs_elbow_window {
                max_abs_elbow_window = elbow_z.abs();
            }
            if elbow_z.abs() > max_abs_elbow_total {
                max_abs_elbow_total = elbow_z.abs();
            }
        }
        t += DT;

        if t >= next_print {
            let e_total = total_energy(&world);
            let de      = e_total - e_initial;
            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;
            // Show window peak (since previous print) so amplitude decay is visible cycle-by-cycle.
            println!("{:<6.2} {:<10.5} {:<10.5} {:<11.5} {:<11.5} {:<11.5}",
                     t, elbow_z, max_abs_elbow_window, de, w_anchor, w_damper);
            next_print += PRINT_INTERVAL;
            // Reset window tracker so next print's max|θ| is for this print's window.
            if t - last_window_reset_time >= 0.85 {
                max_abs_elbow_window = elbow_z.abs();
                last_window_reset_time = t;
            }
        }
    }

    println!();
    println!("Final cumulative max|θ| over whole sim: {:.5}", max_abs_elbow_total);
}
