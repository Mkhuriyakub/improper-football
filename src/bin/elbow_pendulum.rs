// Step 2 of joint-by-joint physics validation.
//
// Releases the lower arm from a 0.30-rad initial flex with gravity on and no
// driving. The lower arm should oscillate around the elbow's Z hinge axis like
// a textbook physical pendulum: amplitude steady (no explicit damping), period
// matching the analytical 2π·√(I_pivot / m·g·L_cm), and the X/Y elbow angles
// staying locked near zero throughout.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 5.0;
const PRINT_INTERVAL: f64 = 0.05;

const INITIAL_FLEX: f64 = 0.30;  // rad — within elbow range, ~17°

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;
const ANCHOR_KD_ANG: f64 =    100.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;  // 1.18

fn main() {
    let mut world = XpbdWorld::new();

    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(upper_def.mass, upper_def.inertia_tensor());
    upper.position = Vec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.orientation = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, INITIAL_FLEX));
    // Place lower arm so its proximal anchor coincides with the elbow at world (0, 1.18, 0)
    // when oriented at INITIAL_FLEX.
    let rotated_anchor = lower.orientation * Vec3::new(0.0, 0.13, 0.0);
    lower.position = Vec3::new(0.0, ELBOW_WORLD_Y, 0.0) - rotated_anchor;
    let lower_idx = world.add_body(lower);

    let mut elbow = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow);

    // Z range widened past the anatomical limit so the pendulum's swing-through
    // (down through 0 to negative flex and back) doesn't bounce off a constraint.
    // Hinge enforced via tight X and Y locks.
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::z(), -1.57, 2.20));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    // Analytical period for a physical pendulum about an axis transverse to the capsule:
    //   I_pivot = I_com_z + m·L_cm²,  I_com_z = (1/12)·m·(3r² + h²)
    //   T_small = 2π·√(I_pivot / (m·g·L_cm))
    let r = lower_def.radius;
    let h = 2.0 * lower_def.half_height;
    let i_com_z  = (1.0 / 12.0) * lower_def.mass * (3.0 * r * r + h * h);
    let l_cm     = lower_def.half_height;
    let i_pivot  = i_com_z + lower_def.mass * l_cm * l_cm;
    let period   = 2.0 * std::f64::consts::PI
        * (i_pivot / (lower_def.mass * 9.81 * l_cm)).sqrt();

    println!("ELBOW PENDULUM TEST  ({:.1}s, dt={:.5}s, {} substeps/frame)",
             SIM_DURATION, DT, SOLVER_ITERATIONS);
    println!("initial flex = {:.3} rad   expected period = {:.4} s ({:.3} Hz)",
             INITIAL_FLEX, period, 1.0 / period);
    println!();
    println!("{:<7} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "t", "elbow_z", "elbow_x", "elbow_y", "omega_z", "uarm_dev");
    println!("{}", "-".repeat(64));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            {
                let body = &mut world.bodies[upper_idx];
                let pos_err = body.position - upper_rest_pos;
                let force   = -ANCHOR_KP_LIN * pos_err - ANCHOR_KD_LIN * body.linear_velocity;
                body.apply_force(force);

                let orient_err = body.orientation.scaled_axis();
                let torque     = -ANCHOR_KP_ANG * orient_err - ANCHOR_KD_ANG * body.angular_velocity;
                body.apply_torque(torque);
            }
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let uarm = &world.bodies[upper_idx];
            let larm = &world.bodies[lower_idx];
            let q_rel = larm.orientation * uarm.orientation.inverse();
            let axis = q_rel.scaled_axis();
            let omega_z = larm.angular_velocity.z - uarm.angular_velocity.z;
            println!("{:<7.3} {:<10.5} {:<10.5} {:<10.5} {:<10.4} {:<10.5}",
                     t, axis.z, axis.x, axis.y, omega_z,
                     uarm.orientation.scaled_axis().norm());
            next_print += PRINT_INTERVAL;
        }
    }
}
