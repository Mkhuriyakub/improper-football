// Step 11 of joint-by-joint physics validation.
//
// Demonstrates ligament tearing as a representable failure mode of the
// biomechanical elbow. Runs the same scenario twice — identical initial
// conditions, identical primitives — but one with an intact ligament
// (tear_threshold = infinity) and one with a ligament that ruptures when
// stretched past 0.20 rad of excess.
//
// Initial flex = 2.50 rad → 0.30 rad past the +2.20 anatomical max. With the
// intact ligament, the soft limit applies up to 45 N·m of restoring torque,
// rapidly returning the joint to range. With the tearing ligament, the
// rupture is triggered on the very first substep (since |excess| = 0.30
// already exceeds the 0.20 threshold) — effective stiffness drops to zero,
// no restoring torque from the limit, and the joint drifts back to range
// only under the much weaker gravity + passive-stiffness + damping budget.
//
// Validation observables: the torn run (a) exhibits no peak limit torque,
// (b) takes substantially longer to return to range, and (c) settles at the
// same gravity+stiffness equilibrium because that part of the joint model
// is unaffected by the tear. The distinct dynamics confirm that the failure
// mode is real in the model and not silently prevented by hard-stop logic.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::math::Vec3;
use player::segment::{BodyType, SegmentId, segment_def};
use nalgebra::UnitQuaternion;

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 6.0;

const INITIAL_FLEX: f64 = 2.50;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;
const ELBOW_STIFFNESS: f64 = 1.5;
const ELBOW_NEUTRAL:   f64 = 0.20;

const ELBOW_LIMIT_MIN: f64 = -0.40;
const ELBOW_LIMIT_MAX: f64 =  2.20;
const ELBOW_LIMIT_STIFFNESS: f64 = 500.0;
const ELBOW_LIMIT_EXPONENT:  f64 =   2.0;

const TEAR_THRESHOLD:    f64 = 0.20;
const POST_TEAR_FACTOR:  f64 = 0.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

struct RunResult {
    label:                String,
    tear_triggered:       bool,
    tear_event_time:      Option<f64>,
    peak_excess:          f64,
    peak_limit_torque:    f64,
    first_in_range_time:  Option<f64>,
    final_angle:          f64,
}

fn run_scenario(label: &str, tear_threshold: f64, post_tear_factor: f64) -> RunResult {
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

    let mut elbow_pos = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow_pos.local_anchor_a = Vec3::new(0.0, -0.15, 0.0);
    elbow_pos.local_anchor_b = Vec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow_pos);

    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, Vec3::y(), -0.05, 0.05));

    let limit = AngularSoftLimit::new(
        upper_idx, lower_idx, Vec3::z(),
        ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX,
        ELBOW_LIMIT_STIFFNESS, ELBOW_LIMIT_EXPONENT,
    ).with_tearing(tear_threshold, post_tear_factor);
    world.angular_soft_limits.push(limit);

    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, Vec3::z(), ELBOW_DAMPING));
    world.angular_springs.push(
        AngularSpring::new(upper_idx, lower_idx, Vec3::z(), ELBOW_STIFFNESS, ELBOW_NEUTRAL));

    let upper_rest_pos = Vec3::new(0.0, UPPER_REST_Y, 0.0);

    let mut t = 0.0;
    let mut peak_excess: f64 = 0.0;
    let mut peak_limit_torque: f64 = 0.0;
    let mut first_in_range_time: Option<f64> = None;
    let mut tear_event_time: Option<f64> = None;
    let mut prev_torn = false;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            let body = &mut world.bodies[upper_idx];
            let pos_err = body.position - upper_rest_pos;
            let force   = -ANCHOR_KP_LIN * pos_err;
            let orient_err = body.orientation.scaled_axis();
            let torque  = -ANCHOR_KP_ANG * orient_err;
            body.apply_force(force);
            body.apply_torque(torque);

            world.substep(SUB_DT);

            let larm = &world.bodies[lower_idx];
            let uarm = &world.bodies[upper_idx];
            let elbow_z = (larm.orientation * uarm.orientation.inverse()).scaled_axis().z;

            let excess = (elbow_z - ELBOW_LIMIT_MAX).max(0.0)
                .max((ELBOW_LIMIT_MIN - elbow_z).max(0.0));
            if excess > peak_excess { peak_excess = excess; }

            let lt = world.angular_soft_limits[0].scalar_torque(elbow_z).abs();
            if lt > peak_limit_torque { peak_limit_torque = lt; }

            if !prev_torn && world.angular_soft_limits[0].torn {
                tear_event_time = Some(t);
                prev_torn = true;
            }

            if first_in_range_time.is_none()
                && elbow_z <= ELBOW_LIMIT_MAX
                && elbow_z >= ELBOW_LIMIT_MIN
            {
                first_in_range_time = Some(t);
            }
        }
        t += DT;
    }

    let final_angle = {
        let larm = &world.bodies[lower_idx];
        let uarm = &world.bodies[upper_idx];
        (larm.orientation * uarm.orientation.inverse()).scaled_axis().z
    };

    RunResult {
        label: label.to_string(),
        tear_triggered: world.angular_soft_limits[0].torn,
        tear_event_time,
        peak_excess,
        peak_limit_torque,
        first_in_range_time,
        final_angle,
    }
}

fn main() {
    println!("ELBOW LIGAMENT TEAR COMPARISON");
    println!("Same scenario, two ligament states. {:.1}s sim, dt={:.5}s", SIM_DURATION, DT);
    println!();
    println!("Common parameters:");
    println!("  initial flex      = {} rad ({:.2}°)",
             INITIAL_FLEX, INITIAL_FLEX.to_degrees());
    println!("  natural Z range   = [{}, {}] rad", ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX);
    println!("  limit stiffness   = {} N·m/rad²   (quadratic)", ELBOW_LIMIT_STIFFNESS);
    println!("  passive stiffness = {} N·m/rad   neutral = {} rad",
             ELBOW_STIFFNESS, ELBOW_NEUTRAL);
    println!("  damping           = {} N·m·s/rad", ELBOW_DAMPING);
    println!();

    let intact = run_scenario("intact ligament", f64::INFINITY, 0.0);
    let torn   = run_scenario("ruptured ligament",
                              TEAR_THRESHOLD, POST_TEAR_FACTOR);

    println!("{:<30} {:<22} {:<22}",
             "Observable", intact.label, torn.label);
    println!("{}", "-".repeat(74));
    println!("{:<30} {:<22} {:<22}",
             "tear threshold (rad)",
             "—",
             format!("{:.3}", TEAR_THRESHOLD));
    println!("{:<30} {:<22} {:<22}",
             "tear triggered",
             "no",
             if torn.tear_triggered { "yes" } else { "no" });
    println!("{:<30} {:<22} {:<22}",
             "tear event time",
             "—",
             torn.tear_event_time.map_or("—".to_string(),
                 |t| format!("{:.4} s", t)));
    println!("{:<30} {:<22} {:<22}",
             "peak excess past limit (rad)",
             format!("{:.5}", intact.peak_excess),
             format!("{:.5}", torn.peak_excess));
    println!("{:<30} {:<22} {:<22}",
             "peak limit torque (N·m)",
             format!("{:.3}", intact.peak_limit_torque),
             format!("{:.3}", torn.peak_limit_torque));
    println!("{:<30} {:<22} {:<22}",
             "time to re-enter range",
             intact.first_in_range_time.map_or("never".to_string(),
                 |t| format!("{:.4} s", t)),
             torn.first_in_range_time.map_or("never".to_string(),
                 |t| format!("{:.4} s", t)));
    println!("{:<30} {:<22} {:<22}",
             "final angle (rad)",
             format!("{:.5}", intact.final_angle),
             format!("{:.5}", torn.final_angle));
    println!("{:<30} {:<22} {:<22}",
             "final angle (deg)",
             format!("{:.2}°", intact.final_angle.to_degrees()),
             format!("{:.2}°", torn.final_angle.to_degrees()));
}
