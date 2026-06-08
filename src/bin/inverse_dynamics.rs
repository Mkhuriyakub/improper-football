// Validates the `InverseDynamics` primitive — gravity-feedforward
// observer that writes per-joint gravity-cancelling torque to each
// configured PD's feedforward field.
//
// Four scenarios:
//   1. Single body above joint, neutral pose — feedforward = 0
//      (gravity moment about joint axis is zero when CG is directly
//      above the joint).
//   2. Same body tilted forward — feedforward = -m·g·h·sin(θ)
//      (matches analytical inverse dynamics).
//   3. Multiple supported bodies — feedforward = sum of individual moments.
//   4. Bilateral sharing (fraction = 0.5) — feedforward halves correctly.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_pd_controller::AngularPDController;
use physics::inverse_dynamics::{IDJoint, InverseDynamics, SupportedBody};
use physics::math::{Mat3, Vec3};
use nalgebra::UnitQuaternion;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

// ─── Scenario 1: vertical chain at neutral ──────────────────────────────

fn scenario_neutral_zero_feedforward() {
    println!("=== Scenario 1: body directly above joint — feedforward = 0 ===");
    let mut world = XpbdWorld::new();

    let mut parent = RigidBody::new(1.0, Mat3::identity());
    parent.position = Vec3::new(0.0, 0.0, 0.0);
    let parent_idx = world.add_body(parent);

    let mut child = RigidBody::new(2.0, Mat3::identity());
    child.position = Vec3::new(0.0, 1.0, 0.0);  // directly above parent
    let child_idx = world.add_body(child);

    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        parent_idx, child_idx, Vec3::new(1.0, 0.0, 0.0), 100.0, 10.0));

    let mut id = InverseDynamics::new();
    id.joints.push(IDJoint {
        pd_idx,
        parent_body: parent_idx,
        anchor_local: Vec3::new(0.0, 0.5, 0.0),  // joint 0.5 m up on parent (world (0,0.5,0))
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![SupportedBody { body: child_idx, fraction: 1.0 }],
    });

    id.update(&mut world);
    let ff = world.angular_pd_controllers[pd_idx].feedforward_torque;
    let pass = approx_eq(ff, 0.0, 1e-9);
    println!("  feedforward = {:+.6} (expected 0) — {}",
             ff, if pass { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 2: body tilted forward ────────────────────────────────────

fn scenario_tilted_forward() {
    println!("=== Scenario 2: body offset forward of joint — feedforward = m·g·Δz ===");
    let mut world = XpbdWorld::new();

    let mut parent = RigidBody::new(1.0, Mat3::identity());
    let parent_idx = world.add_body(parent);

    // Child positioned 0.5 m above and 0.1 m forward of the joint
    // (which is at parent + (0, 0.5, 0) = (0, 0.5, 0)).
    // So child's CG offset from joint: (0, 0.5, 0.1).
    let mut child = RigidBody::new(2.0, Mat3::identity());
    child.position = Vec3::new(0.0, 1.0, 0.1);
    let child_idx = world.add_body(child);
    let _ = parent;

    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        parent_idx, child_idx, Vec3::new(1.0, 0.0, 0.0), 100.0, 10.0));

    let mut id = InverseDynamics::new();
    id.joints.push(IDJoint {
        pd_idx,
        parent_body: parent_idx,
        anchor_local: Vec3::new(0.0, 0.5, 0.0),
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![SupportedBody { body: child_idx, fraction: 1.0 }],
    });

    id.update(&mut world);
    let ff = world.angular_pd_controllers[pd_idx].feedforward_torque;

    // Analytic: gravity force on child = (0, -m·g, 0) = (0, -19.62, 0).
    // r = child.pos - joint_world = (0, 1.0, 0.1) - (0, 0.5, 0) = (0, 0.5, 0.1).
    // r × F = (0.5·0 - 0.1·(-19.62), 0.1·0 - 0·0, 0·(-19.62) - 0.5·0)
    //       = (1.962, 0, 0).
    // axis = world +x. moment_about_axis = +1.962.
    // feedforward = -moment = -1.962.
    let expected_ff = -1.962;
    let pass = approx_eq(ff, expected_ff, 1e-3);
    println!("  feedforward = {:+.4} (expected {:+.4}) — {}",
             ff, expected_ff, if pass { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 3: multiple supported bodies ──────────────────────────────

fn scenario_multiple_supported() {
    println!("=== Scenario 3: two supported bodies — feedforward = sum ===");
    let mut world = XpbdWorld::new();

    let parent = RigidBody::new(1.0, Mat3::identity());
    let parent_idx = world.add_body(parent);

    // Body A: 2 kg, offset +0.1 z. Moment contribution: -1.962.
    // Body B: 3 kg, offset -0.05 z. Moment contribution = (0.5, -0.05) × (0, -29.43, 0)
    //   = (0.5·0 - (-0.05)·(-29.43), ..., ...) = (-1.4715, 0, 0).
    //   moment_about_x = -1.4715.
    // Sum: -1.962 + -1.4715 = -3.4335 (oh wait, signs)
    // Actually: r_A × F_A = (0.5, 0.1) × (0, -19.62, 0) component (0.5·0 - 0.1·(-19.62), ..., ...)
    //   x-component: -0.1·(-19.62) = +1.962. Wait, let me recompute.
    //   r_A = (0, 0.5, 0.1), F_A = (0, -19.62, 0).
    //   cross = (r.y * F.z - r.z * F.y, r.z*F.x - r.x*F.z, r.x*F.y - r.y*F.x)
    //         = (0.5·0 - 0.1·(-19.62), 0.1·0 - 0·0, 0·(-19.62) - 0.5·0)
    //         = (1.962, 0, 0).
    //   x-component: +1.962.
    // r_B = (0, 0.5, -0.05), F_B = (0, -29.43, 0).
    //   cross_x = r.y * F.z - r.z * F.y = 0.5·0 - (-0.05)·(-29.43) = -1.4715.
    //   x-component: -1.4715.
    // Sum moment_about_x = 1.962 + (-1.4715) = 0.4905.
    // feedforward = -0.4905.
    let mut body_a = RigidBody::new(2.0, Mat3::identity());
    body_a.position = Vec3::new(0.0, 1.0, 0.1);
    let a_idx = world.add_body(body_a);

    let mut body_b = RigidBody::new(3.0, Mat3::identity());
    body_b.position = Vec3::new(0.0, 1.0, -0.05);
    let b_idx = world.add_body(body_b);

    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        parent_idx, a_idx, Vec3::new(1.0, 0.0, 0.0), 100.0, 10.0));

    let mut id = InverseDynamics::new();
    id.joints.push(IDJoint {
        pd_idx,
        parent_body: parent_idx,
        anchor_local: Vec3::new(0.0, 0.5, 0.0),
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![
            SupportedBody { body: a_idx, fraction: 1.0 },
            SupportedBody { body: b_idx, fraction: 1.0 },
        ],
    });

    id.update(&mut world);
    let ff = world.angular_pd_controllers[pd_idx].feedforward_torque;
    let expected_ff = -0.4905;
    let pass = approx_eq(ff, expected_ff, 1e-3);
    println!("  feedforward = {:+.4} (expected {:+.4}) — {}",
             ff, expected_ff, if pass { "✓" } else { "✗" });
    println!();
}

// ─── Scenario 4: bilateral sharing ──────────────────────────────────────

fn scenario_bilateral_sharing() {
    println!("=== Scenario 4: bilateral sharing — fraction = 0.5 halves the feedforward ===");
    // Two parents (left and right legs), one shared child (pelvis).
    // Each parent's joint supports the child at fraction 0.5.
    let mut world = XpbdWorld::new();
    let parent_l = RigidBody::new(1.0, Mat3::identity());
    let l_idx = world.add_body(parent_l);
    let parent_r = RigidBody::new(1.0, Mat3::identity());
    let r_idx = world.add_body(parent_r);

    let mut child = RigidBody::new(10.0, Mat3::identity());
    child.position = Vec3::new(0.0, 1.0, 0.1);  // offset forward 0.1 m
    let c_idx = world.add_body(child);

    let pd_l = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        l_idx, c_idx, Vec3::new(1.0, 0.0, 0.0), 100.0, 10.0));
    let pd_r = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        r_idx, c_idx, Vec3::new(1.0, 0.0, 0.0), 100.0, 10.0));

    let mut id = InverseDynamics::new();
    id.joints.push(IDJoint {
        pd_idx: pd_l,
        parent_body: l_idx,
        anchor_local: Vec3::new(0.0, 0.5, 0.0),
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![SupportedBody { body: c_idx, fraction: 0.5 }],
    });
    id.joints.push(IDJoint {
        pd_idx: pd_r,
        parent_body: r_idx,
        anchor_local: Vec3::new(0.0, 0.5, 0.0),
        axis_local: Vec3::new(1.0, 0.0, 0.0),
        supported: vec![SupportedBody { body: c_idx, fraction: 0.5 }],
    });

    id.update(&mut world);
    let ff_l = world.angular_pd_controllers[pd_l].feedforward_torque;
    let ff_r = world.angular_pd_controllers[pd_r].feedforward_torque;

    // Each joint sees only HALF the child's weight.
    // r × F: r = (0, 0.5, 0.1), F = (0, -10·g·0.5, 0) = (0, -49.05, 0)
    //   cross_x = 0.5·0 - 0.1·(-49.05) = 4.905
    // Each feedforward = -4.905 (each leg's hip applies same magnitude).
    let expected_each = -4.905;
    let pass_l = approx_eq(ff_l, expected_each, 1e-3);
    let pass_r = approx_eq(ff_r, expected_each, 1e-3);
    let pass_sum = approx_eq(ff_l + ff_r, -9.81, 1e-3);  // sum = full body comp
    println!("  L feedforward = {:+.4} (expected {:+.4}) — {}",
             ff_l, expected_each, if pass_l { "✓" } else { "✗" });
    println!("  R feedforward = {:+.4} (expected {:+.4}) — {}",
             ff_r, expected_each, if pass_r { "✓" } else { "✗" });
    println!("  L + R = {:+.4} (full gravity comp = {:+.4}) — {}",
             ff_l + ff_r, -9.81, if pass_sum { "✓" } else { "✗" });
    println!();
}

fn main() {
    println!("InverseDynamics (gravity feedforward) validation\n");
    scenario_neutral_zero_feedforward();
    scenario_tilted_forward();
    scenario_multiple_supported();
    scenario_bilateral_sharing();
}
