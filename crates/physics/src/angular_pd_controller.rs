use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Active PD-controlled torque between two bodies along an axis.
///
/// Models the net torque a muscle (or muscle group) produces when commanded
/// to drive a joint toward a target angle. Distinct from `AngularSpring`
/// (passive elastic preload, fixed neutral, always on) and `AngularDamper`
/// (passive viscoelastic resistance, always on) — this is **active willed
/// drive** that can be commanded each frame and turned off.
///
/// The controller is intentionally simple (proportional + derivative on
/// joint angle and angular velocity). Real muscle has additional dynamics
/// (force-length, force-velocity, activation lag, contraction lag) that
/// would be added as wrappers around this primitive when biomechanical
/// fidelity demands them.
pub struct AngularPDController {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame.
    pub axis_local_a: Vec3,
    /// Proportional gain (N·m/rad).
    pub kp: f64,
    /// Derivative gain on relative angular velocity (N·m·s/rad).
    pub kd: f64,
    /// Commanded angle along the axis. External code sets this per-frame to
    /// drive willed motion (ramps, step targets, smooth trajectories, etc.).
    pub target_angle: f64,
    /// Optional torque cap (N·m). The output torque magnitude is clamped to
    /// this value; models the maximum force a muscle can produce. Set to
    /// `f64::INFINITY` for no cap.
    pub torque_limit: f64,
    /// If false, controller produces no torque. Models muscle deactivation.
    pub enabled: bool,
}

impl AngularPDController {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, kp: f64, kd: f64) -> Self {
        Self {
            body_a, body_b, axis_local_a: axis,
            kp, kd,
            target_angle: 0.0,
            torque_limit: f64::INFINITY,
            enabled: true,
        }
    }

    pub fn with_target(mut self, target: f64) -> Self {
        self.target_angle = target;
        self
    }

    pub fn with_torque_limit(mut self, limit: f64) -> Self {
        self.torque_limit = limit;
        self
    }

    /// Scalar torque the controller would apply this substep, given the current
    /// state. Useful for logging the muscle effort.
    pub fn scalar_torque(&self, bodies: &[RigidBody]) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let q_rel = bodies[self.body_b].orientation
            * bodies[self.body_a].orientation.inverse();
        let angle = q_rel.scaled_axis().dot(&n);
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);
        let raw = -self.kp * (angle - self.target_angle) - self.kd * omega_rel_n;
        if raw.abs() > self.torque_limit {
            raw.signum() * self.torque_limit
        } else {
            raw
        }
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        if !self.enabled {
            return;
        }
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let scalar = self.scalar_torque(bodies);
        if scalar == 0.0 {
            return;
        }
        let torque = n * scalar;
        bodies[self.body_b].apply_torque(torque);
        bodies[self.body_a].apply_torque(-torque);
    }
}
