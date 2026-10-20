use nalgebra::UnitQuaternion;
use crate::math::{Vec3, swing_twist_angle};
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
    /// Feedforward torque (N·m) added to the PD output each substep.
    /// Models tonic muscle activation — the baseline effort muscles
    /// produce continuously to maintain posture against gravity. Real
    /// biomechanics: the soleus, quadriceps, erector spinae, etc. all
    /// produce non-zero baseline activation even in "relaxed" standing.
    /// External code can update this per frame to do inverse-dynamics
    /// gravity compensation, anticipated-motion feedforward, etc.
    pub feedforward_torque: f64,
    /// Optional torque cap (N·m). The output torque magnitude (PD + FF
    /// combined) is clamped to this value; models the maximum force a
    /// muscle can produce. Set to `f64::INFINITY` for no cap.
    pub torque_limit: f64,
    /// If false, controller produces no torque. Models muscle deactivation.
    pub enabled: bool,
    /// Rest orientation of body_b in body_a's frame (default identity).
    /// `target_angle` is measured relative to this baseline using
    /// swing-twist factorization.
    pub baseline: UnitQuaternion<f64>,
}

impl AngularPDController {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, kp: f64, kd: f64) -> Self {
        Self {
            body_a, body_b, axis_local_a: axis,
            kp, kd,
            target_angle: 0.0,
            feedforward_torque: 0.0,
            torque_limit: f64::INFINITY,
            enabled: true,
            baseline: UnitQuaternion::identity(),
        }
    }

    pub fn with_feedforward(mut self, ff: f64) -> Self {
        self.feedforward_torque = ff;
        self
    }

    pub fn with_target(mut self, target: f64) -> Self {
        self.target_angle = target;
        self
    }

    pub fn with_torque_limit(mut self, limit: f64) -> Self {
        self.torque_limit = limit;
        self
    }

    /// Set a non-identity rest orientation of body_b in body_a's frame.
    pub fn with_baseline(mut self, baseline: UnitQuaternion<f64>) -> Self {
        self.baseline = baseline;
        self
    }

    fn angle(&self, bodies: &[RigidBody]) -> f64 {
        let q_b_in_a = bodies[self.body_a].orientation.inverse()
                     * bodies[self.body_b].orientation;
        let q_from_rest = q_b_in_a * self.baseline.inverse();
        swing_twist_angle(q_from_rest, self.axis_local_a)
    }

    /// Scalar torque the controller would apply this substep, given the current
    /// state (includes KP + KD + feedforward). Useful for logging the muscle effort.
    pub fn scalar_torque(&self, bodies: &[RigidBody]) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let angle = self.angle(bodies);
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);
        let raw = -self.kp * (angle - self.target_angle) - self.kd * omega_rel_n
                  + self.feedforward_torque;
        if raw.abs() > self.torque_limit {
            raw.signum() * self.torque_limit
        } else {
            raw
        }
    }

    /// KP + feedforward components of the controller torque, with
    /// `torque_limit` applied to the sum. The KD term is integrated
    /// separately by `apply_implicit_damping` and is intentionally NOT
    /// included here.
    pub fn kp_scalar_torque(&self, bodies: &[RigidBody]) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let angle = self.angle(bodies);
        let raw = -self.kp * (angle - self.target_angle) + self.feedforward_torque;
        if raw.abs() > self.torque_limit {
            raw.signum() * self.torque_limit
        } else {
            raw
        }
    }

    /// Implicit (backward-Euler) integration of the KP + feedforward
    /// component, exactly the linear special case of
    /// `AngularSoftLimit::apply_implicit`'s linearized solve:
    ///   ω_rel_new = (ω_rel + dt·w·s(θ)) / (1 − dt²·w·s'(θ))
    /// with s(θ) = −kp·(θ − target) + ff (clamped to torque_limit) and
    /// s' = −kp (0 in the saturated region). Unconditionally stable for
    /// any kp and dt. Must be called once per substep, after
    /// `integrate_velocities` (alongside the other implicit terms).
    ///
    /// This replaced the explicit accumulator-torque KP path (2026-10):
    /// an explicit kp on the foot's few-gram phalanges puts the joint past
    /// the explicit-integration bound (ω·sub_dt = √(kp/I_red)·sub_dt up to
    /// ~6 on toe PIP/DIP joints at kp = 4, I_red ≈ 2·10⁻⁹) — the per-substep
    /// kick (Δω ~10³ rad/s) pumped energy through the ground manifold and
    /// joint constraints, detonating or toppling assemblies that are quiet
    /// with the implicit form. It was the last stiff explicit term left
    /// after the 2026-06 implicit-integration rewrite.
    pub fn apply_implicit_kp(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        if !self.enabled || (self.kp == 0.0 && self.feedforward_torque == 0.0) {
            return;
        }
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let angle = self.angle(bodies);
        let raw = -self.kp * (angle - self.target_angle) + self.feedforward_torque;
        let saturated = raw.abs() > self.torque_limit;
        let s = if saturated { raw.signum() * self.torque_limit } else { raw };
        // Slope of s(θ): −kp in the linear region, 0 where the torque limit
        // clamps (constant-torque region).
        let s_prime = if saturated { 0.0 } else { -self.kp };

        let zero = crate::math::Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let w = n.dot(&(i_inv_a * n)) + n.dot(&(i_inv_b * n));
        if w < 1e-15 {
            return;
        }

        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);

        let omega_new = (omega_rel_n + dt * w * s) / (1.0 - dt * dt * w * s_prime);
        let j = (omega_new - omega_rel_n) / w;

        bodies[self.body_a].angular_velocity -= i_inv_a * n * j;
        bodies[self.body_b].angular_velocity += i_inv_b * n * j;
    }

    /// Implicit-Euler integration of the KD (velocity-damping) term.
    /// Same impulse-splitting math as `AngularDamper::apply_implicit`,
    /// using `self.kd` as the damping coefficient. Unconditionally
    /// stable for any `kd` and `dt`. Must be called once per substep,
    /// after `integrate_velocities`.
    pub fn apply_implicit_damping(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        if !self.enabled || self.kd == 0.0 {
            return;
        }
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let omega_rel = bodies[self.body_b].angular_velocity
            - bodies[self.body_a].angular_velocity;
        let omega_rel_n = omega_rel.dot(&n);
        if omega_rel_n.abs() < 1e-15 {
            return;
        }

        let zero = crate::math::Mat3::zeros();
        let i_inv_a = if bodies[self.body_a].inv_mass == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if bodies[self.body_b].inv_mass == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };
        let w = n.dot(&(i_inv_a * n)) + n.dot(&(i_inv_b * n));
        if w < 1e-15 {
            return;
        }

        let factor = 1.0 + self.kd * dt * w;
        let omega_rel_new = omega_rel_n / factor;
        let delta_omega_rel = omega_rel_new - omega_rel_n;
        let j = delta_omega_rel / w;

        bodies[self.body_a].angular_velocity -= i_inv_a * n * j;
        bodies[self.body_b].angular_velocity += i_inv_b * n * j;
    }
}
