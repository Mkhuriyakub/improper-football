use nalgebra::UnitQuaternion;
use crate::math::{Mat3, Vec3, swing_twist_angle};
use crate::rigid_body::RigidBody;

/// Nonlinear soft range limit between two bodies along an axis fixed in body_a's
/// local frame. Applies torque only when the relative angle is outside the
/// [min_angle, max_angle] range; the torque grows nonlinearly with the excess
/// (`τ = stiffness · |excess|^exponent`, opposing further excess).
///
/// Models the progressive engagement of joint capsule and ligament fibers as a
/// joint approaches its anatomical limit. Unlike `AngularLimitConstraint`, this
/// is **not a hard stop** — the joint can be pushed past the natural range
/// under sufficient load. That makes injury (ligament tear) a representable
/// failure mode of the model: at extreme excess the torque is large but
/// finite, and downstream code can later read the stretch/torque magnitude to
/// trigger a tear by reducing this limit's stiffness.
pub struct AngularSoftLimit {
    pub body_a: usize,
    pub body_b: usize,
    /// Axis in body_a's local frame.
    pub axis_local_a: Vec3,
    pub min_angle: f64,
    pub max_angle: f64,
    /// Stiffness coefficient (N·m / rad^exponent). Effective stiffness is
    /// `stiffness * post_tear_factor` once `torn` is true.
    pub stiffness: f64,
    /// Exponent on |excess|. exponent = 1 is linear, 2 is quadratic, higher is
    /// progressively stiffer near the limit. Real ligament toe/linear regions
    /// are roughly 2–3.
    pub exponent: f64,
    /// If |excess| ever exceeds this threshold, the ligament tears and the
    /// limit's stiffness is permanently scaled by `post_tear_factor`. Default
    /// is `f64::INFINITY` (no tearing).
    pub tear_threshold: f64,
    /// Stiffness multiplier applied after tearing. 0.0 = complete failure (no
    /// restoring torque), 0.05 = severe partial tear, 1.0 = no effect.
    pub post_tear_factor: f64,
    /// Latched on the first substep where |excess| exceeds `tear_threshold`.
    /// Persistent — there is no automatic healing.
    pub torn: bool,
    /// Rest orientation of body_b in body_a's frame (default identity). The
    /// angle compared against [min_angle, max_angle] is measured relative
    /// to this baseline using swing-twist factorization.
    pub baseline: UnitQuaternion<f64>,
}

impl AngularSoftLimit {
    pub fn new(
        body_a: usize, body_b: usize,
        axis: Vec3,
        min_angle: f64, max_angle: f64,
        stiffness: f64, exponent: f64,
    ) -> Self {
        debug_assert!(
            (axis.norm() - 1.0).abs() < 1e-6,
            "AngularSoftLimit axis must be unit length (norm = {})",
            axis.norm()
        );
        Self {
            body_a, body_b, axis_local_a: axis,
            min_angle, max_angle, stiffness, exponent,
            tear_threshold: f64::INFINITY,
            post_tear_factor: 0.0,
            torn: false,
            baseline: UnitQuaternion::identity(),
        }
    }

    /// Builder: enable ligament-tearing failure mode. `threshold` is the |excess|
    /// past the natural range that triggers a tear; `post_tear_factor` is the
    /// remaining fraction of the original stiffness after tearing.
    pub fn with_tearing(mut self, threshold: f64, post_tear_factor: f64) -> Self {
        self.tear_threshold = threshold;
        self.post_tear_factor = post_tear_factor;
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

    fn effective_stiffness(&self) -> f64 {
        if self.torn { self.stiffness * self.post_tear_factor } else { self.stiffness }
    }

    /// Scalar torque about the constraint axis (positive = increases relative angle).
    /// Reflects the current torn-or-intact state.
    pub fn scalar_torque(&self, angle: f64) -> f64 {
        let k = self.effective_stiffness();
        if angle > self.max_angle {
            -k * (angle - self.max_angle).powf(self.exponent)
        } else if angle < self.min_angle {
            k * (self.min_angle - angle).powf(self.exponent)
        } else {
            0.0
        }
    }

    /// Local slope `ds/dθ` of the restoring scalar torque at `angle`
    /// (≤ 0 in the engaged region, 0 inside the range). Used by the implicit
    /// integrator to linearize the nonlinear limit about its operating point.
    fn scalar_torque_slope(&self, angle: f64) -> f64 {
        let k = self.effective_stiffness();
        if angle > self.max_angle {
            let ex = angle - self.max_angle;
            -k * self.exponent * ex.powf(self.exponent - 1.0)
        } else if angle < self.min_angle {
            let ex = self.min_angle - angle;
            -k * self.exponent * ex.powf(self.exponent - 1.0)
        } else {
            0.0
        }
    }

    /// Latch the torn state if the current excess exceeds the tear threshold.
    /// Pulled out of `apply` so the implicit integrator can keep the same
    /// failure-mode behaviour.
    fn update_torn(&mut self, angle: f64) {
        let excess_abs = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            self.min_angle - angle
        } else {
            0.0
        };
        if !self.torn && excess_abs > self.tear_threshold {
            self.torn = true;
        }
    }

    /// Implicit (semi-implicit) limit update: directly modifies angular
    /// velocities instead of routing an explicit torque through the
    /// accumulator. Must be called once per substep, after
    /// `integrate_velocities` (alongside the implicit springs/dampers).
    ///
    /// Linearizes the nonlinear restoring torque `s(θ)` about the current
    /// angle and solves backward-Euler along the axis:
    ///   ω_rel_new = (ω_rel + dt·w·s(θ)) / (1 − dt²·w·s'(θ)),   J = (ω_rel_new − ω_rel)/w,
    /// with `w = n·I_inv_a·n + n·I_inv_b·n`. Since `s'(θ) ≤ 0` in the engaged
    /// region the denominator is ≥ 1, so this is unconditionally stable for
    /// any limit stiffness — the explicit penalty (k up to 10⁴) otherwise
    /// blows up the foot's ~2 g distal phalanges.
    pub fn apply_implicit(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let n = bodies[self.body_a].orientation * self.axis_local_a;
        let angle = self.angle(bodies);

        // Tearing is evaluated on the same substep the threshold is crossed.
        self.update_torn(angle);

        let s = self.scalar_torque(angle);
        if s == 0.0 {
            return;
        }
        let s_prime = self.scalar_torque_slope(angle);

        let zero = Mat3::zeros();
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

    /// Elastic potential energy stored in the limit when out of range.
    /// Uses the current effective stiffness (post-tear if torn).
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let angle = self.angle(bodies);
        let excess = if angle > self.max_angle {
            angle - self.max_angle
        } else if angle < self.min_angle {
            self.min_angle - angle
        } else {
            return 0.0;
        };
        self.effective_stiffness() * excess.powf(self.exponent + 1.0) / (self.exponent + 1.0)
    }
}
