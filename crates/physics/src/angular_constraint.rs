use crate::math::Vec3;
use crate::rigid_body::RigidBody;
use nalgebra::UnitQuaternion;

/// XPBD angular limit constraint.
///
/// Constrains the relative rotation of body_b w.r.t. body_a about one axis to
/// [min_angle, max_angle] radians.  Three constraints per joint (Z, X, Y) give
/// full anatomical limits.
pub struct AngularLimitConstraint {
    pub body_a: usize,
    pub body_b: usize,
    /// Constraint axis expressed in body_a's local frame.
    pub axis_local_a: Vec3,
    pub min_angle: f64,
    pub max_angle: f64,
    pub compliance: f64,
    pub lambda: f64,
}

impl AngularLimitConstraint {
    pub fn new(body_a: usize, body_b: usize, axis: Vec3, min: f64, max: f64) -> Self {
        Self {
            body_a,
            body_b,
            axis_local_a: axis,
            min_angle: min,
            max_angle: max,
            compliance: 0.0,
            lambda: 0.0,
        }
    }

    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let alpha = self.compliance / (dt * dt);

        // Constraint axis in world space (follows body_a orientation).
        let n = bodies[self.body_a].orientation * self.axis_local_a;

        // Relative rotation: q_rel = q_b * q_a^-1
        // Project its rotation vector onto the constraint axis to get the angle.
        let q_rel = bodies[self.body_b].orientation
            * bodies[self.body_a].orientation.inverse();
        let angle = q_rel.scaled_axis().dot(&n);

        let c = if angle < self.min_angle {
            angle - self.min_angle
        } else if angle > self.max_angle {
            angle - self.max_angle
        } else {
            return;
        };

        let i_inv_a = bodies[self.body_a].world_inv_inertia();
        let i_inv_b = bodies[self.body_b].world_inv_inertia();

        let w_a = n.dot(&(i_inv_a * n));
        let w_b = n.dot(&(i_inv_b * n));
        let w = w_a + w_b;
        if w < 1e-10 {
            return;
        }

        let delta_lambda = (-c - alpha * self.lambda) / (w + alpha);
        self.lambda += delta_lambda;

        // Sign convention mirrors PositionalConstraint:
        //   body_a: dθ = -I_a_inv * n * Δλ
        //   body_b: dθ = +I_b_inv * n * Δλ
        let dtheta_a = -(i_inv_a * n) * delta_lambda;
        let dtheta_b = (i_inv_b * n) * delta_lambda;

        bodies[self.body_a].orientation =
            UnitQuaternion::from_scaled_axis(dtheta_a) * bodies[self.body_a].orientation;
        bodies[self.body_b].orientation =
            UnitQuaternion::from_scaled_axis(dtheta_b) * bodies[self.body_b].orientation;
    }

    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
