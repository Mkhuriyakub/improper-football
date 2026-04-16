use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// XPBD positional constraint between two bodies.
/// compliance = 0 gives rigid (PBD), higher values give soft constraints.
pub struct PositionalConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub rest_length: f64,
    pub compliance: f64,
    pub lambda: f64, // accumulated Lagrange multiplier
}

impl PositionalConstraint {
    pub fn new(body_a: usize, body_b: usize, rest_length: f64, compliance: f64) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::zeros(),
            local_anchor_b: Vec3::zeros(),
            rest_length,
            compliance,
            lambda: 0.0,
        }
    }

    /// XPBD constraint solve — updates positions directly.
    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let alpha = self.compliance / (dt * dt);

        let pa = bodies[self.body_a].position
            + bodies[self.body_a].orientation * self.local_anchor_a;
        let pb = bodies[self.body_b].position
            + bodies[self.body_b].orientation * self.local_anchor_b;

        let diff = pb - pa;
        let dist = diff.norm();
        if dist < 1e-10 {
            return;
        }

        let n = diff / dist;
        let c = dist - self.rest_length;

        let w_a = bodies[self.body_a].inv_mass;
        let w_b = bodies[self.body_b].inv_mass;
        let w = w_a + w_b;
        if w < 1e-10 {
            return;
        }

        let delta_lambda = (-c - alpha * self.lambda) / (w + alpha);
        self.lambda += delta_lambda;

        let p = n * delta_lambda;
        bodies[self.body_a].position -= p * w_a;
        bodies[self.body_b].position += p * w_b;
    }

    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
