use crate::math::Vec3;
use crate::rigid_body::RigidBody;
use nalgebra::UnitQuaternion;

/// XPBD positional constraint between two bodies.
/// compliance = 0 gives rigid (PBD), higher values give soft constraints.
pub struct PositionalConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub rest_length: f64,
    pub compliance: f64,
    pub lambda: f64,
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

    /// XPBD constraint solve with full angular correction for extended rigid bodies.
    /// Generalized inverse mass includes the rotational inertia contribution so that
    /// joint anchors (not just body centers) are correctly constrained.
    pub fn solve(&mut self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let alpha = self.compliance / (dt * dt);

        // World-space lever arms from each body center to its joint anchor
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;

        let diff = pb - pa;
        let dist = diff.norm();
        if dist < 1e-10 {
            return;
        }

        let n = diff / dist;
        let c = dist - self.rest_length;

        let inv_mass_a = bodies[self.body_a].inv_mass;
        let inv_mass_b = bodies[self.body_b].inv_mass;
        let i_inv_a = bodies[self.body_a].world_inv_inertia();
        let i_inv_b = bodies[self.body_b].world_inv_inertia();

        // Generalized inverse mass: translational + rotational contribution
        // w_ang = (r × n) · I⁻¹(r × n)
        let ra_cross_n = r_a.cross(&n);
        let rb_cross_n = r_b.cross(&n);
        let w_a = inv_mass_a + ra_cross_n.dot(&(i_inv_a * ra_cross_n));
        let w_b = inv_mass_b + rb_cross_n.dot(&(i_inv_b * rb_cross_n));
        let w = w_a + w_b;
        if w < 1e-10 {
            return;
        }

        let delta_lambda = (-c - alpha * self.lambda) / (w + alpha);
        self.lambda += delta_lambda;

        let p = n * delta_lambda;

        // Linear corrections
        bodies[self.body_a].position -= p * inv_mass_a;
        bodies[self.body_b].position += p * inv_mass_b;

        // Angular corrections: rotate each body so its anchor moves with the position fix
        // Force on A is -p, force on B is +p
        let dtheta_a = i_inv_a * r_a.cross(&(-p));
        let dtheta_b = i_inv_b * r_b.cross(&p);
        bodies[self.body_a].orientation =
            UnitQuaternion::from_scaled_axis(dtheta_a) * bodies[self.body_a].orientation;
        bodies[self.body_b].orientation =
            UnitQuaternion::from_scaled_axis(dtheta_b) * bodies[self.body_b].orientation;
    }

    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
