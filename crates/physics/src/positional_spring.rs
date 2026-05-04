use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Linear elastic spring between two body anchor points. Models the elastic
/// resistance of a joint capsule + cartilage + ligament system that resists
/// separation of the two bones forming a joint.
///
/// Force on body B at its anchor: `F = -stiffness * (offset - rest_length * n_hat)`
/// where `offset = pb - pa` is the world-space vector from anchor A to anchor B.
/// For `rest_length = 0` (typical joint), this is just `F = -stiffness * (pb - pa)`,
/// applied at the anchor on body B with equal-and-opposite at the anchor on A.
/// Lever-arm torques are applied automatically because forces are at the anchors.
///
/// Energy: stores `½ * stiffness * |offset - rest_length * n_hat|²` of elastic PE.
/// Conservative — energy returns when bodies move back together.
pub struct PositionalSpring {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Linear stiffness (N/m). For body-scale joints, ~10⁵ N/m gives realistic
    /// sub-mm deflection under typical loads. Higher = stiffer.
    pub stiffness: f64,
    /// Rest separation between the anchors. 0 for joints; non-zero for
    /// muscle-tendon attachments etc.
    pub rest_length: f64,
}

impl PositionalSpring {
    pub fn new(body_a: usize, body_b: usize, stiffness: f64) -> Self {
        Self {
            body_a, body_b,
            local_anchor_a: Vec3::zeros(),
            local_anchor_b: Vec3::zeros(),
            stiffness,
            rest_length: 0.0,
        }
    }

    /// Elastic potential energy stored in the spring.
    pub fn potential_energy(&self, bodies: &[RigidBody]) -> f64 {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;
        let offset = pb - pa;

        let stretch_vec = if self.rest_length == 0.0 {
            offset
        } else {
            let dist = offset.norm();
            if dist < 1e-12 {
                Vec3::zeros()
            } else {
                offset - (offset / dist) * self.rest_length
            }
        };
        0.5 * self.stiffness * stretch_vec.norm_squared()
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;
        let offset = pb - pa;

        // Force on body B at its anchor (pulls toward A if separated past rest_length).
        let force_on_b = if self.rest_length == 0.0 {
            -self.stiffness * offset
        } else {
            let dist = offset.norm();
            if dist < 1e-12 {
                return;
            }
            let n = offset / dist;
            -self.stiffness * (dist - self.rest_length) * n
        };
        let force_on_a = -force_on_b;

        // Apply force at each anchor: linear force + lever-arm torque.
        bodies[self.body_a].apply_force(force_on_a);
        bodies[self.body_b].apply_force(force_on_b);
        bodies[self.body_a].apply_torque(r_a.cross(&force_on_a));
        bodies[self.body_b].apply_torque(r_b.cross(&force_on_b));
    }
}
