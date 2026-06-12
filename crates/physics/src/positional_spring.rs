use crate::math::{Mat3, Vec3, skew_symmetric};
use crate::rigid_body::RigidBody;

/// Linear elastic spring between two body anchor points. Models the elastic
/// resistance of a joint capsule + cartilage + ligament system that resists
/// separation of the two bones forming a joint.
///
/// For zero `rest_length` (every joint/weld in the project), prefer
/// `apply_implicit` (called after `integrate_velocities`): it integrates the
/// spring — including the lever-arm torque it exerts at the anchors —
/// implicitly, and is unconditionally stable. The explicit `apply` path feeds
/// `r × F` straight into the angular dynamics, whose stability bound
/// (`dt < 2·√(I_red/(k·r²))`) the stiff welds (k = 10⁶ N/m) violate on the
/// foot's small tarsals/phalanges, exploding the assembly within a few frames.
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

    /// Implicit (backward-Euler) spring update for the zero-rest-length case:
    /// directly modifies linear/angular velocities instead of routing an
    /// explicit force through the accumulator. Must be called once per substep,
    /// after `integrate_velocities`. For non-zero `rest_length` it falls back
    /// to the explicit `apply` (no project code uses non-zero rest_length on a
    /// PositionalSpring; tendons are a separate primitive).
    ///
    /// With anchor offsets `r_a, r_b`, current anchor separation `d = pb − pa`,
    /// relative anchor velocity `v_rel`, and the effective inverse-mass operator
    ///   K = (1/m_a + 1/m_b)·I − skew(r_a)·I_inv_a·skew(r_a) − skew(r_b)·I_inv_b·skew(r_b),
    /// the spring impulse on body B is `J = −k·dt·(d + dt·v_rel_new)`. Solving
    ///   (I + k·dt²·K)·v_rel_new = v_rel − k·dt·K·d
    /// then `J = −k·dt·(d + dt·v_rel_new)`. Unconditionally stable for any k/dt,
    /// because it integrates the lever-arm torque (via K's inertia terms)
    /// implicitly rather than as an explicit penalty.
    pub fn apply_implicit(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        if self.rest_length != 0.0 {
            self.apply(bodies);
            return;
        }

        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let pa = bodies[self.body_a].position + r_a;
        let pb = bodies[self.body_b].position + r_b;
        let d = pb - pa;

        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&r_b);
        let v_rel = v_b_anchor - v_a_anchor;

        let zero = Mat3::zeros();
        let inv_m_a = bodies[self.body_a].inv_mass;
        let inv_m_b = bodies[self.body_b].inv_mass;
        let i_inv_a = if inv_m_a == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if inv_m_b == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };

        let sa = skew_symmetric(r_a);
        let sb = skew_symmetric(r_b);
        let k_op = (inv_m_a + inv_m_b) * Mat3::identity()
            - sa * i_inv_a * sa - sb * i_inv_b * sb;

        let k = self.stiffness;
        let sys = Mat3::identity() + k * dt * dt * k_op;
        let rhs = v_rel - k * dt * (k_op * d);
        let v_new = match sys.try_inverse() {
            Some(inv) => inv * rhs,
            None => return,
        };
        let j = -k * dt * (d + dt * v_new);

        bodies[self.body_a].linear_velocity -= inv_m_a * j;
        bodies[self.body_b].linear_velocity += inv_m_b * j;
        bodies[self.body_a].angular_velocity -= i_inv_a * r_a.cross(&j);
        bodies[self.body_b].angular_velocity += i_inv_b * r_b.cross(&j);
    }
}
