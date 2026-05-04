use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Linear viscous damper between two body anchor points. Models the viscous
/// resistance of synovial fluid and connective tissue at a joint. Force on
/// body B at its anchor opposes the relative anchor velocity.
///
/// `F = -damping * v_rel` where `v_rel = v_b_anchor - v_a_anchor`.
/// Always dissipates energy at rate `damping * |v_rel|²`.
pub struct PositionalDamper {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    /// Damping coefficient (N·s/m). Pair with a `PositionalSpring` of stiffness k:
    /// for damping ratio ζ = 0.5–1.0 (well-damped without ringing) at the bodies'
    /// effective mass m_eff, use damping = 2·ζ·√(m_eff·k).
    pub damping: f64,
}

impl PositionalDamper {
    pub fn new(body_a: usize, body_b: usize, damping: f64) -> Self {
        Self {
            body_a, body_b,
            local_anchor_a: Vec3::zeros(),
            local_anchor_b: Vec3::zeros(),
            damping,
        }
    }

    /// Instantaneous power dissipated by this damper (always ≤ 0).
    pub fn power(&self, bodies: &[RigidBody]) -> f64 {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&r_b);
        let v_rel = v_b_anchor - v_a_anchor;
        -self.damping * v_rel.norm_squared()
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&r_b);
        let v_rel = v_b_anchor - v_a_anchor;

        let force_on_b = -self.damping * v_rel;
        let force_on_a = -force_on_b;

        bodies[self.body_a].apply_force(force_on_a);
        bodies[self.body_b].apply_force(force_on_b);
        bodies[self.body_a].apply_torque(r_a.cross(&force_on_a));
        bodies[self.body_b].apply_torque(r_b.cross(&force_on_b));
    }
}
