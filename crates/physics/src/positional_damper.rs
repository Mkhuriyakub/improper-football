use crate::math::{Mat3, Vec3, skew_symmetric};
use crate::rigid_body::RigidBody;

/// Linear viscous damper between two body anchor points. Models the viscous
/// resistance of synovial fluid and connective tissue at a joint. Force on
/// body B at its anchor opposes the relative anchor velocity.
///
/// `F = -damping * v_rel` where `v_rel = v_b_anchor - v_a_anchor`.
/// Always dissipates energy at rate `damping * |v_rel|²`.
///
/// Integrated via `apply_implicit` (called after `integrate_velocities`):
/// unconditionally stable for any damping coefficient and time step. An
/// explicit force-accumulator integration would only be conditionally stable
/// (`sub_dt < 2·m_red/c`) and blows up on small-mass bodies with stiff welds —
/// exactly the failure mode of the foot's tiny tarsals/phalanges, which is why
/// only the implicit path exists.
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

    /// Implicit damping update: directly modifies linear/angular velocities
    /// (does not route through the force/torque accumulator). Must be called
    /// once per substep, after `integrate_velocities`.
    ///
    /// Mirrors `AngularDamper::apply_implicit`, but for a 3-DOF linear damper
    /// applied at the two anchor points. We solve backward-Euler on the
    /// relative anchor velocity:
    ///   (I + c·dt·K) · v_rel_new = v_rel_old,    J = -c·dt·v_rel_new
    /// where K is the 3×3 effective-inverse-mass operator mapping an impulse
    /// at the anchor pair to the change in relative anchor velocity:
    ///   K = (1/m_a + 1/m_b)·I
    ///       − skew(r_a)·I_inv_a·skew(r_a) − skew(r_b)·I_inv_b·skew(r_b).
    /// (−skew(r)·M·skew(r) is symmetric PSD, so I + c·dt·K is SPD/invertible.)
    /// This is unconditionally stable for any damping coefficient and dt.
    pub fn apply_implicit(&self, bodies: &mut Vec<RigidBody>, dt: f64) {
        let r_a = bodies[self.body_a].orientation * self.local_anchor_a;
        let r_b = bodies[self.body_b].orientation * self.local_anchor_b;
        let v_a_anchor = bodies[self.body_a].linear_velocity
            + bodies[self.body_a].angular_velocity.cross(&r_a);
        let v_b_anchor = bodies[self.body_b].linear_velocity
            + bodies[self.body_b].angular_velocity.cross(&r_b);
        let v_rel = v_b_anchor - v_a_anchor;
        if v_rel.norm_squared() < 1e-30 {
            return;
        }

        // Pinned bodies (inv_mass == 0) contribute no inverse mass/inertia.
        let zero = Mat3::zeros();
        let inv_m_a = bodies[self.body_a].inv_mass;
        let inv_m_b = bodies[self.body_b].inv_mass;
        let i_inv_a = if inv_m_a == 0.0 { zero }
                      else { bodies[self.body_a].world_inv_inertia() };
        let i_inv_b = if inv_m_b == 0.0 { zero }
                      else { bodies[self.body_b].world_inv_inertia() };

        let sa = skew_symmetric(r_a);
        let sb = skew_symmetric(r_b);
        let lin = (inv_m_a + inv_m_b) * Mat3::identity();
        let k = lin - sa * i_inv_a * sa - sb * i_inv_b * sb;

        // (I + c·dt·K) v_new = v_rel  →  v_new = (I + c·dt·K)^{-1} v_rel.
        let sys = Mat3::identity() + self.damping * dt * k;
        let v_new = match sys.try_inverse() {
            Some(inv) => inv * v_rel,
            None => return,
        };
        // Impulse on body B at its anchor; body A gets the reaction.
        let j = -self.damping * dt * v_new;

        bodies[self.body_a].linear_velocity -= inv_m_a * j;
        bodies[self.body_b].linear_velocity += inv_m_b * j;
        bodies[self.body_a].angular_velocity -= i_inv_a * r_a.cross(&j);
        bodies[self.body_b].angular_velocity += i_inv_b * r_b.cross(&j);
    }
}
