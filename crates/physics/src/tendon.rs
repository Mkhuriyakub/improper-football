use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// One joint that a tendon crosses. The tendon's "excursion" contribution
/// from this joint is `moment_arm · θ` where θ is the relative rotation of
/// `body_b` w.r.t. `body_a` along `axis_local_a`.
pub struct TendonSpan {
    pub body_a: usize,
    pub body_b: usize,
    /// Joint axis in body_a's local frame (typically the joint's flexion axis).
    pub axis_local_a: Vec3,
    /// Perpendicular distance from joint axis to the tendon path. For a
    /// flexor tendon on the palmar side of a finger joint, this is roughly
    /// 5–8 mm at each joint.
    pub moment_arm: f64,
}

/// Multi-joint tendon coupling. Models a tendon (e.g. flexor digitorum
/// profundus) that runs across multiple joints, coupling their motion.
///
/// Excursion = Σ moment_arm_i · θ_i  across all spans the tendon crosses.
/// Tension = stiffness · (excursion − rest_excursion) + damping · d/dt(excursion)
/// At each span the tendon applies torque  τ_i = −tension · moment_arm_i  on
/// body_b (and equal-and-opposite on body_a) about the joint's axis.
///
/// This is the biomechanical primitive that prevents finger joints from moving
/// independently. With a passive resting tendon length, the tendon constrains
/// the joints to move in a coordinated ratio dictated by their moment arms —
/// the same mechanism that makes a real finger curl as a unit and resists
/// contortions like fully bending the tip while keeping the middle straight.
pub struct Tendon {
    pub spans: Vec<TendonSpan>,
    pub rest_excursion: f64,
    /// N per meter of excursion deviation.
    pub stiffness: f64,
    /// N·s per meter of excursion velocity.
    pub damping: f64,
}

impl Tendon {
    pub fn new(spans: Vec<TendonSpan>, rest_excursion: f64, stiffness: f64, damping: f64) -> Self {
        Self { spans, rest_excursion, stiffness, damping }
    }

    pub fn excursion(&self, bodies: &[RigidBody]) -> f64 {
        let mut total = 0.0;
        for span in &self.spans {
            let n = bodies[span.body_a].orientation * span.axis_local_a;
            let q_rel = bodies[span.body_b].orientation
                * bodies[span.body_a].orientation.inverse();
            let angle = q_rel.scaled_axis().dot(&n);
            total += span.moment_arm * angle;
        }
        total
    }

    pub fn excursion_rate(&self, bodies: &[RigidBody]) -> f64 {
        let mut total = 0.0;
        for span in &self.spans {
            let n = bodies[span.body_a].orientation * span.axis_local_a;
            let omega_rel = bodies[span.body_b].angular_velocity
                - bodies[span.body_a].angular_velocity;
            let omega_n = omega_rel.dot(&n);
            total += span.moment_arm * omega_n;
        }
        total
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let excursion = self.excursion(bodies);
        let excursion_rate = self.excursion_rate(bodies);
        let tension = self.stiffness * (excursion - self.rest_excursion)
                    + self.damping * excursion_rate;
        if tension == 0.0 {
            return;
        }
        for span in &self.spans {
            let n = bodies[span.body_a].orientation * span.axis_local_a;
            let scalar = -tension * span.moment_arm;
            let torque = n * scalar;
            bodies[span.body_b].apply_torque(torque);
            bodies[span.body_a].apply_torque(-torque);
        }
    }
}
