use crate::math::Vec3;
use crate::xpbd::XpbdWorld;

/// One body supported by an inverse-dynamics joint, with the fraction
/// of its weight that flows through this joint. For a tree-only chain
/// the fraction is always 1.0. For a bilateral closed chain (e.g.
/// pelvis supported by two hips), shared bodies have fraction 0.5 per
/// supporter. A controller that infers load sharing from contact
/// forces could update fractions dynamically.
#[derive(Clone, Copy, Debug)]
pub struct SupportedBody {
    pub body: usize,
    pub fraction: f64,
}

/// Description of one joint for inverse dynamics. The observer reads
/// the body positions and computes the gravity moment about the joint
/// axis from all supported bodies, then writes the negation to the
/// joint's `AngularPDController::feedforward_torque`.
///
/// `parent_body`, `anchor_local`, `axis_local` must match the joint's
/// physical setup (same body_a, anchor on body_a, and axis in body_a's
/// local frame as the corresponding `AngularPDController`). The
/// observer doesn't enforce consistency — the binary author is
/// responsible.
pub struct IDJoint {
    /// Index into `world.angular_pd_controllers`.
    pub pd_idx: usize,
    /// Body whose orientation defines the joint axis frame (= PD's body_a).
    pub parent_body: usize,
    /// Joint location in parent's local frame.
    pub anchor_local: Vec3,
    /// Joint axis in parent's local frame.
    pub axis_local: Vec3,
    /// Bodies whose weight is supported THROUGH this joint, with
    /// fractions. For a strict tree chain: fraction = 1.0 per body.
    /// For bilateral shared upper body: pelvis and torso at 0.5 per leg.
    pub supported: Vec<SupportedBody>,
}

/// Inverse-dynamics observer for gravity compensation. Each call to
/// `update()` evaluates the chain's current pose and writes a
/// gravity-cancelling feedforward torque to each configured PD
/// controller. The PD then only handles deviations from the
/// equilibrium pose, not the full gravity load — much higher
/// effective stiffness against disturbance, without per-PD-KP gain
/// inflation.
///
/// Models the inverse-dynamics gravity term `G(q)` from the standard
/// rigid-body equations of motion `M(q)·q̈ + C(q,q̇) + G(q) = τ`. For
/// the static case (q̈ = 0, q̇ = 0), `τ = G(q)`, exactly the
/// gravity-feedforward we want.
///
/// **Approximation:** the project's joints are compliant springs, not
/// rigid kinematic constraints. The springs are stiff enough that the
/// joint-anchor mismatch is sub-millimeter under standing load — the
/// ideal rigid-tree ID is accurate to within that error.
///
/// **Extension path:** add Coriolis/centrifugal terms `C(q,q̇)` to
/// feedforward motion accelerations (running, kicking). Add a
/// trajectory tracking term `M·q̈_desired` for computed-torque
/// control. Both build on this primitive.
///
/// Usage: construct one `InverseDynamics`, populate `joints`, call
/// `update(&mut world)` once per substep (before `world.substep`).
pub struct InverseDynamics {
    pub joints: Vec<IDJoint>,
    pub gravity: Vec3,
    pub enabled: bool,
}

impl InverseDynamics {
    pub fn new() -> Self {
        Self {
            joints: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            enabled: true,
        }
    }

    pub fn with_gravity(mut self, g: Vec3) -> Self {
        self.gravity = g;
        self
    }

    /// Update feedforward torques on all configured PD controllers
    /// based on current world state.
    pub fn update(&self, world: &mut XpbdWorld) {
        if !self.enabled {
            return;
        }
        for j in &self.joints {
            let parent = &world.bodies[j.parent_body];
            let joint_pos = parent.position + parent.orientation * j.anchor_local;
            let axis_world = (parent.orientation * j.axis_local).normalize();

            // Gravity moment about the joint axis = Σ r_b × (m_b · g)
            // projected onto axis_world.
            let mut moment_about_axis = 0.0;
            for sb in &j.supported {
                let body = &world.bodies[sb.body];
                let r = body.position - joint_pos;
                let weight = self.gravity * (body.mass * sb.fraction);
                let moment = r.cross(&weight);
                moment_about_axis += moment.dot(&axis_world);
            }

            // The PD's `feedforward_torque` is the scalar in the
            // torque pair: +scalar applied along axis_world on body_b,
            // -scalar on body_a. To cancel gravity's moment on
            // body_b, set feedforward = -moment_about_axis.
            world.angular_pd_controllers[j.pd_idx].feedforward_torque =
                -moment_about_axis;
        }
    }

    /// Inspect the feedforward torque this observer WOULD set, without
    /// writing to the world. Useful for diagnostics.
    pub fn compute_feedforward(&self, world: &XpbdWorld, joint_idx: usize) -> f64 {
        let j = &self.joints[joint_idx];
        let parent = &world.bodies[j.parent_body];
        let joint_pos = parent.position + parent.orientation * j.anchor_local;
        let axis_world = (parent.orientation * j.axis_local).normalize();
        let mut moment_about_axis = 0.0;
        for sb in &j.supported {
            let body = &world.bodies[sb.body];
            let r = body.position - joint_pos;
            let weight = self.gravity * (body.mass * sb.fraction);
            let moment = r.cross(&weight);
            moment_about_axis += moment.dot(&axis_world);
        }
        -moment_about_axis
    }
}

impl Default for InverseDynamics {
    fn default() -> Self {
        Self::new()
    }
}
