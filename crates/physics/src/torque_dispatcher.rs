use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// One actuator route within a `TorqueDispatcher`. Applies an
/// equal-and-opposite torque pair: `action_body` gets `+τ` about the
/// dispatcher's world axis, `reaction_body` gets `-τ`. Newton's third
/// law for angular motion — net angular momentum is conserved between
/// the two bodies, but external constraints (ground contact, joint
/// constraints with other bodies) can absorb part of the impulse.
///
/// `weight` is the relative weight of this route in the dispatcher's
/// demand split. The dispatcher normalizes the sum of weights to 1.
/// For equal split across N routes, set every route's weight to 1.0.
#[derive(Clone, Copy, Debug)]
pub struct DispatchRoute {
    pub action_body: usize,
    pub reaction_body: usize,
    pub weight: f64,
}

impl DispatchRoute {
    /// Default weight = 1.0.
    pub fn new(action_body: usize, reaction_body: usize) -> Self {
        Self { action_body, reaction_body, weight: 1.0 }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Splits a scalar torque demand across multiple torque-pair routes
/// about a common world-frame axis.
///
/// Designed for whole-body coordinated balance: one high-level
/// controller computes a corrective torque demand (e.g., "200 N·m
/// about world +x to upright the body") and the dispatcher distributes
/// that across the configured actuators (e.g., the two ankles in
/// bilateral standing).
///
/// Pure dispatch — no sensing, no feedback computation. The external
/// controller writes `demand` each substep based on whatever sensor
/// state it observes (CoM position, body tilt, etc.).
///
/// **Per-route torque** = `demand * (weight / total_weight)`, optionally
/// clamped to `per_route_limit`. Each route then writes a torque pair
/// to body accumulators about `axis_world`. Standard explicit-Euler
/// integration applies in `XpbdWorld::integrate_velocities` — the demand
/// is a position-feedback signal (size scales with tilt error), which
/// keeps per-substep impulses bounded as long as the tilt is small.
/// Velocity-feedback (KD) damping should NOT be channeled through the
/// dispatcher — use `AngularDamper` (implicit integration) for that.
pub struct TorqueDispatcher {
    /// Axis in WORLD frame about which all routes apply their torque.
    /// Should be a unit vector.
    pub axis_world: Vec3,
    /// Routes through which the demand is dispatched.
    pub routes: Vec<DispatchRoute>,
    /// Scalar torque demand (N·m) for the current substep. Set
    /// externally before each substep based on sensor state.
    pub demand: f64,
    /// Per-route magnitude cap (N·m). The unclamped per-route value is
    /// `demand * weight / total_weight`. Use `f64::INFINITY` for no cap.
    pub per_route_limit: f64,
    /// If false, applies no torque this substep (regardless of demand).
    pub enabled: bool,
}

impl TorqueDispatcher {
    pub fn new(axis_world: Vec3, routes: Vec<DispatchRoute>) -> Self {
        debug_assert!(
            (axis_world.norm() - 1.0).abs() < 1e-6,
            "TorqueDispatcher axis_world must be unit length (norm = {})",
            axis_world.norm()
        );
        Self {
            axis_world,
            routes,
            demand: 0.0,
            per_route_limit: f64::INFINITY,
            enabled: true,
        }
    }

    pub fn with_per_route_limit(mut self, limit: f64) -> Self {
        self.per_route_limit = limit;
        self
    }

    /// Total weight across all routes. Returns 0.0 for an empty
    /// dispatcher; the dispatcher then applies no torque.
    pub fn total_weight(&self) -> f64 {
        self.routes.iter().map(|r| r.weight).sum()
    }

    /// Per-route torque scalars that would be applied this substep
    /// given the current `demand`. Useful for diagnostics. Length
    /// matches `routes.len()`. Routes with zero total weight all get 0.
    pub fn per_route_torques(&self) -> Vec<f64> {
        let total = self.total_weight();
        if !self.enabled || total <= 0.0 {
            return vec![0.0; self.routes.len()];
        }
        self.routes.iter().map(|r| {
            let unclamped = self.demand * (r.weight / total);
            if unclamped.abs() > self.per_route_limit {
                unclamped.signum() * self.per_route_limit
            } else {
                unclamped
            }
        }).collect()
    }

    /// Write torque pairs to body accumulators. Called once per substep
    /// from `XpbdWorld::substep`.
    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let total = self.total_weight();
        if !self.enabled || total <= 0.0 {
            return;
        }
        for route in &self.routes {
            let unclamped = self.demand * (route.weight / total);
            let scalar = if unclamped.abs() > self.per_route_limit {
                unclamped.signum() * self.per_route_limit
            } else {
                unclamped
            };
            if scalar == 0.0 {
                continue;
            }
            let torque = self.axis_world * scalar;
            bodies[route.action_body].apply_torque(torque);
            bodies[route.reaction_body].apply_torque(-torque);
        }
    }
}
