use crate::rigid_body::RigidBody;
use crate::constraints::PositionalConstraint;
use crate::angular_constraint::AngularLimitConstraint;
use crate::angular_damper::AngularDamper;
use crate::angular_spring::AngularSpring;
use crate::angular_soft_limit::AngularSoftLimit;
use crate::angular_pd_controller::AngularPDController;
use crate::positional_spring::PositionalSpring;
use crate::positional_damper::PositionalDamper;
use crate::tendon::Tendon;
use crate::torque_dispatcher::TorqueDispatcher;
use crate::foot_contact::FootContact;
use crate::contact_constraint::ContactConstraint;
use crate::math::{Vec2, Vec3};
use nalgebra::UnitQuaternion;

pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
pub const SOLVER_ITERATIONS: usize = 30;
/// Default number of sequential-impulse iterations per substep. Each iteration
/// is one Gauss-Seidel pass over all positional and angular constraints,
/// refining the impulses that satisfy the velocity-level constraints.
pub const DEFAULT_SI_ITERATIONS: usize = 8;

/// Read-only summary of one endpoint of a capsule that's currently
/// pressing the ground. Returned by `XpbdWorld::engaged_endpoints`.
/// `contact_point` is the world-frame point on the ground plane (one
/// radius below the end-sphere center); `normal_force` is the magnitude
/// of the upward force the ground is applying at that point.
#[derive(Clone, Copy, Debug)]
pub struct EndpointContact {
    pub contact_point: Vec3,
    pub normal_force: f64,
}

/// World-level ground configuration. The ground is an infinite horizontal
/// plane at `plane_y` (when `Some`); every body whose `capsule_shape` is
/// `Some(...)` collides against it automatically. Setting `plane_y = None`
/// (via `Ground::disabled`) is the canonical way to run the world without
/// gravity-stop — no body collides with the ground, regardless of shape.
///
/// The penalty-spring + bilateral-damping + regularized-Coulomb-friction
/// math (and all its default parameter choices) is identical to the old
/// per-body `GroundContact` it replaces; see the body of
/// `XpbdWorld::apply_ground_contact_all` for the full model.
pub struct Ground {
    pub plane_y: Option<f64>,
    pub stiffness: f64,
    pub damping: f64,
    pub friction_coefficient: f64,
    pub friction_damping: f64,
}

impl Ground {
    pub fn at_y(plane_y: f64) -> Self {
        Self {
            plane_y: Some(plane_y),
            stiffness: 1.0e5,
            damping: 600.0,
            friction_coefficient: 0.6,
            friction_damping: 1000.0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            plane_y: None,
            stiffness: 1.0e5,
            damping: 600.0,
            friction_coefficient: 0.6,
            friction_damping: 1000.0,
        }
    }
}

pub struct XpbdWorld {
    pub bodies: Vec<RigidBody>,
    pub constraints: Vec<PositionalConstraint>,
    pub angular_constraints: Vec<AngularLimitConstraint>,
    pub angular_dampers: Vec<AngularDamper>,
    pub angular_springs: Vec<AngularSpring>,
    pub angular_soft_limits: Vec<AngularSoftLimit>,
    pub angular_pd_controllers: Vec<AngularPDController>,
    pub positional_springs: Vec<PositionalSpring>,
    pub positional_dampers: Vec<PositionalDamper>,
    pub tendons: Vec<Tendon>,
    pub torque_dispatchers: Vec<TorqueDispatcher>,
    pub foot_contacts: Vec<FootContact>,
    pub contact_constraints: Vec<ContactConstraint>,
    /// World-level ground. Every body with `capsule_shape: Some(...)`
    /// collides against this plane each substep.
    pub ground: Ground,
    /// Number of Gauss-Seidel sequential-impulse iterations per substep.
    pub solver_iterations: usize,
}

impl XpbdWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            constraints: Vec::new(),
            angular_constraints: Vec::new(),
            angular_dampers: Vec::new(),
            angular_springs: Vec::new(),
            angular_soft_limits: Vec::new(),
            angular_pd_controllers: Vec::new(),
            positional_springs: Vec::new(),
            positional_dampers: Vec::new(),
            tendons: Vec::new(),
            torque_dispatchers: Vec::new(),
            foot_contacts: Vec::new(),
            contact_constraints: Vec::new(),
            ground: Ground::at_y(0.0),
            solver_iterations: DEFAULT_SI_ITERATIONS,
        }
    }

    /// Set the ground plane to the given world y-coordinate using the
    /// engine's default contact parameters. Replaces any previous
    /// ground configuration.
    pub fn set_ground(&mut self, plane_y: f64) {
        self.ground = Ground::at_y(plane_y);
    }

    /// Disable ground contact entirely. Useful for off-ground tests or
    /// when an external test rig provides the floor. Bodies retain
    /// their `capsule_shape` (for queries / rendering) but the world's
    /// contact pass becomes a no-op.
    pub fn no_ground(&mut self) {
        self.ground = Ground::disabled();
    }

    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        id
    }

    // ─── Center-of-mass observers ─────────────────────────────────────
    //
    // Pure observation primitives — read whole-body or subset-of-body
    // mass, CoM position, CoM velocity. No control logic, no force
    // application. The first ingredient for whole-body coordinated
    // control (the support polygon / ZMP estimator and the multi-actuator
    // dispatcher build on these).
    //
    // Pinned bodies (inv_mass == 0) DO contribute their nominal mass to
    // these aggregates. If the caller wants to exclude pinned bodies
    // from the CoM (e.g., a "system above the pinned shin" CoM), they
    // should use the `_of` variant with an explicit body-index subset.

    /// Sum of `body.mass` across all bodies in the world.
    pub fn total_mass(&self) -> f64 {
        self.bodies.iter().map(|b| b.mass).sum()
    }

    /// Sum of `body.mass` across the specified body indices.
    pub fn mass_of(&self, indices: &[usize]) -> f64 {
        indices.iter().map(|&i| self.bodies[i].mass).sum()
    }

    /// Center-of-mass position of all bodies in the world.
    /// Returns the origin if total mass is zero.
    pub fn com_position(&self) -> Vec3 {
        let m = self.total_mass();
        if m <= 0.0 {
            return Vec3::zeros();
        }
        let weighted: Vec3 = self.bodies.iter()
            .map(|b| b.position * b.mass)
            .sum();
        weighted / m
    }

    /// Center-of-mass position of the specified body subset.
    pub fn com_position_of(&self, indices: &[usize]) -> Vec3 {
        let m = self.mass_of(indices);
        if m <= 0.0 {
            return Vec3::zeros();
        }
        let weighted: Vec3 = indices.iter()
            .map(|&i| self.bodies[i].position * self.bodies[i].mass)
            .sum();
        weighted / m
    }

    /// Center-of-mass velocity (linear) of all bodies in the world.
    pub fn com_velocity(&self) -> Vec3 {
        let m = self.total_mass();
        if m <= 0.0 {
            return Vec3::zeros();
        }
        let weighted: Vec3 = self.bodies.iter()
            .map(|b| b.linear_velocity * b.mass)
            .sum();
        weighted / m
    }

    /// Center-of-mass velocity of the specified body subset.
    pub fn com_velocity_of(&self, indices: &[usize]) -> Vec3 {
        let m = self.mass_of(indices);
        if m <= 0.0 {
            return Vec3::zeros();
        }
        let weighted: Vec3 = indices.iter()
            .map(|&i| self.bodies[i].linear_velocity * self.bodies[i].mass)
            .sum();
        weighted / m
    }

    /// Vertical projection of the whole-world CoM onto the ground
    /// plane (y = 0), as a 2D point in the world XZ plane. This is the
    /// "static ZMP" approximation — exact ZMP equals this only when CoM
    /// horizontal acceleration is zero. Adequate for low-frequency
    /// balance criteria (where the controller's job is to *prevent*
    /// large CoM accelerations).
    pub fn com_ground_projection(&self) -> Vec2 {
        let p = self.com_position();
        Vec2::new(p.x, p.z)
    }

    /// Vertical projection of a subset of bodies' CoM onto y = 0.
    pub fn com_ground_projection_of(&self, indices: &[usize]) -> Vec2 {
        let p = self.com_position_of(indices);
        Vec2::new(p.x, p.z)
    }

    pub fn step(&mut self, dt: f64) {
        let sub_dt = dt / SOLVER_ITERATIONS as f64;
        for _ in 0..SOLVER_ITERATIONS {
            self.substep(sub_dt);
        }
    }

    /// Run one substep using the velocity-level (sequential impulse) solver.
    ///
    /// Order of operations:
    ///   1. Apply external forces/torques (dampers, springs, soft limits, PD,
    ///      tendons) into the accumulators.
    ///   2. Integrate velocities from accumulated forces + gravity.
    ///   3. Solve constraints — multiple Gauss-Seidel iterations of impulse
    ///      application on the constraints. Modifies velocities only.
    ///   4. Integrate positions using the corrected velocities.
    ///   5. Clear force/torque accumulators.
    ///
    /// Notes for callers wrapping this in their own substep loop: forces
    /// must be re-applied each call since accumulators clear at the end of
    /// every substep (same as before the rewrite).
    pub fn substep(&mut self, sub_dt: f64) {
        self.apply_springs();
        self.apply_soft_limits();
        self.apply_pd_controllers();
        self.apply_torque_dispatchers();
        self.apply_positional_springs();
        self.apply_positional_dampers();
        self.apply_tendons();
        self.apply_ground_contact_all();
        self.apply_foot_contacts();
        self.integrate_velocities(sub_dt);
        // Damping applied implicitly AFTER explicit velocity update — this is
        // unconditionally stable for any damping coefficient, fixing the
        // explicit-Euler instability that affected small-inertia bodies.
        // Both passive AngularDampers and the KD-term of PD controllers
        // are integrated implicitly here.
        self.apply_dampers_implicit(sub_dt);
        self.apply_pd_dampers_implicit(sub_dt);
        // Reset contact impulses BEFORE the SI loop. Each substep
        // re-solves contacts from scratch; the SI iterations
        // accumulate impulse for warm-start within this substep but
        // not across substeps.
        for c in &mut self.contact_constraints {
            c.reset_impulses();
        }
        for _ in 0..self.solver_iterations {
            self.solve_constraints(sub_dt);
        }
        self.integrate_positions(sub_dt);
        self.clear_accumulators();
    }

    fn apply_dampers_implicit(&mut self, sub_dt: f64) {
        for damper in &self.angular_dampers {
            damper.apply_implicit(&mut self.bodies, sub_dt);
        }
    }

    fn apply_pd_dampers_implicit(&mut self, sub_dt: f64) {
        for ctrl in &self.angular_pd_controllers {
            ctrl.apply_implicit_damping(&mut self.bodies, sub_dt);
        }
    }

    fn apply_springs(&mut self) {
        for spring in &self.angular_springs {
            spring.apply(&mut self.bodies);
        }
    }

    fn apply_soft_limits(&mut self) {
        for limit in &mut self.angular_soft_limits {
            limit.apply(&mut self.bodies);
        }
    }

    fn apply_pd_controllers(&mut self) {
        for ctrl in &self.angular_pd_controllers {
            ctrl.apply(&mut self.bodies);
        }
    }

    fn apply_torque_dispatchers(&mut self) {
        for d in &self.torque_dispatchers {
            d.apply(&mut self.bodies);
        }
    }

    fn apply_tendons(&mut self) {
        for tendon in &self.tendons {
            tendon.apply(&mut self.bodies);
        }
    }

    fn apply_positional_springs(&mut self) {
        for s in &self.positional_springs {
            s.apply(&mut self.bodies);
        }
    }

    fn apply_positional_dampers(&mut self) {
        for d in &self.positional_dampers {
            d.apply(&mut self.bodies);
        }
    }

    /// World-level ground-contact pass.
    ///
    /// For every body whose `capsule_shape` is `Some(...)` and whose
    /// `inv_mass` is non-zero (pinned bodies pass through unchanged —
    /// the contact would have no effect anyway), check both endpoints
    /// of the capsule against the ground plane. Apply penalty-spring
    /// normal force, bilateral damping, and regularized-Coulomb friction
    /// in the xz tangent plane. Both normal and friction forces are
    /// applied at the actual ground-contact point (one radius below the
    /// end-sphere center) so the friction lever arm is correct.
    ///
    /// The math here is identical to the old per-body `GroundContact::apply`
    /// that this method replaces; see git history for the derivation
    /// notes (penalty spring + bilateral damping rationale, regularized
    /// Coulomb saturation at μ·|F_n|, etc.).
    fn apply_ground_contact_all(&mut self) {
        let Some(plane_y) = self.ground.plane_y else {
            return;
        };
        let stiffness = self.ground.stiffness;
        let damping = self.ground.damping;
        let mu = self.ground.friction_coefficient;
        let friction_damping = self.ground.friction_damping;

        for body_idx in 0..self.bodies.len() {
            let shape = match self.bodies[body_idx].capsule_shape {
                Some(s) => s,
                None => continue,
            };
            if self.bodies[body_idx].inv_mass == 0.0 {
                // Pinned body — contact would have no effect.
                continue;
            }

            let axis_world = self.bodies[body_idx].orientation
                * Vec3::new(0.0, 1.0, 0.0);
            let half = shape.half_h;
            let r = shape.radius;

            // Two endpoints of the capsule's central segment, in world.
            let endpoints = [axis_world * half, -axis_world * half];

            for endpoint_offset in endpoints {
                let endpoint_world = self.bodies[body_idx].position + endpoint_offset;
                // Bottom of the end-sphere = endpoint center - radius (along +y).
                let bottom = endpoint_world.y - r;
                let penetration = plane_y - bottom;
                if penetration <= 0.0 {
                    continue;
                }

                // Actual ground-contact point is one radius below the end-sphere
                // center, on the plane. Use this offset (not the end-sphere
                // offset) for both normal and friction force application so
                // that the friction lever arm is correct.
                let contact_offset = endpoint_offset + Vec3::new(0.0, -r, 0.0);

                // Velocity at the contact point (includes rotation contribution).
                let v_contact = self.bodies[body_idx].linear_velocity
                    + self.bodies[body_idx].angular_velocity.cross(&contact_offset);

                // Normal force: spring + bilateral damping, clamped non-negative
                // (ground can push, not pull — without the clamp, on a high-
                // velocity rebound the damper could pull the body into the
                // ground, which is unphysical).
                let f_spring = stiffness * penetration;
                let f_damp = -damping * v_contact.y;
                let f_n = (f_spring + f_damp).max(0.0);
                let normal_force = Vec3::new(0.0, f_n, 0.0);
                self.bodies[body_idx].apply_force(normal_force);
                self.bodies[body_idx].apply_torque(contact_offset.cross(&normal_force));

                // Friction in the tangent plane (xz). Only meaningful if the
                // ground is actually pushing back (f_n > 0).
                if f_n > 0.0 && mu > 0.0 {
                    let v_t = Vec3::new(v_contact.x, 0.0, v_contact.z);
                    let v_t_speed = v_t.norm();
                    if v_t_speed > 1e-9 {
                        let f_drag = -friction_damping * v_t;
                        let f_drag_mag = f_drag.norm();
                        let f_friction_max = mu * f_n;
                        let friction_force = if f_drag_mag > f_friction_max {
                            f_drag * (f_friction_max / f_drag_mag)
                        } else {
                            f_drag
                        };
                        self.bodies[body_idx].apply_force(friction_force);
                        self.bodies[body_idx]
                            .apply_torque(contact_offset.cross(&friction_force));
                    }
                }
            }
        }
    }

    /// Read-only computation of which capsule endpoints of `body_idx`
    /// are currently pressing the ground, with the contact point and
    /// normal-force magnitude for each. The `[Option; 2]` slot order is
    /// (positive_Y_endpoint, negative_Y_endpoint) — same convention as
    /// `RigidBody::axis_endpoints_world`. An entry is `None` if that
    /// endpoint isn't penetrating, or if the bilateral damping clamped
    /// the normal force to zero (rebound). Returns `[None, None]` if
    /// the ground is disabled, the body has no capsule shape, or the
    /// body is pinned.
    pub fn engaged_endpoints(
        &self,
        body_idx: usize,
    ) -> [Option<EndpointContact>; 2] {
        let Some(plane_y) = self.ground.plane_y else {
            return [None, None];
        };
        let body = &self.bodies[body_idx];
        let shape = match body.capsule_shape {
            Some(s) => s,
            None => return [None, None],
        };
        if body.inv_mass == 0.0 {
            return [None, None];
        }

        let stiffness = self.ground.stiffness;
        let damping = self.ground.damping;
        let axis_world = body.orientation * Vec3::new(0.0, 1.0, 0.0);
        let half = shape.half_h;
        let r = shape.radius;
        let endpoint_offsets = [axis_world * half, -axis_world * half];

        let mut out = [None, None];
        for (i, endpoint_offset) in endpoint_offsets.iter().enumerate() {
            let endpoint_world = body.position + endpoint_offset;
            let bottom = endpoint_world.y - r;
            let penetration = plane_y - bottom;
            if penetration <= 0.0 {
                continue;
            }
            let contact_offset = endpoint_offset + Vec3::new(0.0, -r, 0.0);
            let v_contact = body.linear_velocity
                + body.angular_velocity.cross(&contact_offset);
            let f_spring = stiffness * penetration;
            let f_damp = -damping * v_contact.y;
            let f_n = (f_spring + f_damp).max(0.0);
            if f_n <= 0.0 {
                continue;
            }
            let contact_point = body.position + contact_offset;
            out[i] = Some(EndpointContact {
                contact_point,
                normal_force: f_n,
            });
        }
        out
    }

    /// Total upward normal force the ground is currently exerting on
    /// this body's capsule (sum across both endpoints). Zero if no
    /// endpoint is in active contact.
    pub fn total_normal_force(&self, body_idx: usize) -> f64 {
        self.engaged_endpoints(body_idx).iter()
            .filter_map(|e| e.as_ref().map(|c| c.normal_force))
            .sum()
    }

    /// True if any endpoint of `body_idx` is in active contact with
    /// the ground.
    pub fn is_in_contact(&self, body_idx: usize) -> bool {
        self.engaged_endpoints(body_idx).iter().any(|e| e.is_some())
    }

    fn apply_foot_contacts(&mut self) {
        for c in &self.foot_contacts {
            c.apply(&mut self.bodies);
        }
    }

    fn integrate_velocities(&mut self, dt: f64) {
        for body in &mut self.bodies {
            if body.inv_mass == 0.0 {
                continue;
            }
            let accel = GRAVITY + body.force_accumulator * body.inv_mass;
            body.linear_velocity += accel * dt;

            let alpha = body.world_inv_inertia() * body.torque_accumulator;
            body.angular_velocity += alpha * dt;
        }
    }

    /// Integrate positions using the (constraint-corrected) velocities.
    /// Replaces the old `predict_positions` step — the rename matches the
    /// SI scheme where positions are the *result* of the substep, not a
    /// prediction that gets corrected.
    fn integrate_positions(&mut self, dt: f64) {
        for body in &mut self.bodies {
            if body.inv_mass == 0.0 {
                continue;
            }
            body.position += body.linear_velocity * dt;

            let omega = body.angular_velocity;
            let omega_quat = UnitQuaternion::from_scaled_axis(omega * dt);
            body.orientation = omega_quat * body.orientation;
        }
    }

    fn solve_constraints(&mut self, dt: f64) {
        for i in 0..self.constraints.len() {
            let mut constraint = std::mem::replace(
                &mut self.constraints[i],
                PositionalConstraint::new(0, 0, 0.0, 0.0),
            );
            constraint.solve(&mut self.bodies, dt);
            self.constraints[i] = constraint;
        }
        for i in 0..self.angular_constraints.len() {
            let mut constraint = std::mem::replace(
                &mut self.angular_constraints[i],
                AngularLimitConstraint::new(0, 1, Vec3::z(), 0.0, 0.0),
            );
            constraint.solve(&mut self.bodies, dt);
            self.angular_constraints[i] = constraint;
        }
        for i in 0..self.contact_constraints.len() {
            let mut cc = std::mem::replace(
                &mut self.contact_constraints[i],
                ContactConstraint::new(0, Vec3::zeros()),
            );
            cc.solve(&mut self.bodies, dt);
            self.contact_constraints[i] = cc;
        }
    }

    fn clear_accumulators(&mut self) {
        for body in &mut self.bodies {
            body.clear_accumulators();
        }
    }
}

impl Default for XpbdWorld {
    fn default() -> Self {
        Self::new()
    }
}
