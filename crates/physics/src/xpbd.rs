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
/// Contact itself is resolved by the multi-point SI non-penetration
/// manifold (`XpbdWorld::generate_ground_manifold` + `ContactConstraint`),
/// which is rigid and needs no stiffness/damping tuning. `friction_coefficient`
/// is the Coulomb μ handed to each manifold constraint. The `stiffness`,
/// `damping`, and `friction_damping` fields are legacy penalty-model
/// parameters retained for API compatibility; the manifold ignores them.
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
    /// Per-substep ground contact manifold: one `ContactConstraint` per
    /// capsule endpoint penetrating (or about to penetrate) the ground
    /// plane. Regenerated fresh every substep by `generate_ground_manifold`
    /// and solved together in the SI loop, so a resting body settles onto
    /// multiple simultaneous load-sharing contacts (a stable manifold)
    /// instead of micro-bouncing on its single deepest point. Kept
    /// separate from the user-facing `contact_constraints`.
    pub ground_manifold: Vec<ContactConstraint>,
    /// World-level ground. Every body with `capsule_shape: Some(...)`
    /// collides against this plane each substep.
    pub ground: Ground,
    /// Number of Gauss-Seidel sequential-impulse iterations per substep.
    pub solver_iterations: usize,
    /// Substep dt of the most recent `substep` call. Lets the read-only
    /// `engaged_endpoints` convert a converged manifold impulse
    /// (`lambda_normal`) into a normal force (`lambda_normal / last_sub_dt`).
    last_sub_dt: f64,
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
            ground_manifold: Vec::new(),
            ground: Ground::at_y(0.0),
            solver_iterations: DEFAULT_SI_ITERATIONS,
            last_sub_dt: 0.0,
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
        self.last_sub_dt = sub_dt;
        self.apply_pd_controllers();
        self.apply_torque_dispatchers();
        self.apply_tendons();
        self.apply_foot_contacts();
        self.integrate_velocities(sub_dt);
        // Ground contact is now a multi-point manifold of unilateral SI
        // non-penetration constraints (one per penetrating capsule endpoint),
        // solved together in the SI loop below. This replaces the old
        // per-endpoint penalty impulse, which collapsed a body's geometric
        // footprint to an intermittent single deepest contact and let bodies
        // tip even with the CoM over the base. SI Gauss-Seidel iterations
        // converge to a consistent load-sharing solution across all contacts,
        // so the body settles onto a stable manifold (no penalty bounce, and
        // unconditionally stable for any contact mass). Generated here, BEFORE
        // the SI loop, with zero accumulated impulse — no separate reset
        // needed since it is rebuilt fresh every substep.
        self.generate_ground_manifold();
        // Springs, soft-limits, and dampers applied implicitly AFTER the
        // explicit velocity update — unconditionally stable for any
        // stiffness/damping coefficient, fixing the explicit-Euler instability
        // that affected small-inertia bodies. Passive AngularSprings,
        // AngularSoftLimits, AngularDampers, the KD-term of PD controllers, and
        // the linear PositionalSprings/PositionalDampers are all integrated
        // here. The PositionalSpring lever-arm torque in particular is the one
        // that exploded the foot's small bones when applied explicitly.
        self.apply_positional_springs_implicit(sub_dt);
        self.apply_springs_implicit(sub_dt);
        self.apply_soft_limits_implicit(sub_dt);
        self.apply_dampers_implicit(sub_dt);
        self.apply_pd_dampers_implicit(sub_dt);
        self.apply_positional_dampers_implicit(sub_dt);
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

    fn apply_springs_implicit(&mut self, sub_dt: f64) {
        for spring in &self.angular_springs {
            spring.apply_implicit(&mut self.bodies, sub_dt);
        }
    }

    fn apply_soft_limits_implicit(&mut self, sub_dt: f64) {
        for limit in &mut self.angular_soft_limits {
            limit.apply_implicit(&mut self.bodies, sub_dt);
        }
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

    fn apply_positional_springs_implicit(&mut self, sub_dt: f64) {
        for s in &self.positional_springs {
            s.apply_implicit(&mut self.bodies, sub_dt);
        }
    }

    fn apply_positional_dampers_implicit(&mut self, sub_dt: f64) {
        for d in &self.positional_dampers {
            d.apply_implicit(&mut self.bodies, sub_dt);
        }
    }

    /// Margin (m) within which an endpoint is included in the manifold even
    /// before it actually penetrates. The constraint stays inactive
    /// (`lambda_normal == 0`) until the endpoint truly penetrates, but
    /// pre-generating it lets the SI solver catch fast-approaching contacts
    /// within the substep rather than one substep late.
    const MANIFOLD_MARGIN: f64 = 1.0e-3;

    /// Regenerate the per-substep ground contact manifold.
    ///
    /// For every body whose `capsule_shape` is `Some(...)` and whose
    /// `inv_mass` is non-zero (pinned bodies pass through — contact would
    /// have no effect), check both capsule endpoints against the ground
    /// plane. For each endpoint within `MANIFOLD_MARGIN` of (or below) the
    /// plane, push a fresh rigid `ContactConstraint` anchored at the
    /// end-sphere bottom (expressed in body-local coordinates). These are
    /// then solved together in the SI loop, yielding a consistent
    /// multi-point load-sharing solution instead of the old independent
    /// per-endpoint penalty impulses.
    ///
    /// Because the manifold is rebuilt every substep, the fixed
    /// material-point (local_point) approximation only needs to hold for a
    /// single substep — acceptable for the small per-substep rotation.
    fn generate_ground_manifold(&mut self) {
        self.ground_manifold.clear();
        let Some(plane_y) = self.ground.plane_y else {
            return;
        };
        let mu = self.ground.friction_coefficient;

        for body_idx in 0..self.bodies.len() {
            let shape = match self.bodies[body_idx].capsule_shape {
                Some(s) => s,
                None => continue,
            };
            if self.bodies[body_idx].inv_mass == 0.0 {
                continue;
            }

            let position = self.bodies[body_idx].position;
            let orientation = self.bodies[body_idx].orientation;
            let axis_world = orientation * Vec3::new(0.0, 1.0, 0.0);
            let half = shape.half_h;
            let r = shape.radius;

            // Two endpoints of the capsule's central segment, in world.
            let endpoints = [axis_world * half, -axis_world * half];

            for endpoint_offset in endpoints {
                let endcenter_world = position + endpoint_offset;
                // Bottom of the end-sphere = endpoint center - radius (along +y).
                let bottom_y = endcenter_world.y - r;
                let penetration = plane_y - bottom_y;
                if penetration <= -Self::MANIFOLD_MARGIN {
                    continue;
                }

                // Contact point in world = end-sphere bottom. Express it in
                // body-local so the constraint tracks the material point as
                // the body rotates within this substep.
                let contact_world = endcenter_world - Vec3::new(0.0, r, 0.0);
                let local_point = orientation.inverse() * (contact_world - position);

                // Baumgarte = 0 (no separating-velocity bias). The default
                // Baumgarte term targets a positive (upward) normal velocity
                // proportional to penetration at each contact. For a SINGLE
                // body that just gently pushes out penetration, but for a
                // RIGID MULTI-CONTACT manifold (the whole point here) the
                // contacts sit at slightly different, non-coplanar penetration
                // depths; forcing each one's upward bias simultaneously through
                // the rigid-body coupling levitates the assembly and pumps in
                // real kinetic energy every substep (measured: a 108 N rigid
                // foot gained ~0.16 m/s upward per substep and floated off the
                // ground, then tipped). With bias = 0 the normal constraint is
                // purely a non-penetration velocity floor (v_n ≥ 0) — energy
                // dissipative, never injecting. Penetration never accumulates
                // because the floor stops downward velocity before it deepens.
                self.ground_manifold.push(
                    ContactConstraint::new(body_idx, local_point)
                        .with_plane_y(plane_y)
                        .with_friction(mu)
                        .with_baumgarte(0.0),
                );
            }
        }
    }

    /// Read-only report of which capsule endpoints of `body_idx` are
    /// currently load-bearing on the ground, with the contact point and
    /// normal-force magnitude for each. The `[Option; 2]` slot order is
    /// (positive_Y_endpoint, negative_Y_endpoint) — same convention as
    /// `RigidBody::axis_endpoints_world`.
    ///
    /// This reads the converged ground-contact manifold from the most
    /// recent substep: an endpoint reports `Some` only if it has an active
    /// manifold constraint (`lambda_normal > 0`, i.e. the ground is
    /// actually pushing there). `normal_force` is the converged impulse
    /// divided by the last substep dt (`lambda_normal / last_sub_dt`).
    /// Returns `[None, None]` if the ground is disabled, the body has no
    /// capsule shape, the body is pinned, or no substep has run yet.
    pub fn engaged_endpoints(
        &self,
        body_idx: usize,
    ) -> [Option<EndpointContact>; 2] {
        if self.ground.plane_y.is_none() || self.last_sub_dt <= 0.0 {
            return [None, None];
        }
        let body = &self.bodies[body_idx];
        let shape = match body.capsule_shape {
            Some(s) => s,
            None => return [None, None],
        };
        if body.inv_mass == 0.0 {
            return [None, None];
        }

        // Match each active manifold constraint to an endpoint by which
        // endpoint center its world contact point is closest to (slot order
        // positive_Y, negative_Y — same as the manifold generation order).
        let axis_local = Vec3::new(0.0, shape.half_h, 0.0);
        let endpoint_centers_world = [
            body.position + body.orientation * axis_local,
            body.position - body.orientation * axis_local,
        ];

        let mut out: [Option<EndpointContact>; 2] = [None, None];
        for c in &self.ground_manifold {
            if c.body != body_idx || c.lambda_normal <= 0.0 {
                continue;
            }
            let contact_point = c.contact_point_world(&self.bodies);
            // Assign to the nearer endpoint center.
            let d0 = (contact_point - endpoint_centers_world[0]).norm();
            let d1 = (contact_point - endpoint_centers_world[1]).norm();
            let slot = if d0 <= d1 { 0 } else { 1 };
            let ec = EndpointContact {
                contact_point,
                normal_force: c.lambda_normal / self.last_sub_dt,
            };
            // If two active constraints map to the same slot (shouldn't
            // happen with 2 endpoints), keep the larger-force one.
            match out[slot] {
                Some(existing) if existing.normal_force >= ec.normal_force => {}
                _ => out[slot] = Some(ec),
            }
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
        // Ground contact manifold — the multi-point non-penetration solve.
        // Generated fresh each substep in `generate_ground_manifold`; solved
        // alongside the user-facing contacts so a body settles onto all its
        // load-bearing endpoints simultaneously.
        for i in 0..self.ground_manifold.len() {
            let mut cc = std::mem::replace(
                &mut self.ground_manifold[i],
                ContactConstraint::new(0, Vec3::zeros()),
            );
            cc.solve(&mut self.bodies, dt);
            self.ground_manifold[i] = cc;
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
