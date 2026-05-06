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
use crate::ground_contact::GroundContact;
use crate::math::Vec3;
use nalgebra::UnitQuaternion;

pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
pub const SOLVER_ITERATIONS: usize = 30;
/// Default number of sequential-impulse iterations per substep. Each iteration
/// is one Gauss-Seidel pass over all positional and angular constraints,
/// refining the impulses that satisfy the velocity-level constraints.
pub const DEFAULT_SI_ITERATIONS: usize = 8;

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
    pub ground_contacts: Vec<GroundContact>,
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
            ground_contacts: Vec::new(),
            solver_iterations: DEFAULT_SI_ITERATIONS,
        }
    }

    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        id
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
        self.apply_positional_springs();
        self.apply_positional_dampers();
        self.apply_tendons();
        self.apply_ground_contacts();
        self.integrate_velocities(sub_dt);
        // Damping applied implicitly AFTER explicit velocity update — this is
        // unconditionally stable for any damping coefficient, fixing the
        // explicit-Euler instability that affected small-inertia bodies.
        self.apply_dampers_implicit(sub_dt);
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

    fn apply_ground_contacts(&mut self) {
        for c in &self.ground_contacts {
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
