use crate::rigid_body::RigidBody;
use crate::constraints::PositionalConstraint;
use crate::angular_constraint::AngularLimitConstraint;
use crate::math::Vec3;
use nalgebra::UnitQuaternion;

pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
pub const SOLVER_ITERATIONS: usize = 30;

pub struct XpbdWorld {
    pub bodies: Vec<RigidBody>,
    pub constraints: Vec<PositionalConstraint>,
    pub angular_constraints: Vec<AngularLimitConstraint>,
}

impl XpbdWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            constraints: Vec::new(),
            angular_constraints: Vec::new(),
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

    /// Run one substep: integrate velocities → predict positions → solve constraints → clear.
    /// Call this in a loop with external force application between iterations for per-substep
    /// PD control (forces must be re-applied each call since clear_accumulators is called here).
    pub fn substep(&mut self, sub_dt: f64) {
        self.integrate_velocities(sub_dt);
        self.predict_positions(sub_dt);
        self.reset_lambdas();
        self.solve_constraints(sub_dt);
        // No velocity feedback from constraint corrections: the spine spring kd provides
        // sufficient absolute-frame damping. The forward-lean initial torque (which caused
        // prior ω accumulation) is eliminated by scaling lean with stride frequency.
        self.clear_accumulators();
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

    fn predict_positions(&mut self, dt: f64) {
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

    fn reset_lambdas(&mut self) {
        for c in &mut self.constraints {
            c.reset_lambda();
        }
        for c in &mut self.angular_constraints {
            c.reset_lambda();
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
