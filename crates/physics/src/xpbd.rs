use crate::rigid_body::RigidBody;
use crate::constraints::PositionalConstraint;
use crate::math::Vec3;
use nalgebra::UnitQuaternion;

pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
pub const SOLVER_ITERATIONS: usize = 10;

pub struct XpbdWorld {
    pub bodies: Vec<RigidBody>,
    pub constraints: Vec<PositionalConstraint>,
}

impl XpbdWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            constraints: Vec::new(),
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
            self.integrate_velocities(sub_dt);
            self.predict_positions(sub_dt);
            self.reset_lambdas();
            self.solve_constraints(sub_dt);
            self.update_velocities(sub_dt);
            self.clear_accumulators();
        }
    }

    fn integrate_velocities(&mut self, dt: f64) {
        for body in &mut self.bodies {
            if body.inv_mass == 0.0 {
                continue;
            }
            // Gravity + accumulated forces
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
    }

    fn solve_constraints(&mut self, dt: f64) {
        for i in 0..self.constraints.len() {
            // Temporarily move constraint out to satisfy borrow checker
            let mut constraint = std::mem::replace(
                &mut self.constraints[i],
                PositionalConstraint::new(0, 0, 0.0, 0.0),
            );
            constraint.solve(&mut self.bodies, dt);
            self.constraints[i] = constraint;
        }
    }

    fn update_velocities(&mut self, dt: f64) {
        // Velocities are derived from position deltas in full XPBD;
        // here we rely on integrate_velocities for simplicity in this phase.
        // Full velocity update (for restitution/friction) comes in Phase 3.
        let _ = dt;
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
