use physics::math::{Vec3, Mat3};
use physics::rigid_body::RigidBody;
use crate::aerodynamics::{drag_force, magnus_force, BALL_MASS, BALL_LENGTH, BALL_SHORT_RADIUS};

/// Inertia tensor for a prolate spheroid (football shape).
/// Ix = Iy = m/5 * (a² + b²)   (transverse axes)
/// Iz = 2m/5 * b²               (symmetry/spin axis)
/// where a = half-length, b = equatorial radius
fn football_inertia() -> Mat3 {
    let a = BALL_LENGTH / 2.0;
    let b = BALL_SHORT_RADIUS;
    let i_transverse = BALL_MASS / 5.0 * (a * a + b * b);
    let i_spin = 2.0 * BALL_MASS / 5.0 * (b * b);
    Mat3::from_diagonal(&nalgebra::Vector3::new(i_transverse, i_spin, i_transverse))
}

pub struct Football {
    pub body: RigidBody,
}

impl Football {
    pub fn new() -> Self {
        Self {
            body: RigidBody::new(BALL_MASS, football_inertia()),
        }
    }

    /// Launch the ball as a spiral throw.
    /// `release_velocity`: velocity vector at release
    /// `spin_rate_rpm`: how tight the spiral (typical NFL throw: 600 RPM)
    pub fn throw_spiral(&mut self, release_velocity: Vec3, spin_rate_rpm: f64) {
        self.body.linear_velocity = release_velocity;

        let v_norm = release_velocity.norm();
        if v_norm > 1e-6 {
            // Spin axis aligned with velocity (tight spiral)
            let spin_axis = release_velocity / v_norm;
            let spin_rad_s = spin_rate_rpm * 2.0 * std::f64::consts::PI / 60.0;
            self.body.angular_velocity = spin_axis * spin_rad_s;
        }
    }

    /// Spin axis derived from current angular velocity (unit vector).
    pub fn spin_axis(&self) -> Vec3 {
        let omega = self.body.angular_velocity;
        let norm = omega.norm();
        if norm > 1e-6 { omega / norm } else { Vec3::y() }
    }

    /// Accumulate aerodynamic forces for this frame.
    pub fn apply_aerodynamics(&mut self) {
        let v = self.body.linear_velocity;
        let omega = self.body.angular_velocity;
        let axis = self.spin_axis();

        let drag = drag_force(v, axis);
        let magnus = magnus_force(v, omega);

        self.body.apply_force(drag + magnus);
    }

    /// Advance the ball by dt seconds (standalone, without full XPBD world).
    pub fn step(&mut self, dt: f64) {
        self.apply_aerodynamics();

        // Gravity
        self.body.apply_force(Vec3::new(0.0, -9.81 * BALL_MASS, 0.0));

        // Symplectic Euler: linear motion
        let accel = self.body.force_accumulator * self.body.inv_mass;
        self.body.linear_velocity += accel * dt;
        self.body.position += self.body.linear_velocity * dt;

        // Spin axis precession: a real spiral gyroscopically resists reorientation.
        // Rather than integrating τ = ω×(I·ω) (which blows up at dt=0.01),
        // we model the net aerodynamic pitching moment that slowly aligns the
        // nose toward velocity — the stable long-term behaviour of a tight spiral.
        let v = self.body.linear_velocity;
        let v_norm = v.norm();
        if v_norm > 1e-6 {
            let spin_axis = self.spin_axis();
            let v_hat = v / v_norm;

            // Aerodynamic restoring moment: nudge spin axis toward velocity direction.
            // Rate constant tuned so a 600 RPM spiral stays tight over ~40 m.
            let correction = v_hat - spin_axis * spin_axis.dot(&v_hat);
            let correction_norm = correction.norm();
            if correction_norm > 1e-6 {
                let spin_rate = self.body.angular_velocity.norm();
                // Angular correction capped to avoid over-rotation per step
                let max_correction = 2.0_f64.to_radians() * dt * 60.0;
                let step = (0.8 * correction_norm * dt).min(max_correction);
                let new_axis = (spin_axis + correction / correction_norm * step).normalize();
                self.body.angular_velocity = new_axis * spin_rate;
            }
        }

        // Update orientation from angular velocity
        use nalgebra::UnitQuaternion;
        let omega = self.body.angular_velocity;
        let omega_quat = UnitQuaternion::from_scaled_axis(omega * dt);
        self.body.orientation = omega_quat * self.body.orientation;

        self.body.clear_accumulators();
    }
}

impl Default for Football {
    fn default() -> Self {
        Self::new()
    }
}
