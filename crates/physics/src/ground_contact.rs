use crate::math::Vec3;
use crate::rigid_body::RigidBody;

/// Non-penetration contact between a capsule body and an infinite horizontal
/// ground plane at `plane_y`. Implemented as a penalty spring that activates
/// only when penetrating: the lower of the capsule's two end-spheres pushes
/// up along world +y with a stiff spring + damper.
///
/// Each endpoint is checked independently so a horizontal capsule (both ends
/// touching the ground) gets two contact forces — one at each end — and a
/// tilted capsule with only one end on the ground gets a single force at the
/// lower end (which produces the correct restoring torque about the body
/// center, so the capsule falls flat over time).
///
/// Sign convention: penetration > 0 when the capsule's bottom is below
/// `plane_y`. Spring force = stiffness × penetration, applied along +y.
/// Damper is bilateral while in contact (opposes contact-point velocity
/// in either normal direction), so energy is dissipated during *both* the
/// compression and rebound phases of an impact. With unidirectional
/// damping, the spring fully releases on rebound and the body bounces
/// forever; bilateral damping is what makes a drop actually settle.
///
/// **Friction model** (regularized Coulomb, in the xz tangent plane):
///   - Below a small velocity threshold the friction is viscous
///     (force = -friction_damping × v_tangent), which gives stick-like
///     behavior at rest without the discontinuity of true static/kinetic
///     Coulomb at v=0.
///   - Above the threshold the magnitude saturates at μ × |F_n|, which is
///     the kinetic Coulomb law. The transition is smooth (no jump).
///   - friction_damping = 1000 N·s/m caps to μ·F_n at v ≈ μ·g/k_d ≈
///     0.005 m/s for a 1 kg body — anything moving slower than 5 mm/s is
///     effectively stuck, anything faster slides under saturated kinetic
///     friction. Also keeps the explicit-Euler bound (c·dt/m < 2) safe
///     for contacting bodies down to ~0.1 kg.
///
/// **Lever arm**: both normal force and friction force are applied at the
/// actual ground-contact point (one radius below the end-sphere center),
/// not at the end-sphere center itself. This matters for friction torque:
/// a sliding vertical capsule will tip forward correctly because the
/// friction lever arm includes the radius offset.
///
/// No restitution — the damper takes kinetic energy out and the capsule
/// stops dead.
///
/// Stiffness/damping defaults (for a 1 kg body):
///   - K = 1.0e5 N/m → ω_n ≈ 316 rad/s
///     → static penetration depth at m·g equilibrium ≈ 0.1 mm (effectively
///       rigid for visual purposes)
///   - C = 600 N·s/m ≈ critical damping
///   - μ = 0.6 (dry rubber on turf, typical sports-shoe range 0.5–0.9)
///   - Re-tune K, C, and friction_damping for heavier bodies (70 kg on
///     one foot needs K ≥ 1e6 and friction_damping ≥ 5000 to keep the
///     transition velocity below ~5 mm/s).
pub struct GroundContact {
    pub body: usize,
    pub capsule_half_h: f64,
    pub capsule_radius: f64,
    /// World y-coordinate of the ground plane.
    pub plane_y: f64,
    /// Penalty spring stiffness (N/m) — opposes penetration.
    pub stiffness: f64,
    /// Damping coefficient (N·s/m) on contact-point normal velocity.
    /// Bilateral while in contact.
    pub damping: f64,
    /// Coulomb friction coefficient (dimensionless). 0 = frictionless ice.
    /// Typical sports values: 0.5–0.9 for rubber on turf/concrete.
    pub friction_coefficient: f64,
    /// Tangential viscous coefficient (N·s/m) used to regularize the
    /// Coulomb law near v=0. The friction force magnitude saturates at
    /// μ·|F_n|, so this only affects behavior at low tangential speed.
    pub friction_damping: f64,
}

impl GroundContact {
    pub fn new(body: usize, capsule_half_h: f64, capsule_radius: f64) -> Self {
        Self {
            body,
            capsule_half_h,
            capsule_radius,
            plane_y: 0.0,
            stiffness: 1.0e5,
            damping: 600.0,
            friction_coefficient: 0.6,
            friction_damping: 1000.0,
        }
    }

    pub fn with_plane_y(mut self, plane_y: f64) -> Self {
        self.plane_y = plane_y;
        self
    }

    pub fn with_stiffness(mut self, stiffness: f64) -> Self {
        self.stiffness = stiffness;
        self
    }

    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    pub fn with_friction(mut self, mu: f64) -> Self {
        self.friction_coefficient = mu;
        self
    }

    pub fn with_friction_damping(mut self, damping: f64) -> Self {
        self.friction_damping = damping;
        self
    }

    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        // Capsule axis in world: body's local +y rotated by orientation.
        let body = &bodies[self.body];
        if body.inv_mass == 0.0 {
            return; // Pinned body — contact would have no effect.
        }
        let axis_world = body.orientation * Vec3::new(0.0, 1.0, 0.0);
        let half = self.capsule_half_h;
        let r = self.capsule_radius;

        // Two endpoints of the capsule's central segment, in world.
        let endpoints = [axis_world * half, -axis_world * half];

        for endpoint_offset in endpoints {
            let endpoint_world = bodies[self.body].position + endpoint_offset;
            // Bottom of the end-sphere = endpoint center - radius (along +y).
            let bottom = endpoint_world.y - r;
            let penetration = self.plane_y - bottom;
            if penetration <= 0.0 {
                continue;
            }

            // Actual ground-contact point is one radius below the end-sphere
            // center, on the plane. Use this offset (not the end-sphere
            // offset) for both normal and friction force application so
            // that the friction lever arm is correct.
            let contact_offset = endpoint_offset + Vec3::new(0.0, -r, 0.0);

            // Velocity at the contact point (includes rotation contribution).
            let v_contact = bodies[self.body].linear_velocity
                + bodies[self.body].angular_velocity.cross(&contact_offset);

            // Normal force: spring + bilateral damping, clamped non-negative
            // (ground can push, not pull — without the clamp, on a high-
            // velocity rebound the damper could pull the body into the
            // ground, which is unphysical).
            let f_spring = self.stiffness * penetration;
            let f_damp = -self.damping * v_contact.y;
            let f_n = (f_spring + f_damp).max(0.0);
            let normal_force = Vec3::new(0.0, f_n, 0.0);
            bodies[self.body].apply_force(normal_force);
            bodies[self.body].apply_torque(contact_offset.cross(&normal_force));

            // Friction in the tangent plane (xz). Only meaningful if the
            // ground is actually pushing back (f_n > 0).
            if f_n > 0.0 && self.friction_coefficient > 0.0 {
                let v_t = Vec3::new(v_contact.x, 0.0, v_contact.z);
                let v_t_speed = v_t.norm();
                if v_t_speed > 1e-9 {
                    // Viscous-style drag, capped at μ·|F_n| (regularized
                    // Coulomb). At low speeds (below μ·F_n / friction_damping)
                    // friction is linear in v_t — gives stick-like behavior
                    // at rest. At higher speeds it saturates to kinetic
                    // Coulomb.
                    let f_drag = -self.friction_damping * v_t;
                    let f_drag_mag = f_drag.norm();
                    let f_friction_max = self.friction_coefficient * f_n;
                    let friction_force = if f_drag_mag > f_friction_max {
                        f_drag * (f_friction_max / f_drag_mag)
                    } else {
                        f_drag
                    };
                    bodies[self.body].apply_force(friction_force);
                    bodies[self.body].apply_torque(contact_offset.cross(&friction_force));
                }
            }
        }
    }
}
