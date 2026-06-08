use crate::math::{Vec2, Vec3};
use crate::rigid_body::RigidBody;

/// Read-only summary of one foot-contact point that's currently
/// pressing the ground. Returned by `FootContact::engaged_points`.
#[derive(Clone, Copy, Debug)]
pub struct FootContactPoint {
    /// World-frame position of the contact point on the ground plane.
    pub world_position: Vec3,
    /// Penetration depth (m) at this point.
    pub penetration: f64,
    /// Magnitude of the upward normal force at this point.
    pub normal_force: f64,
}

/// Multi-point ground contact for a foot body, modeling the
/// biomechanical TRIPOD: heel + medial ball (1st metatarsal head) +
/// lateral ball (5th metatarsal head). These three points form a
/// triangular support polygon that is geometrically faithful to a
/// human foot's ground contact area.
///
/// Why this exists: the world-level single-capsule ground contact makes
/// contact only at the two hemispherical endpoints (heel-tip + toe-tip), a
/// degenerate 1D line. Any chain rotation lifts one endpoint, the
/// "polygon" collapses, and bipedal disturbance recovery fails (see
/// `project_capsule_foot_limitation.md`). A 2D triangular footprint
/// gives the body real polygon support — the body can shift weight
/// between heel and ball without lifting the whole foot.
///
/// Each contact point is independent: its own penalty spring
/// + bilateral damper + regularized Coulomb friction. Points are
/// specified in body-local coordinates and transform with the foot's
/// orientation each substep.
///
/// **Local frame convention:** the foot's local axes are arbitrary
/// (user-defined via `local_points`). For the project's standard
/// capsule foot with `baseline = R_x(-π/2)`, the foot lies along the
/// world Z axis at neutral pose; the foot's "bottom" surface is at
/// local -Z (one radius below the central axis). The `tripod()`
/// constructor places points at this local-bottom plane.
pub struct FootContact {
    pub body: usize,
    /// Contact points in body-local frame. Anatomically meaningful
    /// names are user-provided via constructor choice; the apply
    /// logic treats all points uniformly.
    pub local_points: Vec<Vec3>,
    /// World y of the ground plane.
    pub plane_y: f64,
    /// Penalty spring stiffness (N/m), uniform across points.
    pub stiffness: f64,
    /// Bilateral damping (N·s/m on contact-point normal velocity).
    pub damping: f64,
    /// Coulomb friction coefficient.
    pub friction_coefficient: f64,
    /// Tangential viscous coefficient (regularizes Coulomb near v=0).
    pub friction_damping: f64,
}

impl FootContact {
    pub fn new(body: usize, local_points: Vec<Vec3>) -> Self {
        Self {
            body,
            local_points,
            plane_y: 0.0,
            stiffness: 1.0e5,
            damping: 600.0,
            friction_coefficient: 0.6,
            friction_damping: 1000.0,
        }
    }

    /// Biomechanical tripod: heel + medial ball + lateral ball. For a
    /// capsule foot of given `half_h` (length) and `radius`. Points
    /// placed on the foot's bottom surface (local -Z) at anatomically
    /// representative locations.
    ///
    /// `side_sign`: +1.0 for right foot, -1.0 for left foot. Used to
    /// place the medial-ball point on the inside (medial) of the body.
    /// For the right foot the medial side is -x; for the left, +x.
    pub fn tripod(body: usize, half_h: f64, radius: f64, side_sign: f64) -> Self {
        let medial_x = -side_sign * radius * 0.85;
        let lateral_x =  side_sign * radius * 0.85;
        // Heel: back of foot, on the bottom surface. Local +Y is back.
        let heel        = Vec3::new(0.0,        half_h * 0.85, -radius);
        // Ball-medial: forward, big-toe side. Local -Y is forward.
        let ball_medial = Vec3::new(medial_x,  -half_h * 0.65, -radius);
        // Ball-lateral: forward, little-toe side.
        let ball_lateral = Vec3::new(lateral_x, -half_h * 0.65, -radius);
        Self::new(body, vec![heel, ball_medial, ball_lateral])
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

    /// Apply contact + friction forces to the foot body. Called once
    /// per substep from `XpbdWorld::substep`.
    pub fn apply(&self, bodies: &mut Vec<RigidBody>) {
        let body = &bodies[self.body];
        if body.inv_mass == 0.0 {
            return;
        }
        let foot_pos = body.position;
        let foot_orient = body.orientation;

        for &local in &self.local_points {
            let world = foot_pos + foot_orient * local;
            let penetration = self.plane_y - world.y;
            if penetration <= 0.0 {
                continue;
            }
            // Contact-point offset from foot CG.
            let r = world - foot_pos;
            // Velocity at the contact point.
            let v_contact = bodies[self.body].linear_velocity
                + bodies[self.body].angular_velocity.cross(&r);

            // Normal force: spring + bilateral damper, non-negative.
            let f_spring = self.stiffness * penetration;
            let f_damp = -self.damping * v_contact.y;
            let f_n = (f_spring + f_damp).max(0.0);
            if f_n <= 0.0 {
                continue;
            }
            let normal_force = Vec3::new(0.0, f_n, 0.0);
            bodies[self.body].apply_force(normal_force);
            bodies[self.body].apply_torque(r.cross(&normal_force));

            // Friction (regularized Coulomb in xz tangent plane).
            if self.friction_coefficient > 0.0 {
                let v_t = Vec3::new(v_contact.x, 0.0, v_contact.z);
                let v_t_speed = v_t.norm();
                if v_t_speed > 1e-9 {
                    let f_drag = -self.friction_damping * v_t;
                    let f_drag_mag = f_drag.norm();
                    let f_friction_max = self.friction_coefficient * f_n;
                    let friction = if f_drag_mag > f_friction_max {
                        f_drag * (f_friction_max / f_drag_mag)
                    } else {
                        f_drag
                    };
                    bodies[self.body].apply_force(friction);
                    bodies[self.body].apply_torque(r.cross(&friction));
                }
            }
        }
    }

    /// Read-only computation of currently engaged contact points,
    /// with world positions and normal-force magnitudes. Empty if
    /// no point is pressing the ground.
    pub fn engaged_points(&self, bodies: &[RigidBody]) -> Vec<FootContactPoint> {
        let body = &bodies[self.body];
        if body.inv_mass == 0.0 {
            return Vec::new();
        }
        let foot_pos = body.position;
        let foot_orient = body.orientation;
        let mut out = Vec::with_capacity(self.local_points.len());
        for &local in &self.local_points {
            let world = foot_pos + foot_orient * local;
            let penetration = self.plane_y - world.y;
            if penetration <= 0.0 {
                continue;
            }
            let r = world - foot_pos;
            let v_contact = body.linear_velocity + body.angular_velocity.cross(&r);
            let f_spring = self.stiffness * penetration;
            let f_damp = -self.damping * v_contact.y;
            let f_n = (f_spring + f_damp).max(0.0);
            if f_n <= 0.0 {
                continue;
            }
            out.push(FootContactPoint {
                world_position: world,
                penetration,
                normal_force: f_n,
            });
        }
        out
    }

    /// Total upward normal force on the foot from all engaged points.
    pub fn total_normal_force(&self, bodies: &[RigidBody]) -> f64 {
        self.engaged_points(bodies).iter().map(|p| p.normal_force).sum()
    }

    /// True if any contact point is in active contact.
    pub fn is_in_contact(&self, bodies: &[RigidBody]) -> bool {
        !self.engaged_points(bodies).is_empty()
    }

    /// Pressure centroid: force-weighted average of engaged contact
    /// point positions (the "center of pressure" — a standard biomech
    /// quantity that approximates the ZMP under quasi-static conditions).
    /// Returns `None` if no points are in contact.
    pub fn pressure_centroid(&self, bodies: &[RigidBody]) -> Option<Vec2> {
        let pts = self.engaged_points(bodies);
        if pts.is_empty() {
            return None;
        }
        let total_force: f64 = pts.iter().map(|p| p.normal_force).sum();
        if total_force <= 0.0 {
            return None;
        }
        let weighted: Vec2 = pts.iter()
            .map(|p| Vec2::new(p.world_position.x, p.world_position.z) * p.normal_force)
            .sum();
        Some(weighted / total_force)
    }

    /// World-XZ positions of all engaged contact points — convenient
    /// input to convex hull for support polygon construction.
    pub fn support_points_xz(&self, bodies: &[RigidBody]) -> Vec<Vec2> {
        self.engaged_points(bodies).iter()
            .map(|p| Vec2::new(p.world_position.x, p.world_position.z))
            .collect()
    }
}
