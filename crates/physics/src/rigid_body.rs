use crate::math::{Vec3, Quat, Mat3};
use nalgebra::UnitQuaternion;

/// Collision geometry for a capsule body. The capsule's central segment
/// runs along the body's local +Y axis with length `2 * half_h`, and each
/// end is capped by a sphere of radius `radius`. Used by the world-level
/// ground-contact system to detect penetration.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleShape {
    pub half_h: f64,
    pub radius: f64,
}

impl CapsuleShape {
    pub fn new(half_h: f64, radius: f64) -> Self {
        Self { half_h, radius }
    }
}

pub struct RigidBody {
    pub position: Vec3,
    pub orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f64,
    pub inv_mass: f64,
    pub inertia_tensor: Mat3,
    pub inv_inertia_tensor: Mat3,
    pub force_accumulator: Vec3,
    pub torque_accumulator: Vec3,
    /// Optional capsule collision shape. When present, the world's
    /// ground-contact system applies penalty contact to this body.
    /// `None` means the body is not collidable against the ground
    /// (e.g., abstract bodies, pinned anchors, or geometry the user
    /// hasn't characterized yet).
    pub capsule_shape: Option<CapsuleShape>,
}

impl RigidBody {
    pub fn new(mass: f64, inertia_tensor: Mat3) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        // A body constructed with mass <= 0 is pinned (kinematically fixed).
        // Its inverse inertia must be zeroed along with inv_mass: a pinned
        // body with a non-zero inv_inertia_tensor is a "phantom inertia" —
        // SI constraints would deposit angular velocity into it that
        // integrate_positions never consumes (it skips inv_mass == 0 bodies),
        // so the phantom spin persists forever and contaminates every
        // omega_rel read by dampers, springs, PDs, and the constraints'
        // own velocity error.
        let inv_inertia = if mass > 0.0 {
            inertia_tensor.try_inverse().unwrap_or(Mat3::zeros())
        } else {
            Mat3::zeros()
        };
        Self {
            position: Vec3::zeros(),
            orientation: UnitQuaternion::identity(),
            linear_velocity: Vec3::zeros(),
            angular_velocity: Vec3::zeros(),
            mass,
            inv_mass,
            inertia_tensor,
            inv_inertia_tensor: inv_inertia,
            force_accumulator: Vec3::zeros(),
            torque_accumulator: Vec3::zeros(),
            capsule_shape: None,
        }
    }

    /// Construct a capsule rigid body. The capsule's central segment
    /// runs along local +Y with length `2 * half_h`, capped by spheres
    /// of `radius` on each end. The inertia tensor is the standard
    /// solid-cylinder approximation (the rounded caps contribute a
    /// small amount that is ignored — adequate for the body sizes used
    /// in this project).
    ///
    ///   - i_axial   = (1/2) m r²              (about local Y)
    ///   - i_trans   = (1/12) m (3 r² + h²)    (about local X and Z)
    ///
    /// The resulting body has `capsule_shape = Some(...)`, so the
    /// world-level ground-contact system will collide it with the
    /// ground automatically.
    pub fn new_capsule(mass: f64, half_h: f64, radius: f64) -> Self {
        let h = 2.0 * half_h;
        let i_axial = 0.5 * mass * radius * radius;
        let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
        let inertia = Mat3::new(
            i_trans, 0.0, 0.0,
            0.0, i_axial, 0.0,
            0.0, 0.0, i_trans,
        );
        let mut body = Self::new(mass, inertia);
        body.capsule_shape = Some(CapsuleShape::new(half_h, radius));
        body
    }

    /// Pin this body: make it kinematically fixed (infinite mass AND infinite
    /// inertia) and at rest. This is the one correct way to pin an
    /// already-constructed dynamic body.
    ///
    /// Both inverse tensors must be zeroed, not just `inv_mass`: the
    /// integrators and the implicit primitives key off `inv_mass == 0.0`,
    /// but the SI constraint solvers apply angular impulses through
    /// `inv_inertia_tensor` directly. Leaving the dynamic body's inverse
    /// inertia in place creates a phantom inertia — the solver deposits
    /// angular velocity into the pinned body (never cleared, since
    /// integrate_positions skips it), children are driven to track the
    /// phantom spin, and impulses meant to be absorbed rigidly are split
    /// with it, weakening welds to the pinned root. Velocities are zeroed
    /// so any motion the body had at pin time doesn't persist as a frozen,
    /// never-integrated velocity that constraints still read.
    pub fn pin(&mut self) {
        self.inv_mass = 0.0;
        self.inv_inertia_tensor = Mat3::zeros();
        self.linear_velocity = Vec3::zeros();
        self.angular_velocity = Vec3::zeros();
    }

    pub fn apply_force(&mut self, force: Vec3) {
        self.force_accumulator += force;
    }

    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque_accumulator += torque;
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, contact_point: Vec3) {
        self.linear_velocity += impulse * self.inv_mass;
        let r = contact_point - self.position;
        self.angular_velocity += self.world_inv_inertia() * r.cross(&impulse);
    }

    pub fn world_inv_inertia(&self) -> Mat3 {
        let r = self.orientation.to_rotation_matrix();
        r.matrix() * self.inv_inertia_tensor * r.matrix().transpose()
    }

    pub fn clear_accumulators(&mut self) {
        self.force_accumulator = Vec3::zeros();
        self.torque_accumulator = Vec3::zeros();
    }

    /// World-frame positions of the two endpoints of the body's local
    /// +Y axis at the given half-length. For a capsule whose axis is
    /// the local Y axis (the project's convention), `half_length`
    /// should be the capsule's half-height, and the returned points
    /// are the centers of the two rounded caps.
    ///
    /// Returns `(positive_end, negative_end)`: positive_end is the
    /// endpoint on the local +Y side, negative_end on the -Y side.
    /// Whether one of these is "toe" or "heel" depends on the body's
    /// orientation — for a foot at horizontal anatomical pose
    /// (`foot_baseline = R_x(-π/2)`), local +Y maps to world -Z, so
    /// `positive_end` is the heel and `negative_end` is the toe.
    pub fn axis_endpoints_world(&self, half_length: f64) -> (Vec3, Vec3) {
        let local_pos = Vec3::new(0.0, half_length, 0.0);
        let local_neg = Vec3::new(0.0, -half_length, 0.0);
        let world_pos = self.position + self.orientation * local_pos;
        let world_neg = self.position + self.orientation * local_neg;
        (world_pos, world_neg)
    }
}
