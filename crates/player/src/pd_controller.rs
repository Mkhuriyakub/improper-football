use physics::math::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct PdController {
    pub kp: f64,
    pub kd: f64,
}

impl PdController {
    /// Compute joint torque.
    ///
    /// - `orientation_error`: rotation vector from current to target (axis × angle, world space)
    /// - `relative_angular_vel`: angular velocity of child minus parent (world space)
    /// - `_joint_pos`: reserved for future angular-limit clamping
    pub fn compute_torque(
        &self,
        orientation_error: Vec3,
        relative_angular_vel: Vec3,
        _joint_pos: Vec3,
    ) -> Vec3 {
        self.kp * orientation_error - self.kd * relative_angular_vel
    }
}

#[derive(Debug, Clone, Copy)]
pub enum JointKind {
    HipKnee,
    Ankle,
    Spine,
    Shoulder,
    ElbowWrist,
    Neck,
}

pub fn default_pd(joint_type: JointKind) -> PdController {
    match joint_type {
        JointKind::HipKnee    => PdController { kp: 800.0, kd: 80.0 },
        JointKind::Ankle      => PdController { kp: 600.0, kd: 60.0 },
        JointKind::Spine      => PdController { kp: 400.0, kd: 40.0 },
        JointKind::Shoulder   => PdController { kp: 300.0, kd: 30.0 },
        JointKind::ElbowWrist => PdController { kp: 150.0, kd: 15.0 },
        JointKind::Neck       => PdController { kp: 200.0, kd: 20.0 },
    }
}
