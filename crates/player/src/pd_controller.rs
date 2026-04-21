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
        JointKind::HipKnee    => PdController { kp: 2000.0, kd: 120.0 },
        JointKind::Ankle      => PdController { kp: 1500.0, kd:  90.0 },
        JointKind::Spine      => PdController { kp: 3000.0, kd: 180.0 },
        JointKind::Shoulder   => PdController { kp: 1200.0, kd:  80.0 },
        JointKind::ElbowWrist => PdController { kp:  600.0, kd:  40.0 },
        JointKind::Neck       => PdController { kp: 1200.0, kd:  80.0 },
    }
}
