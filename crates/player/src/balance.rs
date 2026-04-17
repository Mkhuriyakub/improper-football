use physics::math::Vec3;
use crate::skeleton::PlayerSkeleton;
use crate::segment::SegmentId;

pub struct BalanceController {
    pub kp: f64,
    pub kd: f64,
}

impl Default for BalanceController {
    fn default() -> Self {
        Self { kp: 1200.0, kd: 120.0 }
    }
}

impl BalanceController {
    /// Horizontal (XZ) midpoint of the two foot positions.
    pub fn support_centre(&self, skel: &PlayerSkeleton) -> Vec3 {
        let lf = skel.bodies[SegmentId::FootL.idx()].position;
        let rf = skel.bodies[SegmentId::FootR.idx()].position;
        Vec3::new((lf.x + rf.x) * 0.5, 0.0, (lf.z + rf.z) * 0.5)
    }

    /// Horizontal offset of CoM from support centre (XZ only).
    pub fn com_offset(&self, skel: &PlayerSkeleton) -> Vec3 {
        let com = skel.centre_of_mass();
        let sc = self.support_centre(skel);
        Vec3::new(com.x - sc.x, 0.0, com.z - sc.z)
    }

    /// Apply corrective torques to ankle and hip joints.
    /// Call once per physics substep before the XPBD solve.
    pub fn apply_corrections(&self, skel: &mut PlayerSkeleton, dt: f64) {
        let offset = self.com_offset(skel);
        let com_vel = skel.bodies[SegmentId::Pelvis.idx()].linear_velocity;
        let com_vel_xz = Vec3::new(com_vel.x, 0.0, com_vel.z);

        let correction_torque = -(self.kp * offset + self.kd * com_vel_xz);

        let ankle_torque = correction_torque * 0.5;
        skel.bodies[SegmentId::FootL.idx()].apply_torque(ankle_torque);
        skel.bodies[SegmentId::FootR.idx()].apply_torque(ankle_torque);
        skel.bodies[SegmentId::Pelvis.idx()].apply_torque(correction_torque * 0.3);

        let _ = dt;
    }
}
