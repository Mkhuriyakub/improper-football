use physics::math::Vec3;
use physics::rigid_body::RigidBody;
use crate::segment::SegmentId;

pub struct BalanceController;

impl BalanceController {
    /// Horizontal CoM-offset correction applied to ankles and pelvis each substep.
    pub fn apply(bodies: &mut Vec<RigidBody>, _dt: f64) {
        let lf = bodies[SegmentId::FootL.idx()].position;
        let rf = bodies[SegmentId::FootR.idx()].position;
        let sc = Vec3::new((lf.x + rf.x) * 0.5, 0.0, (lf.z + rf.z) * 0.5);

        let total_mass: f64 = bodies.iter().map(|b| b.mass).sum();
        let com: Vec3 = bodies.iter().map(|b| b.position * b.mass).sum::<Vec3>() / total_mass;

        let offset = Vec3::new(com.x - sc.x, 0.0, com.z - sc.z);
        let com_vel = bodies[SegmentId::Pelvis.idx()].linear_velocity;
        let com_vel_xz = Vec3::new(com_vel.x, 0.0, com_vel.z);

        let correction = -(1200.0 * offset + 120.0 * com_vel_xz);

        bodies[SegmentId::FootL.idx()].apply_torque(correction * 0.5);
        bodies[SegmentId::FootR.idx()].apply_torque(correction * 0.5);
        bodies[SegmentId::Pelvis.idx()].apply_torque(correction * 0.3);
    }
}

impl Default for BalanceController {
    fn default() -> Self { Self }
}
