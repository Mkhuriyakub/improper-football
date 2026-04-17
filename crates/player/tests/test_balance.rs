use player::balance::BalanceController;
use player::skeleton::PlayerSkeleton;
use player::segment::{BodyType, SegmentId};
use physics::math::Vec3;

#[test]
fn balanced_skeleton_has_near_zero_correction() {
    let skel = PlayerSkeleton::new(BodyType::Receiver, Vec3::zeros());
    let ctrl = BalanceController::default();
    let correction = ctrl.com_offset(&skel);
    assert!(correction.norm() < 0.05, "offset = {}", correction.norm());
}

#[test]
fn leaning_skeleton_has_nonzero_correction() {
    let mut skel = PlayerSkeleton::new(BodyType::Receiver, Vec3::zeros());
    skel.bodies[SegmentId::Chest.idx()].position.x += 0.2;
    let ctrl = BalanceController::default();
    let correction = ctrl.com_offset(&skel);
    assert!(correction.x > 0.01, "expected rightward offset, got {}", correction.x);
}

#[test]
fn support_centre_is_midpoint_of_feet() {
    let skel = PlayerSkeleton::new(BodyType::Lineman, Vec3::new(3.0, 0.0, 2.0));
    let ctrl = BalanceController::default();
    let sc = ctrl.support_centre(&skel);
    let lf = skel.bodies[SegmentId::FootL.idx()].position;
    let rf = skel.bodies[SegmentId::FootR.idx()].position;
    let expected = (lf + rf) * 0.5;
    assert!((sc.x - expected.x).abs() < 0.001);
    assert!((sc.z - expected.z).abs() < 0.001);
}
