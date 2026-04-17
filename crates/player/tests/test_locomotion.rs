use player::locomotion::{StrideGenerator, UpperBodyPose, Role, LegTargets};
use physics::math::Vec3;

#[test]
fn stationary_player_has_zero_stride_freq() {
    let gen = StrideGenerator::default();
    let targets = gen.compute(Vec3::zeros(), 0.0);
    assert!((targets.stride_freq).abs() < 1e-6);
}

#[test]
fn faster_speed_gives_higher_stride_freq() {
    let gen = StrideGenerator::default();
    let slow = gen.compute(Vec3::new(2.0, 0.0, 0.0), 0.0);
    let fast = gen.compute(Vec3::new(7.0, 0.0, 0.0), 0.0);
    assert!(fast.stride_freq > slow.stride_freq,
        "fast={} slow={}", fast.stride_freq, slow.stride_freq);
}

#[test]
fn legs_are_out_of_phase() {
    let gen = StrideGenerator::default();
    let targets = gen.compute(Vec3::new(5.0, 0.0, 0.0), 0.25); // t=0.25s
    // Hip flex angles should differ (out of phase)
    let diff = (targets.hip_flex_l - targets.hip_flex_r).abs();
    assert!(diff > 0.05, "legs should be out of phase, diff={}", diff);
}

#[test]
fn qb_pose_has_nonzero_arm_targets() {
    let pose = UpperBodyPose::for_role(Role::QuarterbackDropback);
    // QB in dropback has right arm raised
    assert!(pose.right_shoulder_flex > 0.1);
}
