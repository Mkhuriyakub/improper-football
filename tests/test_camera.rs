use improper_football::plugins::camera::{spring_follow, CameraConfig};
use bevy::math::Vec3;

#[test]
fn spring_follow_moves_toward_target() {
    let current = Vec3::ZERO;
    let target = Vec3::new(10.0, 5.0, 0.0);
    let result = spring_follow(current, target, 0.1);
    assert!(result.x > 0.0 && result.x < 10.0);
    assert!(result.y > 0.0 && result.y < 5.0);
}

#[test]
fn spring_follow_at_alpha_one_reaches_target() {
    let current = Vec3::ZERO;
    let target = Vec3::new(5.0, 3.0, 1.0);
    let result = spring_follow(current, target, 1.0);
    assert!((result.x - 5.0).abs() < 1e-5);
    assert!((result.y - 3.0).abs() < 1e-5);
}

#[test]
fn camera_config_default_offset_is_above_and_behind() {
    let cfg = CameraConfig::default();
    assert!(cfg.offset.y > 0.0, "camera should be above the field");
}
