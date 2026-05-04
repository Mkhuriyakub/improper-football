use improper_football::render_util::{nalgebra_vec3_to_bevy, nalgebra_quat_to_bevy, rigid_body_to_transform};
use nalgebra::{Vector3, UnitQuaternion};
use bevy::math::Vec3;
use physics::rigid_body::RigidBody;
use physics::math::Mat3;

fn unit_body() -> RigidBody {
    RigidBody::new(1.0, Mat3::identity())
}

#[test]
fn vec3_zero_converts_to_bevy_zero() {
    let v = nalgebra_vec3_to_bevy(Vector3::zeros());
    assert_eq!(v, Vec3::ZERO);
}

#[test]
fn vec3_values_cast_to_f32() {
    let v = nalgebra_vec3_to_bevy(Vector3::new(1.0_f64, 2.0, 3.0));
    assert!((v.x - 1.0_f32).abs() < 1e-6);
    assert!((v.y - 2.0_f32).abs() < 1e-6);
    assert!((v.z - 3.0_f32).abs() < 1e-6);
}

#[test]
fn identity_quaternion_converts_to_bevy_identity() {
    let q = nalgebra_quat_to_bevy(UnitQuaternion::identity());
    assert!((q.w - 1.0_f32).abs() < 1e-6);
    assert!(q.x.abs() < 1e-6);
    assert!(q.y.abs() < 1e-6);
    assert!(q.z.abs() < 1e-6);
}

#[test]
fn rigid_body_at_origin_gives_identity_transform() {
    let body = unit_body();
    let t = rigid_body_to_transform(&body);
    assert_eq!(t.translation, Vec3::ZERO);
    assert!((t.rotation.w - 1.0_f32).abs() < 1e-6);
    assert_eq!(t.scale, Vec3::ONE);
}

#[test]
fn rigid_body_position_maps_to_translation() {
    let mut body = unit_body();
    body.position = Vector3::new(3.0, 1.5, -2.0);
    let t = rigid_body_to_transform(&body);
    assert!((t.translation.x - 3.0_f32).abs() < 1e-4);
    assert!((t.translation.y - 1.5_f32).abs() < 1e-4);
    assert!((t.translation.z + 2.0_f32).abs() < 1e-4);
}
