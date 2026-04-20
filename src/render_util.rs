use nalgebra::{Vector3, UnitQuaternion};
use bevy::math::{Vec3, Quat};
use bevy::transform::components::Transform;
use physics::rigid_body::RigidBody;

pub fn nalgebra_vec3_to_bevy(v: Vector3<f64>) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

pub fn nalgebra_quat_to_bevy(q: UnitQuaternion<f64>) -> Quat {
    Quat::from_xyzw(
        q.i as f32,
        q.j as f32,
        q.k as f32,
        q.w as f32,
    )
}

pub fn rigid_body_to_transform(body: &RigidBody) -> Transform {
    Transform {
        translation: nalgebra_vec3_to_bevy(body.position),
        rotation: nalgebra_quat_to_bevy(body.orientation),
        scale: Vec3::ONE,
    }
}
