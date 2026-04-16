use nalgebra::{Vector3, Matrix3, UnitQuaternion};

pub type Vec3 = Vector3<f64>;
pub type Mat3 = Matrix3<f64>;
pub type Quat = UnitQuaternion<f64>;

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(&b)
}

pub fn skew_symmetric(v: Vec3) -> Mat3 {
    Mat3::new(
         0.0, -v.z,  v.y,
         v.z,  0.0, -v.x,
        -v.y,  v.x,  0.0,
    )
}
