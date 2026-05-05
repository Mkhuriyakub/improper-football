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

/// Extract the signed twist angle of a unit quaternion about a unit axis,
/// using swing-twist decomposition q = q_swing · q_twist where q_twist is
/// rotation purely about `axis` and q_swing is rotation perpendicular to
/// `axis`. Returns the twist angle in radians.
///
/// **Why this exists:** the obvious projection `q.scaled_axis().dot(&n)`
/// reads bivector cross-terms when the rotation is around multiple axes
/// at once — e.g. a Z+X rotation reads non-zero on the Y projection even
/// with no real Y rotation, and tightly-locked off-axes then fight that
/// fake reading and produce jitter. Swing-twist factorization measures
/// the *real* rotation about `axis` (the part that, applied alone, would
/// give the same orientation along that axis) and is invariant to
/// rotations perpendicular to `axis`.
///
/// **Behavior on edge cases:**
/// - Identity rotation → 0
/// - Pure rotation by θ about `axis` → θ (signed)
/// - Pure rotation by θ about an axis perpendicular to `axis` → 0
/// - Always returns in [-π, π] (shortest-arc convention via q.w sign).
///
/// `axis` must be a unit vector. The returned angle is signed by the
/// orientation of the axis: positive rotation by the right-hand rule
/// about `axis`.
pub fn swing_twist_angle(q: UnitQuaternion<f64>, axis: Vec3) -> f64 {
    // Imaginary part of the quaternion projected onto the axis.
    let v = Vec3::new(q.i, q.j, q.k);
    let proj = v.dot(&axis);
    // Twist quaternion (un-normalized) is (q.w, proj·axis). Its squared norm.
    let twist_norm2 = proj * proj + q.w * q.w;
    if twist_norm2 < 1e-18 {
        // Degenerate: rotation is exactly π about an axis perpendicular to
        // `axis` — twist component vanishes. Convention: zero.
        return 0.0;
    }
    let twist_norm = twist_norm2.sqrt();
    // Pick shortest-arc representation (q.w >= 0). Quaternions q and -q
    // represent the same rotation; using the half with positive w avoids
    // angles outside [-π, π].
    let (w, c) = if q.w >= 0.0 {
        (q.w / twist_norm, proj / twist_norm)
    } else {
        (-q.w / twist_norm, -proj / twist_norm)
    };
    // Twist quaternion is normalized to (w, c·axis). Twist angle = 2·atan2(c, w).
    2.0 * c.atan2(w)
}
