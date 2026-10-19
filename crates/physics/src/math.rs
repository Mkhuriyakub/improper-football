use nalgebra::{Vector2, Vector3, Matrix3, UnitQuaternion};

pub type Vec2 = Vector2<f64>;
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

// ─── 2D geometry (used for ground-plane support polygon analysis) ──────

/// Convex hull of a set of 2D points using Andrew's monotone chain.
/// Returns vertices in counter-clockwise order. Removes duplicate
/// vertices on the hull. For 0, 1, or 2 input points, returns those
/// points as-is (no proper "polygon" — caller should handle the
/// degenerate case).
pub fn convex_hull_2d(points: &[Vec2]) -> Vec<Vec2> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    let mut pts: Vec<Vec2> = points.to_vec();
    // Sort by x, then y.
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    fn cross_2d(o: Vec2, a: Vec2, b: Vec2) -> f64 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    // Lower hull.
    let mut hull: Vec<Vec2> = Vec::with_capacity(2 * n);
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len()-2], hull[hull.len()-1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // Upper hull.
    let lower_count = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_count
            && cross_2d(hull[hull.len()-2], hull[hull.len()-1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();  // The last point is the first point.
    hull
}

/// True if `p` is inside (or exactly on the boundary of) the convex
/// polygon `poly`. `poly` must be in counter-clockwise winding order
/// (as returned by `convex_hull_2d`). Returns false for degenerate
/// polygons (fewer than 3 vertices) since "inside a line segment or
/// point" doesn't have a 2D-area meaning.
pub fn point_in_convex_polygon_2d(p: Vec2, poly: &[Vec2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        // Edge direction is b - a; outward normal (for CCW) is rotated 90° CW.
        // Point is inside when on the left of every edge (cross >= 0).
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross < 0.0 {
            return false;
        }
    }
    true
}

/// Signed distance from `p` to the convex polygon `poly`. Positive
/// when `p` is inside, negative when outside, magnitude = Euclidean
/// distance to the nearest edge in either case. CCW winding required.
/// Returns 0.0 for degenerate polygons.
pub fn signed_distance_to_convex_polygon_2d(p: Vec2, poly: &[Vec2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut min_dist = f64::INFINITY;
    let mut all_inside = true;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let ab = b - a;
        let ap = p - a;
        let len2 = ab.dot(&ab);
        let t = if len2 > 1e-18 { (ap.dot(&ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let closest = a + ab * t;
        let d = (p - closest).norm();
        if d < min_dist { min_dist = d; }

        // CCW: point is inside if cross >= 0 for every edge.
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross < 0.0 { all_inside = false; }
    }
    if all_inside { min_dist } else { -min_dist }
}
