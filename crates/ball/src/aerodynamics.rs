use physics::math::Vec3;

// NFL football physical constants
pub const BALL_MASS: f64 = 0.425;          // kg
pub const BALL_LENGTH: f64 = 0.280;        // m (tip to tip)
pub const BALL_SHORT_RADIUS: f64 = 0.085;  // m (equatorial radius)
pub const AIR_DENSITY: f64 = 1.225;        // kg/m³ at sea level

// Aerodynamic coefficients for a prolate spheroid (approximate)
pub const CD_AXIAL: f64 = 0.05;    // drag when flying nose-first (tight spiral)
pub const CD_LATERAL: f64 = 0.60;  // drag when tumbling broadside
pub const CL_MAGNUS: f64 = 0.10;   // Magnus lift coefficient

/// Cross-sectional area depends on the angle of attack.
/// For a prolate spheroid, effective area changes with orientation.
pub fn effective_cross_section(velocity: Vec3, spin_axis: Vec3) -> f64 {
    let v_norm = velocity.norm();
    if v_norm < 1e-6 {
        return std::f64::consts::PI * BALL_SHORT_RADIUS * BALL_SHORT_RADIUS;
    }

    let v_hat = velocity / v_norm;
    // cos(angle) between velocity and spin axis
    let cos_alpha = spin_axis.dot(&v_hat).abs().clamp(0.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).sqrt();

    // Interpolate between nose-on (circle) and broadside (ellipse) cross-sections
    let area_axial = std::f64::consts::PI * BALL_SHORT_RADIUS * BALL_SHORT_RADIUS;
    let area_lateral = BALL_SHORT_RADIUS * BALL_LENGTH; // ellipse approx

    area_axial * cos_alpha + area_lateral * sin_alpha
}

/// Drag force opposing velocity.
pub fn drag_force(velocity: Vec3, spin_axis: Vec3) -> Vec3 {
    let v_norm = velocity.norm();
    if v_norm < 1e-6 {
        return Vec3::zeros();
    }

    let v_hat = velocity / v_norm;
    let cos_alpha = spin_axis.dot(&v_hat).abs().clamp(0.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).sqrt();

    let cd = CD_AXIAL * cos_alpha + CD_LATERAL * sin_alpha;
    let area = effective_cross_section(velocity, spin_axis);

    let drag_mag = 0.5 * AIR_DENSITY * cd * area * v_norm * v_norm;
    -v_hat * drag_mag
}

/// Magnus force from ball spin.
/// F_magnus = CL * rho * A * (omega × v)
/// This curves a spiral throw slightly and causes knuckling on a wobble pass.
pub fn magnus_force(velocity: Vec3, angular_velocity: Vec3) -> Vec3 {
    let v_norm = velocity.norm();
    if v_norm < 1e-6 {
        return Vec3::zeros();
    }

    let area = std::f64::consts::PI * BALL_SHORT_RADIUS * BALL_SHORT_RADIUS;
    let cross = angular_velocity.cross(&velocity);
    cross * (CL_MAGNUS * AIR_DENSITY * area)
}
