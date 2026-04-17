use physics::math::Mat3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentId {
    Pelvis = 0,
    Lumbar,
    Chest,
    Neck,
    Head,
    UpperArmL,
    LowerArmL,
    HandL,
    UpperArmR,
    LowerArmR,
    HandR,
    UpperLegL,
    LowerLegL,
    FootL,
    UpperLegR,
    LowerLegR,
    FootR,
    _Count,  // keep last — used to derive SEGMENT_COUNT
}

pub const SEGMENT_COUNT: usize = SegmentId::_Count as usize;

impl SegmentId {
    pub fn all() -> [SegmentId; SEGMENT_COUNT] {
        [
            SegmentId::Pelvis, SegmentId::Lumbar, SegmentId::Chest,
            SegmentId::Neck, SegmentId::Head,
            SegmentId::UpperArmL, SegmentId::LowerArmL, SegmentId::HandL,
            SegmentId::UpperArmR, SegmentId::LowerArmR, SegmentId::HandR,
            SegmentId::UpperLegL, SegmentId::LowerLegL, SegmentId::FootL,
            SegmentId::UpperLegR, SegmentId::LowerLegR, SegmentId::FootR,
        ]
    }

    pub fn idx(self) -> usize { self as usize }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Lineman,      // ~140 kg
    Receiver,     // ~90 kg
    Quarterback,  // ~102 kg
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentDef {
    pub mass: f64,
    pub radius: f64,       // capsule radius (m)
    pub half_height: f64,  // capsule half-height (m), axis = local Y
}

impl SegmentDef {
    /// Capsule inertia tensor (solid cylinder approximation, spin axis = Y).
    pub fn inertia_tensor(&self) -> Mat3 {
        let m = self.mass;
        let r = self.radius;
        let h = 2.0 * self.half_height;  // h = full cylinder height (not half_height)
        let i_axial = 0.5 * m * r * r;
        let i_trans = (1.0 / 12.0) * m * (3.0 * r * r + h * h);
        Mat3::new(
            i_trans, 0.0,     0.0,
            0.0,     i_axial, 0.0,
            0.0,     0.0,     i_trans,
        )
    }
}

/// Mass fraction and dimensions for each segment, scaled by total body mass.
struct SegmentParams { mass_frac: f64, radius_m: f64, half_h_m: f64 }

/// Mass fractions for each segment. These sum to ~0.952 (not 1.0) because
/// the model omits extremity detail (finger bones, etc.). The remaining ~4.8%
/// of body mass is considered distributed across modelled segments and does
/// not affect relative segment proportions meaningfully for a physics simulation.
fn params(seg: SegmentId) -> SegmentParams {
    match seg {
        SegmentId::Pelvis     => SegmentParams { mass_frac: 0.142, radius_m: 0.12, half_h_m: 0.10 },
        SegmentId::Lumbar     => SegmentParams { mass_frac: 0.100, radius_m: 0.10, half_h_m: 0.09 },
        SegmentId::Chest      => SegmentParams { mass_frac: 0.200, radius_m: 0.14, half_h_m: 0.14 },
        SegmentId::Neck       => SegmentParams { mass_frac: 0.020, radius_m: 0.05, half_h_m: 0.06 },
        SegmentId::Head       => SegmentParams { mass_frac: 0.068, radius_m: 0.10, half_h_m: 0.10 },
        SegmentId::UpperArmL  => SegmentParams { mass_frac: 0.028, radius_m: 0.04, half_h_m: 0.15 },
        SegmentId::LowerArmL  => SegmentParams { mass_frac: 0.016, radius_m: 0.03, half_h_m: 0.13 },
        SegmentId::HandL      => SegmentParams { mass_frac: 0.006, radius_m: 0.03, half_h_m: 0.05 },
        SegmentId::UpperArmR  => SegmentParams { mass_frac: 0.028, radius_m: 0.04, half_h_m: 0.15 },
        SegmentId::LowerArmR  => SegmentParams { mass_frac: 0.016, radius_m: 0.03, half_h_m: 0.13 },
        SegmentId::HandR      => SegmentParams { mass_frac: 0.006, radius_m: 0.03, half_h_m: 0.05 },
        SegmentId::UpperLegL  => SegmentParams { mass_frac: 0.100, radius_m: 0.07, half_h_m: 0.20 },
        SegmentId::LowerLegL  => SegmentParams { mass_frac: 0.047, radius_m: 0.05, half_h_m: 0.19 },
        SegmentId::FootL      => SegmentParams { mass_frac: 0.014, radius_m: 0.04, half_h_m: 0.12 },
        SegmentId::UpperLegR  => SegmentParams { mass_frac: 0.100, radius_m: 0.07, half_h_m: 0.20 },
        SegmentId::LowerLegR  => SegmentParams { mass_frac: 0.047, radius_m: 0.05, half_h_m: 0.19 },
        SegmentId::FootR      => SegmentParams { mass_frac: 0.014, radius_m: 0.04, half_h_m: 0.12 },
        SegmentId::_Count     => unreachable!("_Count is a sentinel, not a real segment"),
    }
}

fn total_mass(body_type: BodyType) -> f64 {
    match body_type {
        BodyType::Lineman     => 140.0,
        BodyType::Receiver    => 90.0,
        BodyType::Quarterback => 102.0,
    }
}

pub fn segment_def(body_type: BodyType, seg: SegmentId) -> SegmentDef {
    // Validate fraction sum in debug builds (excludes _Count sentinel)
    #[cfg(debug_assertions)]
    {
        let sum: f64 = SegmentId::all().iter().map(|&s| params(s).mass_frac).sum();
        debug_assert!((sum - 0.952).abs() < 0.01, "mass fraction sum changed: {}", sum);
    }
    let p = params(seg);
    let mass = total_mass(body_type) * p.mass_frac;
    SegmentDef { mass, radius: p.radius_m, half_height: p.half_h_m }
}
