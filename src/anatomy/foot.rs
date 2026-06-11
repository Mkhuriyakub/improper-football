//! 26-bone anatomical foot assembly.
//!
//! Built from the spec at
//! docs/superpowers/specs/2026-06-09-anatomical-foot-design.md.
//! Every bone is a RigidBody::new_capsule; joints are data-driven
//! from the tables below. Masses are soft-tissue-scaled (x1.70) so
//! the foot segment totals ~1.0 kg; dimensions are osteometric.

use physics::math::{Vec3, Quat};

/// One bone's physical parameters (SI units: kg, m).
#[derive(Clone, Copy, Debug)]
pub struct BoneParams {
    pub name: &'static str,
    pub mass: f64,
    pub half_h: f64,
    pub radius: f64,
}

/// Stable identifier for each of the 26 bones. Array index into the
/// bones table and into FootHandles::bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Bone {
    Calcaneus = 0,
    Talus,
    Navicular,
    Cuboid,
    MedialCuneiform,
    IntermediateCuneiform,
    LateralCuneiform,
    Mt1, Mt2, Mt3, Mt4, Mt5,
    HalluxPp, HalluxDp,
    Toe2Pp, Toe2Mp, Toe2Dp,
    Toe3Pp, Toe3Mp, Toe3Dp,
    Toe4Pp, Toe4Mp, Toe4Dp,
    Toe5Pp, Toe5Mp, Toe5Dp,
}

pub const BONE_COUNT: usize = 26;

/// Bone parameters, indexed by `Bone as usize`. Masses in kg
/// (grams/1000), half_h and radius in m. Values from spec §2.
pub const BONES: [BoneParams; BONE_COUNT] = [
    BoneParams { name: "calcaneus",      mass: 0.247, half_h: 0.0375, radius: 0.0195 },
    BoneParams { name: "talus",          mass: 0.162, half_h: 0.0265, radius: 0.0143 },
    BoneParams { name: "navicular",      mass: 0.051, half_h: 0.0200, radius: 0.0110 },
    BoneParams { name: "cuboid",         mass: 0.056, half_h: 0.0170, radius: 0.0120 },
    BoneParams { name: "med_cuneiform",  mass: 0.043, half_h: 0.0120, radius: 0.0110 },
    BoneParams { name: "int_cuneiform",  mass: 0.022, half_h: 0.0090, radius: 0.0095 },
    BoneParams { name: "lat_cuneiform",  mass: 0.034, half_h: 0.0130, radius: 0.0100 },
    BoneParams { name: "mt1",            mass: 0.094, half_h: 0.0320, radius: 0.0075 },
    BoneParams { name: "mt2",            mass: 0.060, half_h: 0.0365, radius: 0.0053 },
    BoneParams { name: "mt3",            mass: 0.051, half_h: 0.0345, radius: 0.0048 },
    BoneParams { name: "mt4",            mass: 0.041, half_h: 0.0320, radius: 0.0043 },
    BoneParams { name: "mt5",            mass: 0.044, half_h: 0.0300, radius: 0.0053 },
    BoneParams { name: "hallux_pp",      mass: 0.031, half_h: 0.0155, radius: 0.0073 },
    BoneParams { name: "hallux_dp",      mass: 0.015, half_h: 0.0100, radius: 0.0055 },
    BoneParams { name: "toe2_pp",        mass: 0.0077, half_h: 0.0120, radius: 0.0040 },
    BoneParams { name: "toe2_mp",        mass: 0.0031, half_h: 0.0045, radius: 0.0030 },
    BoneParams { name: "toe2_dp",        mass: 0.0020, half_h: 0.0035, radius: 0.0028 },
    BoneParams { name: "toe3_pp",        mass: 0.0071, half_h: 0.0115, radius: 0.0040 },
    BoneParams { name: "toe3_mp",        mass: 0.0029, half_h: 0.0045, radius: 0.0030 },
    BoneParams { name: "toe3_dp",        mass: 0.0019, half_h: 0.0030, radius: 0.0025 },
    BoneParams { name: "toe4_pp",        mass: 0.0065, half_h: 0.0105, radius: 0.0035 },
    BoneParams { name: "toe4_mp",        mass: 0.0026, half_h: 0.0040, radius: 0.0025 },
    BoneParams { name: "toe4_dp",        mass: 0.0017, half_h: 0.0030, radius: 0.0025 },
    BoneParams { name: "toe5_pp",        mass: 0.0061, half_h: 0.0100, radius: 0.0040 },
    BoneParams { name: "toe5_mp",        mass: 0.0022, half_h: 0.0030, radius: 0.0025 },
    BoneParams { name: "toe5_dp",        mass: 0.0024, half_h: 0.0035, radius: 0.0030 },
];

/// Map an anatomical-frame axis (+X anterior, +Z superior, +Y lateral)
/// to the world frame (+z anterior, +y superior, +x lateral).
pub fn anat_to_world(a: Vec3) -> Vec3 {
    Vec3::new(a.y, a.z, a.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bone_table_is_complete_and_masses_sum_to_one_kg() {
        assert_eq!(BONES.len(), BONE_COUNT);
        let total: f64 = BONES.iter().map(|b| b.mass).sum();
        // Spec target ~1.0 kg (soft-tissue-scaled). Allow 5% tolerance.
        assert!((total - 1.0).abs() < 0.05, "foot mass {total} kg, expected ~1.0");
    }

    #[test]
    fn anat_to_world_maps_axes() {
        // anterior +X -> world +z
        assert_eq!(anat_to_world(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 0.0, 1.0));
        // superior +Z -> world +y
        assert_eq!(anat_to_world(Vec3::new(0.0, 0.0, 1.0)), Vec3::new(0.0, 1.0, 0.0));
        // lateral +Y -> world +x
        assert_eq!(anat_to_world(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(1.0, 0.0, 0.0));
    }
}
