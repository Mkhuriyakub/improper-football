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

/// How a joint connects two bones.
#[derive(Clone, Copy, Debug)]
pub enum JointKind {
    /// Articulated: primary axis with ROM soft-limit, passive spring+damper,
    /// and an active PD controller. Off-axes locked.
    Articulated {
        /// Primary axis in the anatomical frame (will be mapped to world).
        axis_anat: Vec3,
        rom_min: f64,
        rom_max: f64,
        /// Active PD gains (postural placeholder; muscle model is Phase 4).
        kp: f64,
        kd: f64,
    },
    /// Stiff weld: behaves as a rigid link (positional + 3-axis angular springs).
    Weld,
}

/// One joint between a parent and child bone.
#[derive(Clone, Copy, Debug)]
pub struct JointSpec {
    pub parent: Bone,
    pub child: Bone,
    pub kind: JointKind,
}

// Shared passive/limit constants (spec §3; analogy-derived where noted).
const ARTIC_PASSIVE_K: f64 = 4.0;     // N·m/rad passive centering
const ARTIC_PASSIVE_C: f64 = 1.0;     // N·m·s/rad
const ARTIC_LIMIT_K: f64 = 8000.0;    // soft-limit stiffness
const ARTIC_LIMIT_EXP: f64 = 2.0;
const OFF_AXIS_K: f64 = 200.0;        // locked off-axis centering
const OFF_AXIS_LIM: f64 = 0.05;       // rad
const OFF_AXIS_LIMIT_K: f64 = 10_000.0;

const JOINT_POS_K: f64 = 60_000.0;    // positional joint stiffness
const JOINT_POS_C: f64 = 150.0;
const WELD_POS_K: f64 = 1_000_000.0;
const WELD_POS_C: f64 = 2_000.0;
const WELD_ANG_K: f64 = 50_000.0;
const WELD_ANG_C: f64 = 500.0;

/// All joints in the foot, spec §3. Axis directions in anatomical frame.
/// ROM in radians. Articulated joints listed first, welds after.
/// NOTE: 21 articulated + 8 welds = 29. The spec counts 22 articulated
/// for the whole assembly; the 22nd is the talocrural (shin↔talus),
/// which the HOST binary creates — it is NOT in this table.
pub const JOINTS: [JointSpec; 29] = [
    // ── Articulated (21; talocrural is host-created, not here) ──
    JointSpec { parent: Bone::Talus, child: Bone::Calcaneus,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.17, rom_max: 0.44, kp: 50.0, kd: 2.0 } }, // subtalar (axis overridden in builder)
    JointSpec { parent: Bone::Talus, child: Bone::Navicular,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.26, 0.0, 0.97),
            rom_min: -0.14, rom_max: 0.14, kp: 30.0, kd: 1.0 } }, // talonavicular
    JointSpec { parent: Bone::Calcaneus, child: Bone::Cuboid,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.26, 0.0, 0.97),
            rom_min: -0.09, rom_max: 0.09, kp: 30.0, kd: 1.0 } }, // calcaneocuboid
    JointSpec { parent: Bone::Navicular, child: Bone::MedialCuneiform,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.09, rom_max: 0.09, kp: 25.0, kd: 0.8 } }, // medial naviculocuneiform (mobile, in vivo)
    JointSpec { parent: Bone::MedialCuneiform, child: Bone::Mt1,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.06, rom_max: 0.06, kp: 25.0, kd: 0.8 } }, // TMT1
    JointSpec { parent: Bone::Cuboid, child: Bone::Mt4,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.17, rom_max: 0.17, kp: 20.0, kd: 0.6 } }, // TMT4
    JointSpec { parent: Bone::Cuboid, child: Bone::Mt5,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.18, rom_max: 0.18, kp: 20.0, kd: 0.6 } }, // TMT5
    JointSpec { parent: Bone::Mt1, child: Bone::HalluxPp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.61, rom_max: 1.22, kp: 30.0, kd: 1.0 } }, // hallux MTP DF70/PF35
    JointSpec { parent: Bone::Mt2, child: Bone::Toe2Pp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.70, rom_max: 0.70, kp: 8.0, kd: 0.3 } },
    JointSpec { parent: Bone::Mt3, child: Bone::Toe3Pp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.70, rom_max: 0.70, kp: 8.0, kd: 0.3 } },
    JointSpec { parent: Bone::Mt4, child: Bone::Toe4Pp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.70, rom_max: 0.70, kp: 8.0, kd: 0.3 } },
    JointSpec { parent: Bone::Mt5, child: Bone::Toe5Pp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.70, rom_max: 0.70, kp: 8.0, kd: 0.3 } },
    JointSpec { parent: Bone::HalluxPp, child: Bone::HalluxDp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -1.57, rom_max: 0.0, kp: 6.0, kd: 0.2 } },
    JointSpec { parent: Bone::Toe2Pp, child: Bone::Toe2Mp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.61, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe2Mp, child: Bone::Toe2Dp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -1.05, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe3Pp, child: Bone::Toe3Mp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.61, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe3Mp, child: Bone::Toe3Dp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -1.05, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe4Pp, child: Bone::Toe4Mp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.61, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe4Mp, child: Bone::Toe4Dp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -1.05, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe5Pp, child: Bone::Toe5Mp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -0.61, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    JointSpec { parent: Bone::Toe5Mp, child: Bone::Toe5Dp,
        kind: JointKind::Articulated { axis_anat: Vec3::new(0.0, 1.0, 0.0),
            rom_min: -1.05, rom_max: 0.0, kp: 4.0, kd: 0.15 } },
    // ── Welds (8): 4 spanning-tree + 4 loop-closure ──
    JointSpec { parent: Bone::Navicular, child: Bone::IntermediateCuneiform, kind: JointKind::Weld },
    JointSpec { parent: Bone::Navicular, child: Bone::LateralCuneiform, kind: JointKind::Weld },
    JointSpec { parent: Bone::IntermediateCuneiform, child: Bone::Mt2, kind: JointKind::Weld }, // TMT2
    JointSpec { parent: Bone::LateralCuneiform, child: Bone::Mt3, kind: JointKind::Weld }, // TMT3
    JointSpec { parent: Bone::MedialCuneiform, child: Bone::IntermediateCuneiform, kind: JointKind::Weld }, // intercuneiform
    JointSpec { parent: Bone::IntermediateCuneiform, child: Bone::LateralCuneiform, kind: JointKind::Weld }, // intercuneiform
    JointSpec { parent: Bone::LateralCuneiform, child: Bone::Cuboid, kind: JointKind::Weld }, // cuneocuboid
    JointSpec { parent: Bone::Cuboid, child: Bone::Navicular, kind: JointKind::Weld }, // cuboideonavicular
];

/// Subtalar axis in the anatomical frame: 42° inclination above the
/// transverse plane, 23° medial deviation from the foot long axis.
/// Spec §3. Returned as a unit vector in anatomical (+X ant, +Y lat,
/// +Z sup) coordinates for a right foot (medial = -Y).
pub fn subtalar_axis_anat() -> Vec3 {
    let incl = 42.0_f64.to_radians();
    let dev = 23.0_f64.to_radians();
    // Start along +X (anterior), tilt up by incl about +Y, deviate
    // medially (toward -Y) by dev about +Z.
    let x = incl.cos() * dev.cos();
    let y = -incl.cos() * dev.sin(); // medial = -Y
    let z = incl.sin();
    Vec3::new(x, y, z).normalize()
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

    #[test]
    fn subtalar_axis_has_expected_inclination() {
        let a = subtalar_axis_anat();
        // inclination above transverse plane = asin(z component)
        let incl_deg = a.z.asin().to_degrees();
        assert!((incl_deg - 42.0).abs() < 1.0, "subtalar incl {incl_deg}, expected 42");
        assert!(a.y < 0.0, "subtalar axis should deviate medially (-Y)");
        assert!((a.norm() - 1.0).abs() < 1e-9, "axis must be unit");
    }
}
