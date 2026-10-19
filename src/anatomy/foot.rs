//! 26-bone anatomical foot assembly.
//!
//! Built from the spec at
//! docs/superpowers/specs/2026-06-09-anatomical-foot-design.md.
//! Every bone is a RigidBody::new_capsule; joints are data-driven
//! from the tables below. Masses are soft-tissue-scaled (x1.70) so
//! the foot segment totals ~1.0 kg; dimensions are osteometric.

use physics::math::{Vec3, Quat};
use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_spring::AngularSpring;
use physics::angular_damper::AngularDamper;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;

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

// Shared passive/limit constants (spec §3). Published moment-angle data
// exists for only three foot joints; these shared values are SIZED to land
// in those curves' ranges and inherited by analogy everywhere else:
// - Soft-limit: torque = K·(Δθ)² past the ROM bound, so 8000·(0.05)² =
//   20 N·m at 0.05 rad past the talocrural dorsiflexion limit — inside
//   Riener & Edrich (1999)'s ~20-30 N·m functional-extreme passive range.
// - Passive centering 4.0 N·m/rad is order-matched to the stiffest
//   published first-MTP passive quasi-stiffness (Heng et al. 2016,
//   0.66-53.4 N·mm/deg ≈ up to ~3 N·m/rad).
const ARTIC_PASSIVE_K: f64 = 4.0;     // N·m/rad passive centering (primary axis)
const ARTIC_PASSIVE_C: f64 = 1.0;     // N·m·s/rad
const ARTIC_LIMIT_K: f64 = 8000.0;    // primary-axis ROM soft-limit stiffness
const ARTIC_LIMIT_EXP: f64 = 2.0;
const OFF_AXIS_LIM: f64 = 0.05;       // off-axis lock half-range (rad)

// Positional coupling stiffness, mapped to XPBD constraint compliance (1/k).
// The damping constants are retained for the call signature but the XPBD
// constraint supplies its own (energy-stable) damping via velocity solve.
const JOINT_POS_K: f64 = 60_000.0;    // articulated-joint positional stiffness
const JOINT_POS_C: f64 = 150.0;
const WELD_POS_K: f64 = 1_000_000.0;  // weld positional stiffness (near-rigid)
const WELD_POS_C: f64 = 2_000.0;

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
            rom_min: -0.61, rom_max: 1.22, kp: 30.0, kd: 1.0 } }, // hallux MTP DF70/PF35 (clinical ROM; passive stiffness context: Heng 2016)
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
    // Spherical components directly: anterior (+X) = cos(incl)cos(dev),
    // medial (-Y) = cos(incl)sin(dev), superior (+Z) = sin(incl).
    let x = incl.cos() * dev.cos();
    let y = -incl.cos() * dev.sin(); // medial = -Y
    let z = incl.sin();
    Vec3::new(x, y, z).normalize()
}

/// Body indices and PD-controller indices for the assembled foot.
pub struct FootHandles {
    /// World body index for each Bone (index by `bone as usize`).
    pub bodies: [usize; BONE_COUNT],
    /// PD controller index for each articulated joint, keyed by child Bone.
    /// `usize::MAX` for welded children (no PD).
    pub pd_by_child: [usize; BONE_COUNT],
}

impl FootHandles {
    pub fn body(&self, b: Bone) -> usize { self.bodies[b as usize] }
}

/// Adds the positional coupling that holds two bones together at a joint.
///
/// Uses the engine's XPBD `PositionalConstraint` (3-DOF ball-and-socket,
/// solved in the SI loop) rather than a penalty spring+damper. A penalty
/// spring with anchor lever arms injects energy on stiff multibody systems
/// (the anchor offset `d` goes stale as the body rotates within the substep),
/// which exploded the 26-bone assembly. The XPBD constraint is symplectic and
/// energy-stable by construction; `compliance = 1/k` reproduces the intended
/// joint stiffness (compliance 0 = perfectly rigid).
fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    anchor_parent: Vec3, anchor_child: Vec3, k: f64, _c: f64,
) {
    let mut c = PositionalConstraint::new(parent, child, 0.0, 1.0 / k);
    c.local_anchor_a = anchor_parent;
    c.local_anchor_b = anchor_child;
    world.constraints.push(c);
}

pub fn weld(
    world: &mut XpbdWorld, a: usize, b: usize,
    anchor_a: Vec3, anchor_b: Vec3,
) {
    add_elastic_joint(world, a, b, anchor_a, anchor_b, WELD_POS_K, WELD_POS_C);
    // Rigid orientation lock on all three axes via the engine's SI angular
    // constraint with a degenerate range [0, 0]: every deviation is "out of
    // range" and is pulled back to zero, so the pair acts as a bilateral
    // angular weld. This replaces three stiff penalty AngularSprings, which —
    // applied as independent Gauss-Seidel passes about x/y/z on a few-gram
    // tarsal — fought each other and injected energy that exploded the
    // assembly. The SI constraint is iterated together with all others and is
    // energy-stable.
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_constraints.push(
            AngularLimitConstraint::new(a, b, axis, 0.0, 0.0));
    }
}

/// Adds an articulated joint: positional anchor, primary-axis ROM
/// soft-limit + passive spring/damper + PD controller, and two locked
/// off-axes. `axis_world` is the primary axis already mapped to world.
/// `baseline` is the child's rest orientation relative to the parent
/// (identity if both share orientation). Returns the PD index.
fn add_articulated(
    world: &mut XpbdWorld, parent: usize, child: usize,
    anchor_parent: Vec3, anchor_child: Vec3,
    axis_world: Vec3, rom_min: f64, rom_max: f64, kp: f64, kd: f64,
    baseline: Quat,
) -> usize {
    add_elastic_joint(world, parent, child, anchor_parent, anchor_child, JOINT_POS_K, JOINT_POS_C);

    let primary = axis_world.normalize();
    world.angular_soft_limits.push(
        AngularSoftLimit::new(parent, child, primary, rom_min, rom_max, ARTIC_LIMIT_K, ARTIC_LIMIT_EXP)
            .with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(parent, child, primary, ARTIC_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(parent, child, primary, ARTIC_PASSIVE_K, 0.0).with_baseline(baseline));
    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(parent, child, primary, kp, kd).with_baseline(baseline));

    // Two off-axes orthogonal to the primary, locked. Pick a helper axis
    // not near-parallel to the primary (|x| < 0.9) so the first cross
    // product is well-conditioned and doesn't degenerate toward zero.
    let helper = if primary.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    let off1 = primary.cross(&helper).normalize();
    let off2 = primary.cross(&off1).normalize();
    for off in [off1, off2] {
        // Off-axis lock as an SI angular constraint with a small ±OFF_AXIS_LIM
        // range (a near-rigid hinge with a few degrees of physiological slop).
        // Penalty springs/soft-limits here are two orthogonal stiff stacks that
        // fight each other in Gauss-Seidel on the few-gram phalanges and slowly
        // pump energy; the iterated SI constraint is energy-stable.
        world.angular_constraints.push(
            AngularLimitConstraint::new(parent, child, off, -OFF_AXIS_LIM, OFF_AXIS_LIM));
        world.angular_dampers.push(AngularDamper::new(parent, child, off, ARTIC_PASSIVE_C));
    }
    pd_idx
}

/// Anterior-posterior shift of the whole foot (every bone except the
/// talus) relative to the talus/ankle. The talus stays under the shin
/// (z=0); shifting the rest of the foot backward by this amount moves the
/// ground support polygon so its centroid sits closer to under the ankle
/// rather than far forward of it. With the un-shifted layout the talocrural
/// joint sits at the heel end of a forward-extending footprint, so a load
/// CoM directly over the ankle projects to the rear edge of the support
/// polygon. (Diagnostic note: centring the polygon is necessary but NOT
/// sufficient for passive standing balance under a tall heavy load — see
/// the ankle_anatomical binary's findings — but it is correct geometry and
/// maximises the passive stability margin.)
const FOOT_Z_SHIFT: f64 = -0.05;

/// Rest position of each bone's CENTER relative to the foot origin
/// (ground point below the talus), in world meters (x lat, y up, z ant).
/// Starting layout per spec §2 column structure; refined by stage tests.
const REST_POS: [Vec3; BONE_COUNT] = [
    // Plantar-contact bones (heel, metatarsal heads, phalanges) are placed
    // so their capsule bottoms (center.y − radius) lie on a common sole
    // plane at y = 0.018 above the foot origin — a FLAT sole, not the
    // graded heights of the original layout (which left the heel ~1 cm
    // below the forefoot, so only the heel touched and the foot toppled).
    // Midfoot bones (navicular, cuboid, cuneiforms) stay above the sole.
    Vec3::new( 0.005, 0.0375, -0.045), // calcaneus (heel: behind; bottom=0.018)
    Vec3::new( 0.000, 0.0570,  0.000), // talus (on top, at ankle)
    Vec3::new(-0.008, 0.0450,  0.030), // navicular (medial midfoot, arch apex)
    Vec3::new( 0.022, 0.0300,  0.020), // cuboid (lateral midfoot)
    Vec3::new(-0.015, 0.0380,  0.052), // medial cuneiform
    Vec3::new(-0.002, 0.0400,  0.050), // intermediate cuneiform
    Vec3::new( 0.012, 0.0380,  0.050), // lateral cuneiform
    Vec3::new(-0.018, 0.0255,  0.095), // mt1 (bottom=0.018)
    Vec3::new(-0.006, 0.0233,  0.100), // mt2 (bottom=0.018)
    Vec3::new( 0.006, 0.0228,  0.098), // mt3 (bottom=0.018)
    Vec3::new( 0.018, 0.0223,  0.092), // mt4 (bottom=0.018)
    Vec3::new( 0.028, 0.0233,  0.085), // mt5 (bottom=0.018)
    Vec3::new(-0.020, 0.0253,  0.135), // hallux_pp (bottom=0.018)
    Vec3::new(-0.021, 0.0235,  0.158), // hallux_dp (bottom=0.018)
    Vec3::new(-0.008, 0.0220,  0.140), // toe2_pp (bottom=0.018)
    Vec3::new(-0.008, 0.0210,  0.156), // toe2_mp (bottom=0.018)
    Vec3::new(-0.008, 0.0208,  0.166), // toe2_dp (bottom=0.018)
    Vec3::new( 0.004, 0.0220,  0.138), // toe3_pp (bottom=0.018)
    Vec3::new( 0.004, 0.0210,  0.153), // toe3_mp (bottom=0.018)
    Vec3::new( 0.004, 0.0205,  0.162), // toe3_dp (bottom=0.018)
    Vec3::new( 0.016, 0.0215,  0.132), // toe4_pp (bottom=0.018)
    Vec3::new( 0.016, 0.0205,  0.146), // toe4_mp (bottom=0.018)
    Vec3::new( 0.016, 0.0205,  0.154), // toe4_dp (bottom=0.018)
    Vec3::new( 0.027, 0.0220,  0.122), // toe5_pp (bottom=0.018)
    Vec3::new( 0.027, 0.0205,  0.134), // toe5_mp (bottom=0.018)
    Vec3::new( 0.027, 0.0210,  0.143), // toe5_dp (bottom=0.018)
];

/// Rest orientation of each bone. Most bones lie with their long axis
/// (local +y) pointing anterior (world +z) → rotate local +y to +z via
/// R_x(-π/2). For Phase 1 ALL bones use this anterior-pointing baseline —
/// the capsule shape is approximate for blocky tarsals anyway (spec §6).
fn bone_baseline() -> Quat {
    Quat::from_scaled_axis(Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0))
}

/// Local-frame anchor on the body at `bone_idx` for a joint whose world
/// location is `joint_world`. Expresses that single world point in the
/// body's local frame: `R⁻¹ · (joint_world − center)`.
///
/// Both bodies of a joint pass the SAME `joint_world` (the midpoint of their
/// centers), so at rest both anchors map back to the same world point and the
/// positional spring sees zero offset → zero preload. The earlier
/// "point toward the neighbor, scaled to half_h" scheme placed the two
/// anchors up to ~2 cm apart for tightly-packed tarsals, which a 10⁶ N/m weld
/// spring turned into multi-kN build-time forces that exploded the small
/// bones on substep 0.
pub fn anchor_at(world: &XpbdWorld, bone_idx: usize, joint_world: Vec3) -> Vec3 {
    let b = &world.bodies[bone_idx];
    b.orientation.inverse() * (joint_world - b.position)
}

/// Attach a proximal segment (the shin) to the talus via the talocrural
/// joint: dorsi/plantarflexion about world +x (mediolateral), with the Y
/// and Z off-axes locked. The segment is assumed identity-oriented; the
/// talus carries `bone_baseline()`, so the joint's rest offset is
/// `bone_baseline()` and all angular primitives are baseline-aware.
///
/// Unlike the foot's internal joints (which use baseline-free
/// AngularLimitConstraints, valid only when both bones share orientation),
/// this joint crosses the identity→baseline boundary, so it uses
/// baseline-aware penalty soft-limits. Those are stable here because the
/// shin/talus are kg-scale (the constraint treatment was only needed for
/// the few-gram foot bones). Returns the talocrural PD index.
pub fn attach_talocrural(
    world: &mut XpbdWorld, seg_idx: usize, foot: &FootHandles,
    rom_min: f64, rom_max: f64, kp: f64, kd: f64,
) -> usize {
    let talus_idx = foot.body(Bone::Talus);
    let joint_world = world.bodies[talus_idx].position; // talus center
    let anchor_seg = anchor_at(world, seg_idx, joint_world);
    let anchor_talus = anchor_at(world, talus_idx, joint_world);
    add_elastic_joint(world, seg_idx, talus_idx, anchor_seg, anchor_talus, JOINT_POS_K, JOINT_POS_C);

    let baseline = bone_baseline();
    let primary = Vec3::new(1.0, 0.0, 0.0); // DF/PF
    world.angular_soft_limits.push(
        AngularSoftLimit::new(seg_idx, talus_idx, primary, rom_min, rom_max, ARTIC_LIMIT_K, ARTIC_LIMIT_EXP)
            .with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(seg_idx, talus_idx, primary, ARTIC_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(seg_idx, talus_idx, primary, ARTIC_PASSIVE_K, 0.0).with_baseline(baseline));
    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(seg_idx, talus_idx, primary, kp, kd).with_baseline(baseline));

    // Y and Z off-axes locked with baseline-aware penalty soft-limits.
    for off in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(
            AngularSoftLimit::new(seg_idx, talus_idx, off, -OFF_AXIS_LIM, OFF_AXIS_LIM, ARTIC_LIMIT_K, ARTIC_LIMIT_EXP)
                .with_baseline(baseline));
        world.angular_dampers.push(AngularDamper::new(seg_idx, talus_idx, off, ARTIC_PASSIVE_C));
    }
    pd_idx
}

/// Build the 26-bone foot into `world`. `attach` is the world position
/// of the talus center (the host places it at the shin's distal end).
/// The talocrural joint (shin↔talus) is created by the HOST, not here.
/// Returns FootHandles with all 26 body indices and the per-child PD
/// indices for the 21 articulated joints (welded children stay usize::MAX).
pub fn build_foot(world: &mut XpbdWorld, attach: Vec3) -> FootHandles {
    let baseline = bone_baseline();
    // Foot origin = ground point below the talus. talus REST_POS.y is its
    // height above origin, so origin = attach - (0, talus.y, 0).
    let talus_rest = REST_POS[Bone::Talus as usize];
    let origin = attach - Vec3::new(0.0, talus_rest.y, 0.0);

    let mut bodies = [usize::MAX; BONE_COUNT];
    for i in 0..BONE_COUNT {
        let p = &BONES[i];
        let mut body = RigidBody::new_capsule(p.mass, p.half_h, p.radius);
        let mut rest = REST_POS[i];
        if i != Bone::Talus as usize { rest.z += FOOT_Z_SHIFT; }
        body.position = origin + rest;
        body.orientation = baseline;
        bodies[i] = world.add_body(body);
    }

    let mut pd_by_child = [usize::MAX; BONE_COUNT];

    for spec in JOINTS.iter() {
        let parent_idx = bodies[spec.parent as usize];
        let child_idx = bodies[spec.child as usize];
        let parent_pos = world.bodies[parent_idx].position;
        let child_pos = world.bodies[child_idx].position;
        // Joint sits at the midpoint of the two bone centers; both anchors
        // resolve to this same world point, so the rest preload is zero.
        let joint_world = (parent_pos + child_pos) * 0.5;
        let anchor_p = anchor_at(world, parent_idx, joint_world);
        let anchor_c = anchor_at(world, child_idx, joint_world);

        match spec.kind {
            JointKind::Weld => {
                weld(world, parent_idx, child_idx, anchor_p, anchor_c);
            }
            JointKind::Articulated { axis_anat, rom_min, rom_max, kp, kd } => {
                // Subtalar is the one joint whose axis isn't a table literal:
                // its oblique Manter/Inman axis (42° incl, 23° dev — the
                // sprain-mechanism axis) is computed in subtalar_axis_anat().
                let axis_a = if spec.parent == Bone::Talus && spec.child == Bone::Calcaneus {
                    subtalar_axis_anat()
                } else {
                    axis_anat
                };
                let axis_world = anat_to_world(axis_a);
                // Both bones share bone_baseline(), so child-relative-to-parent
                // rest orientation is identity.
                let pd = add_articulated(
                    world, parent_idx, child_idx, anchor_p, anchor_c,
                    axis_world, rom_min, rom_max, kp, kd,
                    Quat::identity());
                pd_by_child[spec.child as usize] = pd;
            }
        }
    }

    FootHandles { bodies, pd_by_child }
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
        // medial deviation from the foot long axis, in the transverse plane.
        let dev_deg = (-a.y).atan2(a.x).to_degrees();
        assert!((dev_deg - 23.0).abs() < 1.0, "subtalar deviation {dev_deg}, expected 23");
        assert!((a.norm() - 1.0).abs() < 1e-9, "axis must be unit");
    }

    #[test]
    fn build_foot_places_26_bodies_above_ground() {
        let mut world = XpbdWorld::new();
        let attach = Vec3::new(0.0, 0.30, 0.0); // talus 30 cm up
        let h = build_foot(&mut world, attach);
        assert_eq!(world.bodies.len(), BONE_COUNT);
        // Layout sanity check: with the talus held 30 cm up, no bone should
        // sit below the world ground at all. The 5 cm bound is a loose guard
        // against a grossly misplaced REST_POS entry (a real bone dips at most
        // ~3 cm below the foot origin, far above the world ground here).
        // True ground-clearance at the standing pose is tested in the settle
        // and static-balance tests (Tasks 6-7).
        for i in 0..BONE_COUNT {
            let b = &world.bodies[h.bodies[i]];
            let lowest = b.position.y - BONES[i].radius - BONES[i].half_h;
            assert!(lowest > -0.05, "bone {} starts {lowest} below ground", BONES[i].name);
        }
        // Talus should be at the attach point.
        let talus = &world.bodies[h.body(Bone::Talus)];
        assert!((talus.position - attach).norm() < 1e-9);
    }

    #[test]
    fn foot_settles_on_ground_without_exploding() {
        let mut world = XpbdWorld::new();
        world.set_ground(0.0);
        // Drop the foot from a low height; talus 6 cm up.
        let h = build_foot(&mut world, Vec3::new(0.0, 0.06, 0.0));
        let dt = 1.0 / 120.0;
        let iters = 60;
        let sub_dt = dt / iters as f64;
        for _ in 0..240 { // 2 seconds
            for _ in 0..iters { world.substep(sub_dt); }
        }
        // No NaN anywhere.
        for i in 0..BONE_COUNT {
            let p = world.bodies[h.bodies[i]].position;
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                "bone {} went non-finite", BONES[i].name);
        }
        // Calcaneus (heel) should be resting near the ground, not flown away.
        let heel = world.bodies[h.body(Bone::Calcaneus)].position;
        assert!(heel.y < 0.10 && heel.y > -0.02, "heel y={} after settle", heel.y);
    }

    #[test]
    fn build_foot_creates_expected_joint_counts() {
        let mut world = XpbdWorld::new();
        let h = build_foot(&mut world, Vec3::new(0.0, 0.30, 0.0));
        // 21 articulated joints in the foot module → 21 PD controllers.
        // (The 22nd articulated joint, talocrural, is host-created.)
        let pd_count = h.pd_by_child.iter().filter(|&&x| x != usize::MAX).count();
        assert_eq!(pd_count, 21, "expected 21 articulated PDs in the foot module");
        assert_eq!(world.angular_pd_controllers.len(), 21);
    }
}
