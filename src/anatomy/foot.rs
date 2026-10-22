//! 26-bone anatomical foot assembly.
//!
//! Built from the spec at
//! docs/superpowers/specs/2026-06-09-anatomical-foot-design.md.
//! Every bone is a RigidBody::new_capsule; joints are data-driven
//! from the tables below. Masses are soft-tissue-scaled (x1.70) so
//! the foot segment totals ~1.0 kg; dimensions are osteometric.
//!
//! Coaxial solve (Stage 2, re-landed 2026-10-21 on the coupled in-loop
//! engine): every articulated joint's PRIMARY axis is ONE `JointAxis` —
//! active PD + passive spring + passive damper + nonlinear ROM soft-limit
//! solved together inside the SI Gauss-Seidel loop with a shared
//! denominator and the L2 warm-start pair. The legacy sequential passes
//! attenuated each term's gain by 1/(1 + dt²·w·k) (~94% on gram-scale toe
//! joints, docs/findings.md 2026-10) and transmitted only 1/D of the
//! demanded torque through a blocking contact (docs/findings.md
//! 2026-10-21); the in-loop JointAxis removes both. This module pushes NO
//! angular_springs, angular_soft_limits, or angular_pd_controllers on
//! primary axes anymore. Off-axis locks stay SI `AngularLimitConstraint`s
//! with `AngularDamper`s (different axes — not coaxial, so nothing for the
//! combined solve to combine), so every joint keeps constraints on every
//! anatomical axis.

use physics::math::{Vec3, Quat};
use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::joint_axis::JointAxis;
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
/// (grams/1000), half_h and radius in m. Masses and radii from spec §2
/// (masses stay literature-anchored).
///
/// `half_h` is the capsule SEGMENT half-length, sized so segment + end
/// caps = anatomical bone length (half_h = length/2 − radius, floored at
/// 1 mm). The spec's §2 derivation set half_h = length/2 directly, which
/// ignored that a capsule's surface extends one `radius` beyond each
/// segment end — every bone rendered/collided `2·radius` too long (~14%
/// oversize on slender metatarsals, 55–70% on the blocky tarsals, which
/// made the midfoot an overlapping blob). For bones where radius ≥
/// length/2 (e.g., the intermediate cuneiform) the segment collapses
/// toward a sphere — correct and more honest for blocky bones.
pub const BONES: [BoneParams; BONE_COUNT] = [
    BoneParams { name: "calcaneus",      mass: 0.247, half_h: 0.0180, radius: 0.0195 },
    BoneParams { name: "talus",          mass: 0.162, half_h: 0.0122, radius: 0.0143 },
    BoneParams { name: "navicular",      mass: 0.051, half_h: 0.0090, radius: 0.0110 },
    BoneParams { name: "cuboid",         mass: 0.056, half_h: 0.0050, radius: 0.0120 },
    BoneParams { name: "med_cuneiform",  mass: 0.043, half_h: 0.0010, radius: 0.0110 },
    BoneParams { name: "int_cuneiform",  mass: 0.022, half_h: 0.0010, radius: 0.0095 },
    BoneParams { name: "lat_cuneiform",  mass: 0.034, half_h: 0.0030, radius: 0.0100 },
    BoneParams { name: "mt1",            mass: 0.094, half_h: 0.0245, radius: 0.0075 },
    BoneParams { name: "mt2",            mass: 0.060, half_h: 0.0312, radius: 0.0053 },
    BoneParams { name: "mt3",            mass: 0.051, half_h: 0.0297, radius: 0.0048 },
    BoneParams { name: "mt4",            mass: 0.041, half_h: 0.0277, radius: 0.0043 },
    BoneParams { name: "mt5",            mass: 0.044, half_h: 0.0247, radius: 0.0053 },
    BoneParams { name: "hallux_pp",      mass: 0.031, half_h: 0.0082, radius: 0.0073 },
    BoneParams { name: "hallux_dp",      mass: 0.015, half_h: 0.0045, radius: 0.0055 },
    BoneParams { name: "toe2_pp",        mass: 0.0077, half_h: 0.0080, radius: 0.0040 },
    BoneParams { name: "toe2_mp",        mass: 0.0031, half_h: 0.0015, radius: 0.0030 },
    BoneParams { name: "toe2_dp",        mass: 0.0020, half_h: 0.0010, radius: 0.0028 },
    BoneParams { name: "toe3_pp",        mass: 0.0071, half_h: 0.0075, radius: 0.0040 },
    BoneParams { name: "toe3_mp",        mass: 0.0029, half_h: 0.0015, radius: 0.0030 },
    BoneParams { name: "toe3_dp",        mass: 0.0019, half_h: 0.0010, radius: 0.0025 },
    BoneParams { name: "toe4_pp",        mass: 0.0065, half_h: 0.0070, radius: 0.0035 },
    BoneParams { name: "toe4_mp",        mass: 0.0026, half_h: 0.0015, radius: 0.0025 },
    BoneParams { name: "toe4_dp",        mass: 0.0017, half_h: 0.0010, radius: 0.0025 },
    BoneParams { name: "toe5_pp",        mass: 0.0061, half_h: 0.0060, radius: 0.0040 },
    BoneParams { name: "toe5_mp",        mass: 0.0022, half_h: 0.0010, radius: 0.0025 },
    BoneParams { name: "toe5_dp",        mass: 0.0024, half_h: 0.0010, radius: 0.0030 },
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

/// Body indices and JointAxis indices for the assembled foot.
pub struct FootHandles {
    /// World body index for each Bone (index by `bone as usize`).
    pub bodies: [usize; BONE_COUNT],
    /// `world.joint_axes` index of each articulated joint's primary-axis
    /// combined solve (PD + spring + damper + ROM limit), keyed by child
    /// Bone. `usize::MAX` for welded children (no articulated joint).
    pub ja_by_child: [usize; BONE_COUNT],
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

/// Adds an articulated joint: positional anchor, ONE primary-axis
/// `JointAxis` (combined ROM soft-limit + passive spring/damper + active
/// PD — one backward-Euler solve inside the SI loop, one shared
/// denominator; no split-pass gain attenuation, no intra-joint ordering
/// sensitivity, full force transmission through blocking contacts), and
/// two locked off-axes. `axis_world` is the primary axis already mapped to
/// world. `baseline` is the child's rest orientation relative to the
/// parent (identity if both share orientation). Returns the joint_axes
/// index.
fn add_articulated(
    world: &mut XpbdWorld, parent: usize, child: usize,
    anchor_parent: Vec3, anchor_child: Vec3,
    axis_world: Vec3, rom_min: f64, rom_max: f64, kp: f64, kd: f64,
    baseline: Quat,
) -> usize {
    add_elastic_joint(world, parent, child, anchor_parent, anchor_child, JOINT_POS_K, JOINT_POS_C);

    let primary = axis_world.normalize();
    let ja_idx = world.joint_axes.len();
    world.joint_axes.push(
        JointAxis::new(parent, child, primary)
            .with_baseline(baseline)
            .with_soft_limit(rom_min, rom_max, ARTIC_LIMIT_K, ARTIC_LIMIT_EXP)
            .with_passive_spring(ARTIC_PASSIVE_K, 0.0)
            .with_passive_damping(ARTIC_PASSIVE_C)
            .with_pd(kp, kd));

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
    ja_idx
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
    // 2026-10 compaction note: the original z-spacing was tuned while every
    // capsule was 2·radius oversized (the spec's half_h = length/2 error).
    // With true extents the hindfoot/midfoot chain could no longer span it —
    // calcaneus↔cuboid↔mt4 left 13-24 mm of empty space at each joint and
    // the subtalar pair sat 33 mm apart. The calcaneus moves 30 mm anterior
    // and the cuboid 20 mm anterior so every JOINTS pair is again within a
    // few mm of surface contact (joint spaces, not disarticulations). Sole
    // plane (bottom = 0.018) and the forefoot/toe layout are unchanged.
    Vec3::new( 0.005, 0.0375, -0.015), // calcaneus (heel: behind; bottom=0.018)
    Vec3::new( 0.000, 0.0570,  0.000), // talus (on top, at ankle)
    Vec3::new(-0.008, 0.0450,  0.030), // navicular (medial midfoot, arch apex)
    Vec3::new( 0.022, 0.0300,  0.040), // cuboid (lateral midfoot)
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
/// Unlike the foot's internal joints (whose off-axes use baseline-free
/// AngularLimitConstraints, valid only when both bones share orientation),
/// this joint crosses the identity→baseline boundary, so EVERY axis here
/// is a baseline-aware `JointAxis`: the primary carries the full combined
/// stack (limit + spring + damper + PD), and each off-axis is a combined
/// two-term solve of its penalty lock (soft-limit ±OFF_AXIS_LIM) with its
/// damper — the same coaxial pair the legacy build ran as two sequential
/// passes, now sharing one denominator. Penalty locks are stable here
/// because the shin/talus are kg-scale (the SI-constraint treatment was
/// only needed for the few-gram foot bones). Every anatomical axis keeps
/// its constraint. Returns the talocrural's joint_axes index.
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
    let ja_idx = world.joint_axes.len();
    world.joint_axes.push(
        JointAxis::new(seg_idx, talus_idx, primary)
            .with_baseline(baseline)
            .with_soft_limit(rom_min, rom_max, ARTIC_LIMIT_K, ARTIC_LIMIT_EXP)
            .with_passive_spring(ARTIC_PASSIVE_K, 0.0)
            .with_passive_damping(ARTIC_PASSIVE_C)
            .with_pd(kp, kd));

    // Y and Z off-axes: baseline-aware penalty lock + damper, combined.
    for off in [Vec3::y(), Vec3::z()] {
        world.joint_axes.push(
            JointAxis::new(seg_idx, talus_idx, off)
                .with_baseline(baseline)
                .with_soft_limit(-OFF_AXIS_LIM, OFF_AXIS_LIM, ARTIC_LIMIT_K, 2.0)
                .with_passive_damping(ARTIC_PASSIVE_C));
    }
    ja_idx
}

/// Build the 26-bone foot into `world`. `attach` is the world position
/// of the talus center (the host places it at the shin's distal end).
/// The talocrural joint (shin↔talus) is created by the HOST, not here.
/// Returns FootHandles with all 26 body indices and the per-child
/// JointAxis indices for the 21 articulated joints (welded children stay
/// usize::MAX).
pub fn build_foot(world: &mut XpbdWorld, attach: Vec3) -> FootHandles {
    // The design's L1 convergence lever, scoped to this assembly (spec
    // §2.6; measured in Commit C): chained JointAxes under ground contact
    // need SI sweeps proportional to the L2 warm seeds' velocity
    // injection (w·λ ≈ 10³ rad/s at toe scale, ∝ sub_dt). At the engine
    // default of 8 the residual seed velocity reaches integrate_positions
    // and the toe chains detonate (toe-ray reproducer: DIP swings ±2.4 rad
    // past its ROM, |ω| spikes 4.8e4 rad/s); at 16 they are bounded but
    // sustain episodic 20–230 rad/s chatter (lift phases, grazing
    // contacts); at 32 the ray is clean in every phase at sub_dt 1/7200
    // and the demo foot's toe curl goes fully quiet (0 sign flips above
    // the 0.5 rad/s judder floor vs the legacy 76.6/s). Hosts running a
    // COARSER substep need proportionally more sweeps (the dual-rate test
    // below matches 1/3600 with 64). The global default stays 8 so legacy
    // (non-JointAxis) worlds remain bit-identical (docs/findings.md,
    // Stage-2 re-land entry). max(): never LOWER a host's own setting.
    world.solver_iterations = world.solver_iterations.max(32);
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

    let mut ja_by_child = [usize::MAX; BONE_COUNT];

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
                let ja = add_articulated(
                    world, parent_idx, child_idx, anchor_p, anchor_c,
                    axis_world, rom_min, rom_max, kp, kd,
                    Quat::identity());
                ja_by_child[spec.child as usize] = ja;
            }
        }
    }

    FootHandles { bodies, ja_by_child }
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

    /// E4e (Commit C): the coupled solve's dt-independence claim (design
    /// §2.5), asserted at ASSEMBLY level on a grounded toe press (anchored
    /// talus, hallux MTP ramped to +0.3 rad at the demo's
    /// muscles-don't-step 0.2 rad/s, 1 s settle + 2.5 s run), at
    /// sub_dt = 1/7200 vs 1/3600 — compared at MATCHED CONVERGENCE
    /// EFFORT: 32 sweeps @1/7200 vs 64 @1/3600, i.e. the same 230k SI
    /// sweeps per simulated second. The L2 warm seeds re-inject w·λ ∝ dt
    /// of velocity every substep, so the per-substep sweep demand scales
    /// with dt: at FIXED 32 sweeps the coarse rate under-converges and
    /// the settled angle drops to +0.1704 (measured — the honest
    /// assembly-level limitation, recorded in docs/findings.md
    /// 2026-10-21; single joints are exact at any count per E1/E6).
    /// Matching sweeps-per-second is the apples-to-apples comparison of
    /// the COUPLED FIXED POINT itself, which is what §2.5 claims is
    /// dt-independent.
    ///
    /// BAND DERIVATION (measured 2026-10-21): θ = +0.2347 @1/7200×32 vs
    /// +0.2343 @1/3600×64 — 0.17% apart; band 2% = E1(iii)'s cross-rate
    /// band applied to the settled pose (12× margin). The pp LIFT is NOT
    /// banded across rates: measured 2.41 vs 1.79 mm (±40% scatter across
    /// the sweep grid at fixed physics — the mm-scale lift rides on the
    /// statically indeterminate contact/weld preload split documented at
    /// xpbd.rs's KNOWN LIMITATION), so it is asserted qualitatively only:
    /// the press genuinely lifts the pp (≥ 0.8 mm; grid minimum 0.90 mm,
    /// and the BLOCKED Stage-2 branch measured 0.6 mm — this bites the
    /// transmission gap). The legacy pipeline flipped grounded outcomes
    /// qualitatively with sub_dt (E2's flag-off receipt), so the angle
    /// band is a real discriminator. BITE-VERIFIED: asserting the naive
    /// fixed-sweeps pair (32/32) fails the 2% angle band at 27% — the
    /// matched-effort comparison is load-bearing, not cosmetic.
    #[test]
    fn grounded_toe_press_is_substep_rate_independent() {
        let run = |sub_dt: f64, iters: usize| -> (f64, f64) {
            let mut world = XpbdWorld::new();
            world.set_ground(0.0);
            let foot = build_foot(&mut world, Vec3::new(0.0, 0.039, 0.0));
            world.solver_iterations = iters;
            world.bodies[foot.body(Bone::Talus)].pin();
            let ja_idx = foot.ja_by_child[Bone::HalluxPp as usize];
            let pp_idx = foot.body(Bone::HalluxPp);
            let steps_per_s = (1.0 / sub_dt).round() as usize;
            for _ in 0..steps_per_s {
                world.substep(sub_dt); // settle
            }
            let pp_y0 = world.bodies[pp_idx].position.y;
            let total = (2.5 * steps_per_s as f64) as usize;
            for i in 0..total {
                let t = i as f64 * sub_dt;
                world.joint_axes[ja_idx].target_angle = (0.2 * t).min(0.3);
                world.substep(sub_dt);
            }
            let theta = world.joint_axes[ja_idx].angle(&world.bodies);
            let lift = world.bodies[pp_idx].position.y - pp_y0;
            (theta, lift)
        };
        let (th_f, lift_f) = run(1.0 / 7200.0, 32);
        let (th_c, lift_c) = run(1.0 / 3600.0, 64);
        println!(
            "dual-rate toe press: θ {th_f:+.4} / {th_c:+.4} rad, lift {:.2} / {:.2} mm",
            lift_f * 1000.0, lift_c * 1000.0
        );
        assert!(
            th_f > 0.1,
            "rig broken: grounded press did not dorsiflex (θ = {th_f})"
        );
        assert!(
            (th_f - th_c).abs() <= 0.02 * th_f.abs(),
            "settled MTP angle differs across substep rates at matched convergence \
             effort: {th_f} vs {th_c} rad (> 2%)"
        );
        assert!(
            lift_f >= 0.0008 && lift_c >= 0.0008,
            "grounded press failed to lift the pp at one rate: {lift_f} / {lift_c} m \
             (≥ 0.8 mm; the blocked force-transmission branch measured 0.6 mm)"
        );
    }

    #[test]
    fn build_foot_creates_expected_joint_counts() {
        let mut world = XpbdWorld::new();
        let h = build_foot(&mut world, Vec3::new(0.0, 0.30, 0.0));
        // 21 articulated joints in the foot module → 21 primary-axis
        // JointAxes. (The 22nd articulated joint, talocrural, is
        // host-created.)
        let ja_count = h.ja_by_child.iter().filter(|&&x| x != usize::MAX).count();
        assert_eq!(ja_count, 21, "expected 21 articulated JointAxes in the foot module");
        assert_eq!(world.joint_axes.len(), 21);
        // Stage-2 invariant: the module composes NO legacy coaxial
        // primitives on primary axes anymore (off-axis dampers are the
        // only legacy angular primitives left: 2 per articulated joint).
        assert_eq!(world.angular_springs.len(), 0);
        assert_eq!(world.angular_soft_limits.len(), 0);
        assert_eq!(world.angular_pd_controllers.len(), 0);
        assert_eq!(world.angular_dampers.len(), 42);
    }
}
