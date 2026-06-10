# 26-Bone Anatomical Foot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Phase-1 single-capsule foot with a 26-bone anatomical foot (distinct rigid body per bone, anatomically-axed articulated joints, welded rigid midfoot, per-bone ground contact) so that static balance under load holds on a real tripod support polygon with passive structure alone.

**Architecture:** A reusable `improper_football::anatomy::foot` module builds the whole foot assembly from data tables (bone parameters + joint specs, all sourced from the spec) using a small set of construction helpers. Positions are derived by kinematic chaining from bone lengths — no hand-placed magic coordinates. The validation binary (`ankle_anatomical.rs`) and the visual demo (`ankle_anatomical_demo.rs`) both consume the module, eliminating the current build_world duplication. The build is staged hindfoot → midfoot → forefoot → toes, with a stability/geometry checkpoint after each stage so solver cost and instability are caught early.

**Tech Stack:** Rust 2021, `physics` crate (XPBD), nalgebra. Bevy 0.14 for the demo only. Reference spec: `docs/superpowers/specs/2026-06-09-anatomical-foot-design.md`.

---

## Background the implementer must know

**World frame:** y-up. Foot extends +z (anterior), +x lateral (right foot), +y superior. Capsule long axis is local **+y** (the ground-contact code reads `orientation * (0,1,0)` as the capsule axis).

**Anatomical → world axis mapping.** The spec specifies joint axes in an anatomical frame: **+X anterior, +Z superior, +Y lateral**. World is **+z anterior, +y superior, +x lateral**. So an anatomical axis `(ax, ay, az)` maps to world `(ay, az, ax)`. A helper `anat_to_world(Vec3) -> Vec3` encodes this; every spec axis passes through it.

**Existing primitives (verified against the codebase):**
- `RigidBody::new_capsule(mass, half_h, radius)` — sets mass, capsule inertia, and `capsule_shape` (so the body collides with the world ground automatically).
- `XpbdWorld` has `set_ground(y)` (default ground is already at y=0), `engaged_endpoints(body_idx)`, `total_normal_force(body_idx)`, `is_in_contact(body_idx)`.
- Angular primitives: `AngularSpring::new(a,b,axis,k,baseline_angle)`, `AngularDamper::new(a,b,axis,c)`, `AngularSoftLimit::new(a,b,axis,min,max,k,exp)`, `AngularPDController::new(a,b,axis,kp,kd)`. The spring/limit/PD have a `.with_baseline(quat)` builder for non-identity rest orientation.
- `swing_twist_angle(q, axis)` in `physics::math`.
- Positional joint: `PositionalSpring`/`PositionalDamper` with public `local_anchor_a`/`local_anchor_b`. The existing `add_elastic_joint` and `weld_bodies` helpers in `ankle_anatomical.rs` wrap these — they move into the foot module.

**Kinematic chaining for placement.** Each bone is positioned relative to its parent by placing the joint at the parent's distal tip and the child's proximal tip, using `place_child`. Joint anchors are at `±half_h` along each bone's long axis (its column direction). Lateral offsets between the medial/central/lateral columns come from bone widths. This makes every coordinate a function of the spec's dimension data — nothing hand-placed.

**Column assignment (anatomical):**
- Medial column: talus → navicular → medial cuneiform → MT1 → hallux PP → hallux DP
- Central column: (under talus) intermediate cuneiform → MT2 → toe2 PP/MP/DP; lateral cuneiform → MT3 → toe3 PP/MP/DP
- Lateral column: calcaneus → cuboid → MT4 → toe4 PP/MP/DP; MT5 → toe5 PP/MP/DP

**Important reality flag — positions are starting values.** The chaining scheme produces an anatomically-plausible starting layout, but exact rest positions of irregular tarsals are approximate. Each stage's checkpoint test (does it rest stably on the ground at the expected height? does the support polygon look right?) is what validates and refines the positions. When a checkpoint fails on geometry, adjust the column offsets/arch heights in the data table — do NOT fudge the test thresholds.

---

## File structure

- **Create:** `src/anatomy/mod.rs` — declares `pub mod foot;`
- **Create:** `src/anatomy/foot.rs` — the foot assembly: bone-parameter table, joint-spec tables, construction helpers, `FootHandles` struct, `build_foot(world, shin_idx, attach_anchor) -> FootHandles`, and unit tests.
- **Modify:** `src/lib.rs` — add `pub mod anatomy;`
- **Modify:** `src/bin/ankle_anatomical.rs` — replace inline foot/forefoot construction with `anatomy::foot::build_foot`; keep shin + stick + tests; update tests for the new structure.
- **Modify:** `src/bin/ankle_anatomical_demo.rs` — consume the foot module; render every bone.

---

## Task 1: Anatomy module scaffold + bone parameter table

**Files:**
- Create: `src/anatomy/mod.rs`
- Create: `src/anatomy/foot.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Wire the module**

Add to `src/lib.rs` (after the existing `pub mod` lines):

```rust
pub mod anatomy;
```

Create `src/anatomy/mod.rs`:

```rust
//! Anatomical assemblies built from physics primitives.
pub mod foot;
```

- [ ] **Step 2: Write the bone parameter table in `src/anatomy/foot.rs`**

```rust
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
```

- [ ] **Step 3: Write a unit test verifying the table integrity**

Add to the bottom of `src/anatomy/foot.rs`:

```rust
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
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p improper-football --lib anatomy::foot 2>&1 | tail -15`
Expected: both tests PASS. (If the mass-sum test fails, the BONES table was transcribed wrong — fix the table, not the test.)

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/anatomy/mod.rs src/anatomy/foot.rs
git commit -m "anatomy: foot module scaffold + 26-bone parameter table"
```

---

## Task 2: Joint spec tables + construction helpers

**Files:**
- Modify: `src/anatomy/foot.rs`

- [ ] **Step 1: Add the joint spec types and tables**

Insert above the `#[cfg(test)]` block:

```rust
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
    // Subtalar: incline 42°, deviate 23° medial. Axis built in code (see
    // subtalar_axis_anat()), so the table carries a placeholder unit axis
    // that the builder overrides for Bone::Calcaneus parent==Talus.
    JointSpec { parent: Bone::Talus, child: Bone::Calcaneus,
        kind: JointKind::Articulated { axis_anat: Vec3::new(1.0, 0.0, 0.0),
            rom_min: -0.17, rom_max: 0.44, kp: 50.0, kd: 2.0 } }, // ev 10°/inv 25°
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
    // MTP ×5 (axis mediolateral = anatomical +Y; dorsi = +, plantar = -)
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
    // IP hallux, PIP/DIP toes (flexion-dominant, axis mediolateral)
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
```

NOTE: count the array — 21 articulated + 8 welds = 29. The `[JointSpec; 29]` length is asserted at compile time; if you add/remove a row, update the count. (The 22nd articulated joint in the spec's whole-assembly count is the host-created talocrural.)

- [ ] **Step 2: Add the subtalar oblique-axis helper**

The subtalar axis is special (Manter/Inman oblique). Add:

```rust
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
```

- [ ] **Step 3: Add a test for the subtalar axis**

In the tests module:

```rust
#[test]
fn subtalar_axis_has_expected_inclination() {
    let a = subtalar_axis_anat();
    // inclination above transverse plane = asin(z component)
    let incl_deg = a.z.asin().to_degrees();
    assert!((incl_deg - 42.0).abs() < 1.0, "subtalar incl {incl_deg}, expected 42");
    assert!(a.y < 0.0, "subtalar axis should deviate medially (-Y)");
    assert!((a.norm() - 1.0).abs() < 1e-9, "axis must be unit");
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p improper-football --lib anatomy::foot 2>&1 | tail -15`
Expected: all (now 3) tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/anatomy/foot.rs
git commit -m "anatomy: foot joint spec tables + subtalar oblique axis"
```

---

## Task 3: Construction helpers (bodies, joints, welds) + FootHandles

**Files:**
- Modify: `src/anatomy/foot.rs`

- [ ] **Step 1: Add imports and the FootHandles struct**

At the top of `foot.rs`, extend imports:

```rust
use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_spring::AngularSpring;
use physics::angular_damper::AngularDamper;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use nalgebra::UnitQuaternion;
```

Add the handles struct:

```rust
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
```

- [ ] **Step 2: Add the positional-joint and weld helpers**

```rust
fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    anchor_parent: Vec3, anchor_child: Vec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = anchor_parent; s.local_anchor_b = anchor_child;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = anchor_parent; d.local_anchor_b = anchor_child;
    world.positional_dampers.push(d);
}

fn weld(
    world: &mut XpbdWorld, a: usize, b: usize,
    anchor_a: Vec3, anchor_b: Vec3,
) {
    add_elastic_joint(world, a, b, anchor_a, anchor_b, WELD_POS_K, WELD_POS_C);
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_springs.push(AngularSpring::new(a, b, axis, WELD_ANG_K, 0.0));
        world.angular_dampers.push(AngularDamper::new(a, b, axis, WELD_ANG_C));
    }
}
```

- [ ] **Step 3: Add the articulated-joint helper**

```rust
/// Adds an articulated joint: positional anchor, primary-axis ROM
/// soft-limit + passive spring/damper + PD controller, and two locked
/// off-axes. `axis_world` is the primary axis already mapped to world.
/// `baseline` is the child's rest orientation relative to the parent
/// (identity if both share orientation). Returns the PD index.
fn add_articulated(
    world: &mut XpbdWorld, parent: usize, child: usize,
    anchor_parent: Vec3, anchor_child: Vec3,
    axis_world: Vec3, rom_min: f64, rom_max: f64, kp: f64, kd: f64,
    baseline: UnitQuaternion<f64>,
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

    // Two off-axes orthogonal to the primary, locked.
    let helper = if primary.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    let off1 = primary.cross(&helper).normalize();
    let off2 = primary.cross(&off1).normalize();
    for off in [off1, off2] {
        world.angular_soft_limits.push(
            AngularSoftLimit::new(parent, child, off, -OFF_AXIS_LIM, OFF_AXIS_LIM, OFF_AXIS_LIMIT_K, ARTIC_LIMIT_EXP)
                .with_baseline(baseline));
        world.angular_dampers.push(AngularDamper::new(parent, child, off, ARTIC_PASSIVE_C));
        world.angular_springs.push(
            AngularSpring::new(parent, child, off, OFF_AXIS_K, 0.0).with_baseline(baseline));
    }
    pd_idx
}
```

- [ ] **Step 4: Build (lib only) to confirm helpers compile**

Run: `cargo build -p improper-football --lib 2>&1 | tail -15`
Expected: `Finished` (warnings about unused helpers are fine — Task 4 wires them).

- [ ] **Step 5: Commit**

```bash
git add src/anatomy/foot.rs
git commit -m "anatomy: foot construction helpers + FootHandles"
```

---

## Task 4: Bone placement table + `build_foot` (bodies only, no joints yet)

This task places all 26 bones in space via the column-chaining scheme and adds them to the world, returning a partial `FootHandles` (joints come in Task 5). The placement table encodes each bone's column position and arch height as offsets; positions are computed from these plus the bone half_h values.

**Files:**
- Modify: `src/anatomy/foot.rs`

- [ ] **Step 1: Add the rest-position table**

The talus is the reference: it sits at the ankle, at world position passed in by the host (the `attach` argument). Every other bone's rest position is an offset from the foot origin (the point on the ground directly below the talus center). Offsets are in world meters: `(x lateral, y up, z anterior)`. These are starting values derived from anatomical proportions (heel behind, arch raised, toes forward and low); they are refined by the stage checkpoints.

```rust
/// Rest position of each bone's CENTER relative to the foot origin
/// (ground point below the talus), in world meters (x lat, y up, z ant).
/// Starting layout per spec §2 column structure; refined by stage tests.
const REST_POS: [Vec3; BONE_COUNT] = [
    Vec3::new( 0.005, 0.030, -0.045), // calcaneus (heel: behind, low)
    Vec3::new( 0.000, 0.057,  0.000), // talus (on top, at ankle)
    Vec3::new(-0.008, 0.045,  0.030), // navicular (medial midfoot, arch apex)
    Vec3::new( 0.022, 0.030,  0.020), // cuboid (lateral midfoot)
    Vec3::new(-0.015, 0.038,  0.052), // medial cuneiform
    Vec3::new(-0.002, 0.040,  0.050), // intermediate cuneiform
    Vec3::new( 0.012, 0.038,  0.050), // lateral cuneiform
    Vec3::new(-0.018, 0.028,  0.095), // mt1
    Vec3::new(-0.006, 0.026,  0.100), // mt2
    Vec3::new( 0.006, 0.026,  0.098), // mt3
    Vec3::new( 0.018, 0.026,  0.092), // mt4
    Vec3::new( 0.028, 0.026,  0.085), // mt5
    Vec3::new(-0.020, 0.020,  0.135), // hallux_pp
    Vec3::new(-0.021, 0.018,  0.158), // hallux_dp
    Vec3::new(-0.008, 0.020,  0.140), // toe2_pp
    Vec3::new(-0.008, 0.018,  0.156), // toe2_mp
    Vec3::new(-0.008, 0.017,  0.166), // toe2_dp
    Vec3::new( 0.004, 0.020,  0.138), // toe3_pp
    Vec3::new( 0.004, 0.018,  0.153), // toe3_mp
    Vec3::new( 0.004, 0.017,  0.162), // toe3_dp
    Vec3::new( 0.016, 0.020,  0.132), // toe4_pp
    Vec3::new( 0.016, 0.018,  0.146), // toe4_mp
    Vec3::new( 0.016, 0.017,  0.154), // toe4_dp
    Vec3::new( 0.027, 0.020,  0.122), // toe5_pp
    Vec3::new( 0.027, 0.018,  0.134), // toe5_mp
    Vec3::new( 0.027, 0.017,  0.143), // toe5_dp
];

/// Rest orientation of each bone. Most bones lie with their long axis
/// (local +y) pointing anterior (world +z) → rotate local +y to +z via
/// R_x(-π/2). Tarsals that are taller than long (cuneiforms) keep their
/// long axis vertical-ish; for Phase 1 simplicity ALL bones use the
/// anterior-pointing baseline — the capsule shape is approximate for
/// blocky tarsals anyway (spec §6 risk).
fn bone_baseline() -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0))
}
```

- [ ] **Step 2: Add `build_foot` (bodies only for now)**

```rust
/// Build the 26-bone foot into `world`. `attach` is the world position
/// of the talus center (the host places it at the shin's distal end).
/// The talocrural joint (shin↔talus) is created by the HOST, not here.
/// Returns FootHandles with all body indices; PD indices are filled in
/// Task 5 (joints).
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
        body.position = origin + REST_POS[i];
        body.orientation = baseline;
        bodies[i] = world.add_body(body);
    }

    FootHandles { bodies, pd_by_child: [usize::MAX; BONE_COUNT] }
}
```

- [ ] **Step 3: Add a placement sanity test**

```rust
#[test]
fn build_foot_places_26_bodies_above_ground() {
    let mut world = XpbdWorld::new();
    let attach = Vec3::new(0.0, 0.30, 0.0); // talus 30 cm up
    let h = build_foot(&mut world, attach);
    assert_eq!(world.bodies.len(), BONE_COUNT);
    // Every bone's lowest point should be at or above the ground (y=0),
    // allowing 5 mm penetration tolerance for the starting layout.
    for i in 0..BONE_COUNT {
        let b = &world.bodies[h.bodies[i]];
        let lowest = b.position.y - BONES[i].radius - BONES[i].half_h;
        assert!(lowest > -0.05, "bone {} starts {lowest} below ground", BONES[i].name);
    }
    // Talus should be at the attach point.
    let talus = &world.bodies[h.body(Bone::Talus)];
    assert!((talus.position - attach).norm() < 1e-9);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p improper-football --lib anatomy::foot 2>&1 | tail -20`
Expected: all PASS. If the placement test fails because a bone starts below ground, adjust that bone's `REST_POS` y (raise it) — the layout is starting values.

- [ ] **Step 5: Commit**

```bash
git add src/anatomy/foot.rs
git commit -m "anatomy: bone placement table + build_foot (bodies)"
```

---

## Task 5: Wire all joints into `build_foot`

**Files:**
- Modify: `src/anatomy/foot.rs`

- [ ] **Step 1: Add a helper to compute a joint anchor on a bone**

The anchor where a joint sits on a bone is the point on the bone's surface facing its neighbor. With the column-chaining approximation, use the vector from the bone center toward the neighbor, scaled to the bone's half-extent. Add:

```rust
/// Local-frame anchor on `bone` pointing toward `toward` (a world point).
/// Converts the world direction into the bone's local frame and scales
/// to roughly the bone's half-length so the anchor sits on its surface.
fn anchor_toward(world: &XpbdWorld, bone_idx: usize, bone: Bone, toward: Vec3) -> Vec3 {
    let b = &world.bodies[bone_idx];
    let dir_world = (toward - b.position).normalize();
    let dir_local = b.orientation.inverse() * dir_world;
    dir_local * BONES[bone as usize].half_h
}
```

- [ ] **Step 2: Extend `build_foot` to add joints after the bodies loop**

Insert before the `FootHandles { ... }` return:

```rust
    let mut pd_by_child = [usize::MAX; BONE_COUNT];

    for spec in JOINTS.iter() {
        let parent_idx = bodies[spec.parent as usize];
        let child_idx = bodies[spec.child as usize];
        let parent_pos = world.bodies[parent_idx].position;
        let child_pos = world.bodies[child_idx].position;
        let anchor_p = anchor_toward(world, parent_idx, spec.parent, child_pos);
        let anchor_c = anchor_toward(world, child_idx, spec.child, parent_pos);

        match spec.kind {
            JointKind::Weld => {
                weld(world, parent_idx, child_idx, anchor_p, anchor_c);
            }
            JointKind::Articulated { axis_anat, rom_min, rom_max, kp, kd } => {
                // Subtalar uses the special oblique axis.
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
                    UnitQuaternion::identity());
                pd_by_child[spec.child as usize] = pd;
            }
        }
    }

    FootHandles { bodies, pd_by_child }
```

(Replace the previous bare `FootHandles { bodies, pd_by_child: [...] }` return with the loop above followed by the populated return.)

- [ ] **Step 3: Add a joint-count test**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p improper-football --lib anatomy::foot 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/anatomy/foot.rs
git commit -m "anatomy: wire all 30 foot joints into build_foot"
```

---

## Task 6: Hindfoot stability checkpoint (staged validation)

Before trusting the full assembly, verify the foot alone (no shin/stick) settles stably on the ground. This is the first dynamics test and the first solver-cost measurement.

**Files:**
- Modify: `src/anatomy/foot.rs` (add an integration test)

- [ ] **Step 1: Add a settle test**

```rust
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
```

- [ ] **Step 2: Run the test (also times it)**

Run: `cargo test -p improper-football --lib anatomy::foot::tests::foot_settles -- --nocapture 2>&1 | tail -20`
Expected: PASS. Note the wall-clock time reported by the test harness.

**If it explodes (NaN) or the heel flies off:** this is a real stability or placement problem, not a test to relax. Likely causes and fixes:
- Bones starting interpenetrating → ground/contact spikes. Spread the `REST_POS` so bones don't overlap at start.
- Tiny distal phalanges unstable → check the per-bone stability bound (see spec §6); if a 2 g phalanx is the culprit, its joint stiffness may exceed the explicit-damper limit — reduce its joint K or raise its mass slightly within anatomical range.
- Weld springs injecting energy → lower `WELD_ANG_K` until stable, document the value.
Report findings; do not move on with a failing checkpoint.

- [ ] **Step 3: Commit**

```bash
git add src/anatomy/foot.rs
git commit -m "anatomy: hindfoot/full-foot settle checkpoint passes"
```

---

## Task 7: Integrate the foot into `ankle_anatomical.rs`

Replace the inline foot/forefoot/talus construction with `build_foot`, keeping shin + stick + talocrural + ground + the existing test framework.

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Replace `build_world`'s foot section**

Remove the old foot, forefoot, talus bodies and the subtalar/MTP joint construction. Keep: shin, stick, the stick weld, ground. Add the foot via the module. The new `build_world`:

```rust
use improper_football::anatomy::foot::{self, Bone, FootHandles};

struct Assembly {
    world: XpbdWorld,
    foot: FootHandles,
    shin_idx: usize,
    stick_idx: usize,
    talocrural_pd_idx: usize,
}

fn build_world() -> Assembly {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    // Shin: vertical, distal (bottom) end at the ankle (talus attach point).
    let ankle_y = 0.075; // talus center height above ground at rest
    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    shin.position = Vec3::new(0.0, ankle_y + SHIN_HALF_H, 0.0);
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // Foot: talus attaches at the shin's distal end.
    let attach = Vec3::new(0.0, ankle_y, 0.0);
    let foot = foot::build_foot(&mut world, attach);

    // Talocrural joint: shin ↔ talus. Axis world +x (mediolateral),
    // anatomical lateral-inferior/posterior tilt folded in via anat map.
    let talus_idx = foot.body(Bone::Talus);
    add_elastic_joint(&mut world, shin_idx, talus_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0), Vec3::new(0.0, 0.0, 0.0),
        60_000.0, 150.0);
    let tc_axis = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        shin_idx, talus_idx, tc_axis, -0.35, 0.70, 10_000.0, 2.0));
    world.angular_dampers.push(AngularDamper::new(shin_idx, talus_idx, tc_axis, 1.2));
    world.angular_springs.push(AngularSpring::new(shin_idx, talus_idx, tc_axis, 4.0, 0.0));
    // Capture the talocrural PD index BEFORE pushing (it lands right
    // after the foot's 21 PDs, but capture dynamically — never hardcode).
    let talocrural_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        shin_idx, talus_idx, tc_axis, 80.0, 1.0));
    for off in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            shin_idx, talus_idx, off, -0.05, 0.05, 10_000.0, 2.0));
        world.angular_dampers.push(AngularDamper::new(shin_idx, talus_idx, off, 1.2));
        world.angular_springs.push(AngularSpring::new(shin_idx, talus_idx, off, 200.0, 0.0));
    }

    // Load stick welded to shin top.
    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    stick.position = Vec3::new(0.0, ankle_y + 2.0 * SHIN_HALF_H + STICK_HALF_H, 0.0);
    stick.orientation = id;
    let stick_idx = world.add_body(stick);
    weld_bodies(&mut world, shin_idx, stick_idx,
        Vec3::new(0.0, SHIN_HALF_H, 0.0), Vec3::new(0.0, -STICK_HALF_H, 0.0));

    Assembly { world, foot, shin_idx, stick_idx, talocrural_pd_idx }
}
```

Keep the `weld_bodies` helper local to the binary (or import from the foot module if you make it `pub`). Delete the now-unused foot/forefoot/subtalar/MTP constants and the old `AnkleHandles` struct.

- [ ] **Step 2: Update `test_sustained_stability` and `test_static_balance` to the new `Assembly`**

```rust
fn test_sustained_stability() -> TestResult {
    let mut a = build_world();
    let mut blew_up = false;
    simulate(&mut a.world, 10.0, |w, _t| {
        if !w.bodies.iter().all(|b| b.position.y.is_finite()) { blew_up = true; }
    });
    if blew_up { return TestResult::fail("sustained stability", "non-finite position".into()); }
    TestResult::pass("sustained stability")
}

fn test_static_balance() -> TestResult {
    let mut a = build_world();
    let mut max_tilt = 0.0_f64;
    simulate(&mut a.world, 5.0, |w, _t| {
        let tilt = w.bodies[a.stick_idx].orientation.scaled_axis().norm();
        if tilt > max_tilt { max_tilt = tilt; }
    });
    let deg = max_tilt.to_degrees();
    if deg < 5.0 { TestResult::pass("static balance") }
    else { TestResult::fail("static balance", format!("max tilt {deg:.2}° (limit 5°)")) }
}
```

- [ ] **Step 3: Build**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -15`
Expected: `Finished`. Fix any leftover references to deleted constants/structs.

- [ ] **Step 4: Run — this is the headline test**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -10`
Expected: `sustained stability: PASS`. For `static balance`: the goal is PASS (tripod holds the load). If it still tips:
- Check the support polygon: print the ground-contact points (`world.engaged_endpoints` for calcaneus, MT1, MT5). They should form a triangle spanning the foot. If they're collinear or tiny, the bone positions need adjustment (widen the lateral column offset, ensure heel and metatarsal heads actually reach the ground).
- This is genuine tuning of the `REST_POS` table, not threshold-fudging. Report the contact geometry you observe.

- [ ] **Step 5: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle: integrate 26-bone foot; static balance on tripod"
```

---

## Task 8: Foot-specific validation tests

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Center-of-pressure migration test**

Add a helper that computes the foot's center of pressure (force-weighted average of all ground contact points across all foot bones), then a test that commands a heel→toe roll (via the talocrural PD) and asserts CoP moves anteriorly (world +z increases monotonically over the roll).

```rust
fn center_of_pressure(world: &XpbdWorld, foot: &FootHandles) -> Option<Vec3> {
    let mut num = Vec3::zeros();
    let mut den = 0.0;
    for i in 0..foot::BONE_COUNT {
        let idx = foot.bodies[i];
        for ep in world.engaged_endpoints(idx).iter().flatten() {
            num += ep.contact_point * ep.normal_force;
            den += ep.normal_force;
        }
    }
    if den > 1e-6 { Some(num / den) } else { None }
}

fn test_cop_migration() -> TestResult {
    let mut a = build_world();
    // Settle.
    simulate(&mut a.world, 1.0, |_, _| {});
    let cop_start = center_of_pressure(&a.world, &a.foot);
    // Command talocrural into plantarflexion (heel rises, weight to toes).
    a.world.angular_pd_controllers[a.talocrural_pd_idx].target_angle = 0.5;
    let mut cop_end = None;
    simulate(&mut a.world, 1.5, |w, _t| { cop_end = center_of_pressure(w, &a.foot); });
    match (cop_start, cop_end) {
        (Some(s), Some(e)) if e.z > s.z + 0.01 => TestResult::pass("CoP migration"),
        (Some(s), Some(e)) => TestResult::fail("CoP migration",
            format!("CoP z {:.3}→{:.3}, expected anterior shift", s.z, e.z)),
        _ => TestResult::fail("CoP migration", "no ground contact".into()),
    }
}
```

The talocrural PD index comes from `a.talocrural_pd_idx` (captured in `build_world`, Task 7). Never hardcode it.

- [ ] **Step 2: Arch compression test**

```rust
fn test_arch_compression() -> TestResult {
    let mut a = build_world();
    let nav = a.foot.body(Bone::Navicular);
    let y0 = a.world.bodies[nav].position.y;
    simulate(&mut a.world, 2.0, |_, _| {});
    let y1 = a.world.bodies[nav].position.y;
    // Navicular (arch apex) should drop a little under load but not collapse.
    let drop = y0 - y1;
    if drop >= 0.0 && drop < 0.02 { TestResult::pass("arch compression") }
    else { TestResult::fail("arch compression", format!("navicular dropped {drop:.4} m")) }
}
```

- [ ] **Step 3: MTP push-off commanded-motion test**

```rust
fn test_mtp_pushoff() -> TestResult {
    let mut a = build_world();
    // Command hallux MTP into dorsiflexion (toes up).
    let pd = a.foot.pd_by_child[Bone::HalluxPp as usize];
    a.world.angular_pd_controllers[pd].target_angle = 0.8;
    simulate(&mut a.world, 1.5, |_, _| {});
    let angle = joint_angle_for_pd(&a.world, pd);
    if (angle - 0.8).abs() < 0.1 { TestResult::pass("MTP push-off") }
    else { TestResult::fail("MTP push-off", format!("hallux MTP {angle:.3}, target 0.8")) }
}
```

Add a small `joint_angle_for_pd` helper that reads the PD's bodies/axis and returns `swing_twist_angle`.

- [ ] **Step 4: Wire the new tests into `main` and run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -15`
Expected: CoP migration, arch compression, MTP push-off all PASS (plus the earlier stability/balance tests). Tune `REST_POS` / gains if a test fails on geometry — report specifics, never relax thresholds silently.

- [ ] **Step 5: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle: CoP migration, arch compression, MTP push-off tests"
```

---

## Task 9: Inversion-stability + ROM + passive-moment tests

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Subtalar inversion ROM test**

Command the subtalar PD (child = Calcaneus) to its inversion limit and verify it clamps within ROM+10%.

```rust
fn test_subtalar_rom() -> TestResult {
    let mut a = build_world();
    let pd = a.foot.pd_by_child[Bone::Calcaneus as usize];
    a.world.angular_pd_controllers[pd].target_angle = 1.0; // beyond inv limit 0.44
    simulate(&mut a.world, 2.0, |_, _| {});
    let angle = joint_angle_for_pd(&a.world, pd);
    let limit = 0.44 * 1.10;
    if angle <= limit { TestResult::pass("subtalar ROM clamp") }
    else { TestResult::fail("subtalar ROM clamp", format!("{angle:.3} > {limit:.3}")) }
}
```

- [ ] **Step 2: Talocrural passive-moment test**

Pin shin and talus, force talocrural to 0.05 rad past the dorsiflexion limit, disable the talocrural PD, apply only passive structures, read the resulting torque (≈25 N·m within 30%, per spec). Mirror the Phase-1 test 9 approach (set `inv_mass=0` on shin and talus, set talus orientation, clear accumulators, apply springs+limits once, read `torque_accumulator`).

```rust
fn test_talocrural_passive_moment() -> TestResult {
    let mut a = build_world();
    let talus = a.foot.body(Bone::Talus);
    a.world.angular_pd_controllers[a.talocrural_pd_idx].enabled = false;
    a.world.bodies[a.shin_idx].inv_mass = 0.0;
    a.world.bodies[talus].inv_mass = 0.0;
    a.world.bodies[talus].orientation = UnitQuaternion::from_scaled_axis(Vec3::new(-0.40, 0.0, 0.0));
    a.world.bodies[a.shin_idx].clear_accumulators();
    a.world.bodies[talus].clear_accumulators();
    for s in &a.world.angular_springs { s.apply(&mut a.world.bodies); }
    for l in &mut a.world.angular_soft_limits { l.apply(&mut a.world.bodies); }
    let m = a.world.bodies[talus].torque_accumulator.x.abs();
    if (m - 25.0).abs() < 7.5 { TestResult::pass("talocrural passive moment") }
    else { TestResult::fail("talocrural passive moment", format!("{m:.2} N·m, expected 25±7.5")) }
}
```

(Verify `AngularSpring::apply`/`AngularSoftLimit::apply` signatures against the crate; adjust if they take different arguments. If `enabled` isn't a field on `AngularPDController`, instead remove/skip the PD — check the type.)

- [ ] **Step 3: Wire into `main`, run all**

Run: `cargo run --release --bin ankle_anatomical 2>&1`
Expected: full suite PASS. Report any failures with measured values.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle: subtalar ROM + talocrural passive-moment tests"
```

---

## Task 10: Performance measurement + main summary

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add timing to the full run**

Wrap the test suite in `std::time::Instant` timing; print total wall-clock and a real-time ratio (sim seconds / wall seconds) for the 10 s stability test.

```rust
fn main() {
    let start = std::time::Instant::now();
    let tests: Vec<fn() -> TestResult> = vec![
        test_sustained_stability, test_static_balance, test_cop_migration,
        test_arch_compression, test_mtp_pushoff, test_subtalar_rom,
        test_talocrural_passive_moment,
    ];
    let mut pass = 0;
    for t in tests {
        let r = t();
        println!("  [{}] {}: {}", if r.passed {"PASS"} else {"FAIL"}, r.name, r.message);
        if r.passed { pass += 1; }
    }
    let n = 7;
    println!("SUMMARY: {pass}/{n} passed in {:.1?}", start.elapsed());
    if pass < n { std::process::exit(1); }
}
```

- [ ] **Step 2: Run and record performance**

Run: `cargo run --release --bin ankle_anatomical 2>&1`
Expected: all PASS. Record the total time. The 28-body assembly will be much slower than the 5-body Phase-1 (~1 s for 10 s sim). If the 10 s stability test no longer runs faster than real-time, note it — that's the spec's flagged solver-cost risk materializing, and a candidate for a follow-up optimization task (not addressed here unless it's so slow tests are impractical, i.e. >60 s total).

- [ ] **Step 3: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle: full validation suite + performance timing"
```

---

## Task 11: Update the visual demo to the 26-bone foot

**Files:**
- Modify: `src/bin/ankle_anatomical_demo.rs`

- [ ] **Step 1: Replace the duplicated build_world with the shared module**

Delete the demo's duplicated constants/build_world. Use the same `build_world`/`Assembly` approach as the binary (factor the shared `build_world` into the foot module or a small shared helper if convenient; otherwise mirror the binary's `build_world`). Spawn one capsule mesh per foot bone by iterating `foot.bodies`, plus shin and stick.

```rust
for i in 0..foot::BONE_COUNT {
    let p = &foot::BONES[i];
    let mesh = meshes.add(Capsule3d::new(p.radius as f32, (2.0 * p.half_h) as f32));
    let mat = materials.add(StandardMaterial { base_color: bone_color(i), ..default() });
    commands.spawn((PbrBundle {
        mesh, material: mat,
        transform: rigid_body_to_transform(&world.bodies[a.foot.bodies[i]]),
        ..default()
    }, BodyRef(a.foot.bodies[i])));
}
```

Add a `bone_color(i)` that tints tarsals/metatarsals/phalanges differently so the structure is visible.

- [ ] **Step 2: Build**

Run: `cargo build --release --bin ankle_anatomical_demo 2>&1 | tail -15`
Expected: `Finished`.

- [ ] **Step 3: Run briefly to confirm the foot renders and stands**

Run: `cargo run --release --bin ankle_anatomical_demo` (close after observing)
Expected: a recognizable multi-bone foot on the ground supporting the shin + stick, standing on its tripod (or tipping realistically if balance is marginal — but it should not clip through the floor, and the foot should show distinct bones).

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical_demo.rs
git commit -m "demo: render 26-bone foot from shared anatomy module"
```

---

## Task 12: Memory update

**Files:**
- Create: `C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\project_anatomical_foot.md`
- Modify: `C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\MEMORY.md`

- [ ] **Step 1: Write the memory file**

```markdown
---
name: 26-bone anatomical foot + world-level ground
description: Foot is now 26 distinct bones (src/anatomy/foot.rs) with anatomical oblique-axis joints and per-bone ground contact. Ground is a world-level property (XpbdWorld), not per-body GroundContact. Spec 2026-06-09, plan 2026-06-10.
metadata:
  type: project
---
The single-capsule foot was replaced with a 26-bone anatomical model after it failed static balance (knife-edge support → tipped to 90-180°).

**Two coupled changes:**
1. **World-level ground** — `GroundContact` (per-body opt-in) deleted. `XpbdWorld` owns a `Ground` (set via `set_ground(y)`); any `RigidBody` with `capsule_shape` (from `new_capsule`) collides automatically. Queries `engaged_endpoints`/`total_normal_force`/`is_in_contact` moved to the world. The old `with_plane_y(-radius)` disable-hack is gone. 18 binaries migrated.
2. **26-bone foot** — `src/anatomy/foot.rs`, reusable module (used by `ankle_anatomical` binary + demo). Data-driven: `BONES` parameter table + `JOINTS` spec table consumed by helpers. 22 articulated joints (oblique anatomical axes via `anat_to_world`), 8 welds (4 tree + 4 midfoot loop-closure). Positions derived by column-chaining, tunable via `REST_POS`.

**How to apply:** build feet/limbs via `anatomy` modules, not inline in binaries. New bodies should use `new_capsule` so they respect the ground. See [[project_four_phase_plan]] — this merged the Phase-1 foot with the deferred Phase-2 multi-region contact. Muscle model still Phase 4 (PDs are placeholders).

**Open:** solver cost at 28 bodies (measured in the validation binary); per-bone REST_POS positions are anatomically-plausible starting values refined against the CoP/balance tests, not cadaver-exact.
```

- [ ] **Step 2: Add the index line to MEMORY.md**

Append:

```markdown
- [26-bone anatomical foot + world-level ground](project_anatomical_foot.md) — foot is per-bone in src/anatomy/foot.rs; ground is a world property; 18 binaries migrated
```

- [ ] **Step 3: Verify the whole suite still passes**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -3` and `cargo build --release 2>&1 | tail -3`
Expected: validation suite passes; full workspace builds.

- [ ] **Step 4: Commit**

```bash
git add "C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\project_anatomical_foot.md" "C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\MEMORY.md"
git commit -m "memory: 26-bone foot + world-level ground"
```

---

## Notes for the implementing engineer

- **Tune positions, not thresholds.** When a balance/CoP/arch test fails, the fix is almost always in `REST_POS` (bone layout) or joint gains — never relaxing a test bound. The tests encode the spec's success criteria.
- **Frame discipline.** Every spec axis is anatomical; pass it through `anat_to_world` exactly once. The subtalar axis is built in anatomical coords by `subtalar_axis_anat()` then mapped — don't double-map.
- **Anchor approximation.** `anchor_toward` puts joint anchors on the line between bone centers. For tightly-packed tarsals this is approximate; if a joint visibly pulls bones together wrongly, replace that joint's anchors with explicit offsets. Flag it; don't silently special-case.
- **Solver cost is the known risk.** Measure at the Task 6 checkpoint and Task 10. If the assembly is too slow to iterate on, raise it before pressing forward — a follow-up optimization (broad-phase contact culling, per-island solving) may be needed, but that's out of this plan's scope.
- **PD index convention.** The foot module creates 21 PDs (creation order in `JOINTS`, indices 0..20); the host's talocrural PD is added next (index 21). Always read indices from `FootHandles::pd_by_child` and `Assembly::talocrural_pd_idx` — never hardcode the numbers.
