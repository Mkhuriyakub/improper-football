# Anatomical Ankle Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-body ankle assembly (talocrural + subtalar + MTP joints, with talus and forefoot as new bodies) using existing engine primitives, plus a validation test rig that exercises the unit on its own with 9 PASS/FAIL scenarios from the spec.

**Architecture:** Two new files in `src/bin/`. The text validation binary (`ankle_anatomical.rs`) is self-contained, prints PASS/FAIL per scenario, exits with non-zero status on failure. The visual companion (`ankle_anatomical_demo.rs`) shares the world setup and renders the assembly with a live-angles UI panel. No engine changes needed — Phase 1 composes existing primitives (`AngularPDController`, `AngularSpring`, `AngularDamper`, `AngularSoftLimit`, `PositionalSpring`, `PositionalDamper`, `GroundContact`).

**Tech Stack:** Rust 2021, project's physics crate (XPBD substrate), nalgebra for math. Bevy 0.14 for the visual demo only. Reference: `docs/superpowers/specs/2026-05-18-anatomical-ankle-phase1-design.md`.

---

## File structure

- **Create:** `src/bin/ankle_anatomical.rs`
  - Constants from spec
  - Helper functions: `capsule_inertia`, `place_child`, `add_elastic_joint`, `weld_bodies`, `compute_swing_twist_angle`
  - `AnkleHandles` struct (body and PD indices returned from world setup)
  - `build_world() -> (XpbdWorld, AnkleHandles)`
  - `TestResult` struct + `run_test` helper
  - One function per test scenario (9 total)
  - `main()` that runs all tests and prints summary
- **Create:** `src/bin/ankle_anatomical_demo.rs`
  - Same world setup (factored into a shared helper or duplicated)
  - Bevy app: camera, lights, capsule meshes per body, ground plane
  - UI overlay with live joint angles + scenario name
  - Scenario state machine cycling through commands

## Geometric layout reference (used throughout)

At neutral pose, all positions in world coordinates:

```
Body         | Center position       | Length | Notes
-------------|-----------------------|--------|-----------------------------
Foot         | (0, 0.032, +0.09)     | 0.18 m | baseline R_x(-π/2); heel at world (0,0.032,0), MTP at (0,0.032,+0.18)
Forefoot     | (0, 0.032, +0.215)    | 0.07 m | baseline R_x(-π/2); tip at world (0,0.032,+0.25)
Talus        | (0, 0.057, 0)         | 0.05 m | identity orientation; subtalar at bottom, talocrural at top
Shin         | (0, 0.292, 0)         | 0.42 m | identity orientation; bottom at (0,0.082,0) = talocrural
Stick (load) | (0, 0.902, 0)         | 0.80 m | identity orientation; bottom at (0,0.502,0) = shin top
```

Joint anchors (local frames):

```
Joint       | Body A (parent) | Anchor on A          | Body B (child) | Anchor on B
------------|-----------------|----------------------|----------------|--------------------
Talocrural  | shin            | (0, -0.21, 0)        | talus          | (0, +0.025, 0)
Subtalar    | talus           | (0, -0.025, 0)       | foot           | (0, +0.09, 0)
MTP         | foot            | (0, -0.09, 0)        | forefoot       | (0, +0.035, 0)
Stick weld  | shin            | (0, +0.21, 0)        | stick          | (0, -0.40, 0)
```

---

## Task 1: Create binary shell with constants and helpers

**Files:**
- Create: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Create the file with imports and constants**

Write to `src/bin/ankle_anatomical.rs`:

```rust
// Phase 1 anatomical ankle — text validation binary.
//
// Builds the multi-body ankle assembly (talocrural + subtalar + MTP)
// per the spec at docs/superpowers/specs/2026-05-18-anatomical-ankle-phase1-design.md
// and runs 9 PASS/FAIL test scenarios.
//
// All masses, lengths, ROMs, and gains come from the spec.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3, Mat3, swing_twist_angle};
use nalgebra::UnitQuaternion;

// ─── Simulation timing ────────────────────────────────────────────────

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ─── Body parameters (from spec §3) ───────────────────────────────────

const SHIN_MASS:    f64 = 3.00;
const SHIN_HALF_H:  f64 = 0.21;
const SHIN_RADIUS:  f64 = 0.05;

const TALUS_MASS:   f64 = 0.05;
const TALUS_HALF_H: f64 = 0.025;
const TALUS_RADIUS: f64 = 0.020;

const FOOT_MASS:    f64 = 0.85;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;

const FOREFOOT_MASS:    f64 = 0.10;
const FOREFOOT_HALF_H:  f64 = 0.035;
const FOREFOOT_RADIUS:  f64 = 0.020;

const STICK_MASS:    f64 = 7.00;  // load above shin; sized so total above-ankle ≈ 10 kg
const STICK_HALF_H:  f64 = 0.40;
const STICK_RADIUS:  f64 = 0.08;

// ─── Joint parameters (from spec §4) ──────────────────────────────────

// Talocrural (shin ↔ talus)
const TALOCRURAL_JOINT_K: f64 = 60_000.0;
const TALOCRURAL_JOINT_C: f64 =    150.0;
const TALOCRURAL_PRIMARY_KP: f64 = 80.0;
const TALOCRURAL_PRIMARY_KD: f64 =  1.0;
const TALOCRURAL_PASSIVE_K:  f64 =  4.0;
const TALOCRURAL_PASSIVE_C:  f64 =  1.2;
const TALOCRURAL_ROM_MIN: f64 = -0.35;
const TALOCRURAL_ROM_MAX: f64 = +0.70;
const TALOCRURAL_LIMIT_K: f64 = 10_000.0;
const TALOCRURAL_LIMIT_EXP: f64 = 2.0;
const TALOCRURAL_OFF_AXIS_K:    f64 = 200.0;
const TALOCRURAL_OFF_AXIS_LIM:  f64 = 0.05;
const TALOCRURAL_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

// Subtalar (talus ↔ foot)
const SUBTALAR_JOINT_K: f64 = 60_000.0;
const SUBTALAR_JOINT_C: f64 =    150.0;
const SUBTALAR_PRIMARY_KP: f64 = 50.0;
const SUBTALAR_PRIMARY_KD: f64 =  2.0;
const SUBTALAR_PASSIVE_K:  f64 =  3.5;
const SUBTALAR_PASSIVE_C:  f64 =  1.0;
const SUBTALAR_ROM_MIN: f64 = -0.26;
const SUBTALAR_ROM_MAX: f64 = +0.52;
const SUBTALAR_LIMIT_K: f64 = 8_000.0;
const SUBTALAR_LIMIT_EXP: f64 = 2.0;
const SUBTALAR_TWIST_K: f64 = 8.0;
const SUBTALAR_TWIST_C: f64 = 0.5;
const SUBTALAR_TWIST_LIM: f64 = 0.10;
const SUBTALAR_TWIST_LIMIT_K: f64 = 10_000.0;
const SUBTALAR_OFF_AXIS_K:    f64 = 200.0;
const SUBTALAR_OFF_AXIS_LIM:  f64 = 0.05;
const SUBTALAR_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

// MTP (foot ↔ forefoot)
const MTP_JOINT_K: f64 = 60_000.0;
const MTP_JOINT_C: f64 =    150.0;
const MTP_PRIMARY_KP: f64 = 30.0;
const MTP_PRIMARY_KD: f64 =  0.5;
const MTP_PASSIVE_K:  f64 =  5.0;
const MTP_PASSIVE_C:  f64 =  1.5;
const MTP_ROM_MIN: f64 = -0.35;
const MTP_ROM_MAX: f64 = +0.87;
const MTP_LIMIT_K: f64 = 5_000.0;
const MTP_LIMIT_EXP: f64 = 2.0;
const MTP_OFF_AXIS_K:    f64 = 100.0;
const MTP_OFF_AXIS_LIM:  f64 = 0.05;
const MTP_OFF_AXIS_LIMIT_K: f64 = 5_000.0;

// Stick weld to shin (stiff rigid attachment via springs)
const WELD_POS_K: f64 = 1_000_000.0;
const WELD_POS_C: f64 =     2_000.0;
const WELD_ANG_K: f64 =    50_000.0;
const WELD_ANG_C: f64 =       500.0;

fn main() {
    println!("PHASE 1 ANATOMICAL ANKLE VALIDATION");
    println!("(scaffold — tests not yet implemented)");
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished `release` profile`. There may be "unused" warnings; ignore them — they go away as later tasks use the constants.

- [ ] **Step 3: Add helper functions below the constants, above `fn main()`**

Insert these helpers after the constant block and before `fn main()`:

```rust
fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(
        i_trans, 0.0, 0.0,
        0.0, i_axial, 0.0,
        0.0, 0.0, i_trans,
    )
}

fn place_child(
    parent: &RigidBody, parent_distal: Vec3, child_proximal: Vec3,
    rel_rot: UnitQuaternion<f64>,
) -> (Vec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: Vec3, ca: Vec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = pa; s.local_anchor_b = ca;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = pa; d.local_anchor_b = ca;
    world.positional_dampers.push(d);
}

/// Welds two bodies together via stiff positional + 3-axis angular springs.
/// Use to attach the load stick rigidly to the shin top.
fn weld_bodies(
    world: &mut XpbdWorld, body_a: usize, body_b: usize,
    anchor_a: Vec3, anchor_b: Vec3,
) {
    // Positional weld
    add_elastic_joint(world, body_a, body_b, anchor_a, anchor_b, WELD_POS_K, WELD_POS_C);
    // 3-axis angular weld: K and 0-baseline springs on x, y, z to lock orientation.
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_springs.push(
            AngularSpring::new(body_a, body_b, axis, WELD_ANG_K, 0.0));
        world.angular_dampers.push(
            AngularDamper::new(body_a, body_b, axis, WELD_ANG_C));
    }
}

/// Read the signed angle of body_b relative to body_a about an axis in
/// body_a's local frame. Same convention as AngularPDController.
fn joint_angle(
    world: &XpbdWorld, body_a: usize, body_b: usize, axis_local_a: Vec3,
) -> f64 {
    let q_b_in_a = world.bodies[body_a].orientation.inverse()
                 * world.bodies[body_b].orientation;
    swing_twist_angle(q_b_in_a, axis_local_a)
}
```

- [ ] **Step 4: Build and verify**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished` with possibly some unused-function warnings.

- [ ] **Step 5: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: binary shell with constants and helpers"
```

---

## Task 2: Build world function — bodies

**Files:**
- Modify: `src/bin/ankle_anatomical.rs` (add struct and function)

- [ ] **Step 1: Add `AnkleHandles` struct after the helpers**

Insert after the helper functions:

```rust
#[derive(Clone, Copy)]
struct AnkleHandles {
    shin_idx: usize,
    talus_idx: usize,
    foot_idx: usize,
    forefoot_idx: usize,
    stick_idx: usize,
    talocrural_pd_idx: usize,
    subtalar_pd_idx: usize,
    mtp_pd_idx: usize,
}
```

- [ ] **Step 2: Add `build_world` function — bodies only**

Insert below the struct:

```rust
fn build_world() -> (XpbdWorld, AnkleHandles) {
    let mut world = XpbdWorld::new();
    let id = UnitQuaternion::identity();

    // Foot baseline: R_x(-π/2). Local +Y maps to world -Z (heel direction),
    // local -Y to world +Z (toe direction).
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    // Foot: center at world (0, 0.032, +0.09); heel at world (0, 0.032, 0).
    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    foot.position = Vec3::new(0.0, FOOT_RADIUS, 0.09);
    foot.orientation = foot_baseline;
    let foot_idx = world.add_body(foot);

    // Forefoot: attached to foot's toe end (world (0,0.032,+0.18)) at its
    // own heel end (local +Y). Carries foot baseline.
    let mut forefoot = RigidBody::new(FOREFOOT_MASS,
        capsule_inertia(FOREFOOT_MASS, FOREFOOT_RADIUS, FOREFOOT_HALF_H));
    let (pos, orient) = place_child(&world.bodies[foot_idx],
        Vec3::new(0.0, -FOOT_HALF_H, 0.0),         // foot toe end (foot local)
        Vec3::new(0.0, +FOREFOOT_HALF_H, 0.0),      // forefoot heel end (forefoot local)
        id);
    forefoot.position = pos;
    forefoot.orientation = orient;
    let forefoot_idx = world.add_body(forefoot);

    // Talus: above foot's heel (subtalar joint at world (0,0.032,0)).
    // Talus is at identity orientation.
    let mut talus = RigidBody::new(TALUS_MASS,
        capsule_inertia(TALUS_MASS, TALUS_RADIUS, TALUS_HALF_H));
    talus.position = Vec3::new(0.0, FOOT_RADIUS + TALUS_HALF_H, 0.0);  // (0, 0.057, 0)
    talus.orientation = id;
    let talus_idx = world.add_body(talus);

    // Shin: above talus (talocrural joint at world (0, 0.082, 0)).
    let mut shin = RigidBody::new(SHIN_MASS,
        capsule_inertia(SHIN_MASS, SHIN_RADIUS, SHIN_HALF_H));
    let talocrural_world_y = FOOT_RADIUS + 2.0 * TALUS_HALF_H;  // 0.082
    shin.position = Vec3::new(0.0, talocrural_world_y + SHIN_HALF_H, 0.0);  // (0, 0.292, 0)
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // Stick (load): above shin, welded to shin top.
    let mut stick = RigidBody::new(STICK_MASS,
        capsule_inertia(STICK_MASS, STICK_RADIUS, STICK_HALF_H));
    let shin_top_y = talocrural_world_y + 2.0 * SHIN_HALF_H;  // 0.502
    stick.position = Vec3::new(0.0, shin_top_y + STICK_HALF_H, 0.0);  // (0, 0.902, 0)
    stick.orientation = id;
    let stick_idx = world.add_body(stick);

    // PD indices: stub for now (filled in later tasks).
    let talocrural_pd_idx = usize::MAX;
    let subtalar_pd_idx = usize::MAX;
    let mtp_pd_idx = usize::MAX;

    (world, AnkleHandles {
        shin_idx, talus_idx, foot_idx, forefoot_idx, stick_idx,
        talocrural_pd_idx, subtalar_pd_idx, mtp_pd_idx,
    })
}
```

- [ ] **Step 3: Update `main` to call `build_world` and print positions**

Replace `main` with:

```rust
fn main() {
    let (world, h) = build_world();
    println!("Body positions at neutral:");
    println!("  foot     y={:.4}, z={:+.4}", world.bodies[h.foot_idx].position.y,    world.bodies[h.foot_idx].position.z);
    println!("  forefoot y={:.4}, z={:+.4}", world.bodies[h.forefoot_idx].position.y, world.bodies[h.forefoot_idx].position.z);
    println!("  talus    y={:.4}, z={:+.4}", world.bodies[h.talus_idx].position.y,   world.bodies[h.talus_idx].position.z);
    println!("  shin     y={:.4}, z={:+.4}", world.bodies[h.shin_idx].position.y,    world.bodies[h.shin_idx].position.z);
    println!("  stick    y={:.4}, z={:+.4}", world.bodies[h.stick_idx].position.y,   world.bodies[h.stick_idx].position.z);
}
```

- [ ] **Step 4: Run and verify positions match the spec layout**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -10`
Expected output:
```
Body positions at neutral:
  foot     y=0.0320, z=+0.0900
  forefoot y=0.0320, z=+0.2150
  talus    y=0.0570, z=+0.0000
  shin     y=0.2920, z=+0.0000
  stick    y=0.9020, z=+0.0000
```

- [ ] **Step 5: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: bodies placed per spec geometry"
```

---

## Task 3: Add talocrural joint

**Files:**
- Modify: `src/bin/ankle_anatomical.rs` (extend `build_world`)

- [ ] **Step 1: Add talocrural joint code after `shin_idx` is added but before stick body creation**

Find the line `// Stick (load): above shin, welded to shin top.` and insert before it:

```rust
    // ── Talocrural joint (shin ↔ talus) ─────────────────────────────
    // Positional joint at shin bottom / talus top.
    add_elastic_joint(&mut world, shin_idx, talus_idx,
        Vec3::new(0.0, -SHIN_HALF_H, 0.0),
        Vec3::new(0.0, +TALUS_HALF_H, 0.0),
        TALOCRURAL_JOINT_K, TALOCRURAL_JOINT_C);

    // Primary axis: world +x at neutral, in shin local frame.
    let tc_primary = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_ROM_MIN, TALOCRURAL_ROM_MAX,
        TALOCRURAL_LIMIT_K, TALOCRURAL_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_K, 0.0));
    let talocrural_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_PRIMARY_KP, TALOCRURAL_PRIMARY_KD));

    // Off-axes (Y, Z) locked: strong passive + soft-limit at narrow ROM.
    for off_axis in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            shin_idx, talus_idx, off_axis,
            -TALOCRURAL_OFF_AXIS_LIM, TALOCRURAL_OFF_AXIS_LIM,
            TALOCRURAL_OFF_AXIS_LIMIT_K, TALOCRURAL_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_OFF_AXIS_K, 0.0));
    }
```

- [ ] **Step 2: Update the return statement to use the real `talocrural_pd_idx`**

Find the line `let talocrural_pd_idx = usize::MAX;` and remove it (the new code defines this variable).

- [ ] **Step 3: Build and check no errors**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished`. There may be warnings about unused `subtalar_pd_idx` and `mtp_pd_idx`; that's expected (Tasks 4 and 5 fill them).

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: talocrural joint (PF/DF + locked off-axes)"
```

---

## Task 4: Add subtalar joint

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add subtalar joint code after the talocrural block**

Below the talocrural block (still inside `build_world` before the stick), insert:

```rust
    // ── Subtalar joint (talus ↔ foot) ───────────────────────────────
    // The foot baseline R_x(-π/2) is applied at this joint: rel_rot in
    // place_child was foot_baseline. The positional joint uses the
    // anatomical anchor points.
    add_elastic_joint(&mut world, talus_idx, foot_idx,
        Vec3::new(0.0, -TALUS_HALF_H, 0.0),
        Vec3::new(0.0, +FOOT_HALF_H, 0.0),  // foot heel anchor in foot local
        SUBTALAR_JOINT_K, SUBTALAR_JOINT_C);

    // Primary axis: world +z at neutral (inversion/eversion).
    // In talus local frame (talus is at identity), axis = (0, 0, 1).
    let st_primary = Vec3::new(0.0, 0.0, 1.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_ROM_MIN, SUBTALAR_ROM_MAX,
        SUBTALAR_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_K, 0.0)
        .with_baseline(foot_baseline));
    let subtalar_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_PRIMARY_KP, SUBTALAR_PRIMARY_KD)
        .with_baseline(foot_baseline));

    // Twist axis (Y in talus local): very limited passive ROM.
    let st_twist = Vec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_twist,
        -SUBTALAR_TWIST_LIM, SUBTALAR_TWIST_LIM,
        SUBTALAR_TWIST_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_K, 0.0)
        .with_baseline(foot_baseline));

    // Other off-axis (X in talus local): locked, prevents PF/DF leaking
    // into the subtalar joint (that's the talocrural's job).
    let st_locked = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_locked,
        -SUBTALAR_OFF_AXIS_LIM, SUBTALAR_OFF_AXIS_LIM,
        SUBTALAR_OFF_AXIS_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_OFF_AXIS_K, 0.0)
        .with_baseline(foot_baseline));
```

- [ ] **Step 2: Remove the stub `let subtalar_pd_idx = usize::MAX;` from the bottom of `build_world`**

- [ ] **Step 3: Build and verify**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished`. Only `mtp_pd_idx` may be unused.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: subtalar joint (inv/ev + twist + locked x)"
```

---

## Task 5: Add MTP joint

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add MTP joint code below the subtalar block**

```rust
    // ── MTP joint (foot ↔ forefoot) ─────────────────────────────────
    add_elastic_joint(&mut world, foot_idx, forefoot_idx,
        Vec3::new(0.0, -FOOT_HALF_H, 0.0),
        Vec3::new(0.0, +FOREFOOT_HALF_H, 0.0),
        MTP_JOINT_K, MTP_JOINT_C);

    // Primary axis: world +x at neutral. In foot local (after baseline),
    // axis (1, 0, 0) still maps to world +x. The MTP PD's baseline is
    // identity (no offset between foot and forefoot at rest).
    let mtp_primary = Vec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_ROM_MIN, MTP_ROM_MAX,
        MTP_LIMIT_K, MTP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_K, 0.0));
    let mtp_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_PRIMARY_KP, MTP_PRIMARY_KD));

    // Off-axes (Y, Z) locked.
    for off_axis in [Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            foot_idx, forefoot_idx, off_axis,
            -MTP_OFF_AXIS_LIM, MTP_OFF_AXIS_LIM,
            MTP_OFF_AXIS_LIMIT_K, MTP_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            foot_idx, forefoot_idx, off_axis, MTP_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            foot_idx, forefoot_idx, off_axis, MTP_OFF_AXIS_K, 0.0));
    }
```

- [ ] **Step 2: Remove the stub `let mtp_pd_idx = usize::MAX;`**

- [ ] **Step 3: Build**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished`.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: MTP joint (toe dorsiflexion)"
```

---

## Task 6: Add load stick weld + ground contact

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: After the stick body is added but before the return statement, add weld and ground contact**

Find the line `let stick_idx = world.add_body(stick);` and insert below it (and before the `(world, AnkleHandles { ... })` return):

```rust
    // ── Stick welded to shin top ────────────────────────────────────
    weld_bodies(&mut world, shin_idx, stick_idx,
        Vec3::new(0.0, +SHIN_HALF_H, 0.0),
        Vec3::new(0.0, -STICK_HALF_H, 0.0));

    // ── Ground contact on foot only ─────────────────────────────────
    // Foot uses single-capsule GroundContact (multi-region foot is
    // Phase 2). Other bodies' contact (forefoot when toes are up,
    // etc.) is intentionally not modeled in Phase 1.
    world.ground_contacts.push(
        GroundContact::new(foot_idx, FOOT_HALF_H, FOOT_RADIUS));
```

- [ ] **Step 2: Build**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished`.

- [ ] **Step 3: Run to verify positions are still correct (chain unchanged at startup)**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -10`
Expected: same body positions as Task 2 step 4.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: load stick welded to shin + ground contact"
```

---

## Task 7: Test framework (TestResult + simulate helper)

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `TestResult` struct and `simulate` helper after `build_world`**

Insert these after the `build_world` function:

```rust
struct TestResult {
    name: &'static str,
    passed: bool,
    message: String,
}

impl TestResult {
    fn pass(name: &'static str) -> Self {
        Self { name, passed: true, message: String::from("PASS") }
    }
    fn fail(name: &'static str, message: String) -> Self {
        Self { name, passed: false, message }
    }
}

/// Run the simulation for `duration` seconds, optionally calling
/// `per_frame` once per frame (after each `SOLVER_ITERATIONS` substeps).
/// Returns the world afterward.
fn simulate(
    world: &mut XpbdWorld,
    duration: f64,
    mut per_frame: impl FnMut(&mut XpbdWorld, f64),
) {
    let mut t = 0.0;
    while t < duration {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;
        per_frame(world, t);
    }
}

/// Check that all body positions are finite (no NaN/inf).
fn all_positions_finite(world: &XpbdWorld) -> bool {
    world.bodies.iter().all(|b|
        b.position.x.is_finite() && b.position.y.is_finite() && b.position.z.is_finite())
}
```

- [ ] **Step 2: Build**

Run: `cargo build --release --bin ankle_anatomical 2>&1 | tail -5`
Expected: `Finished`. There will be unused warnings on `TestResult` constructors; that's fine.

- [ ] **Step 3: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test framework scaffolding"
```

---

## Task 8: Test 8 — sustained stability (sanity check)

Run this test FIRST (despite being labeled #8 in the spec) because it's the foundation: if the assembly explodes on a 10-second run with no commands, all other tests are moot.

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_sustained_stability` function**

Insert after the `simulate` helper:

```rust
fn test_sustained_stability() -> TestResult {
    let (mut world, _h) = build_world();
    let mut blew_up = false;
    simulate(&mut world, 10.0, |w, _t| {
        if !all_positions_finite(w) {
            blew_up = true;
        }
    });
    if blew_up {
        return TestResult::fail("8: sustained stability",
            "non-finite body position detected".into());
    }
    TestResult::pass("8: sustained stability")
}
```

- [ ] **Step 2: Update `main` to run just this test**

Replace `main` with:

```rust
fn main() {
    let r = test_sustained_stability();
    println!("{}: {}", r.name, r.message);
    if !r.passed {
        std::process::exit(1);
    }
}
```

- [ ] **Step 3: Run the binary**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -3`
Expected: `8: sustained stability: PASS`

If it FAILS or hangs:
- If it produces NaN: the assembly has a stability bug. Check that the talus mass is at least 0.05 kg (we calculated 9× substep margin in the spec, so this should not happen).
- If it doesn't terminate within ~30 seconds: real-time performance issue, not a simulation bug.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 8 (sustained stability) passes"
```

---

## Task 9: Test 1 — static balance under load

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_static_balance` function**

Insert after `test_sustained_stability`:

```rust
fn test_static_balance() -> TestResult {
    let (mut world, h) = build_world();
    // PD targets are 0 (default). Just let the chain settle.
    let mut max_stick_tilt = 0.0_f64;
    simulate(&mut world, 5.0, |w, _t| {
        // Stick tilt = scaled-axis magnitude (small-angle approx).
        let tilt = w.bodies[h.stick_idx].orientation.scaled_axis().norm();
        if tilt > max_stick_tilt { max_stick_tilt = tilt; }
    });
    let tilt_deg = max_stick_tilt.to_degrees();
    if tilt_deg < 5.0 && all_positions_finite(&world) {
        TestResult::pass("1: static balance")
    } else {
        TestResult::fail("1: static balance",
            format!("max stick tilt = {:.2}° (limit 5°)", tilt_deg))
    }
}
```

- [ ] **Step 2: Update `main` to run both tests**

```rust
fn main() {
    let mut results = vec![];
    results.push(test_sustained_stability());
    results.push(test_static_balance());
    for r in &results {
        println!("{}: {}", r.name, r.message);
    }
    if results.iter().any(|r| !r.passed) {
        std::process::exit(1);
    }
}
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -5`
Expected:
```
8: sustained stability: PASS
1: static balance: PASS
```

If test 1 fails with max tilt > 5°, the load is too heavy or the passive ankle stiffness is too low. Verify constants match the spec. (Spec STICK_MASS = 7 kg, STICK_HALF_H = 0.40 — these should produce stable balance.)

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 1 (static balance) passes"
```

---

## Task 10: Test 2 — talocrural commanded motion

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_talocrural_commanded` function**

```rust
fn test_talocrural_commanded() -> TestResult {
    let (mut world, h) = build_world();
    // Command talocrural to dorsiflex 0.3 rad (-0.3 since +PF, -DF).
    world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = -0.3;
    simulate(&mut world, 2.0, |_, _| {});
    let actual = joint_angle(&world,
        world.angular_pd_controllers[h.talocrural_pd_idx].body_a,
        world.angular_pd_controllers[h.talocrural_pd_idx].body_b,
        world.angular_pd_controllers[h.talocrural_pd_idx].axis_local_a);
    let err = (actual - (-0.3)).abs();
    if err < 0.05 {
        TestResult::pass("2: talocrural commanded")
    } else {
        TestResult::fail("2: talocrural commanded",
            format!("target=-0.30, actual={:+.3}, err={:.3} (limit 0.05)", actual, err))
    }
}
```

- [ ] **Step 2: Add to `main`'s results vector**

```rust
    results.push(test_talocrural_commanded());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -6`
Expected:
```
8: sustained stability: PASS
1: static balance: PASS
2: talocrural commanded: PASS
```

If test 2 fails: the PD didn't drive the joint to target. Likely the load resistance overwhelms the postural-grade KP=80. Note: the chain may tip during this test (the foot's heel lifts, etc.) — that's expected. We're testing joint angle, not chain pose. If the joint truly can't reach -0.3 rad under load, KP or KD may need tuning here, OR the test may need a lower load (e.g., reduce STICK_MASS to 3 kg for active control tests).

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 2 (talocrural commanded motion) passes"
```

---

## Task 11: Test 3 — subtalar commanded motion

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_subtalar_commanded`**

```rust
fn test_subtalar_commanded() -> TestResult {
    let (mut world, h) = build_world();
    // Command subtalar to invert 0.26 rad (positive = inversion).
    world.angular_pd_controllers[h.subtalar_pd_idx].target_angle = 0.26;
    simulate(&mut world, 2.0, |_, _| {});
    let actual = joint_angle(&world,
        world.angular_pd_controllers[h.subtalar_pd_idx].body_a,
        world.angular_pd_controllers[h.subtalar_pd_idx].body_b,
        world.angular_pd_controllers[h.subtalar_pd_idx].axis_local_a);
    let err = (actual - 0.26).abs();
    if err < 0.05 {
        TestResult::pass("3: subtalar commanded")
    } else {
        TestResult::fail("3: subtalar commanded",
            format!("target=+0.26, actual={:+.3}, err={:.3} (limit 0.05)", actual, err))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_subtalar_commanded());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -7`
Expected: `3: subtalar commanded: PASS`

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 3 (subtalar commanded motion) passes"
```

---

## Task 12: Test 4 — MTP commanded motion

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_mtp_commanded`**

```rust
fn test_mtp_commanded() -> TestResult {
    let (mut world, h) = build_world();
    // Command MTP to dorsiflex toes 0.52 rad (toes up, push-off pose).
    world.angular_pd_controllers[h.mtp_pd_idx].target_angle = 0.52;
    simulate(&mut world, 2.0, |_, _| {});
    let actual = joint_angle(&world,
        world.angular_pd_controllers[h.mtp_pd_idx].body_a,
        world.angular_pd_controllers[h.mtp_pd_idx].body_b,
        world.angular_pd_controllers[h.mtp_pd_idx].axis_local_a);
    let err = (actual - 0.52).abs();
    if err < 0.05 {
        TestResult::pass("4: MTP commanded")
    } else {
        TestResult::fail("4: MTP commanded",
            format!("target=+0.52, actual={:+.3}, err={:.3} (limit 0.05)", actual, err))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_mtp_commanded());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -8`
Expected: `4: MTP commanded: PASS`

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 4 (MTP commanded motion) passes"
```

---

## Task 13: Test 5 — return to neutral

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_return_to_neutral`**

```rust
fn test_return_to_neutral() -> TestResult {
    let (mut world, h) = build_world();

    // Phase 1: drive all joints away from neutral for 2 s.
    world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = -0.3;
    world.angular_pd_controllers[h.subtalar_pd_idx].target_angle    =  0.26;
    world.angular_pd_controllers[h.mtp_pd_idx].target_angle         =  0.52;
    simulate(&mut world, 2.0, |_, _| {});

    // Phase 2: command all back to 0, run 3 s, then check.
    world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = 0.0;
    world.angular_pd_controllers[h.subtalar_pd_idx].target_angle    = 0.0;
    world.angular_pd_controllers[h.mtp_pd_idx].target_angle         = 0.0;
    simulate(&mut world, 3.0, |_, _| {});

    let pds = [h.talocrural_pd_idx, h.subtalar_pd_idx, h.mtp_pd_idx];
    let mut max_err = 0.0_f64;
    for &pd_idx in &pds {
        let pd = &world.angular_pd_controllers[pd_idx];
        let actual = joint_angle(&world, pd.body_a, pd.body_b, pd.axis_local_a);
        if actual.abs() > max_err { max_err = actual.abs(); }
    }
    if max_err < 0.02 {
        TestResult::pass("5: return to neutral")
    } else {
        TestResult::fail("5: return to neutral",
            format!("max residual angle = {:+.3} rad (limit 0.02)", max_err))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_return_to_neutral());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -9`
Expected: `5: return to neutral: PASS`

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 5 (return to neutral) passes"
```

---

## Task 14: Test 6 — disturbance recovery

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_disturbance_recovery`**

```rust
fn test_disturbance_recovery() -> TestResult {
    let (mut world, h) = build_world();

    // Let the chain settle for 1 s.
    simulate(&mut world, 1.0, |_, _| {});

    // Apply 50 N horizontal force in +z direction to the stick for 40 ms
    // (= 2 N·s impulse, "moderate shoulder bump"). Since the existing
    // engine integrates external forces per substep, we set the stick's
    // linear velocity directly: Δv = F·dt / m = 2 / 7 ≈ 0.286 m/s in +z.
    world.bodies[h.stick_idx].linear_velocity.z += 50.0 * 0.040 / STICK_MASS;

    // Run 1.5 s of recovery.
    let mut min_post_tilt = f64::INFINITY;
    simulate(&mut world, 1.5, |w, _t| {
        let tilt = w.bodies[h.stick_idx].orientation.scaled_axis().norm();
        if tilt < min_post_tilt { min_post_tilt = tilt; }
    });
    let final_tilt = world.bodies[h.stick_idx].orientation.scaled_axis().norm();
    if final_tilt.to_degrees() < 5.0 {
        TestResult::pass("6: disturbance recovery")
    } else {
        TestResult::fail("6: disturbance recovery",
            format!("final stick tilt = {:.2}° (limit 5°)", final_tilt.to_degrees()))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_disturbance_recovery());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -10`
Expected: `6: disturbance recovery: PASS`

If it FAILS: the disturbance exceeds the postural-grade ankle's restoring capacity. This is actually a known limit (Phase 1 = postural placeholder, not full muscle). If the recovery just barely fails, that's a valid Phase 1 outcome and the test reports it honestly. Don't tune the disturbance down to force a pass.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 6 (disturbance recovery)"
```

---

## Task 15: Test 7 — ROM clamping

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_rom_clamping`**

```rust
fn test_rom_clamping() -> TestResult {
    let (mut world, h) = build_world();
    // Command talocrural way past the dorsi limit (-0.35).
    world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = -1.0;
    simulate(&mut world, 2.0, |_, _| {});
    let pd = &world.angular_pd_controllers[h.talocrural_pd_idx];
    let actual = joint_angle(&world, pd.body_a, pd.body_b, pd.axis_local_a);
    // ROM is [-0.35, +0.70]. 10% past lower limit = -0.385.
    let limit_with_margin = TALOCRURAL_ROM_MIN
        - 0.10 * (TALOCRURAL_ROM_MAX - TALOCRURAL_ROM_MIN).abs();
    if actual >= limit_with_margin {
        TestResult::pass("7: ROM clamping")
    } else {
        TestResult::fail("7: ROM clamping",
            format!("actual={:+.3} below limit-margin {:+.3}", actual, limit_with_margin))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_rom_clamping());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -11`
Expected: `7: ROM clamping: PASS`

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 7 (ROM clamping) passes"
```

---

## Task 16: Test 9 — passive moment past ROM limit

This test is more involved — it requires measuring the passive resistance torque at a forced angle. We forcibly set the talus orientation to put talocrural at -0.40 rad, then read the angular torque output from the spring + soft-limit primitives.

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Add `test_passive_moment_past_rom`**

```rust
fn test_passive_moment_past_rom() -> TestResult {
    let (mut world, h) = build_world();

    // Disable the active PD so we measure only passive resistance.
    world.angular_pd_controllers[h.talocrural_pd_idx].enabled = false;

    // Pin shin and talus to fixed orientations. We'll set talus to a
    // -0.40 rad rotation about world +x (talocrural joint past dorsi).
    world.bodies[h.shin_idx].inv_mass = 0.0;
    world.bodies[h.talus_idx].inv_mass = 0.0;
    world.bodies[h.talus_idx].orientation =
        UnitQuaternion::from_scaled_axis(Vec3::new(-0.40, 0.0, 0.0));

    // Clear torque accumulators and apply just the spring + soft limit.
    world.bodies[h.talus_idx].clear_accumulators();
    world.bodies[h.shin_idx].clear_accumulators();

    // Apply springs and soft limits in isolation (no substep).
    for spring in &world.angular_springs {
        spring.apply(&mut world.bodies);
    }
    for limit in &mut world.angular_soft_limits {
        limit.apply(&mut world.bodies);
    }

    // Read the torque on the talus about world +x.
    let measured_torque = world.bodies[h.talus_idx].torque_accumulator.x.abs();

    // Spec expectation: ≈ 25 N·m within 30% (range 17.5 to 32.5).
    let expected = 25.0;
    let tolerance = 0.30 * expected;
    let in_range = (measured_torque - expected).abs() < tolerance;
    if in_range {
        TestResult::pass("9: passive moment past ROM")
    } else {
        TestResult::fail("9: passive moment past ROM",
            format!("measured={:.2} N·m, expected {:.1}±{:.1}",
                measured_torque, expected, tolerance))
    }
}
```

- [ ] **Step 2: Add to `main`'s results**

```rust
    results.push(test_passive_moment_past_rom());
```

- [ ] **Step 3: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1 | tail -12`
Expected: `9: passive moment past ROM: PASS`

If it FAILS with a value > 32.5 N·m: the soft-limit K is too high. Reduce TALOCRURAL_LIMIT_K. If it fails LOW (<17.5): increase TALOCRURAL_LIMIT_K. The spec's K=10000, exp=2 should give 10000·(0.05)² = 25 N·m at 0.05 past limit, plus ~1.6 N·m from the linear passive spring (K=4·0.4) = 26.6 N·m total. That's within tolerance.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: test 9 (passive moment past ROM) passes"
```

---

## Task 17: Main function with full test summary

**Files:**
- Modify: `src/bin/ankle_anatomical.rs`

- [ ] **Step 1: Replace `main` with full summary version**

```rust
fn main() {
    println!("PHASE 1 ANATOMICAL ANKLE VALIDATION");
    println!("{}", "─".repeat(60));

    let tests: Vec<fn() -> TestResult> = vec![
        test_sustained_stability,
        test_static_balance,
        test_talocrural_commanded,
        test_subtalar_commanded,
        test_mtp_commanded,
        test_return_to_neutral,
        test_disturbance_recovery,
        test_rom_clamping,
        test_passive_moment_past_rom,
    ];

    let mut results = vec![];
    for t in tests {
        let r = t();
        let tag = if r.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {}: {}", tag, r.name, r.message);
        results.push(r);
    }

    let pass_count = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("{}", "─".repeat(60));
    println!("SUMMARY: {} / {} tests passed", pass_count, total);

    if pass_count < total {
        std::process::exit(1);
    }
}
```

- [ ] **Step 2: Run**

Run: `cargo run --release --bin ankle_anatomical 2>&1`
Expected: all 9 tests print `[PASS]`, summary says `9 / 9 tests passed`.

- [ ] **Step 3: Commit**

```bash
git add src/bin/ankle_anatomical.rs
git commit -m "ankle phase 1: full test summary with exit code"
```

---

## Task 18: Bevy visual demo — shared world setup

**Files:**
- Create: `src/bin/ankle_anatomical_demo.rs`

The visual demo uses the same world setup. To avoid duplication, we'll factor `build_world` into a shared module. The simplest project-style approach is to duplicate the small setup code in the demo binary (the project doesn't use library-style sharing for bin/* files).

- [ ] **Step 1: Create the demo binary with shared constants/world**

Write to `src/bin/ankle_anatomical_demo.rs`. To keep this manageable: copy the constants, helpers, `AnkleHandles` struct, and `build_world` function VERBATIM from `ankle_anatomical.rs`. (Duplication accepted here — the alternative is moving to a shared module which is out of project pattern.)

Top of file:

```rust
// Phase 1 anatomical ankle — Bevy visual companion.
//
// Mirrors the world setup in ankle_anatomical.rs (the text validation
// binary) and adds rendering + a live joint-angle UI. Cycles through
// scenarios 1-6 with pauses for viewer comprehension.

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::ground_contact::GroundContact;
use physics::math::{Vec3 as NaVec3, Mat3, swing_twist_angle};
use improper_football::render_util::rigid_body_to_transform;
```

Then paste the constant block, the helper functions, `AnkleHandles`, and `build_world` from `ankle_anatomical.rs`. Replace `Vec3` with `NaVec3` everywhere in those pasted blocks (Bevy's prelude exports its own `Vec3` and we need to disambiguate). The `joint_angle` helper is not needed in the demo.

- [ ] **Step 2: Add a stub `main` that just builds the world**

```rust
fn main() {
    let (_world, _h) = build_world();
    println!("ankle_anatomical_demo: world built (rendering TODO)");
}
```

- [ ] **Step 3: Build**

Run: `cargo build --release --bin ankle_anatomical_demo 2>&1 | tail -5`
Expected: `Finished`.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical_demo.rs
git commit -m "ankle phase 1 demo: shared world setup (stub main)"
```

---

## Task 19: Bevy visual demo — scene setup

**Files:**
- Modify: `src/bin/ankle_anatomical_demo.rs`

- [ ] **Step 1: Add `PhysicsWorld` resource and `BodyRef` component**

After the `build_world` function, add:

```rust
const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    handles: AnkleHandles,
    sim_time: f64,
    scenario: usize,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct Readout;
```

- [ ] **Step 2: Add `setup` system that spawns meshes for each body**

```rust
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (world, handles) = build_world();

    let foot_mat     = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let forefoot_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.65, 0.45, 0.30), ..default() });
    let talus_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.90, 0.85, 0.70), ..default() });
    let shin_mat     = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let stick_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });

    let foot_mesh     = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let forefoot_mesh = meshes.add(Capsule3d::new(FOREFOOT_RADIUS as f32, (2.0 * FOREFOOT_HALF_H) as f32));
    let talus_mesh    = meshes.add(Capsule3d::new(TALUS_RADIUS as f32, (2.0 * TALUS_HALF_H) as f32));
    let shin_mesh     = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let stick_mesh    = meshes.add(Capsule3d::new(STICK_RADIUS as f32, (2.0 * STICK_HALF_H) as f32));

    commands.spawn((PbrBundle {
        mesh: foot_mesh, material: foot_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.foot_idx]),
        ..default()
    }, BodyRef(handles.foot_idx)));
    commands.spawn((PbrBundle {
        mesh: forefoot_mesh, material: forefoot_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.forefoot_idx]),
        ..default()
    }, BodyRef(handles.forefoot_idx)));
    commands.spawn((PbrBundle {
        mesh: talus_mesh, material: talus_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.talus_idx]),
        ..default()
    }, BodyRef(handles.talus_idx)));
    commands.spawn((PbrBundle {
        mesh: shin_mesh, material: shin_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.shin_idx]),
        ..default()
    }, BodyRef(handles.shin_idx)));
    commands.spawn((PbrBundle {
        mesh: stick_mesh, material: stick_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.stick_idx]),
        ..default()
    }, BodyRef(handles.stick_idx)));

    // Ground plane
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(2.0, 2.0));
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_mat,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.5, 0.7, 1.5)
            .looking_at(Vec3::new(0.0, 0.4, 0.0), Vec3::Y),
        ..default()
    });

    // Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight { illuminance: 10_000.0, shadows_enabled: false, ..default() },
        transform: Transform::from_xyz(1.5, 3.0, 1.5).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight { color: Color::WHITE, brightness: 220.0 });

    // UI readout
    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 16.0, color: Color::WHITE, ..default() },
    ).with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, handles, sim_time: 0.0, scenario: 0,
    });
}
```

- [ ] **Step 3: Update `main` to start a Bevy app**

```rust
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Phase 1 anatomical ankle — visual companion".into(),
                resolution: (1100.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .run();
}
```

- [ ] **Step 4: Build (this is a big compile — may take 30+ seconds)**

Run: `cargo build --release --bin ankle_anatomical_demo 2>&1 | tail -5`
Expected: `Finished`. May take 1-2 minutes if it's the first Bevy build in this clone.

- [ ] **Step 5: Run briefly to confirm window opens**

Run: `cargo run --release --bin ankle_anatomical_demo 2>&1 &`
Expected: a window opens showing the assembly (foot on ground, talus, shin, stick stacked vertically). No physics yet — just a static snapshot.

Close the window. (On Windows, click the X. The shell job will terminate.)

- [ ] **Step 6: Commit**

```bash
git add src/bin/ankle_anatomical_demo.rs
git commit -m "ankle phase 1 demo: scene setup, static snapshot renders"
```

---

## Task 20: Bevy visual demo — physics stepping + transform sync

**Files:**
- Modify: `src/bin/ankle_anatomical_demo.rs`

- [ ] **Step 1: Add `physics_step` and `sync_transforms` systems**

After the `setup` function:

```rust
fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut q: Query<(&BodyRef, &mut Transform)>,
) {
    for (body_ref, mut t) in &mut q {
        *t = rigid_body_to_transform(&phys.world.bodies[body_ref.0]);
    }
}
```

- [ ] **Step 2: Wire systems into the app in `main`**

Update the `App::new()` call to:

```rust
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Phase 1 anatomical ankle — visual companion".into(),
                resolution: (1100.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, sync_transforms)
        .run();
}
```

- [ ] **Step 3: Build and run briefly**

Run: `cargo build --release --bin ankle_anatomical_demo 2>&1 | tail -5`
Expected: `Finished`.

Run: `cargo run --release --bin ankle_anatomical_demo 2>&1 &`
Expected: window opens, assembly settles under gravity (foot slightly squishes onto ground, all bodies stable).

Close the window.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical_demo.rs
git commit -m "ankle phase 1 demo: physics stepping + transform sync"
```

---

## Task 21: Bevy visual demo — scenario cycling + UI readout

**Files:**
- Modify: `src/bin/ankle_anatomical_demo.rs`

- [ ] **Step 1: Add scenario state and cycling system**

After the `sync_transforms` function:

```rust
const SCENARIO_DURATION: f64 = 4.0;  // seconds per scenario
const SCENARIO_NAMES: [&str; 6] = [
    "1: static balance",
    "2: talocrural dorsiflex (-0.3 rad)",
    "3: subtalar invert (+0.26 rad)",
    "4: MTP toe-up (+0.52 rad)",
    "5: return to neutral",
    "6: disturbance recovery",
];

fn cycle_scenarios(mut phys: ResMut<PhysicsWorld>) {
    let elapsed = phys.sim_time;
    let scenario_idx = (elapsed / SCENARIO_DURATION) as usize % SCENARIO_NAMES.len();
    if scenario_idx == phys.scenario {
        return;  // no change
    }
    phys.scenario = scenario_idx;

    // Reset targets for clean transitions
    let h = phys.handles;
    phys.world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = 0.0;
    phys.world.angular_pd_controllers[h.subtalar_pd_idx].target_angle = 0.0;
    phys.world.angular_pd_controllers[h.mtp_pd_idx].target_angle = 0.0;

    match scenario_idx {
        0 => { /* static balance: all 0 */ }
        1 => { phys.world.angular_pd_controllers[h.talocrural_pd_idx].target_angle = -0.3; }
        2 => { phys.world.angular_pd_controllers[h.subtalar_pd_idx].target_angle = 0.26; }
        3 => { phys.world.angular_pd_controllers[h.mtp_pd_idx].target_angle = 0.52; }
        4 => { /* return to neutral: all 0 (same as scenario 0) */ }
        5 => {
            // Disturbance: kick the stick once with a +z velocity bump.
            let stick_idx = h.stick_idx;
            phys.world.bodies[stick_idx].linear_velocity.z += 50.0 * 0.040 / STICK_MASS;
        }
        _ => {}
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut q: Query<&mut Text, With<Readout>>,
) {
    let h = phys.handles;
    let talocrural = swing_twist_angle(
        phys.world.bodies[h.shin_idx].orientation.inverse()
            * phys.world.bodies[h.talus_idx].orientation,
        NaVec3::new(1.0, 0.0, 0.0));
    let subtalar = swing_twist_angle(
        phys.world.bodies[h.talus_idx].orientation.inverse()
            * phys.world.bodies[h.foot_idx].orientation,
        NaVec3::new(0.0, 0.0, 1.0));
    let mtp = swing_twist_angle(
        phys.world.bodies[h.foot_idx].orientation.inverse()
            * phys.world.bodies[h.forefoot_idx].orientation,
        NaVec3::new(1.0, 0.0, 0.0));
    let stick_tilt = phys.world.bodies[h.stick_idx].orientation.scaled_axis().norm();

    for mut text in &mut q {
        text.sections[0].value = format!(
            "t = {:.2} s\n\nScenario: {}\n\nJoint angles:\n  talocrural: {:+.3} rad ({:+.1}°)\n  subtalar:   {:+.3} rad ({:+.1}°)\n  MTP:        {:+.3} rad ({:+.1}°)\n\nStick tilt: {:.2}°",
            phys.sim_time,
            SCENARIO_NAMES[phys.scenario],
            talocrural, talocrural.to_degrees(),
            subtalar, subtalar.to_degrees(),
            mtp, mtp.to_degrees(),
            stick_tilt.to_degrees(),
        );
    }
}
```

- [ ] **Step 2: Wire the new systems into `main`**

Update the `add_systems(Update, ...)` line:

```rust
        .add_systems(Update, (sync_transforms, cycle_scenarios, update_readout))
```

- [ ] **Step 3: Build and run**

Run: `cargo build --release --bin ankle_anatomical_demo 2>&1 | tail -5`
Expected: `Finished`.

Run: `cargo run --release --bin ankle_anatomical_demo`
Expected: window opens. UI shows live joint angles and current scenario. Every 4 s, the scenario advances. Talocrural commands dorsiflex the foot; subtalar inverts; MTP toes-up. Each scenario lasts 4 s, then returns to neutral.

Close the window when satisfied.

- [ ] **Step 4: Commit**

```bash
git add src/bin/ankle_anatomical_demo.rs
git commit -m "ankle phase 1 demo: scenario cycling + live UI readout"
```

---

## Task 22: Update project memory marking Phase 1 done

**Files:**
- Create: `C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\project_phase1_ankle_complete.md`
- Modify: `C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\MEMORY.md`

- [ ] **Step 1: Create the memory file**

Write to `C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\project_phase1_ankle_complete.md`:

```markdown
---
name: Phase 1 anatomical ankle complete
description: Multi-body ankle assembly (talocrural + subtalar + MTP) with placeholder-PD active control. All 9 validation tests pass. Spec at docs/superpowers/specs/2026-05-18-anatomical-ankle-phase1-design.md.
type: project
---
Phase 1 of the anatomical ankle build-out is complete as of 2026-06-07.

**What was built:**
- 4 bodies (shin, talus, foot, forefoot) with anatomical mass distribution (Zatsiorsky/Seluyanov split)
- 3 joints (talocrural shin↔talus PF/DF, subtalar talus↔foot inv/ev, MTP foot↔forefoot toe dorsi)
- Each joint has full biological constraints on every anatomical axis (primary + off-axes locked) per the discipline rule
- Placeholder PD actuators on each primary axis (NOT a real muscle model — Phase 4)
- Validation binary `src/bin/ankle_anatomical.rs` with 9 PASS/FAIL test scenarios
- Bevy visual demo `src/bin/ankle_anatomical_demo.rs` cycling through scenarios

**What was NOT built (deferred per spec):**
- Hill-type muscle model — Phase 4
- Multi-region foot contact (heel pad vs forefoot) — Phase 2
- Plantar fascia / windlass — Phase 3
- Stretch reflex / proprioception — Phase 5
- Oblique anatomical joint axes (talocrural ~10° from frontal, subtalar 42°/23°)

**How to apply:**
- The ankle assembly is now ready to be wired into the bilateral chain in `balance_bilateral.rs` (replace single-joint ankle with the new structure)
- Subsequent phases (2-5) revisit and extend the ankle; do NOT call this "fully functional" until Phase 4 muscle model is added.
- Reference values (KP, KD, K_soft_limit, ROM) for each joint live in the spec § "Joint specifications"; copy from there rather than `balance_bilateral.rs` if reusing.
```

- [ ] **Step 2: Add an index entry to `MEMORY.md`**

Find the last line of `MEMORY.md` (the most recent bullet) and append:

```markdown
- [Phase 1 anatomical ankle complete](project_phase1_ankle_complete.md) — multi-body talocrural+subtalar+MTP with placeholder PDs, all 9 validation tests pass; deferred items per spec
```

- [ ] **Step 3: Verify the build is fully clean and tests still pass**

Run: `cargo run --release --bin ankle_anatomical 2>&1`
Expected: all 9 tests PASS.

Run: `cargo build --release 2>&1 | tail -3`
Expected: `Finished` with no errors. Pre-existing warnings (from other binaries) are acceptable.

- [ ] **Step 4: Commit**

```bash
git add C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\project_phase1_ankle_complete.md C:\Users\Mkhur\.claude\projects\c--Users-Mkhur-OneDrive-Desktop-Github-improper-football\memory\MEMORY.md
git commit -m "memory: Phase 1 ankle complete"
```

---

## Notes for the implementing engineer

**Before starting, verify API references against existing code:**
- `src/bin/balance_bilateral.rs` — uses `XpbdWorld`, `RigidBody`, all the angular primitives. Mirror its import paths and field-access patterns.
- `src/bin/balance_bilateral_demo.rs` (Bevy demo) — copy its Bevy app structure, mesh helpers, and the `rigid_body_to_transform` import path used in this plan's Task 18. If the project's helper has a different module name or signature, update Task 18 accordingly.
- `src/bin/inverse_dynamics.rs` — uses the same access patterns this plan needs (`clear_accumulators`, `torque_accumulator`, `enabled` field on PD, etc.). Confirms these APIs exist.

If a primitive in this plan doesn't compile (e.g., `AngularSoftLimit::with_baseline` not found), check the latest signature in the relevant `crates/physics/src/*.rs` file. The plan was written against the engine state as of 2026-06-07; field/method names may have evolved.

**Sign conventions** (consistent with project convention; also documented in spec §3 "Frame convention"):
- **Talocrural:** positive angle = plantarflexion (foot pointing down, like ballet pointe). Negative = dorsiflexion. This is because the foot has baseline `R_x(-π/2)`: a positive rotation about world +x at the talocrural joint plus the baseline produces toes-down rotation of the foot.
- **Subtalar:** positive angle = inversion (sole tilts inward). Negative = eversion.
- **MTP:** positive angle = toe dorsiflexion (toes up, push-off pose). Negative = toe plantarflexion (toe-curling).

**Foot baseline:** `R_x(-π/2)` rotates the foot's local +Y axis to world -Z (heel) and local -Y to world +Z (toe). The foot's local +X stays world +X (rotation about x doesn't move x). This baseline is applied at the subtalar joint via `.with_baseline(foot_baseline)` on the subtalar's angular structures.

**If a test fails:** the spec lists per-test pass criteria. If a value is just slightly off (e.g., commanded motion target reached within 0.06 rad instead of 0.05), the diagnostic should report the actual value — do NOT lower the threshold without consulting the design spec. If a test fails because of unexpected physics behavior (chain explodes, joint goes to NaN), investigate root cause before adjusting parameters.

**Engine primitives used:** the spec §4 "Engine primitives required" lists what this Phase 1 build depends on. All are existing engine code; no new primitives in Phase 1.
