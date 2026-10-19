# improper-football

A biomechanically-grounded human motion physics engine in Rust, built from
first principles on an XPBD (extended position-based dynamics) substrate with
sequential-impulse constraints, implicit integration for stiff components, and
anatomically-faithful joint modeling.

The long-term goal is a general human-motion simulator that integrates the
best ideas from the leading physics engines — MuJoCo's soft constraints,
Drake's inverse dynamics, Bullet's sequential-impulse contacts, OpenSim's
musculoskeletal fidelity — into one coherent system, applied to (American)
football motion as the driving use case.

## Highlights

**26-bone anatomical foot** (`src/anatomy/foot.rs`) — every tarsal,
metatarsal, and phalanx is a distinct rigid body with literature-sourced
dimensions and mass distribution (Zatsiorsky/de Leva segment data,
osteometric bone studies). 21 articulated joints with anatomically-oblique
axes — the subtalar joint uses Manter/Inman's 42°-inclination,
23°-medial-deviation axis, the mechanism behind inversion ankle sprains —
plus 8 stiff welds for the cadaver-confirmed minimal-motion midfoot joints.
Data-driven construction: bone and joint parameter tables consumed by
generic builders, no hand-placed magic numbers.

**Multi-point manifold ground contact** — ground contact is solved as
simultaneous sequential-impulse non-penetration constraints across all
contacting capsule endpoints (Bullet SI-style, with a Coulomb friction
cone), not independent penalty forces. This was motivated by a measured
failure: independent per-point penalty contact collapsed a rigid foot's
84 cm² footprint to one intermittent contact point (0.39 cm²), so bodies
tipped over instead of standing. With the manifold, the same assembly
stands at 2.3° tilt on 36 simultaneous contacts. The diagnostic that
isolated this (`src/bin/foot_rigid_balance_diag.rs`) is kept as a standing
acceptance gate.

**Stiffness-safe integration** — explicit-Euler force application is
unstable for stiff springs/dampers on small bodies (the foot's 2-gram
distal phalanges violate `dt < 2m/c` by >10×). All springs, dampers, and
soft limits integrate implicitly (backward-Euler velocity solves,
unconditionally stable); joint position coupling and angular locks use
XPBD/SI constraints. The engine survived its hardest test to date: a
28-body assembly spanning a 3500:1 mass ratio (7 kg load stick to 2 g toe
bones) settling stably on the ground.

**Biological constraints always** — every joint constrains every anatomical
axis: primary axis with ROM soft-limits and passive stiffness/damping from
published moment-angle data (Riener & Edrich 1999, Heng 2016), off-axes
locked with physiological slop. Per-joint parameters are data, not
engine-level constants — biological variance is the design.

**Inverse dynamics** (`crates/physics/src/inverse_dynamics.rs`) — per-joint
gravity feedforward G(q) computed each substep and fed to the PD
controllers (Drake/MuJoCo computed-torque style), with fractional load
sharing for closed chains (bilateral stance).

**Regression-tested engine** — `crates/physics/tests/engine_regression.rs`
guards contact behavior, small-body stability, energy boundedness, and the
manifold multi-contact property. Tests were adversarially verified to fail
on revert before being trusted.

## Architecture

```
crates/physics/          The engine (no app dependencies)
  src/xpbd.rs            World, substep loop, SI solver, ground manifold
  src/rigid_body.rs      Capsule bodies
  src/contact_constraint.rs  Unilateral SI contact + friction cone
  src/angular_*.rs       Springs, dampers, soft limits, PD controllers,
                         angular SI constraints (all implicit/SI)
  src/inverse_dynamics.rs    Gravity feedforward G(q)
  tests/engine_regression.rs Engine acceptance suite

src/anatomy/             Reusable anatomical assemblies (data-driven)
  foot.rs                26-bone foot: bone/joint tables + builders

src/bin/                 Validation + visual binaries (one concern each)
  ankle_anatomical.rs    Foot+ankle validation suite (PASS/FAIL, CI-ready)
  ankle_anatomical_demo.rs   Bevy visual companion
  foot_rigid_balance_diag.rs Contact-model acceptance diagnostic
  balance_*.rs, elbow_*.rs, ...  Incremental motion milestones
```

Physics runs at 120 Hz frames × 60 substeps (sub-dt ≈ 139 µs). Rendering
(Bevy 0.14) is strictly an observer — the engine has no dependency on it.

## Running

```bash
# Engine regression suite + anatomy unit tests
cargo test -p physics
cargo test -p improper-football --lib

# Foot + ankle validation (prints PASS/FAIL, exits nonzero on failure)
cargo run --release --bin ankle_anatomical
cargo run --release --bin ankle_anatomical -- --diagnose   # support-polygon trace

# Contact-model acceptance diagnostic (rigid assembly must stand)
cargo run --release --bin foot_rigid_balance_diag

# Visual demos (Bevy)
cargo run --release --bin ankle_anatomical_demo
cargo run --release --bin balance_bilateral_demo
```

## Method

The project is built motion-by-motion in small verified increments, with a
spec → plan → implementation → review cycle for each milestone
(`docs/superpowers/specs/`, `docs/superpowers/plans/`). Two principles are
enforced throughout:

- **Diagnose, don't tune.** When something fails, the root cause is isolated
  with a decisive experiment before anything is changed. Example: when the
  foot wouldn't balance, a fully-welded-rigid variant separated "the contact
  model can't form a support base" (true, fixed with manifold contact) from
  "standing needs active control" (false at this stage).
- **Honest scoping.** The anatomy is real but actuation is still placeholder
  PD control; sustained articulated standing awaits the plantar
  fascia/windlass mechanism (Phase 3) and Hill-type muscles (Phase 4). Tests
  assert what the system actually does, not what we wish it did — the
  validation suite's comments document exactly where each limit lies.

## Roadmap

1. ~~Full-body joint topology with biological constraints~~ (done)
2. ~~Ground contact: world-level ground, manifold multi-point SI contact~~ (done)
3. Plantar fascia / windlass arch stiffening → articulated weight-bearing
4. Hill-type muscle model (force-length/velocity, activation dynamics,
   tendon elasticity) replacing placeholder PDs
5. Proprioception / reflex loops; coordinated whole-body control
6. Sport motion: gait, cutting, jumping — then football.
