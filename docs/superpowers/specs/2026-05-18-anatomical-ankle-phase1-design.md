# Anatomically-correct ankle — Phase 1 design

**Status:** Draft for review
**Date:** 2026-05-18
**Phase:** 1 of 5 (anatomy first; Phases 2-5 are deferred)

## 1. Context and motivation

Up to now the project has built a chain of joints (ankle, knee, hip, lumbar, etc.) at varying levels of fidelity, with each binary stripping or adding constraints as needed. Recent debugging revealed that bilateral disturbance recovery fails because the chain has under-constrained off-axis modes, and the controller can only operate in the sagittal plane. Before attempting a sophisticated whole-body controller, we should make sure the body parts the controller will use are actually complete.

Phase 1 builds the **ankle assembly** as the first fully-anatomical unit: talocrural + subtalar + MTP, with active control authority on each axis, plus a validation test rig that exercises the unit on its own.

Subsequent phases (deferred):
- Phase 2: multi-region foot contact (heel pad vs forefoot vs toes)
- Phase 3: plantar fascia / windlass mechanism
- Phase 4: Hill-type muscle model (replaces `AngularPDController` with `Muscle`)
- Phase 5: proprioception / stretch reflex

## 2. Goals and validation criteria

**Goal:** Replace the current 3-DOF lumped ankle with an anatomically-correct multi-body ankle that respects human joint topology, with active PD control on each joint's primary axis and strong passive constraints on off-axes.

**Done means** (validation criteria, in test-rig order):
1. **Static balance** under load (a stick of standing-equivalent mass) for ≥5 s, stick stays vertical within 5°, no joint exceeds posture-PD equilibrium beyond a few degrees, no drift.
2. **Active commanded motion** — set each primary-axis target_angle, the joint reaches that angle within ±0.05 rad of target within 1 s and holds. Verifies the ankle can be *driven*, not just held passively.
3. **Ankle-strategy disturbance recovery** — small impulse on the load stick, the ankle's own active PDs (no upper-body control) bring CoM back over support polygon within 1.5 s.
4. **ROM compliance** — commanded targets outside anatomical range hit soft limits; joint stays within ROM ±10%.
5. **Static moment at ROM end** matches anatomy — peak passive resistance at end of dorsiflexion ROM ≈ 50 N·m within 30% of published values.
6. **No NaN / no explosion** — simulation stable for 10 s under normal commanded conditions.

**Non-goals** (deferred to later phases):
- Multi-region foot contact (Phase 2)
- Plantar fascia / windlass (Phase 3)
- Hill-type muscle model (Phase 4)
- Stretch reflex (Phase 5)
- Oblique anatomical joint axes (Phase 1 uses orthogonal axes)
- Heel pad vs forefoot pad differential compliance (Phase 2)

## 3. Anatomical structure

### Bodies (4 total)

| Body | Mass | Length | Role |
|---|---|---|---|
| Shin | 3.0 kg (existing) | 0.42 m | unchanged |
| Talus (new) | 0.05 kg | 0.02 m capsule | tiny gimbal between shin and foot |
| Foot | 0.70 kg (was 1.0) | 0.135 m | hindfoot through midfoot, heel to MTP |
| Forefoot (new) | 0.25 kg | 0.045 m | toes, MTP to tip |

Total foot-region mass stays at 1.0 kg; redistributed across three bodies.

### Joints (3 total)

| Joint | Between | Active DOF | Off-axes |
|---|---|---|---|
| Talocrural | shin ↔ talus | hinge about world +x (PF/DF) | Y, Z locked with stiff passives |
| Subtalar | talus ↔ foot | inversion/eversion about world +z | Y (twist): limited passive ROM; X: locked |
| MTP | foot ↔ forefoot | toe dorsiflexion about world +x | Y, Z locked |

### Layout at neutral

```
       shin (vertical, knee at top, ankle attachment at bottom)
          |
       (talocrural joint, world +x axis)
          • talus
          |
       (subtalar joint, world +z axis)
          |
       foot (heel ─────────── MTP)
                                  • forefoot ────→ toe tips
```

### Why split lumped 3-DOF ankle

Real human ankle isn't a single 3-DOF ball. Talocrural and subtalar have different axes and different roles. A lumped joint can't model how inversion-while-plantarflexing differs from inversion-while-dorsiflexing (which is exactly the mechanism behind sprained ankles). Phase 1 uses orthogonal axes for simplicity; oblique axes are a future refinement.

### Why split foot and forefoot

The MTP joint enables toe dorsiflexion, which is essential for proper push-off and heel-to-toe rocker. Even before adding active push-off (Phase 4), having MTP as a real joint allows the correct rolling kinematics.

## 4. Joint specifications

### Talocrural (shin ↔ talus)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in shin local) | (1, 0, 0) | world +x at neutral |
| Primary ROM | [-0.35, +0.70] rad | dorsi ~20°, plantar ~40° |
| Active PD: KP / KD | 80 / 8 (N·m/rad, N·m·s/rad) | gastroc-soleus + tib anterior lumped |
| Passive K / C | 4 / 1.2 | existing ankle values |
| Soft-limit K, exp | 100 / 2.0 | progressive end-of-range resistance |
| Off-axes (Y, Z) | locked: K=200, ROM ±0.05, soft-limit K=400 | strong off-axis lock |
| Feedforward | exposed via `with_feedforward()` | for future inverse-dynamics use |

### Subtalar (talus ↔ foot)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in talus local) | (0, 0, 1) | world +z at neutral; inversion/eversion |
| Primary ROM | [-0.26, +0.52] rad | eversion ~15°, inversion ~30° |
| Active PD: KP / KD | 50 / 5 | peroneals + tib posterior |
| Passive K / C | 3.5 / 1.0 | |
| Soft-limit K, exp | 80 / 2.0 | |
| Twist axis (Y in talus local) | very limited: ROM ±0.10, K=8, C=0.5, soft-limit K=200 | |
| Other off-axis (X in talus local) | locked: K=200, ROM ±0.05, soft-limit K=400 | |
| Foot baseline `R_x(-π/2)` | applied at this joint | foot lies horizontal at neutral |

### MTP (foot ↔ forefoot)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in foot local) | (1, 0, 0) | world +x at neutral; toe dorsiflexion |
| Primary ROM | [-0.087, +0.87] rad | plantar ~5°, dorsi ~50° |
| Active PD: KP / KD | 30 / 3 | flexor digitorum + extensor digitorum |
| Passive K / C | 5 / 1.5 | |
| Soft-limit K, exp | 50 / 2.0 | |
| Off-axes (Y, Z) | locked: K=100, ROM ±0.05 | |

### Gain-selection rationale

- PD KP values are sized so each joint can produce its anatomical peak torque at ~half its ROM. Example: peak ankle moment during walking is ~50 N·m; 80 × 0.6 rad ≈ 48 N·m. These are not muscle peak forces — those arrive in Phase 4.
- KD ≈ 10% of KP for ~critical damping with the engine's implicit-integration `AngularDamper`.
- Starting values; tuning during validation may adjust.

### Engine primitives required

All existing (no new engine work in Phase 1):
- `AngularPDController` with implicit KD and feedforward field
- `AngularSpring`, `AngularDamper`, `AngularSoftLimit`
- `PositionalSpring` + `PositionalDamper` for joint anchors

## 5. Validation test rig

### Binary: `src/bin/ankle_anatomical.rs`

A text-mode validation binary that prints PASS/FAIL per test scenario and a summary at end. Exits with non-zero code on any failure (for future regression detection).

### World setup at startup

- Ground plane at y=0
- Foot (0.70 kg) — capsule, horizontal baseline `R_x(-π/2)`, settled on ground
- Forefoot (0.25 kg) — capsule, connected to foot via MTP joint
- Talus (0.05 kg) — small capsule above foot
- Shin (3.0 kg) — vertical, top is "ankle attachment" point
- Load stick (test-dependent mass, typically 10 kg) — vertical capsule, pinned at shin top via rigid 6-DOF joint

Foot uses single-capsule `GroundContact`. Multi-region foot contact is Phase 2.

### Test scenarios

| # | Test | Pass criterion | Duration |
|---|---|---|---|
| 1 | Static balance under 10 kg stick | stick tilt < 5°, no joint past posture-PD equilibrium | 5 s |
| 2 | Talocrural commanded motion | target = -0.3 rad. After 1 s, actual within ±0.05 rad | 2 s |
| 3 | Subtalar commanded motion | target = +0.26 rad. Within ±0.05 rad | 2 s |
| 4 | MTP commanded motion | target = +0.52 rad. Within ±0.05 rad | 2 s |
| 5 | Return-to-neutral | After 2-4, set targets to 0. All within ±0.02 rad | 3 s |
| 6 | Ankle-strategy disturbance recovery | Impulse +2 N·s on stick (world +z direction). Ankle alone brings tilt < 5° within 1.5 s | 3 s |
| 7 | ROM clamping | target = -1.0 rad (beyond dorsi limit -0.35). Actual stays within ROM by ≤ 10% | 2 s |
| 8 | Sustained stability | Normal load, no disturbance, no commands — no NaN, no joint runaway | 10 s |
| 9 | Peak passive moment at ROM end | At max dorsiflexion, passive resistance ≈ 50 N·m within 30% | snapshot |

### Visual companion: `src/bin/ankle_anatomical_demo.rs`

A Bevy app that visualizes the assembly with a UI panel of live joint angles. Cycles through scenarios 1-6 with brief pauses for viewer comprehension.

### Why this matters

Existing balance binaries test the *chain above* the ankle. This is the first binary that validates the ankle **as a unit** — that it can be commanded to specific configurations, that each joint reaches commanded targets, that disturbance recovery works without upper-body control. That's the bar for "fully functional ankle."

## 6. Out of scope

### Deferred to Phase 2 (multi-region foot contact)
- Heel pad vs forefoot pad differential compliance
- 2D footprint via tripod / multi-region `ContactConstraint`
- Heel-to-toe rocker demonstration

### Deferred to Phase 3 (plantar fascia / windlass)
- MTP-angle to foot-arch-stiffness coupling
- `AnatomicalCoupling` primitive

### Deferred to Phase 4 (Hill-type muscle model)
- Replace `AngularPDController` with `Muscle` primitive
- Individual muscle bodies (gastroc, soleus, tibialis anterior, posterior, peroneals, etc.)
- Tendon compliance (Achilles, plantar)
- Activation dynamics with neural delay
- Force-length and force-velocity curves
- Active push-off power test (requires muscle)

### Deferred to Phase 5 (proprioception / reflexes)
- Muscle spindle and Ia afferent model
- Stretch reflex / spinal reflex arcs

### Phase 1 deferred design choices
- Oblique anatomical joint axes (Phase 1 uses orthogonal axes — orthogonal is sufficient for functional balance; oblique is a future refinement)
- Off-axis ROM fine-tuning against cadaver studies (Phase 1 uses functional-feel values)

### Not in any current phase
- Forward / inverse kinematics for limb endpoint placement
- Skin / fat soft tissue
- Vasculature, temperature, fatigue
- Pain / injury modeling

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Talus's small mass / inertia could trigger the small-body instability (see `project_bilateral_pd_limit.md`) | Talus's joint dampers use the implicit `AngularDamper::apply_implicit`. Keep talus mass ≥ 0.05 kg. If still unstable, can raise talus mass to 0.10 kg without affecting overall fidelity. |
| Joint stacking with 3 joints close together (talocrural, subtalar, MTP all near each other) may have explicit-integration coupling issues | All angular dampers and the PD KD term use implicit integration (the engine fix from 2026-05-08). Should be safe. |
| Test rig's load stick attachment via 6-DOF rigid joint — what about reaction torques? | The stick is pinned via stiff positional + angular springs (effectively a rigid weld). For testing the ankle, the stick is a load to balance; its own kinematics don't matter. |
| Tuning the active PD gains may require iteration | Listed values are starting points; the validation harness will reveal whether they need adjustment. |

## 8. Effort estimate

**1-2 weeks of focused work**, broken down roughly:
- Days 1-2: world setup, body definitions, joint creation, basic geometry. First "it stands" milestone.
- Days 3-4: Wire up active PDs, off-axis locks. Validate commanded motion (tests 2-5).
- Days 5-6: Disturbance recovery test, ROM compliance, peak moment. Tune gains as needed.
- Days 7-8: Visual companion binary. Documentation.
- Buffer: ~3 days for unexpected issues (small-body coupling, tuning surprises).

## 9. Acceptance

Phase 1 complete when:
- All 9 test scenarios pass in `ankle_anatomical.rs`
- Visual demo runs smoothly (no NaN, animations look anatomical)
- Code reviewed for project-standard biological constraints discipline
- Memory updated: new memory `project_phase1_ankle_complete.md` noting completion

Subsequent phases revisit this design with knowledge gained.
