# Anatomical ankle — Phase 1: joint topology and passive structure

**Note on naming:** Phase 1 builds the **kinematic structure and passive constraints** of an anatomical ankle, plus placeholder PD actuators. "Fully functional" requires Phase 4 (Hill-type muscle model). The original framing "fully functional ankle in Phase 1" was wrong — corrected here.

---


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
5. **Static moment at ROM end** matches anatomy — peak passive resistance at end of talocrural dorsiflexion ≈ 30 N·m within 30% of Riener & Edrich 1999 values.
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

| Body | Mass | Length | Radius | Role |
|---|---|---|---|---|
| Shin | 3.0 kg (existing) | 0.42 m | 0.05 m | unchanged |
| Talus (new) | 0.05 kg | 0.05 m capsule | 0.020 m | gimbal between shin and foot (~anatomical talus size) |
| Foot | 0.85 kg (was 1.0) | 0.18 m | 0.032 m | hindfoot+midfoot+metatarsals, heel to MTP heads |
| Forefoot (new) | 0.10 kg | 0.07 m | 0.020 m | phalanges (toes only), MTP to tip |

Total foot-region mass stays at 1.0 kg; redistributed across three bodies. Mass split based on Zatsiorsky/Seluyanov segmental data (phalanges ≈ 10% of foot mass, talus alone ≈ 5%).

Total foot+forefoot length = 0.25 m, matches anatomical adult foot (was 0.18 m, anatomically short).

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
       (talocrural joint, axis = world +x at neutral)
          • talus (identity orientation, follows shin frame)
          |
       (subtalar joint, axis = world +z at neutral)
          |
       foot (heel ────────────── MTP) — baseline R_x(-π/2), local +Y → world -Z (heel)
                                  • forefoot ────→ toe tips (foot baseline carried through)
```

**Frame convention:** "world +x at neutral" means: when all bodies are at their starting orientations, the joint's axis points along world +x. Axes are specified in the *parent* body's local frame. The shin is at identity, so its (1,0,0) maps directly to world +x. The talus is also at identity, so subtalar axis (0,0,1) in talus-local = world +z. The foot has baseline `R_x(-π/2)` applied; its local (1,0,0) still maps to world +x (rotation about x doesn't move x), so MTP axis (1,0,0) in foot-local = world +x.

### Why split lumped 3-DOF ankle

Real human ankle isn't a single 3-DOF ball. Talocrural and subtalar have different axes and different roles. A lumped joint can't model how inversion-while-plantarflexing differs from inversion-while-dorsiflexing (which is exactly the mechanism behind sprained ankles). Phase 1 uses orthogonal axes for simplicity; oblique axes are a future refinement.

### Why split foot and forefoot

The MTP joint enables toe dorsiflexion, which is essential for proper push-off and heel-to-toe rocker. Even before adding active push-off (Phase 4), having MTP as a real joint allows the correct rolling kinematics.

## 4. Joint specifications

### Talocrural (shin ↔ talus)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in shin local) | (1, 0, 0) | world +x at neutral. **Simplification:** real talocrural axis is oblique (~10° from frontal); orthogonal is Phase 1. |
| Primary ROM | [-0.35, +0.70] rad | dorsi ~20°, plantar ~40° (Roaas & Andersson 1982) |
| Active PD: KP / KD | placeholder for muscle (see §4 note) | not anatomical peak; replaced by Hill model in Phase 4 |
| Passive K / C | 4 / 1.2 | mid-range stiffness from Riener & Edrich 1999 |
| Soft-limit K, exp | 10000 / 2.0 | sized so passive resistance at limit ≈ 30 N·m, matching Riener & Edrich curves |
| Off-axes (Y, Z) | locked: K=200, ROM ±0.05 rad, soft-limit K=10000 | strong off-axis lock |
| Feedforward | exposed via `with_feedforward()` | for inverse-dynamics gravity compensation |

### Subtalar (talus ↔ foot)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in talus local) | (0, 0, 1) | world +z at neutral. **Simplification:** real subtalar axis is oblique (42° from horizontal, 23° from sagittal — Manter 1941); Phase 1 uses orthogonal. |
| Primary ROM | [-0.26, +0.52] rad | eversion ~15°, inversion ~30° |
| Active PD: KP / KD | placeholder for muscle (see §4 note) | replaced by Hill model in Phase 4 |
| Passive K / C | 3.5 / 1.0 | |
| Soft-limit K, exp | 8000 / 2.0 | ~25 N·m at limit |
| Twist axis (Y in talus local) | very limited: ROM ±0.10, K=8, C=0.5, soft-limit K=10000 | |
| Other off-axis (X in talus local) | locked: K=200, ROM ±0.05, soft-limit K=10000 | |
| Foot baseline `R_x(-π/2)` | applied at this joint | foot lies horizontal at neutral |

### MTP (foot ↔ forefoot)

| Property | Value | Notes |
|---|---|---|
| Primary axis (in foot local) | (1, 0, 0) | world +x at neutral; toe dorsiflexion |
| Primary ROM | [-0.35, +0.87] rad | plantar ~20° (clawing), dorsi ~50° (push-off) |
| Active PD: KP / KD | placeholder for muscle (see §4 note) | replaced by Hill model in Phase 4 |
| Passive K / C | 5 / 1.5 | |
| Soft-limit K, exp | 5000 / 2.0 | ~15 N·m at limit |
| Off-axes (Y, Z) | locked: K=100, ROM ±0.05, soft-limit K=5000 | |

### Active PD gain rationale (and a real limitation)

**The active PDs are a placeholder for muscle.** A simple PD cannot reproduce:
- Force-length and force-velocity dependence
- Activation dynamics (~50 ms electromechanical delay)
- Antagonist co-contraction
- Tendon elasticity (Achilles stores/returns energy)
- Real ankle peak torque dynamic range (postural ~10-20 N·m, walking peak ~100-150 N·m at heel-off, running 200+, jumping 250+) — Winter, Kuo et al.

The Phase 1 PD is sized for **postural control only** (~50 N·m peak output, matches quiet-standing demand). Phase 4 (Hill-type muscle model) replaces these with real muscle dynamics. Calling Phase 1 a "research-grade" ankle is incorrect; Phase 1 is **"joint topology + passive structure + placeholder actuator."** The complete ankle requires Phase 4.

**KP selection:** target ~50 N·m at typical operating range (~half ROM), giving KP_talocrural = 80, KP_subtalar = 50 (smaller muscles), KP_MTP = 30 (intrinsic foot muscles only). These produce postural-grade torques.

**KD selection:** critical damping is `C = 2·sqrt(KP·I_red)` where I_red is the joint's reduced inertia. With the implicit `AngularDamper` integration, any value is stable. Per-joint values:
- Talocrural: I_red ≈ I_talus_about_axis ≈ 5e-5 kg·m² (talus is the small body). C_crit ≈ 2·sqrt(80·5e-5) = 0.13. Use KD = 1.0 for over-damped postural feel.
- Subtalar: I_red dominated by foot, ≈ 1e-3 kg·m². C_crit ≈ 0.45. Use KD = 2.0.
- MTP: I_red dominated by forefoot, ≈ 1e-4 kg·m². C_crit ≈ 0.11. Use KD = 0.5.

These compute differently from the 10%-of-KP rule of thumb; updated values reflect actual inertia.

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
- Foot (0.85 kg, 0.18 m, radius 0.032 m) — capsule, horizontal baseline `R_x(-π/2)`, settled on ground
- Forefoot (0.10 kg, 0.07 m, radius 0.020 m) — capsule, connected to foot via MTP joint
- Talus (0.05 kg, 0.05 m, radius 0.020 m) — capsule above foot
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
| 6 | Ankle-strategy disturbance recovery | Apply 50 N horizontal force for 40 ms (= 2 N·s impulse, ≈ "moderate shoulder bump" magnitude; references Hsiao-Wecksler et al. perturbation studies). Ankle alone brings tilt < 5° within 1.5 s | 3 s |
| 7 | ROM clamping | target = -1.0 rad (beyond dorsi limit -0.35). Actual stays within ROM by ≤ 10% | 2 s |
| 8 | Sustained stability | Normal load, no disturbance, no commands — no NaN, no joint runaway | 10 s |
| 9 | Peak passive moment at ROM end | At max talocrural dorsiflexion (-0.35 rad), passive resistance ≈ 30 N·m within 30% (matches Riener & Edrich 1999 curve at functional ROM limit) | snapshot |

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
- **Oblique anatomical joint axes** — Phase 1 uses orthogonal:
  - Talocrural real axis: ~10° from frontal plane (Inman 1976). Phase 1 uses (1,0,0).
  - Subtalar real axis: 42° from horizontal, 23° from sagittal (Manter 1941). Phase 1 uses (0,0,1).
  - Consequence: foot supination/pronation coupling and sprain-mechanism dynamics absent. Adequate for static balance; revisit for gait.
- Off-axis ROM fine-tuning against cadaver studies (Phase 1 uses functional-feel values)

### Not in any current phase
- Forward / inverse kinematics for limb endpoint placement
- Skin / fat soft tissue
- Vasculature, temperature, fatigue
- Pain / injury modeling

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Talus small-body coupling instability | **Explicit check:** talus inertia I_talus_trans = (1/12)·0.05·(3·0.020² + 0.05²) = 1.2e-5 kg·m². Positional joint K = 60000 N/m at the talocrural anchor with lever arm r ≈ 0.025 m gives effective angular stiffness K_ang = K·r² = 37.5 N·m/rad. Coupled mode ω = sqrt(K_ang/I) = sqrt(37.5/1.2e-5) = 1768 rad/s. Stable explicit-Euler bound sub_dt < 2/ω = 1.1 ms. Project sub_dt = 1.39e-4 s — within margin (8×). With the implicit damper engine fix, this is comfortable. |
| Joint stacking with 3 joints close together | All angular dampers and PD KD terms use implicit integration (engine fix 2026-05-08). Safe. |
| Load stick attachment | Pinned via stiff positional + angular springs (effectively rigid weld). |
| Tuning the active PD gains may require iteration | Listed values are starting points; validation harness reveals adjustments. |
| Phase 1 axis simplifications limit realism | Talocrural (real axis oblique ~10° from frontal — Inman 1976) and subtalar (oblique 42°/23° — Manter 1941) use orthogonal axes in Phase 1. This means foot supination/pronation coupling and sprain-mechanism dynamics are absent. Acceptable for static balance and commanded motion; revisit when adding gait. |

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
