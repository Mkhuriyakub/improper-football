# Anatomical foot — 26-bone high-fidelity design

**Status:** Implemented, with outcome corrections (see addendum)
**Date:** 2026-06-09
**Supersedes:** the foot portion of `2026-05-18-anatomical-ankle-phase1-design.md` (single foot capsule + lumped forefoot are deleted)

## Outcome addendum (2026-06 → 2026-07)

The foot was built as designed (26 bodies, 21 articulated + 8 welded joints,
per-bone ground contact). Three outcomes diverged from this spec and are the
record of what was learned:

1. **§5 test 1 ("static balance, no postural controller — the headline") was
   re-scoped.** Diagnosis (`foot_rigid_balance_diag`) proved the layering:
   the fully-RIGID assembly stands (so contact and load are fine), but the
   bare ligamentous articulated foot slowly twists off its support through
   the compliant subtalar/midfoot joints. Anatomically, weight-bearing arch
   rigidity comes from the plantar fascia / windlass (Phase 3) and intrinsic
   muscles (Phase 4) — neither in scope here. The shipped test asserts what
   this phase actually achieves: the footprint bears full body weight over a
   distributed manifold (`test_footprint_bears_full_weight`).
2. **Ground contact had to become a multi-point SI manifold** (engine work
   this spec didn't anticipate): independent per-endpoint contact collapsed
   the footprint to one intermittent point. See the manifold commit and the
   engine regression test.
3. **Placement:** the plan's column-chaining placement was replaced during
   integration by an explicit `REST_POS` coordinate table with a flattened
   sole plane — documented in `src/anatomy/foot.rs`.
4. **§5 tests 6–8 (inversion resistance, ROM clamping, talocrural passive
   moment) are not yet implemented** — tracked as open validation work.
5. **(2026-10) The §2 half_h derivation ignored capsule cap extension and
   was corrected.** §2 set `half_h = length/2`, but a capsule's surface
   extends one `radius` beyond each segment end, so every bone was
   `2·radius` oversized (~14% on slender metatarsals, 55–70% on blocky
   tarsals — the midfoot rendered as an overlapping blob). `half_h` is now
   `max(length/2 − radius, 1 mm)` so segment + caps = anatomical length;
   REST_POS's hindfoot/midfoot spacing (tuned under the oversize) was
   compacted to match. The correction also unmasked the explicit PD
   proportional term as unstable on true toe inertias (now implicit —
   see `docs/findings.md`), and §5 test 5's original pass had depended on
   the oversized geometry (see the test's re-scope note in
   `ankle_anatomical.rs`).

## 1. Context and motivation

The Phase 1 ankle used a single foot capsule plus a lumped forefoot. That gave a 1-D (knife-edge) base of support: the chain tips over under load because a single capsule can't form a real support polygon (confirmed empirically — static balance test reached 90–180° tilt). The fix is structural, not control: build the foot as it actually is.

This spec replaces the foot with a **26-bone anatomical model** for high-fidelity sport biomechanics — injury mechanics (inversion sprain, turf toe), gait-cycle center-of-pressure migration, push-off power, and cutting/landing. Every bone is its own rigid body; every motion-producing joint is articulated with anatomical (oblique) axes; every bone has collision geometry so any foot surface contacts the ground when it geometrically should.

### Design decisions (locked during brainstorming)

- **Detail level: C2** — all 26 individual bones as distinct bodies (not lumped regions).
- **Joint articulation: option B** — motion-producing joints articulated with anatomical ROM; cadaver-confirmed minimal-motion joints stiff-welded (bones stay distinct for mass, contact, and future fracture-site fidelity).
- **Axes: anatomical everywhere** — literature-derived oblique axis orientations, no orthogonal placeholders.
- **Ground contact: option B** — every bone has an anatomically-sized capsule; the lowest surface contacts the ground in any pose (relies on the world-level ground refactor of 2026-06-09).

## 2. The 26 bones

Each bone is one `RigidBody::new_capsule(mass, half_h, radius)`. The talus already exists in the assembly (its mass is corrected here from the Phase-1 placeholder to its anatomical value). The **shin (3.0 kg) and load stick are retained unchanged** from the existing assembly; the talocrural joint is rewired to the corrected talus, and the subtalar joint is rewired from talus↔(lumped foot) to talus↔calcaneus.

### Mass model (honest note)

There is **no published dataset of individual foot-bone masses in grams** — this is a genuine literature gap. Masses below are derived: bone dimensions are literature-grounded (osteometric studies, see §7), volumes estimated from bounding dimensions with per-class fill factors, then rescaled. Skeletal foot mass ≈ 590 g; the remaining ~410 g of the 1.0 kg anthropometric foot segment (de Leva 1.37%, Dempster/Winter 1.45% of body mass) is soft tissue. To match segment inertia in the dynamics sim, **all bone masses are scaled ×1.70** so each capsule carries its share of soft tissue and the segment totals ~1.0 kg.

### Confidence legend
- **[A]** dimensions from direct osteometric study
- **[B]** dimension sourced, mass derived
- **[C]** estimated (no per-bone literature; interpolated from proportions, labeled)

### Body table

Masses in grams (sim uses kg). Capsule `half_h` = bone long-axis length / 2; `radius` = mean of the two cross-section half-dimensions. Values in meters.

| # | Bone | Mass (g) | half_h (m) | radius (m) | Conf |
|---|---|---|---|---|---|
| 1 | Calcaneus | 247 | 0.0375 | 0.0195 | [A] |
| 2 | Talus (reused, mass corrected) | 162 | 0.0265 | 0.0143 | [A] |
| 3 | Navicular | 51 | 0.0200 | 0.0110 | [A] |
| 4 | Cuboid | 56 | 0.0170 | 0.0120 | [A] |
| 5 | Medial cuneiform | 43 | 0.0120 | 0.0110 | [A] |
| 6 | Intermediate cuneiform | 22 | 0.0090 | 0.0095 | [A] |
| 7 | Lateral cuneiform | 34 | 0.0130 | 0.0100 | [A] |
| 8 | Metatarsal 1 | 94 | 0.0320 | 0.0075 | [A] |
| 9 | Metatarsal 2 | 60 | 0.0365 | 0.0053 | [A]/[B] |
| 10 | Metatarsal 3 | 51 | 0.0345 | 0.0048 | [A]/[B] |
| 11 | Metatarsal 4 | 41 | 0.0320 | 0.0043 | [A]/[B] |
| 12 | Metatarsal 5 | 44 | 0.0300 | 0.0053 | [A]/[B] |
| 13 | Hallux proximal phalanx | 31 | 0.0155 | 0.0073 | [A] |
| 14 | Hallux distal phalanx | 15 | 0.0100 | 0.0055 | [B] |
| 15 | Toe 2 proximal phalanx | 7.7 | 0.0120 | 0.0040 | [C] |
| 16 | Toe 2 middle phalanx | 3.1 | 0.0045 | 0.0030 | [C] |
| 17 | Toe 2 distal phalanx | 2.0 | 0.0035 | 0.0028 | [C] |
| 18 | Toe 3 proximal phalanx | 7.1 | 0.0115 | 0.0040 | [C] |
| 19 | Toe 3 middle phalanx | 2.9 | 0.0045 | 0.0030 | [C] |
| 20 | Toe 3 distal phalanx | 1.9 | 0.0030 | 0.0025 | [C] |
| 21 | Toe 4 proximal phalanx | 6.5 | 0.0105 | 0.0035 | [C] |
| 22 | Toe 4 middle phalanx | 2.6 | 0.0040 | 0.0025 | [C] |
| 23 | Toe 4 distal phalanx | 1.7 | 0.0030 | 0.0025 | [C] |
| 24 | Toe 5 proximal phalanx | 6.1 | 0.0100 | 0.0040 | [C] |
| 25 | Toe 5 middle phalanx | 2.2 | 0.0030 | 0.0025 | [C] |
| 26 | Toe 5 distal phalanx | 2.4 | 0.0035 | 0.0030 | [C] |
| | **Total** | **~997** | | | |

Relative breakdown: tarsals ~61%, metatarsals ~29%, phalanges ~10% (of skeletal mass; same proportions after the ×1.70 soft-tissue scaling).

## 3. Joint articulation map (option B)

Anatomical frame: **+X anterior** (foot long axis ≈ 2nd ray), **+Z superior**, **+Y lateral** (right foot). Axis orientations are specified as unit directions in this frame; the engine's `AngularSpring`/`AngularSoftLimit`/`AngularPDController` already accept arbitrary axes.

### Articulated joints (anatomical axis + ROM + passive soft-limit)

| Joint | Bodies | Axis orientation | ROM | Conf |
|---|---|---|---|---|
| Talocrural | shin ↔ talus | mediolateral; lateral end inferior ~8°, posterior ~6° (Inman) | DF 20°, PF 45° | axis HIGH |
| Subtalar | talus ↔ calcaneus | incline 42° above horizontal, deviate 23° medial from long axis (Inman/Manter) | inv 25°, ev 10° | axis HIGH |
| Talonavicular | talus ↔ navicular | Chopart longitudinal+oblique (Manter, nominal) | sag ~8°, frontal ~15° | axis MED, ROM HIGH |
| Calcaneocuboid | calcaneus ↔ cuboid | Chopart oblique (Manter, nominal) | sag ~5°, frontal ~8° | axis MED |
| Medial naviculocuneiform | navicular ↔ medial cuneiform | sagittal-dominant | ~5° (in vivo: mobile, not rigid) | MED — see note |
| TMT1 (1st ray) | medial cuneiform ↔ MT1 | sagittal-dominant, coupled frontal | ~3.5° + ~5 mm sag translation | HIGH mobile |
| TMT4 | cuboid ↔ MT4 | sagittal | ~9.6° | HIGH mobile |
| TMT5 | cuboid ↔ MT5 | sagittal | ~10° | HIGH mobile |
| MTP hallux | MT1 ↔ hallux PP | mediolateral | DF 70°, PF 35° | HIGH |
| MTP lesser ×4 | MT2–5 ↔ toe PP | mediolateral | DF 40°, PF 40° | HIGH |
| IP hallux | hallux PP ↔ DP | mediolateral | flex 90° | MED-HIGH |
| PIP ×4 | toe PP ↔ MP | mediolateral | flex 35° | MED |
| DIP ×4 | toe MP ↔ DP | mediolateral | flex 60° | MED |

**Medial naviculocuneiform note:** cadaver/clinical convention treats this as rigid, but in-vivo bone-pin studies (Nester 2007, Lundgren 2008) found motion "far greater than expected." For dynamic gait fidelity it is articulated (compliant), not welded. This is a refinement from the brainstorm, where all three cuneonavicular joints were going to be welded.

### Stiff-welded joints (cadaver-confirmed <3–5° motion; bones stay distinct)

Welded via the `weld_bodies` pattern (stiff positional + 3-axis angular springs). Two roles:

*Spanning-tree welds* (the bone's only structural parent — connects it into the kinematic tree):
- Naviculo-intermediate cuneiform (navicular ↔ intermediate cuneiform)
- Naviculo-lateral cuneiform (navicular ↔ lateral cuneiform)
- TMT2 (intermediate cuneiform ↔ MT2) — rigid central ray
- TMT3 (lateral cuneiform ↔ MT3) — rigid central ray

*Loop-closure welds* (extra connections that close kinematic loops and rigidify the midfoot block — the implementer should expect closed chains here, not a pure tree):
- Intercuneiform ×2 (medial–intermediate, intermediate–lateral)
- Cuneocuboid (lateral cuneiform ↔ cuboid)
- Cuboideonavicular (cuboid ↔ navicular)

### Windlass mechanism

Hicks' windlass (MTP dorsiflexion tensions the plantar fascia and stiffens the arch) is modeled as **MTP-dorsiflexion increasing midtarsal passive stiffness** (per Heng 2016), not an independent joint axis. Full active windlass / plantar fascia is deferred to Phase 3; Phase 1 carries only passive arch stiffness via the soft-limits on the Chopart and naviculocuneiform joints.

### Passive stiffness sourcing

Only three foot joints have published moment-angle data: talocrural (Riener & Edrich 1999 double-exponential; ~15–25 N·m at ROM limit), subtalar (~0.5 N·m/deg inversion, ~0.9 eversion; cadaver), first MTP (Heng 2016, 0.66–53.4 N·mm/deg). All other joints get nonlinear soft-limits assigned by analogy (same `K·(angle_past_limit)^exp` form used in Phase 1), tuned during validation. Each is flagged in code as analogy-derived.

## 4. Ground contact model

Every bone carries its anatomically-sized capsule (`capsule_shape` set by `new_capsule`). The world-level ground (refactor of 2026-06-09) applies contact to whichever surface is lowest, automatically, in any pose:
- **Normal stance/gait**: calcaneus (heel), then metatarsal heads (ball), then phalanges (toes) — the rolling center-of-pressure emerges from geometry.
- **Inversion sprain**: lateral surface of cuboid + MT5 contacts as the foot rolls over its lateral edge.
- **Deep flexion / pointe**: dorsal toe and metatarsal surfaces contact.

No bone is exempt from the ground. This is the whole point of the redesign: a permanent physical floor, no per-surface opt-outs.

The natural **tripod** (calcaneus + 1st MT head + 5th MT head) forms a ~5–6 cm wide support polygon at neutral stance — enough that static balance under load should hold with passive structure alone (no postural controller). That is the primary success criterion below.

## 5. Validation

Extends `src/bin/ankle_anatomical.rs` (and its visual companion). The single-capsule foot tests are replaced/augmented:

| # | Test | Pass criterion |
|---|---|---|
| 1 | Static balance under 10 kg load, tripod support, **no postural controller** | stick tilt < 5° for 5 s |
| 2 | Sustained stability (no NaN) with full 28-body assembly | stable 10 s |
| 3 | Center-of-pressure migration during a commanded heel→toe roll | CoP moves monotonically heel→ball→toe |
| 4 | Medial longitudinal arch compression under load | navicular drops a bounded amount, recovers on unload |
| 5 | Commanded MTP dorsiflexion (push-off pose) | hallux + lesser MTPs reach target ±0.05 rad |
| 6 | Inversion resistance | foot resists inversion up to ROM limit; subtalar oblique axis produces inversion-plantarflexion coupling (sprain kinematic) past threshold |
| 7 | ROM clamping on representative joints (talocrural, MTP, subtalar) | stays within ROM +10% |
| 8 | Passive moment at talocrural ROM end | ≈ 25 N·m within 30% (Riener & Edrich) |

Test 1 flipping from FAIL (single capsule) to PASS (tripod, passive) is the headline result that justifies the redesign.

## 6. Risks and open questions

| Risk | Mitigation |
|---|---|
| **Solver cost**: 28 bodies (shin + 26 foot bones + load stick), 22 articulated joints, 8 welds (4 spanning-tree + 4 loop welds that rigidify the midfoot block), ~26 ground-contact checks per substep — far heavier than the 5-body Phase-1 assembly (which ran 10 s in ~1 s). | Measure early. Build staged (hindfoot → midfoot → forefoot), checking step time at each stage. If real-time is lost, profile before optimizing; the solver iteration count may need tuning per body count. |
| **Small-bone stability**: distal phalanges are ~2 g with ~3 mm radius — smaller than the talus that drove the Phase-1 stability margin calc. | Apply the same implicit-damper treatment; compute the coupled-mode stability bound for the smallest bone before committing its mass/joint stiffness (per `project_damping_stability`). |
| **Chopart axis confidence**: the transverse-tarsal (Manter) axis angles are low-N classic data, flagged MED confidence. | Use as nominal starting values; mark in code. Tune against the CoP-migration and arch tests, not treated as ground truth. |
| **Welded-joint count**: 7 welds each add stiff springs that can inject solver energy. | Reuse the validated `weld_bodies` pattern; watch the energy artifact (`project_engine_energy_artifact`) at higher body counts. |
| **Capsules are poor shapes for blocky tarsals** (calcaneus, cuneiforms are not capsule-like). | Acceptable for Phase-1 contact/inertia fidelity. If contact behavior is wrong for blocky bones, a convex-hull or box collider is a future engine addition (out of scope here). |

## 7. Out of scope (deferred)

- Hill-type muscle model (Phase 4) — actuation stays placeholder-PD
- Active plantar fascia / full windlass (Phase 3) — only passive arch stiffness here
- Proprioception / stretch reflex (Phase 5)
- Individual ligament modeling — joints use lumped soft-limit + spring, not per-ligament springs
- Non-capsule colliders (box/convex-hull for blocky tarsals) — future engine work
- Per-bone fracture mechanics — bones are rigid; distinct bodies leave the door open but no fracture model now

## 8. Acceptance

Complete when:
- All 28 bodies build and the assembly is stable for 10 s (no NaN)
- Static balance under load passes with passive structure alone (no postural controller) — the headline criterion
- CoP migration, arch compression, MTP push-off, and inversion tests pass
- Solver runs at acceptable speed (target: faster than real-time for the validation binary; exact bar set after first measurement)
- Memory updated noting the 26-bone foot completion and the world-level ground refactor it depends on

## 9. Citations

Bone dimensions: South Indian cuboid (PMC10837486); talus morphometry (Natl J Clin Anat 2022); SE-Asian calcaneus (PMC11116927); navicular (Surg Radiol Anat 2023); cuneiforms CT (PMC6551459); tarsal dimorphism (Harris & Case 2012); metatarsal lengths (PMC4799308); hallux/MT1 geometry (PMC5115701). Segment mass: de Leva 1996; Dempster/Winter. Joint axes/ROM: Manter 1941; Inman 1976; Lundgren 2008; Nester 2007; Ouzounian & Shereff 1989. Passive stiffness: Riener & Edrich 1999; Heng 2016; IRCOBI 2000 (subtalar).
