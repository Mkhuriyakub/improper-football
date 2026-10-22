# Engineering findings

Dated record of the non-obvious things this project learned the hard way —
each entry is a diagnosed root cause, not a tuning anecdote. Newest last.
Code references are to the state at the entry's date; the README describes
the current system.

## 2026-05-04 — Axis-angle projection contaminates joint angles

Measuring a joint angle as `scaled_axis().dot(axis)` mixes in bivector
cross-terms whenever the relative rotation has components off the measured
axis — every off-axis wobble bled into the primary-axis reading and the PD
fought phantom error. Fix: swing-twist factorization in every angular
primitive, with an optional `baseline` quaternion so joints with
non-identity rest poses measure deviation from rest, not from identity.

## 2026-05 — Explicit damping has a mass-dependent stability cliff

Explicit-Euler dampers are stable only for `sub_dt < 2·m_red/c`. A damping
coefficient tuned for a 3 kg shin silently blows up a 50 g foot in the same
chain. Treating damping as a per-body-scaled parameter helped, but the real
lesson landed later (see 2026-06): stop integrating stiff terms explicitly
at all.

## 2026-05-11 — A capsule foot is a knife edge, and no controller fixes that

After exhausting controller-side fixes for bilateral disturbance recovery
(CoM-based ankle PD, hip strategy, lift-off detection, velocity feedback),
the root cause was structural: a single-capsule foot contacts the ground
along a line, so any chain rotation lifts one end and the support polygon
collapses. A penalty-spring tripod primitive (`FootContact`) passed in
isolation but exploded under chain load — an underdamped multi-point
penalty resonance. Both dead ends pointed the same way: real foot geometry
and a different contact formulation.

## 2026-05 — The bilateral buckle is an off-axis cascade, not a gain problem

A per-substep trace of the post-disturbance collapse showed a sequence:
yaw-mode excitation, then roll, then sagittal buckle. A single-axis sagittal
controller cannot suppress it at any gain; the fix is 6-DOF whole-body
control (future work), not stiffer PDs. Separately: gravity-feedforward
inverse dynamics G(q) is exactly zero at the neutral standing pose, so it
cannot help there by construction — it pays off in non-neutral poses.

## 2026-06 — 2-gram bones forced the implicit-integration rewrite

The 26-bone foot's distal phalanges (~2 g) violate the explicit bound for
every stiff term by 5–25×. All springs, dampers, and soft limits moved to
backward-Euler velocity solves (unconditionally stable); welds and off-axis
locks moved to iterated SI constraints, because three orthogonal stiff
penalty springs solved as independent Gauss-Seidel passes fight each other
and pump energy even when individually implicit. Joint anchors moved to the
midpoint of the two bone centers: surface-point anchors up to 2 cm apart
turned a 10⁶ N/m weld into a substep-0 explosion. Result: a 28-body
assembly spanning a 3500:1 mass ratio settles stably. Gotcha for future
work: the SI angular constraint has no baseline support, so joints crossing
a rest-orientation boundary must use baseline-aware penalty primitives
(fine at kg-scale masses).

## 2026-06 — Independent contact points collapse to one; manifold contact

Measured: a rigid foot with an 84 cm² geometric footprint made at most ONE
simultaneous ground contact (0.39 cm² effective) under independent
per-endpoint contact — so a rigid body with its CoM over its base tipped
over. Fix: regenerate a contact manifold every substep (one SI
non-penetration constraint per near-plane capsule endpoint) and solve them
together; Gauss-Seidel converges to a consistent multi-point distribution.
Critical detail: the manifold constraints run Baumgarte = 0. The standard
separating-velocity bias injects kinetic energy into rigid multi-contact
bodies at slightly non-coplanar depths — the assembly levitated, then
tipped. With a pure dissipative velocity floor, the same assembly stands at
2.3° on 36 simultaneous contacts.

## 2026-06 → 07 — The layered balance finding

With contact fixed, standing decomposed into three separable layers, each
with a decisive experiment:

1. **Contact** — the fully-rigid assembly (everything welded) stands.
   So the contact model and the load are fine.
2. **Load bearing** — the articulated foot's footprint carries full body
   weight (~108 N over 35 contacts, CoM inside the polygon).
3. **Structural rigidity** — the bare ligamentous articulated foot slowly
   twists off its support through the compliant oblique subtalar and
   midfoot joints (~0.5 s onset, yaw-led).

An earlier conclusion — "passive balance is fundamentally impossible, this
needs active center-of-pressure control" — was **disproved** by experiment 1.
The actual gap is anatomical: weight-bearing arch rigidity comes from the
plantar fascia / windlass mechanism and intrinsic foot muscles, which are
exactly the deferred Phase 3/4 structures. The validation suite asserts
layer 2 (what this phase achieves) and documents layer 3 as the next
milestone, rather than asserting a goal the anatomy can't yet support.

## 2026-07 — Validation honesty beats validation green

Two examples now encoded in the suite: the center-of-pressure test verifies
a +124 mm anterior CoP migration under a plantarflexion command (the
heel-to-toe mechanism works), and the arch-compression test passes its
stated bounds while its doc comment records that the measured 6 µm navicular
drop is ~3 orders below the physiological ~5 mm — the midfoot is
near-rigid in the sagittal plane until Phase 3 adds its compliance together
with the windlass stabilizers that make that compliance survivable.

## 2026-10 — Rendering exposed a geometry bug the numeric tests couldn't see

Visual inspection of `foot_anatomical_demo` showed the midfoot as an
overlapping blob. Root cause: the foot spec derived every bone's capsule
`half_h` as anatomical length / 2, ignoring that a capsule's surface
extends one `radius` beyond each segment end — every bone was 2·radius
oversized (~14% on slender metatarsals, 55–70% on blocky tarsals). Every
numeric validation passed on the wrong shapes: the flat-sole plane is
radius-based (unaffected), masses were untouched, and the footprint was
merely generous. Two lessons: visual inspection is a validation channel,
not decoration; and a rest-pose layout tuned against wrong geometry stores
the error — with true extents the hindfoot/midfoot chain could no longer
span REST_POS's spacing (the subtalar pair sat 33 mm apart, bridged before
by capsule fat), so the calcaneus and cuboid moved anterior to restore
surface-contact joint spacing. A rest-pose joint-gap audit (center distance
vs summed capsule extents, per JOINTS pair) now prints in the demo's
`--probe-toe5` mode.

## 2026-10 — The explicit PD proportional term was the last energy pump

The pinky-toe "spasm" seen in the demo was diagnosed by probe
(`foot_anatomical_demo --probe-toe5`): with PD kp zeroed (kd kept) the toe
chains are kinematically quiet (ω ≤ 0.5 rad/s, sub-0.03 mm motion); with
kp active they limit-cycle, and on the corrected (smaller) bone inertias
the pump grew categorical — the undamped bare foot toppled in <0.5 s and
the loaded world's toe chains detonated. Root cause: the 2026-06 implicit
rewrite moved every stiff term to backward-Euler velocity solves EXCEPT
the PD proportional term, which still fired a per-substep explicit kick
(Δω = kp·err·dt/I ≈ 10³ rad/s on 10⁻⁹–10⁻⁶ kg·m² toe inertias; the shared
c = 1.0 dampers that contained it are ~90× supracritical for a toe and
rectified the kick into wrong-direction drift through the Baumgarte
anchors). Fix: `AngularPDController::apply_implicit_kp` — the same
linearized backward-Euler solve as `AngularSoftLimit`, applied BEFORE the
dampers so they dissipate its impulse within the same substep (ordering
matters: applied after the dampers, the undamped excursion reaches the SI
loop and detonates). Residual toe jitter with the demo's extra dampers is
millimeter-scale and bounded. Knock-on: the MTP push-off test's original
pass had depended on the oversized geometry (the free foot pitching over
fat toes); it is re-scoped to airborne tracking (original 0.8 ± 0.15
criterion) plus grounded lift-off under a stabilized shin — holding a
free-standing toe-stand awaits the Phase 3/4 arch stabilizers.

## 2026-10 — Sequential implicit passes attenuate PD gain by 1/(1+dt²·w·kp)

Locking the implicit-KP fix in with an asserting regression ray
(engine_regression.rs: metatarsal-rooted toe5 chain on the ground manifold)
both re-verified the fix and quantified a bias it carries. The fix itself:
the explicit PD proportional term was the last stiff explicit term after the
2026-06 rewrite and pumped energy through the contact manifold on gram-scale
bodies — visible as toe jitter, categorical once the bone-geometry correction
shrank toe inertias — cured by integrating kp implicitly, with the ordering
load-bearing: moved after the damper passes, the undamped kp excursion
reaches the SI loop and the regression ray limit-cycles at 264 rad/s
sustained (vs 0.167 in the correct order), and against the pre-fix explicit
engine the ray keeps only 5–20% command authority (DIP 0.014 rad vs 0.210
for a 0.3 rad press — the damper-rectification mechanism measured on the
pinned hallux). The quantified discovery: because each stiff term is
integrated one-at-a-time, every backward-Euler pass divides its contribution
by 1 + dt²·w·k — effective proportional gain kp_eff = kp/(1 + dt²·w·kp) —
negligible on kg-scale segments but ~94% attenuation on gram-scale toe
joints at the current substep rate (dt²·w·kp ≈ 19: w ≈ 2.5·10⁸, kp = 4,
sub_dt = 1/7200). Equilibria therefore do NOT follow the rigid split
kp/(kp+K)·target at toe scale, and command flips barely move through the
~90×-supracritical shared c = 1.0 dampers. A known, documented bias — to be
removed by a combined coaxial-stiffness solve (all same-axis stiff terms in
one linearized solve) when Phase 3 needs accurate toe tracking.

## 2026-10-19 — Joint Baumgarte β = 0.2 acquitted; the May 0.024 J/s artifact is gone

An engine review flagged the joint constraints' unscaled per-substep
Baumgarte (β = 0.2 at sub_dt = 1/7200 → ~1440/s feedback, vs contact's
deliberately rescaled ~12/s) as a classic energy injector and the prime
suspect for the May "~0.024 J/s solver injection" accepted back then as a
known limitation. A three-scene sweep (β ∈ {0.2, 0.02, 0.002} on BOTH joint
constraint types together; energy = ΣKE + gravitational PE per frame;
constraint quality = max anchor separation + max angular-lock violation)
measured the opposite of the prediction:

| scene | β | tail rate (J/s) | peak \|v\| (m/s) | max anchor sep (m) | max ang viol (rad) |
|---|---|---|---|---|---|
| settling chain (6 s, tail 2 s) | 0.2 | −5.6e-5 | 0.98 | 4.6e-8 | 0 |
| | 0.02 | −5.6e-5 | 0.98 | 4.5e-7 | 0 |
| | 0.002 | −5.8e-5 | 0.98 | 4.7e-6 | 0 |
| 26-bone foot settle (4 s, tail 1 s) | 0.2 | +1.5e-3 | 0.57 | 1.1e-4 | 2.4e-3 |
| | 0.02 | +1.8e-3 | 0.57 | 4.8e-4 | 8.9e-3 |
| | 0.002 | +1.6e-3 | **6.50** | **9.6e-3** | **1.2e-1** |
| chain + 5 mm anchor error (6 s, tail 2 s) | 0.2 | −4.4e-6 | 0.90 | 1.3e-8 | 0 |
| | 0.02 | −1.4e-5 | 0.98 | 2.6e-6 | 0 |
| | 0.002 | −5.2e-6 | 0.98 | 4.1e-6 | 0 |

Three findings. (1) **Energy does not track β** — the predicted injection
does not exist. The reason is structural: a BILATERAL constraint reclaims
its own bias velocity. The Baumgarte bias accelerates the bodies toward
closure, but the next substeps' solves command progressively smaller biases
and decelerate them again — the transient energy is pulled back out by the
same constraint. Contact's documented launch failure (the reason IT runs
β = 0.002) is a UNILATERAL artifact: a launched body separates, the
constraint deactivates, and the bias energy escapes before being reclaimed.
The 5 mm-assembly-error scene showed the mechanism directly: the predicted
"1440/s × 5 mm = 7.2 m/s one-substep launch" measured as a 2.06 m/s,
+0.41 J transient decaying geometrically (~0.8×/substep) and fully
reabsorbed within ~20 substeps (3 ms) — no clamp needed (zero-preload
assembly, as `foot.rs::anchor_at` guarantees, remains good practice).
(2) **Constraint quality tracks β directly** — β = 0.2 dominates every
scene. Lowering β starves the foot's [0,0]-range weld locks of the position
feedback that holds them against sustained gravity load: at β = 0.002 the
welds gape 9.6 mm / 0.12 rad, the assembly goes spongy, and a bone is
ejected at 6.5 m/s. Defaults stay 0.2; both constraints gained a
`with_baumgarte` builder for future experiments. (3) **The May artifact is
resolved as HISTORICAL**: the 0.024 J/s was measured (elbow-pendulum rigs,
2026-04) on the old textbook-XPBD scheme — position projection plus
`v += dx/dt` velocity feedback, where each correction deposits ½m(dv)² that
nothing reclaims. The 2026-06 velocity-level SI rewrite removed that scheme,
and with it the injection: the same settling-chain scene now NET-DISSIPATES
0.150 J over 3 s with a −5.6e-5 J/s steady tail at every β, and its
regression bound is tightened from +0.2 J to +1 mJ. The foot scene's small
positive tail (+1.5e-3 J/s at 4 s) is not a solver injection either: over a
10 s run its energy is flat from t = 1 s to t = 8 s (+1.3e-4 J/s) with
episodic bumps from the documented bare-foot structural twist (PD and
ROM-limit events), and it does not track β.

## 2026-10-21 — A joint pre-pass under a blocking contact transmits 1/D of its torque; fixed by moving JointAxis into the SI loop (+ warm-start pair)

THE RECEIPT. A pinned-parent toe pressing its grounded tip (E6's rig,
crates/physics/tests/coupled_joint.rs) transmitted 0.90% of its 1.2 N·m
static PD demand through the contact at sub_dt = 1/7200, and 0.45% at
1/3600 — a dt-DEPENDENT, three-orders force-transmission gap. On the
Commit B E1 rig (foot-family damping, w ≈ 3.8e6) the fraction is 1/616;
the design contract's diagnosis rig measured 1/1880. The divisor is
exactly the joint's own implicit denominator D = 1 + dt·w·(d_act+d_pas)
+ dt²·w·(k_act+k_pas).

THE MECHANISM (the ω⁺ booking). `JointAxis::apply_implicit` — like every
sequential implicit pass — computes ω⁺ = (ω + dt·w·τ₀)/D and books the
velocity-feedback terms (−kd·ω⁺, −kp·dt·ω⁺, −c·ω⁺) against ITS OWN
predicted end-of-substep velocity. When the SI contact loop afterwards
annihilates that velocity (a wedged toe), the feedback was charged against
motion that never happened: with ω ≈ 0 the transmitted torque collapses to
τ₀/D per substep, damping-dominated at toe scale (D − 1 ∝ dt·w·(kd+c), so
the gap doubles when dt halves). The saturated branch transmitted
correctly all along — a constant torque has no velocity feedback to
mis-book — which is what the burst/ratchet artifacts of the artifact-era
CoP figures rode on.

THE FIX (design contract: docs/superpowers/specs/
2026-10-21-coupled-joint-contact-solve.md). Each JointAxis runs INSIDE the
SI Gauss-Seidel loop as an iterated incremental soft constraint — the
exact constraint-space re-parameterization of the same backward-Euler
solve (residual ω⁺ + γΛ + b = 0, γ = 1/(dt·(d + dt·k)), b = −γ·dt·τ₀),
split into an active/passive 2×2 mini-block solved directly each sweep,
with the torque budget as an accumulated-impulse clamp |λ_a| ≤ dt·τ_max.
Free space is algebraically unchanged (iteration 1 lands on the pre-pass
fixed point; measured trajectory divergence ≤ 7.3e-14 rad/s over 600
substeps); under contact the joint and the contacts exchange impulses
within each sweep and the coupled fixed point transmits the full static
demand, dt-independently.

CONVERGENCE, MEASURED (E6, transmitted-torque fraction vs iterations).
Cold start on the worst-case alignment (pinned proximal + grounded child,
a → 1): 0.069/0.132/0.241/0.412 at 4/8/16/32 iterations @1/7200, halving
at 1/3600 — the per-sweep residual tracks (D−1)/D, so the escalation
ladder's L1 (16 iterations) measured 0.050 on E1's 10% band and did NOT
land. L2 (the warm-start pair) landed on that trigger: joint lambdas carry
across substeps (clamped to the current ±cap in `prepare`, applied at the
contact-reset slot — see the seed-placement amendment below) AND
ground-manifold lambdas carry across the per-substep regeneration, matched
by (body, endpoint slot). With both sides carried the cross-substep fixed
point Λ* = dt·τ₀ is exact for ANY alignment — verified by measurement:
the same sweep reads 0.9999/0.9999/1.0000/1.0000 @1/7200 and
1.0002/1.0000/1.0000/1.0000 @1/3600. One side alone is NOT enough
(bite-measured: joint seed with the manifold carry disabled = 0.049 — the
contact side forgets the reaction each substep and iteration 1 re-crushes
the joint lambda to ~τ₀/D).

GATES (all green, flag default now true): E1 static transmission (2.4 N·m
demand through a 23.7 mm lever = 101.4 N at the tip; transmitted fraction
1.0000, ΔF_n = Δτ/l to 4 digits, 2% cross-rate band met at 0.5%); E2
heel-lift threshold (τ* = W·l exact by anchor placement; outcome AND
settled lift height dt-independent, where the legacy path's outcome
qualitatively flips with sub_dt); E5 pressed-phase energy monotone over
5 s (max rise 0.0 J; an anti-damper injector pumps 10.9 J and fails it);
T1–T5, the A–D equivalences, engine_regression 17, ankle_anatomical.
One honest rig note: E2 runs μ = 0.05 because its pivot contact's
spin-friction couple adds a REAL ±μ·N·R/kp hysteresis band to the settled
lift height (±48% at μ = 0.6, measured landing anywhere inside the band by
rate and iteration count) — friction path-dependence, physics, not a
solver defect; the rig isolates the dt-independence claim from it.

The legacy pre-pass (`apply_implicit`) and the sequential legacy
primitives keep the documented gap and stay as the flag-off rollback path;
un-migrated assemblies are unaffected until they adopt JointAxis (Commit C
re-lands the foot).

SEED-PLACEMENT AMENDMENT (2026-10-21, pre-Commit-C gates; spec doc
amendment of the same date). The joint-side warm seed originally applied
inside `prepare`, BEFORE the legacy implicit passes — the exact
rectification channel this entry diagnosed for the manifold seed (which
already applies at contact-reset time). A MIXED world — a grounded
JointAxis body sharing bodies with legacy dampers/springs/PDs, which is
precisely the mid-migration state Commit C creates — re-exposed it, and
no gate covered that composition. Two canaries now do
(crates/physics/tests/coupled_joint.rs), both bite-verified against the
pre-amendment placement: a legacy damper on the SAME pair into a pinned
parent silently steals the ground reaction (F_n 0.21 N of a ~101 N lever
demand while the λ diagnostic reads full transmission and the world stays
quiet — the statics assertion is the one that bites); a legacy articulated
stack on an ADJACENT free body turns the seed (Δω = w·dt·τ₀ ≈ 1.3e3 rad/s
at toe scale) into a per-substep kick — sustained 151/1441 rad/s limit
cycle, 5.2e-2 J energy pump, sign-flipped transmission. Fix: `prepare` is
now read-only (clamps and PARKS the seed); `JointAxis::apply_warm_start`
applies it at the contact-reset slot alongside the manifold seeds, after
every legacy pass and before the SI sweeps — which is all the
cross-substep fixed point ever needed (admission never saw the manifold
seeds either, so §2.3's original rationale was already moot). All gates
re-measured green at the amended placement: the figures quoted above are
the current ones. Also hardened in the same batch: `warm_start`
cone-projects the carried FRICTION seed against the constraint's current
cone (μ ≤ 0 zeroes it) — un-projected, a friction impulse carried onto a
μ = 0 ground re-applies every substep and never unwinds (measured: a
coasting capsule reverses and gains speed without bound,
carried_friction_seed_is_cone_projected_when_mu_drops).
