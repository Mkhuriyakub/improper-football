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
