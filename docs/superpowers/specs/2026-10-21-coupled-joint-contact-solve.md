# Design: coupled joint+contact solve (JointAxis inside the SI loop)

**Date:** 2026-10-21
**Status:** APPROVED by the user (design-only; this document is the implementation contract)
**Base:** `feature/rendering` @ 79bae46. Blocked Stage-2 migration preserved at branch `stage2-coaxial-foot-blocked` (commit 7dae7e0).
**Scope:** the fix for the diagnosed force-transmission gap; the blocker for the foot's JointAxis migration; THE Phase 3 (windlass) prerequisite.

All file paths are repository-relative.

## 0. Recommendation in one paragraph

Move each `JointAxis` into the SI Gauss-Seidel loop as an **iterated, incremental soft constraint** (Catto gamma/bias form — the exact constraint-space re-parameterization of the existing backward-Euler solve), implemented as a per-joint **2x2 active/passive mini-block** with an **accumulated-impulse clamp** `|lambda_act| <= dt*torque_limit` as the torque budget. Linearize once per substep at theta_n (positions are frozen during the loop, so per-iteration re-linearization is mathematically identical — this question dissolves). Baseline is cold-start (joint lambda reset per substep, contacts unchanged); two pre-designed escalation levers — joint-lambda warm-start and ground-manifold-lambda warm-start by (body, endpoint) identity — are specified and land only if the measured convergence demands them. Free-space behavior is algebraically identical to Stage 1 (Section 2.4: the in-loop constraint reaches the Stage-1 fixed point in ONE iteration when nothing else acts on the pair), and the coupled fixed point transmits the full static demand tau0 through contact, dt-independently (Section 2.5). Option (b) is rejected quantitatively (it fixes at most a factor-k of a factor-1880 deficit, Section 3b); option (c) is rejected on implementation risk, not correctness (Section 3c).

## 1. Root cause, formalized

Stage-1 `JointAxis::apply_implicit` (crates/physics/src/joint_axis.rs:359) computes

    omega_plus = (omega + dt*w*tau0) / D,
    D = 1 + dt*w*(d_act + d_pas) + dt^2*w*(k_act + k_pas)

and applies J = (omega_plus - omega)/w — a model in which *its own impulse determines the end-of-substep velocity*. When the SI contact loop afterwards annihilates omega_plus (wedged toe), the booked velocity-feedback terms (-kd*omega_plus, -kp*dt*omega_plus, -c*omega_plus) were charged against motion that never happened; with omega ~ 0 the transmitted torque is J/dt = tau0/D — the measured 1/1880, doubling at half dt because D-1 is proportional to dt*w*(kd+c), damping-dominated at MTP scale (w ~ 1.0e7 back-solves exactly from D = 1880 at sub_dt = 1/7200). The correct coupled statics: contact holds omega_plus = 0, so the joint must book against the *realized* velocity and transmit tau0 in full; the ground supplies the reaction. The saturated branch's sigma*tau_max transmits correctly because a constant torque has no velocity feedback to mis-book — the code hint confirms the diagnosis.

## 2. The chosen formulation (option a, block form)

### 2.1 Constraint-space re-parameterization (exact, not approximate)

Backward Euler for the joint's total coaxial torque, rearranged as an impulse equation: the substep impulse Lambda = dt*tau0 - dt*(d_tot + dt*k_tot)*omega_plus. Dividing through:

    residual:  omega_plus + gamma*Lambda + b = 0,
    gamma = 1/(dt*(d_tot + dt*k_tot)),
    b = -gamma*dt*tau0

Per GS iteration, with current axis-relative velocity omega_i (which now *includes every contact impulse applied so far this sweep*) and accumulated Lambda:

    delta_Lambda = -(omega_i + gamma*Lambda + b)/(w + gamma)
    Lambda += delta_Lambda; apply delta_Lambda along n to both bodies.

This is the standard Catto soft-constraint. **Fixed-point identity (verified algebraically):** the fixed point is

    lambda_star = (dt*tau0 - omega*dt*(d_tot + dt*k_tot))/D

which equals the Stage-1 impulse exactly (expand lambda_star = -(omega + b)/(w + gamma) and multiply numerator and denominator by dt*(d_tot + dt*k_tot); the denominator becomes w*dt*(d_tot + dt*k_tot) + 1 = D).

### 2.2 Active/passive split: the 2x2 mini-block (the torque budget)

Two stacked scalar constraints on the same axis:
- **active**: gamma_a from k_act = kp, d_act = kd; bias b_a from tau_act0 = kp*(target - theta_n) + ff
- **passive**: gamma_p from k_pas = K_sp - s_lim'(theta_n) + ext_stiffness, d_pas = c + ext_damping; bias b_p from tau_pas0 = -K_sp*(theta_n - rest) + s_lim(theta_n) + ext_torque

Do NOT iterate them as two independent GS rows — two near-hard constraints on the identical axis Gauss-Seidel at rate rho = w^2/((w+gamma_a)(w+gamma_p)) ~ 1 (the SPLIT converges glacially even though the total converges fast, and the clamp/diagnostics read the split). Instead solve both deltas directly each iteration, with u the current axis-relative velocity:

    (w+gamma_a)*D_a + w*D_p        = r_a = -(u + gamma_a*lambda_a + b_a)
    w*D_a        + (w+gamma_p)*D_p = r_p = -(u + gamma_p*lambda_p + b_p)
    det = w*(gamma_a + gamma_p) + gamma_a*gamma_p > 0

Determinant positivity IS the A-stability certificate: every k and d >= 0 (soft limit contributes -s' >= 0; ext_* asserted >= 0) implies gamma > 0 implies det > 0 — the same non-negativity invariant Stage 1 carries, translated.

**Clamp (the budget):** if |lambda_a + D_a| > dt*torque_limit, set lambda_a <- sigma*dt*torque_limit (sigma = sign of the unclamped end-state demand — preserves the Stage-1 sigma-sign-flip semantics, since the unclamped solution IS the end-of-substep demand), then re-solve the passive row alone:

    D_p = -(u + w*D_a + gamma_p*lambda_p + b_p)/(w + gamma_p)

"Muscle saturates, ligaments don't" — structurally identical to Stage 1's saturated branch; the accumulated clamp makes sustained active torque exactly +/-tau_max, dt-independently (cap = dt*tau_max scales with dt, so the torque cap does not).

Verified against Stage 1: the unsaturated 2x2 fixed point gives the identical omega_plus as the shared-denominator formula (eliminate the lambdas: u_plus = (omega + dt*w*(tau_a0 + tau_p0))/D). The saturated branch reproduces omega_plus = (omega + dt*w*(sigma*tau_max + tau_p0))/(1 + dt*w*d_pas + dt^2*w*k_pas).

**Degenerate branches (must be explicit in code):**
- `d_i + dt*k_i ~ 0` with nonzero tau_i0 (pure feedforward / pure ext_torque): gamma -> infinity limit = constant impulse; set lambda_i = dt*tau_i0 on iteration 1 (apply the delta), exclude that row from subsequent solves. This is exactly "constant torque rides outside the denominator".
- Active absent (disabled, or kp = kd = 0 and ff = 0): 1-D passive solve. Passive absent: 1-D active with clamp. Both absent: skip.
- Both bodies pinned (w < 1e-15): skip, but still consume ext_* (same as today).

### 2.3 Lifecycle: prepare (once per substep), solve_iteration (x n_iter), finalize (once)

**`prepare(bodies, dt)`** — runs in the substep where `apply_joint_axes` runs today, but **moved BEFORE `generate_ground_manifold`** (so speculative-contact admission sees any warm-start velocity change; currently nothing depends on the slot since the legacy implicit passes already run after manifold generation — note this relocation in the commit message):
1. Consume ext_torque/ext_stiffness/ext_damping (same per-substep semantics; tendon/windlass injections are set before `world.substep()` exactly as documented today). Keep the ext_* >= 0 debug_assert here.
2. Tear latch at theta_n (unchanged semantics, same substep the threshold is crossed).
3. Compute and CACHE: n = q_a * axis_local_a, theta_n, i_inv_a*n, i_inv_b*n (zero vectors for pinned bodies, post-f11d73f convention), w, gamma_a, b_a, gamma_p, b_p, cap = dt*torque_limit. Positions are frozen during the SI loop, so all of these are constant for the substep; only omega varies between iterations. (Perf: avoids 8x world_inv_inertia and 8x swing-twist per joint.)
4. lambda_a = lambda_p = 0 (baseline cold start). If the L2 warm-start lever is enabled: clamp the carried lambda_a to the current +/-cap, apply both carried impulses to the bodies, keep the lambdas seeded.

**`solve_iteration(bodies, dt)`** — called inside `solve_constraints` (crates/physics/src/xpbd.rs), placed **after the AngularLimitConstraints, before contact_constraints and ground_manifold** (contacts keep the last word on non-penetration within each sweep, matching the existing ordering philosophy — ground_manifold is already solved last). Read u = (omega_b - omega_a) . n fresh from the bodies with the cached n; run the 2x2 block; apply deltas via the cached i_inv*n vectors.

**Convergence skip:** skip applying when |D_a| + |D_p| <= 1e-13 * (|lambda_a| + |lambda_p| + tiny). This makes free-space iterations 2..8 true no-ops (bit-stability for T2's 1e-12 trajectory comparisons) and is a perf win; 1e-13 relative is far below any physical residual so it cannot mask real coupling.

**`finalize` (post-loop, before integrate_positions):** write diagnostics — `applied_total_torque = (lambda_a + lambda_p)/dt`; `applied_active_torque = sigma*torque_limit` **assigned directly, NOT lambda_a/dt**, when the loop ended clamped (T3 asserts bit equality `== tau_max`; (tau_max*dt)/dt is not guaranteed bit-exact in f64), else lambda_a/dt. Store the lambdas for the next substep if warm-start is enabled. Note the diagnostic now reports the REALIZED transmitted torque including contact coupling — this is precisely the receipt experiment E1 reads.

**Resulting substep order (flag on):** 1 dispatchers/tendons -> 2 integrate_velocities -> 3 joint prepare (+ warm-start if L2) -> 4 generate_ground_manifold -> 5 legacy implicit passes (unchanged) -> 6 contact reset -> 7 SI loop x solver_iterations (positional, angular limits, JOINT AXES, user contacts, ground manifold) -> 8 joint finalize -> 9 integrate_positions -> 10 clear accumulators.

### 2.4 Free-space exactness (Stage 1 preserved — the T1-T5 guarantee)

With no other constraint touching the pair, iteration 1 solves the same linear 1-D (or 2x2) system Stage 1 solves — GS is exact in one step for an isolated linear constraint — landing on the identical fixed point (algebra in 2.1/2.2); iterations 2..8 produce deltas at roundoff scale, suppressed by the convergence skip. With warm-start enabled, the free-space case still lands exactly: pre-applying lambda_prev then solving incrementally cancels analytically (lambda_1 = -(omega + b)/(w + gamma), independent of lambda_prev — derived, not assumed). So there is no stale-impulse hazard when contact vanishes mid-flight: the first sweep unwinds a stale warm impulse exactly.

Honest caveat: floating-point op ORDER changes (gamma-form vs D-form; the warm-start add/subtract). T1 (+/-0.005), T3 (+/-0.005 plus the bit-equality handled in 2.3), T4, T5 have ample headroom. T2 asserts trajectory match < 1e-12 over seconds; with the convergence skip this should hold (code iteration-1 math with the same expression shapes as `apply_implicit` where cheap). Sanctioned fallback if measurement shows an ulp-walk: re-derive the T2 tolerance to <= 1e-10 with an explicit fp-associativity justification comment — a tolerance re-derivation, never a semantics change.

### 2.5 Coupled statics and dt-independence

At the converged fixed point with contact holding omega_plus = 0: the residuals force lambda_a + lambda_p = dt*(tau_act0 + tau_pas0) (clamped case: sigma*dt*tau_max + dt*tau_pas0), i.e. transmitted torque = full static demand, a function of theta only — **dt-independent by construction**. The contact's lambda_n picks up the reaction; the lever statics (E1's ~100 N) follow.

### 2.6 Convergence analysis — the honest part, and the escalation ladder

GS convergence per sweep for a joint-vs-contact pair: rho = c^2/((w_j + gamma)*w_c) <= a*(D-1)/D, where a = (J_j M^-1 J_c^T)^2/(w_j*w_c) is the mass-metric alignment. Estimated for a DIP-scale toe tip on ground (I ~ 5e-9 kg*m^2, m ~ 2.4 g, 4 mm lever): a ~ 0.4-0.5, so the cold-start residual after 8 sweeps is ~0.45^8 ~ 2e-3 — essentially fully converged within one substep.

**Worst case a -> 1** (contact exactly blocks the joint axis through the mass matrix): rho -> (D-1)/D ~ 0.9995 at toe scale; 8 cold iterations then transmit only ~8*tau0/D per substep — the bug would survive. Critically (derived during design): for an UNclamped soft constraint the incremental update is mathematically an overwrite from omega_tilde = omega_i - w*lambda, so **joint-side warm-starting alone does NOT accumulate across substeps in the degenerate case** (the contact zeroes the velocity signal at each substep's end, and iteration 1 re-crushes lambda to tau0/D). The static memory must live on BOTH sides: with **ground-manifold lambda carried across regeneration** (matched by (body_idx, endpoint slot 0/1) — the manifold has a stable identity) AND joint lambda carried, the cross-substep fixed point is exact for any alignment: Lambda* = (w*Lambda* + gamma*dt*tau0)/(w + gamma) implies Lambda* = dt*tau0. The relaxation time is ~ (D/n_iter)*dt, which SHRINKS with dt (proportional to dt^2 in the damping-dominated case) — so even the transient is dt-convergent.

**Decision: baseline cold, measure, escalate.** The real foot's alignments are expected non-degenerate (the 0.44 estimate; chains give the proximal side freedom); acceptance experiments plus the convergence sweep (E6) decide. Escalation ladder, each a small gated commit, triggered ONLY by E1/E2/E6 misses:
- **L1:** solver_iterations 8 -> 16 (perf headroom is enormous, Section 2.8). Trigger: E6 shows transmitted-torque fraction materially improved at 16 and E1/E2 bands missed at 8.
- **L2:** joint-lambda + manifold-lambda warm-start pair (both fully specified above; manifold warm-start likely IMPROVES every contact test's convergence, but it touches all contact greens, hence its own commit plus a full gate re-run; the release-projection and speculative logic need no change — a carried lambda at a vanished endpoint simply is not re-created, and a carried lambda unwinds within one sweep via the lambda >= 0 clamp). Trigger: E1/E2 still missed at L1, or E6 shows a near-degenerate alignment (residual tracking (D-1)/D).
- **L3:** joint + adjacent-contacts direct mini-block inside GS (designed only if L2 fails — not expected, since L2's fixed point is exact for any alignment).

### 2.7 Energy / A-stability story

- Fixed point = coupled backward-Euler + contact complementarity: unconditionally A-stable; the denominator->=1 certificate translates to gamma > 0 iff all k, d >= 0, enforced by the same debug_assert on ext_* (kept, now in prepare).
- Unsaturated iterates: each 2x2 block step is an exact coordinate-descent (prox) step on the regularized dual quadratic; PGS truncation is the same regime the existing SI loop already lives in. Friction cones make the global problem nonconvex — pre-existing, unchanged.
- Saturated branch: constant bounded torque that only replaces a larger demand — Stage 1's argument verbatim.
- No formal no-injection proof exists for PGS+friction anywhere; the house enforcement is measurement: E5 (settle-chain +1 mJ, manifold tests, T4a restitution <= 1, T5 stress) gates every stage.

### 2.8 Perf at 7200 substeps/s

Per joint per iteration: 1 dot product, ~30 flops, 2 angular-velocity updates (cached i_inv*n). ~30 axes x 8 iters x 7200 Hz ~ 1.7M block-solves/s ~ 1e8 flops/s — noise next to the existing per-iteration AngularLimitConstraint work (which recomputes quaternion swing-twist x8; the joint block does not, thanks to prepare caching). The expensive Stage-1 work (angle(), world_inv_inertia) moves from once to once. 16 iterations (L1) remains trivially affordable.

### 2.9 Compatibility answers

- **Friction / speculative contacts:** no change. Joint impulses within the sweep shift normal forces before friction clamps — that is the point; the speculative-friction policy and release projection are untouched. Gates: the friction trio + fast-impact tests.
- **Tendon injection (ext_*):** consumed at prepare, folded into b_p/gamma_p — injection point and semantics ("set before world.substep") unchanged. The Phase-3 windlass plugs in here and gets full contact coupling for free.
- **ContactConstraint/manifold:** baseline NO change. Iteration count stays 8 pending E6. The only candidate change is the L2 warm-start lever (own commit, own gates).
- **Multiple JointAxes on one body pair** (talocrural primary + two off-axis locks, the blocked-branch structure): each is an independent GS row; orthogonal axes couple only weakly through anisotropic inertia — handled by the sweep.

## 3. Alternatives, honestly weighed

**(b) Staggered re-solve — REJECTED, quantitatively.** The pass-based solve carries no accumulated impulse; each outer [joint pass -> contact loop] adds ~tau0/D*dt of impulse (the pass re-books from the freshly annihilated velocity) and the next substep starts from zero again. Sustained transmission after k outer loops ~ k*tau0/D: k = 4 turns 1/1880 into 4/1880 — three orders short — and the dt-dependence (D proportional to dt*w*d) survives at full strength. Exactness requires k ~ D ~ 2000 outer loops. The only repair is adding impulse accumulation + warm start, at which point it IS option (a) with a strictly worse loop structure (full contact re-convergence per joint update = wasted work, plus ordering artifacts). The fixed point it converges to at practical k is simply "slightly-less-throttled", still dt-dependent.

**(c) Island direct solve — REJECTED for now, with respect.** Correctness is unmatched (exact coupled NCP per substep; no convergence caveats, no warm-start subtleties). Perf is actually fine (a 28-body island with ~50 rows is a ~1e6-flop dense solve; x7200/s ~ 7 Gflop/s, borderline but feasible). The honest costs: island discovery + J M^-1 J^T assembly + an LCP/NCP solver with friction cones and the saturation complementarity — the single most defect-prone component class in physics engines, replacing the solver path under every currently-green test, maintained by one developer. Rollback is a full revert, not a flag. It remains the documented escalation if (a) fails E1/E2 even at L2 — which the convergence math says it will not, since L2's fixed point is exact for any alignment.

**(d) Hybrids.** The chosen design already is one: a direct per-joint 2x2 block embedded in GS. "In-loop only for joints in contact islands, pass-based elsewhere" was considered and rejected: in free space the in-loop form is exact anyway (2.4), so the split buys nothing and costs a mode seam. Larger joint+contact blocks are the L3 contingency.

## 4. Acceptance experiments (built as gates, red-first)

**E1 — Static force transmission (the ~100 N check).** New test in crates/physics/tests/engine_regression.rs (or a new coupled_joint.rs). Rig: pinned proximal body; one JointAxis (kp sized so demand ~ 2.4 N*m at the settled error); child capsule toe resting on ground; lever l = horizontal distance contact<->axis sized so tau/l ~ 100 N (e.g. 24 mm). Two phases (neutral, pressed); after 1 s settle measure:
  (i) `applied_total_torque` vs the state-computed static demand kp*(target - theta_inf) + ff - K_sp*(theta_inf - rest) + s_lim(theta_inf) — within **10%**;
  (ii) Delta(total_normal_force) vs tau_measured/l — within **10%** (friction + weight-share cross-talk is why not 2%);
  (iii) the runs at sub_dt **1/7200 and 1/3600** agree with each other within **2%**.
Red-first receipt: with the flag off, record the current numbers (expected ~demand/1880) in the test's doc comment.

**E2 — Grounded dt-independence (heel-lift threshold).** Rig: pinned "shin"; one foot capsule (m = 1 kg, half_h 0.1, r 0.02) lying along +z on both endpoints; ankle JointAxis about x. From the rig's statics compute the heel-unload threshold torque tau* (moment balance about the ball contact); command PD targets realizing **0.8*tau*** and **1.25*tau***. Assert: lift/no-lift OUTCOME identical at **1/7200 and 1/3600**; for the lifting case the settled heel height agrees within **10%** across rates; for the non-lifting case the heel stays loaded (lambda_n > 0) at both rates. (This is the scenario class that qualitatively flips today.)

**E3 — Stage 1 free-space exactness.** T1-T5 (crates/physics/tests/joint_axis.rs) green through the new default path. Tolerance re-derivation only per 2.4's fallback rule (T2 <= 1e-10 cap, justified in-comment).

**E4 — Stage-2 gates on the re-landed foot** (cherry-pick 7dae7e0): pump canary rebuilt on a JointAxis toe ray with bands re-derived against REAL physics — cross-checked against statics, not just "whatever it measures": the DIP press settle should now obey the honest torque balance (PD vs spring vs limit slope vs contact wedge); state the rigid-split-with-contact prediction and band around it. MTP Part B grounded lift: physics predicts the lift should now EXCEED the legacy value (full torque transmits where legacy relied on bursts) — re-measure, expect >= 1 mm with margin. Demo scenario measured targets re-derived. Ankle CoP under the 0.2 rad PF command: the true value is UNKNOWN; qualitative prediction: anterior shift ~ tau_transmitted/W, growing until the CoP reaches the forefoot edge, then heel-rise takes over — expect >> the blocked branch's 3.3 mm; whether it approaches the artifact-era +124 mm (nearly the whole foot length — plausibly ratchet-inflated heel-lift geometry) is exactly what the re-derivation determines. The >= 5 mm band in src/bin/ankle_anatomical.rs should pass with margin; the documented value gets replaced.

**E5 — Energy.** settling_chain_does_not_gain_energy (+1 mJ bound), manifold-level test, friction trio, fast-impact, T4a restitution, T5 — all green at every stage. Add to E1 a 5 s pressed-phase total-mechanical-energy monotonicity check (no pump under the sustained honest contact fight).

**E6 — Convergence instrumentation (diagnostic, #[ignore] test or a bin).** On E1's rig and the blocked-branch toe5 rig: transmitted-torque fraction vs solver_iterations in {4, 8, 16, 32} x {cold, L2 warm} x {1/7200, 1/3600}. This measurement decides whether L1/L2 land, and it documents the margin in the findings entry.

## 5. Staging and rollback (house style: additive first, migration gated)

1. **Commit A (additive, zero behavior change):** `JointAxis::{prepare, solve_iteration, finalize}` + lambda fields; `XpbdWorld.coupled_joint_solve: bool` **default false** (flag-off = exact current pre-pass path; `apply_implicit` retained); unit tests: flag-on free-space single-substep equivalence to `apply_implicit` (including saturated and degenerate-gamma branches), E6 harness.
2. **Commit B (gates red -> green):** E1/E2 written against flag-off (red; numbers recorded as receipts in doc comments), then asserted flag-on (green); **flip the default to true in this commit** with the full suite + T1-T5 + E5 green. Rollback at any later point = flip one default.
3. **Commit C (Stage-2 re-land):** cherry-pick 7dae7e0; re-derive E4 bands; ankle_anatomical + foot_anatomical_demo gates green.
4. **Commit D (cleanup, only after C survives):** remove the flag and the `apply_joint_axes` pre-pass from substep; keep `apply_implicit` as the documented single-solve primitive / executable spec.
5. **L1/L2 levers** land as separate gated commits only if E6/E1/E2 demand them (triggers in 2.6).

**Legacy passes and un-migrated binaries:** untouched at every stage. A world with empty `joint_axes` executes an identical substep (the new code adds only empty loops), so every legacy demo/binary is bit-identical. The legacy sequential primitives KEEP the documented force-transmission gap — acceptable because they are off the Phase-3 critical path; the story: migrate per-assembly onto JointAxis, gated on the E-experiments; the legacy coaxial passes become deprecated-but-frozen (deprecation itself is future work, not this change).

## 6. Documentation corrections that become due

- **docs/findings.md:** new dated entry (the 1/1880 receipt, the omega_plus-booking mechanism, the in-loop fix, E6 convergence numbers). Amend the 2026-07 entry: the **+124 mm CoP migration** is an artifact-era number produced by burst-transient ratcheting (41-207 rad/s per-pass transients) — record the re-derived value. Amend both 2026-10 entries that promise "to be removed by a combined coaxial-stiffness solve": the pass-based combined solve preserved free space but throttled under contact; the actual fix required in-loop coupling.
- **crates/physics/tests/engine_regression.rs:** the pump canary's EQUILIBRIUM BAND paragraph (cites the attenuation formula and the combined-solve promise) — repoint to the new findings entry; the legacy-ray test itself stays untouched and green (legacy path unchanged); a parallel JointAxis-ray canary is added in Commit C.
- **src/bin/ankle_anatomical.rs:** CoP test comment re-derivation; MTP Part B re-scope note updated with the honest grounded number.
- **README.md:** Phase-3 prerequisite status; any line citing the CoP migration figure or toe behavior.
- **crates/physics/src/joint_axis.rs / xpbd.rs doc comments:** the "runs FIRST in this slot" / ordering narrative rewritten for the in-loop scheme.

## 7. Risks

1. **GS convergence at degenerate alignment** -> residual throttling / dt-dependent force-development lag (worst (D/n_iter)*dt ~ 33 ms toe @1/7200, 0.26 s @1/3600). Mitigation: E6 measures; L1 -> L2 ladder; L2's fixed point is exact for any alignment. Likelihood judged low (alignment estimate ~0.44).
2. **fp-noise vs T2's 1e-12** -> convergence skip (primary), tolerance re-derivation <= 1e-10 (fallback, justified).
3. **Manifold warm-start blast radius (L2 only)** -> own commit, full contact-suite + E5 gates; carried lambda at vanished endpoints is simply not recreated; within-sweep unwind via the lambda >= 0 clamp.
4. **T3 bit-equality** -> assign sigma*tau_max to the diagnostic directly, never divide the clamped impulse (2.3).
5. **Re-landed foot shows NEW real dynamics** (honest ~100 N toe forces may expose genuine structural responses — heel rise, arch twist). That is physics, and Phase 3's job; distinguish via energy gates + statics cross-checks, not band-widening.
6. **Energy under warm-start + truncation** — no formal proof; enforced by E5 at every stage; rollback flag live through Commit C.

## 8. Resolved/open choices (with decisions)

- Per-iteration vs once-per-substep linearization: **identical by construction** (positions frozen in the loop); compute once in prepare.
- Cold-start baseline vs warm-start-from-day-one: **cold** (measurement-driven escalation; the warm pair touches contact greens).
- Joint row position in the sweep: **after angular limits, before contacts** (contacts keep the last word); low stakes, E6 can A/B it.
- solver_iterations: **stay 8** pending E6.
- Accumulated-passive clamp: **no** (ligaments don't saturate — Stage-1 semantics).
- Keep a pass-based path for non-contact joints: **no** (uniform in-loop path; free space is exact anyway).

## 9. Critical files for implementation

- crates/physics/src/joint_axis.rs — prepare/solve_iteration/finalize, the 2x2 block, lambda fields, degenerate branches
- crates/physics/src/xpbd.rs — substep ordering, solve_constraints insertion, the coupled_joint_solve flag, prepare slot before generate_ground_manifold
- crates/physics/tests/joint_axis.rs — T1-T5 must stay green through the new path (T2's 1e-12 and T3's bit-equality are the sharp edges)
- crates/physics/tests/engine_regression.rs — E1/E2/E5/E6 live here; build_toe_ray/toe5_ray are the rig templates; pump-canary doc corrections
- crates/physics/src/contact_constraint.rs — unchanged in baseline; the L2 manifold warm-start lever and the lambda-identity carry live here if needed
- src/anatomy/foot.rs (+ branch stage2-coaxial-foot-blocked, commit 7dae7e0) and src/bin/ankle_anatomical.rs — the Stage-2 re-land and CoP gate re-derivation

## Amendment (2026-10-21): the joint warm seed applies at contact-reset time, not in prepare

Section 2.3 placed `prepare` before `generate_ground_manifold` and had it apply the L2 joint-lambda warm seed there, "so speculative-contact admission sees any warm-start velocity change". Two facts, found while landing the pre-Commit-C mixed-primitive gates, change that placement:

1. **The admission rationale was already moot in practice.** The manifold-side seeds were moved to apply at the contact-reset step (after the legacy implicit passes) when the L2 pair landed, so admission never saw THEIR velocity either; the cross-substep fixed point never depended on admission seeing seeds — only on the seeds being applied before the SI sweeps read velocities, which the reset slot satisfies.

2. **Applied in prepare, the joint seed re-opens the exact rectification channel this batch diagnosed for the manifold seed.** The seed (Δω = w·dt·τ₀, ~1.3e3 rad/s at toe scale under a 2.4 N·m demand) is read by any legacy implicit primitive sharing a body with the joint — precisely the mid-migration state Commit C creates. Measured on the mixed canaries (crates/physics/tests/coupled_joint.rs): a legacy damper on the same pair into a pinned parent silently steals the ground reaction (0.21 N arrives of a ~101 N lever demand, while the λ diagnostic reads full transmission and the world stays quiet); a legacy articulated stack on an adjacent free body turns the seed into a per-substep kick — sustained 151/1441 rad/s limit cycle, 5.2e-2 J energy pump, sign-flipped transmission.

Resolution: `prepare` still runs before manifold generation (read-only on the bodies — it consumes ext_*, runs the tear latch, caches the solve state, and clamps + parks the seed); the parked impulse is applied by the new `JointAxis::apply_warm_start` at the contact-reset slot, immediately alongside the manifold seeds — after every legacy implicit pass, before the SI sweeps. With the amendment both canaries are quiet and transmit exactly (ΔF_n = Δτ/l to 4 digits), and the full gate set stays green: E1 fraction 1.0000 at both rates and 8/16/32 iterations; E6's warm table 0.9999/0.9999/1.0000/1.0000 @1/7200, 1.0002/1.0000/1.0000/1.0000 @1/3600.
