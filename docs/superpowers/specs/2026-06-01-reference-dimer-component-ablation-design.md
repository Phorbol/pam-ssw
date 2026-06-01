# Reference Dimer Component Ablation Design

Date: 2026-06-01

## Objective

PAM-SSW has diverged from the reference SSW implementation along several axes at once: seed selection, direction generation, proposal construction, relaxation policy, and duplicate handling. The next work should isolate those changes instead of adding another tuned Direct-QP variant.

The immediate objective is to answer three questions with real C60/CuO/PdO evidence:

1. Does the archive/UCB seed selector improve over the reference Metropolis chain when the proposal mechanism is held fixed?
2. Does the current scored direction pool help, or does it damage C60 exploration through unproductive momentum/archive-momentum directions?
3. Does the original SSW dimer-constrained soft mode outperform the scored direction pool when embedded in PAM-SSW's archive/UCB framework?

The primary target is an interpretable component comparison, not a new production default.

## Reference Algorithm

The reference code under `/tmp/ssw-reference` uses this loop:

1. Relax the initial structure to a true local minimum.
2. Generate an initial direction with `sample_mixed_mode`.
   - Global random mode sampled from a normalized Gaussian.
   - Local bond-forming mode from a random atom pair farther than `min_pair_distance`.
   - Mixed direction `N0 = normalize(N_global + lambda * N_local)`.
3. Rotate `N0` with `BiasedDimerRotator`.
   - Fixed midpoint at the current structure.
   - Dimer endpoint `R1 = R0 + delta_R * N`.
   - Bias force constrains the dimer direction near `N0`.
   - Final curvature is computed from finite-difference forces along the rotated dimer direction.
4. Add a Gaussian bias centered at the current point along the rotated direction.
5. Displace by `ds` and relax on the modified PES.
6. Remove bias and relax on the true PES.
7. Accept or reject the trial with Metropolis.

The reference method therefore couples the soft mode to the bias+relax proposal path. It does not use a direction pool, UCB seed selection, Direct-QP, archive novelty scoring, or duplicate-rescue logic.

## Current PAM-SSW Axes

The current implementation combines these mechanisms:

- Seed selector: `archive_ucb` or `metropolis_chain`.
- Direction engine: scored candidate pool using momentum, random, bond, bond-form, bond-break, Ritz, regularized Ritz, evolved, and archive-momentum candidates.
- Direction-type memory: optional UCB-style bonuses for direction kinds.
- Proposal path: `bias_relax` or `direct_qp`.
- Post-step correction: optional Direct-QP micro-relax.
- Acceptance/archive: true quench, deduplication, archive insertion, and archive-based reward updates.

Because these were changed together, a better or worse benchmark result cannot currently be attributed to one component.

## Design Options

### Option A: External Reference Only

Run `/tmp/ssw-reference` as an external baseline and compare it against existing PAM-SSW variants.

This is the lowest implementation cost, but it does not answer whether the dimer soft mode helps inside PAM-SSW's archive/UCB workflow. It also leaves diagnostics split across two runners.

### Option B: Opt-In Reference Dimer Direction Engine

Port the reference `sample_mixed_mode` and `BiasedDimerRotator` into PAM-SSW as an opt-in direction engine:

`direction_engine = "scored_pool" | "reference_dimer"`

With `reference_dimer`, each walk step obtains one direction and curvature from the reference dimer routine. That direction replaces `SoftModeOracle.choose_direction` for the step, but the rest of PAM-SSW can remain unchanged.

This is the recommended path. It allows controlled hybrids:

- UCB selector + reference dimer + bias+relax.
- Metropolis selector + current direction pool + bias+relax.
- UCB selector + reference dimer + Direct-QP.

### Option C: Full Reference SSW Reimplementation Inside PAM-SSW

Recreate the entire reference loop in the PAM-SSW runner, including Metropolis-only acceptance and reference adaptive climbing.

This gives the cleanest internal baseline, but it is too large for the next step and duplicates the already available `/tmp/ssw-reference` baseline.

## Recommended Architecture

Implement Option B after this design is approved for implementation.

Add a small module, `pamssw/reference_dimer.py`, with:

- `sample_global_mode`
- `sample_local_bond_mode`
- `sample_mixed_mode`
- `ReferenceDimerRotator`
- `ReferenceDimerResult`

Add configuration fields:

- `direction_engine: str = "scored_pool"`
- `reference_dimer_delta: float = 0.005`
- `reference_dimer_bias_strength: float = 500.0`
- `reference_dimer_max_steps: int = 15`
- `reference_dimer_rotation_tol: float = 0.03`
- `reference_dimer_angular_step: float = 0.05`
- `reference_dimer_lambda_min: float = 0.1`
- `reference_dimer_lambda_max: float = 1.5`
- `reference_dimer_min_pair_distance: float = 3.0`

Add a new `DirectionCandidateKind.REFERENCE_DIMER` for diagnostics. The returned `DirectionChoice` should include:

- normalized direction,
- directional curvature from the dimer calculation,
- candidate count equal to one,
- score unset or zero,
- diagnostics recorded separately.

Do not remove the existing direction pool. The first implementation must be opt-in so benchmark comparisons remain controlled.

## Integration Semantics

For `proposal_step_mode="bias_relax"`:

1. Generate reference dimer direction and curvature.
2. Use the dimer direction in the existing Gaussian bias construction.
3. Use the dimer curvature as the direction curvature input where applicable.
4. Continue with the existing modified-PES proposal relax and true quench.

For `proposal_step_mode="direct_qp"`:

1. Generate reference dimer direction and curvature.
2. Use the dimer curvature as the `directional_curvature` for rank-1 Direct-QP.
3. Avoid an additional true HVP unless a config flag explicitly requests verification.
4. Continue with Direct-QP step, gated micro-relax, and true quench.

The Direct-QP mode is diagnostic only in the first matrix. A worse result would indicate that Direct-QP lacks enough basin-settling even when the direction is improved.

## Diagnostics

Add per-run summary stats:

- `reference_dimer_steps`
- `reference_dimer_mean_rotations`
- `reference_dimer_converged_fraction`
- `reference_dimer_mean_curvature`
- `reference_dimer_mean_abs_dot_initial`
- `reference_dimer_force_evaluations_estimated`

Add per-direction diagnostics rows when direction diagnostics are enabled:

- direction kind,
- curvature,
- dimer rotations,
- dimer converged,
- dot between final dimer direction and initial mixed direction,
- local pair selected by `sample_mixed_mode`,
- lambda used by `sample_mixed_mode`.

Existing direction productivity stats should include `reference_dimer` so it can be compared against momentum, random, bond, and archive-momentum.

## Benchmark Matrix

### Phase 1: C60 Component Matrix

Run C60 with 3 seeds and 40 trials per seed.

Variants:

1. `reference_original`
   - External `/tmp/ssw-reference`.
   - Metropolis selector, reference dimer soft mode, bias+relax.

2. `paw_current_bias_relax`
   - PAM-SSW current production-like bias+relax.
   - Archive/UCB selector, scored direction pool, bias+relax.

3. `paw_metropolis_pool_bias_relax`
   - PAM-SSW Metropolis chain selector.
   - Current scored direction pool, bias+relax.
   - Isolates selector impact.

4. `paw_ucb_reference_dimer_bias_relax`
   - Archive/UCB selector.
   - Reference dimer direction engine, bias+relax.
   - Tests the main simplification hypothesis.

5. `paw_ucb_pool_no_momentum_bias_relax`
   - Archive/UCB selector.
   - Current scored direction pool with momentum and archive-momentum disabled or filtered.
   - Tests whether C60 failure is mainly momentum pollution.

6. `paw_ucb_reference_dimer_direct_qp_adaptive50`
   - Archive/UCB selector.
   - Reference dimer direction engine.
   - Direct-QP rank-1 with adaptive50 micro-relax.
   - Tests whether Direct-QP remains useful with a stronger direction source.

If runtime must be reduced, run variants 1, 2, 4, and 5 first. That reduced matrix still answers the most important direction-engine question.

### Phase 2: Slab Sanity Matrix

Run CuO and PdO with 3 seeds and 40 trials per seed after Phase 1 shows the implementation is stable.

Variants:

1. `reference_original`
2. `paw_current_bias_relax`
3. `paw_ucb_reference_dimer_bias_relax`

Direct-QP slab variants are not part of this first reference-dimer matrix because fixed-kappa Direct-QP is already a strong slab result. The slab question here is whether reference dimer is a generally useful direction engine.

## Metrics

Primary metrics:

- best energy,
- mean best energy over seeds,
- best energy per 1k force evaluations,
- number of minima,
- duplicate rate,
- force evaluations,
- wall time.

Trace metrics:

- best energy versus trial,
- best energy versus force evaluations,
- accepted new basin count versus trial,
- duplicate count versus trial.

Direction metrics:

- selected count by direction kind,
- productive count by direction kind,
- duplicate rate by direction kind when recoverable,
- reference dimer convergence and rotation statistics.

Geometry metrics:

- nearest-neighbor minimum and median,
- fragmentation rejection count,
- unphysical energy-drop rejection count,
- C60 integrity checks for collapse or fragmentation.

## Decision Criteria

Promote reference dimer direction engine for further work if:

- `paw_ucb_reference_dimer_bias_relax` beats `paw_current_bias_relax` on C60 mean best energy or force-normalized best energy, and
- duplicate rate does not increase materially, and
- geometry diagnostics remain physically sane.

Keep archive/UCB selector if:

- `paw_current_bias_relax` beats `paw_metropolis_pool_bias_relax`, or
- `paw_ucb_reference_dimer_bias_relax` beats `reference_original` with similar force budget.

Deprecate or disable momentum-like directions for C60 if:

- `paw_ucb_pool_no_momentum_bias_relax` beats `paw_current_bias_relax`, or
- direction diagnostics show momentum/archive-momentum dominate selected directions while producing mostly duplicate basins.

Do not promote Direct-QP for C60 if:

- `paw_ucb_reference_dimer_direct_qp_adaptive50` remains worse than `paw_ucb_reference_dimer_bias_relax` at comparable force budget.

## Execution Contract

No benchmark execution is part of this design step.

Before running benchmarks, create a run ledger directory:

`runs/20260601-reference-dimer-component-ablation/`

Required files:

- `run_manifest.yaml`
- `questions.md`
- `plan.md`
- `event_log.jsonl`
- `artifacts.json`
- `status.md`
- `summary.md`

Every implementation command, test command, and benchmark command should be logged there before execution begins.

## Risks

- The reference dimer uses several force calls per walk step. It may improve quality but lose force-normalized efficiency.
- A short dimer budget may make the soft-mode comparison unfair. Record convergence and rotation counts so underpowered dimer runs are identifiable.
- `/tmp/ssw-reference` and PAM-SSW may not have identical default relaxation tolerances, so external reference comparisons must record exact parameters.
- Direct-QP with dimer direction may still fail if the missing piece is full basin settling rather than direction quality.
- Slab systems may prefer simpler fixed-kappa Direct-QP behavior; C60 results should not be generalized without CuO/PdO checks.

## Acceptance Checklist

The design is ready for implementation planning when:

- The user agrees that Option B is the first implementation target.
- The Phase 1 matrix variants are accepted or reduced explicitly.
- The benchmark systems and seeds are fixed.
- The output ledger path is accepted.
- The external reference baseline is allowed to run from `/tmp/ssw-reference`.
