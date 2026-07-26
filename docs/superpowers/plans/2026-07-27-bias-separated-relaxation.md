# Bias-Separated Proposal Relaxation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by
> task. Every behavior change is test-driven.

**Goal:** Determine whether analytic Gaussian-bias separation can reduce
biased proposal-relax force evaluations, while preserving exact accounting,
convergence semantics, and all existing SSW defaults.

**Architecture:** Add component evaluation and telemetry at the existing
`ProposalPotential -> Relaxer` boundary. First remove report-only repeated
evaluations. Then implement one safe L-BFGS core with total-secant and
bias-separated modes. Validate on deterministic objectives before executing a
staged, paired CUDA MACE C60/PdO comparison.

**Tech stack:** Python, NumPy, SciPy, ASE, pytest, existing MACE/CUDA runtime.

**Authoritative design:**
`docs/superpowers/specs/2026-07-27-bias-separated-relaxation-design.md`

---

## Task 1: Component evaluation contract

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_bias.py`
- Modify: `tests/unit/test_relax.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] Write failing tests for a frozen `RelaxEvaluation` value object and
  `ProposalPotential.evaluate_parts()`.
- [ ] Require exact total = true + bias + softening energy/gradient sums.
- [ ] Require one and only one raw calculator call per component evaluation.
- [ ] Test multiple hills, no hills, fixed atoms, and a stable periodic MIC
  branch.
- [ ] Implement the smallest component contract; keep
  `ProposalPotential.evaluate()` returning its current two-tuple.
- [ ] Run the focused tests and commit.

## Task 2: P0 relaxation cache and convergence telemetry

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `pamssw/result.py`
- Modify: `pamssw/walker.py`
- Modify: `pamssw/exploration/runner.py`
- Modify: `tests/unit/test_relax.py`
- Modify: `tests/integration/test_posterior_exploration_runner.py`

- [ ] Write failing call-trace tests proving current SciPy, FIRE, and ASE
  L-BFGS paths re-evaluate initial/final report points.
- [ ] Add defaulted `RelaxTelemetry` to `RelaxResult`.
- [ ] Add an exact-point reporting cache that never suppresses an
  optimizer-requested new point.
- [ ] Record backend, objective calls, cache hits, explicit finalization calls,
  unified convergence, and termination reason.
- [ ] Make bootstrap fail closed when the unified per-atom force certificate
  is not met.
- [ ] Verify existing optimizer trajectories and defaults are unchanged apart
  from removal of report-only duplicate calls.
- [ ] Run focused accounting/runner tests and commit.

## Task 3: Safe L-BFGS core, total-secants first

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `pamssw/config.py`
- Modify: `tests/unit/test_relax.py`
- Modify: `tests/unit/test_config.py`

- [ ] Write failing unit tests for the two-loop recursion, inverse scaling,
  numerical curvature rejection, total-gradient descent check, atomic
  displacement limit, Armijo acceptance, and explicit line-search failure.
- [ ] Implement `safe-lbfgs-total` as proposal-only and opt-in.
- [ ] Add no new public numerical parameters.
- [ ] Keep default proposal/quench optimizer values unchanged.
- [ ] Reject use as `quench_optimizer`.
- [ ] Run focused tests and commit.

## Task 4: Bias-separated secant mode

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_relax.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] Write failing tests that isolate the only difference:
  `y = delta(total_gradient)` versus
  `y = delta(total_gradient - bias_gradient)`.
- [ ] Require both modes to generate the same first step.
- [ ] Require explicit rejection when component evaluation is unavailable or
  local softening is enabled.
- [ ] Record per-hill MIC image signatures; if an accepted endpoint changes a
  signature, reject that secant, clear history, and count a branch reset in
  both custom modes.
- [ ] Implement `bias-separated-lbfgs` by reusing the same safe L-BFGS core.
- [ ] Record accepted/rejected secants and bias contribution to
  `s^T y`.
- [ ] Run focused tests and commit.

## Task 5: Existing FIRE2 control and search-path integration

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Modify: `pamssw/exploration/ssw_worker.py` only if diagnostics require it
- Modify: relevant unit/integration tests

- [ ] Add `ase-fire2` as a proposal-only, capability-checked backend.
- [ ] Pass component evaluation only at biased proposal relax call sites.
- [ ] Aggregate minimal relaxation telemetry without storing large vectors.
- [ ] Preserve exact purpose accounting and action caps for every backend.
- [ ] Run all unit/integration tests and commit.

## Task 6: Deterministic local-objective ablation

**Files:**
- Create: `benchmarks/bias_relaxation_compare.py`
- Create: `tests/unit/test_bias_relaxation_compare.py`
- Create: `runs/<timestamp>-bias-relaxation-analytic/...`

- [ ] Write schema tests for raw per-case output without rankings.
- [ ] Add fixed quadratic/anisotropic objectives with one and multiple hills.
- [ ] Run the four backends with identical starts and stopping controls.
- [ ] Record objective calls, convergence, final force/energy, termination,
  and secant diagnostics.
- [ ] Reject, rather than tune, a candidate that fails correctness.
- [ ] Register commands, logs, artifacts, and summary under the research run
  ledger.
- [ ] Commit the harness, not generated run artifacts unless repository policy
  requires them.

## Task 7: CUDA MACE G1 screen

**Files:**
- Create: `runs/<timestamp>-bias-relaxation-mace-g1/plan.md`
- Create: `runs/<timestamp>-bias-relaxation-mace-g1/questions.md`
- Create: `runs/<timestamp>-bias-relaxation-mace-g1/run_manifest.yaml`
- Create: `runs/<timestamp>-bias-relaxation-mace-g1/event_log.jsonl`
- Create: run-scoped driver and result artifacts

- [ ] Freeze HEAD, environment, input/model hashes, complete configs, seeds,
  backend matrix, and stop rules.
- [ ] Set `proposal_trust_radius=None` for every arm and record each backend's
  effective coordinate/step/PBC constraint semantics.
- [ ] Verify CUDA, model, C60/PdO inputs, masks, PBC, and single points.
- [ ] Execute the exact G1 matrix serially by campaign with the existing
  two-worker ThreadPool action pool.
- [ ] Do not retry after a campaign has made physical calls.
- [ ] Validate exact purpose counts, budget caps, geometries, active forces,
  convergence telemetry, and wall time.
- [ ] Stop failed candidates; write a raw comparison and claim boundary.

## Task 8: CUDA MACE G2 paired expansion

**Files:**
- Create a new run-scoped G2 ledger and driver.

- [ ] Include `ase-fire` and the complete total/bias-separated pair only when
  both custom modes passed G1; never advance only one custom arm.
- [ ] Execute paired seeds 42, 43, and 44 with all G1 controls frozen.
- [ ] Report raw paired differences and medians for proposal calls,
  convergence, total calls, wall time, and bootstrap-relative best energy.
- [ ] Do not add significance, optimizer rankings, or retuning.
- [ ] Decide each experimental block:
  keep, reject, or defer with explicit evidence.

## Task 9: Final verification and branch handoff

**Files:**
- Modify: `README.md`
- Modify: relevant design/plan status sections

- [ ] Run the full pytest suite, compileall, import smoke, and
  `git diff --check`.
- [ ] Request independent scientific-spec and code-quality reviews.
- [ ] Remove any block that lacks independent positive evidence, or leave it
  explicitly opt-in and experimental if the evidence boundary requires more
  data.
- [ ] Update README with proven behavior, rejected ideas, and unproven scope.
- [ ] Push the feature branch and create a dependent pull request against the
  exact-accounting runner branch.
