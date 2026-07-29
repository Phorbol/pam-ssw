# Uphill Propagation U0/U1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the same frozen SSW walk prefix and direction, whether a calibrated fixed Gaussian, an analytic curvature-matched Gaussian without feedback, or the current fully adaptive Gaussian produces the most useful biased-PES proposal per force evaluation.

**Architecture:** Keep production defaults unchanged. Extend the existing optimizer-neutral `ProposalRelaxationTask` replay boundary with one last-Gaussian retargeting operation and an observer that records true/bias/total energies from evaluations the optimizer already performs. Put calibration and paired comparison in a run-local experiment; do not add OPES, a new bias shape, Direct-QP, or full-walk policy selection in U0/U1.

**Tech Stack:** Python, NumPy, pytest, ASE/MACE through the existing calculator interfaces and `EvalCounter`.

---

## Scientific contract

- The primary comparison is a conditional one-step intervention on a frozen current-policy walk prefix.
- All arms share the same bias prefix, last-bias center, direction, optimizer, force certificate, maximum iterations, and calculator model.
- The last Gaussian and its explicit displacement are the only changed quantities.
- The first-step current tasks calibrate one fixed `(sigma, weight)` pair per system using component-wise medians. Calibration tasks are not pooled with evaluation tasks.
- The current arm replays the captured task byte-for-byte.
- The curvature-matched arm removes feedback scaling. For a captured final bias with unclipped current weight, recover the local matched weight from a separately recorded or recomputed inner curvature:

  ```text
  weight = sigma_base**2 * max(inner_curvature + target_negative_curvature, 0)
  ```

- U0/U1 makes no full-search superiority claim. A winning local arm must later pass paired full-walk and multi-seed C60/PdO validation.

## Existing baseline limitation

The clean worktree baseline is `1452 passed, 7 skipped, 88 failed`. All 88 failures require ignored historical GPU output ledgers that are absent from a fresh worktree. Targeted tests and the core non-artifact suite must remain green; the missing-ledger failures are not to be edited or suppressed in this work.

### Task 1: Retarget one frozen Gaussian step

**Files:**
- Modify: `pamssw/proposal_replay.py`
- Modify: `tests/unit/test_fixed_proposal_replay.py`

- [x] **Step 1: Write a failing immutability and geometry test**

Add a test that captures a one-bias task, calls:

```python
retargeted = retarget_last_gaussian(
    task,
    sigma=0.25,
    weight=0.4,
)
```

and asserts:

```python
assert task.biases[-1].sigma != retargeted.biases[-1].sigma
assert retargeted.biases[-1].sigma == 0.25
assert retargeted.biases[-1].weight == 0.4
delta = (
    retargeted.initial_state.flatten_positions()
    - retargeted.biases[-1].center
)
assert np.dot(delta, retargeted.biases[-1].direction) == pytest.approx(0.25)
np.testing.assert_allclose(
    delta
    - np.dot(delta, retargeted.biases[-1].direction)
    * retargeted.biases[-1].direction,
    0.0,
    atol=1.0e-12,
)
```

Also assert all earlier biases are numerically unchanged and the source task remains immutable.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/unit/test_fixed_proposal_replay.py::test_retarget_last_gaussian_changes_only_last_bias_and_explicit_displacement
```

Expected: import failure because `retarget_last_gaussian` does not exist.

- [x] **Step 3: Implement the minimal retargeting function**

In `pamssw/proposal_replay.py`, add:

```python
def retarget_last_gaussian(
    task: ProposalRelaxationTask,
    *,
    sigma: float,
    weight: float,
) -> ProposalRelaxationTask:
    if not task.biases:
        raise ValueError("proposal task has no Gaussian bias")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")
    last = task.biases[-1]
    center_state = State(
        numbers=task.initial_state.numbers.copy(),
        positions=last.center.reshape(task.initial_state.n_atoms, 3).copy(),
        cell=None if task.initial_state.cell is None else task.initial_state.cell.copy(),
        pbc=task.initial_state.pbc,
        fixed_mask=task.initial_state.fixed_mask.copy(),
        metadata=task.initial_state.metadata.copy(),
    )
    trial_state = CartesianCoordinates.from_state(center_state).displace(
        TangentVector(last.direction),
        sigma,
    )
    biases = (
        *task.biases[:-1],
        GaussianBiasTerm(
            center=last.center,
            direction=last.direction,
            sigma=sigma,
            weight=weight,
        ),
    )
    return ProposalRelaxationTask(
        initial_state=trial_state,
        biases=biases,
        softening=task.softening,
        fmax=task.fmax,
        maxiter=task.maxiter,
        coordinate_trust_radius=task.coordinate_trust_radius,
    )
```

Import `CartesianCoordinates` and `TangentVector` from `pamssw.coordinates`.

- [x] **Step 4: Run targeted tests and verify GREEN**

Run:

```bash
pytest -q tests/unit/test_fixed_proposal_replay.py
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add pamssw/proposal_replay.py tests/unit/test_fixed_proposal_replay.py
git commit -m "Add fixed-prefix Gaussian retargeting"
```

### Task 2: Observe proposal components without observer-only PES calls

**Files:**
- Modify: `pamssw/proposal_replay.py`
- Modify: `tests/unit/test_fixed_proposal_replay.py`

- [x] **Step 1: Write failing observed-replay tests**

Add tests for a new:

```python
ObservedProposalReplayResult
replay_proposal_task_observed(...)
```

The tests must assert:

```python
observed.evaluation_counts.total == observed.result.telemetry.evaluator_calls
observed.observer_only_force_evaluations == 0
observed.initial.true_energy == pytest.approx(expected_true_initial)
observed.initial.total_energy == pytest.approx(expected_total_initial)
observed.final.total_energy == pytest.approx(observed.result.energy)
observed.direction_progress == pytest.approx(
    np.dot(
        observed.result.state.flatten_positions() - task.biases[-1].center,
        task.biases[-1].direction,
    )
)
observed.orthogonal_displacement_norm >= 0.0
```

Use the analytic quadratic calculator so expected true and bias components can be evaluated analytically in the test without calling the replay calculator.

- [x] **Step 2: Run the tests and verify RED**

Run the two new test node IDs. Expected: missing observed replay API.

- [x] **Step 3: Implement an internal recording proposal**

Add a private subclass of `ProposalPotential` that records the `RelaxEvaluation` returned by each normal backend call, keyed by the exact flattened positions. The observer must run inside `evaluate_parts`; it must not call the calculator itself.

Add immutable:

```python
@dataclass(frozen=True)
class ProposalPointObservation:
    true_energy: float
    bias_energy: float
    softening_energy: float
    total_energy: float

@dataclass(frozen=True)
class ObservedProposalReplayResult:
    result: RelaxResult
    evaluation_counts: EvaluationCounts
    wall_time_s: float
    certificate_satisfied: bool
    initial: ProposalPointObservation
    final: ProposalPointObservation
    direction_progress: float
    orthogonal_displacement_norm: float
    observer_only_force_evaluations: int = 0
```

`replay_proposal_task_observed` must otherwise use the same `Relaxer`, optimizer, purpose, fmax, maxiter, and trust-radius code path as `replay_proposal_task`. It must fail closed if the optimizer never evaluated the exact initial or final state; do not silently add a reporting evaluation.

- [x] **Step 4: Deduplicate ordinary and observed replay execution**

Extract only the shared construction/execution lines needed to prevent the two replay functions from drifting. Do not change the public behavior or return type of `replay_proposal_task`.

- [x] **Step 5: Run targeted and core replay tests**

Run:

```bash
pytest -q \
  tests/unit/test_fixed_proposal_replay.py \
  tests/unit/test_proposal_energy_trace.py \
  tests/unit/test_walker_policy.py -k 'proposal or trust or bias'
```

Expected: pass.

- [x] **Step 6: Commit**

```bash
git add pamssw/proposal_replay.py tests/unit/test_fixed_proposal_replay.py
git commit -m "Observe frozen proposal energetics without extra PES calls"
```

### Task 3: Add a run-local three-arm paired analyzer

**Files:**
- Create: `runs/20260730-uphill-propagation-u0-u1/run_ablation.py`
- Create: `runs/20260730-uphill-propagation-u0-u1/analyze_results.py`
- Create: `runs/20260730-uphill-propagation-u0-u1/protocol.md`
- Create: `tests/unit/test_uphill_propagation_ablation.py`

- [x] **Step 1: Write failing arm-construction tests**

The run-local module must define exactly:

```python
ARM_IDS = (
    "current_full",
    "curvature_matched_no_feedback",
    "fixed_calibrated",
)
```

Test that:

- `current_full` returns the source task unchanged;
- `fixed_calibrated` applies the supplied system median sigma and weight;
- `curvature_matched_no_feedback` uses a supplied `base_sigma`, `inner_curvature`, and `target_negative_curvature`;
- no arm changes prefix biases;
- invalid/nonfinite calibration values fail closed.

- [x] **Step 2: Run the arm tests and verify RED**

Expected: the run-local module or helpers do not exist.

- [x] **Step 3: Implement pure arm construction**

Use:

```python
weight = base_sigma * base_sigma * max(
    inner_curvature + target_negative_curvature,
    0.0,
)
```

Do not add feedback gamma, clipping, Bayesian selection, or a composite score.

- [x] **Step 4: Write failing calibration/analyzer tests**

Calibration must:

- require at least two finite first-step tasks per system;
- use component-wise medians of sigma and weight;
- store task IDs and source hashes;
- never use evaluation-arm outcomes.

The analyzer must retain a metric vector:

```text
certificate_satisfied
biased_proposal_relax_force_evaluations
wall_time_s
initial/final true energy
initial/final bias energy
direction_progress
orthogonal_displacement_norm
endpoint position hash
```

It must report paired differences and per-system Pareto relations without a weighted score.

- [x] **Step 5: Implement calibration and analysis**

Write JSON rows and a summary containing exact protocol/config/source hashes. Reject incomplete task-arm matrices, duplicate rows, nonfinite values, unequal source task IDs, or nonzero observer-only force evaluations.

- [x] **Step 6: Run run-local tests**

Run:

```bash
pytest -q tests/unit/test_uphill_propagation_ablation.py
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add \
  runs/20260730-uphill-propagation-u0-u1 \
  tests/unit/test_uphill_propagation_ablation.py
git commit -m "Add paired uphill propagation U0 U1 experiment"
```

### Task 4: Analytic smoke and production gate

**Files:**
- Modify: `runs/20260730-uphill-propagation-u0-u1/protocol.md`
- Create after execution: `runs/20260730-uphill-propagation-u0-u1/analytic_smoke.json`
- Create after execution: `runs/20260730-uphill-propagation-u0-u1/final_report.md`

- [x] **Step 1: Run an analytic quadratic/double-well smoke**

Use at least two task seeds, all three arms, and both first- and later-bias frozen prefixes. Require exact task-arm matrix closure, zero unattributed force evaluations, zero observer-only evaluations, and finite component energies.

- [x] **Step 2: Review the smoke before GPU execution**

Stop if arm construction changes prefix biases, current replay is not byte-equivalent to the source task, or component observations add calculator calls.

- [x] **Step 3: Run the bounded C60/PdO GPU screen**

Use independent calibration and evaluation action seeds. Start with 8 calibration first-step tasks and 16 evaluation prefixes per system, split across bias counts 1, 3, 5, or the largest capturable count. Use the same MACE model, dtype, constraints, optimizer, fmax, and iteration cap for every arm. Do not run a 200/500 macro-step production search in U0/U1.

- [x] **Step 4: Analyze without tuning**

An arm advances only if it is Pareto-nondominated within paired uncertainty on proposal force evaluations, certificate coverage, true-energy progress, and endpoint diversity for at least one system. Mixed C60/PdO evidence permits a system-conditional hypothesis; it does not permit parameter tuning on this corpus.

- [x] **Step 5: Write the claim-bounded report**

The report must distinguish:

- code-path verification;
- fixed-prefix local mechanism evidence;
- cost evidence;
- full-walk/global-search behavior not yet tested;
- the next gate: displacement/width ratio \(r=a/\ell\) or full-walk confirmation.

- [x] **Step 6: Run verification**

Run:

```bash
pytest -q \
  tests/unit/test_fixed_proposal_replay.py \
  tests/unit/test_uphill_propagation_ablation.py \
  tests/unit/test_walker_policy.py \
  tests/integration/test_epam_accounting.py
git status --short
```

Expected: targeted tests pass and only planned files are changed.

- [x] **Step 7: Commit**

```bash
git add runs/20260730-uphill-propagation-u0-u1
git commit -m "Record uphill propagation U0 U1 evidence"
```

