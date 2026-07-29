# Krylov Curvature--Overlap Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the complete zero-extra-HVP Ritz curvature--anchor-overlap frontier and attach it to the previously validated C60 terminal outcomes through exact proposal replay.

**Architecture:** Extend the existing Krylov result with immutable per-Ritz-point diagnostics computed from the already stored `Q`, total `HQ`, and true `HQ`.  Keep lowest-Ritz selection unchanged, serialize the full spectrum through the existing direction diagnostics seam, and run a 12-case proposal-only replay that must match the prior escape hashes and selected traces before referencing prior terminal outcomes.

**Tech Stack:** Python, NumPy, pytest, ASE, MACE/CUDA, existing `BudgetedCalculator`, `SurfaceWalker`, and `solve_krylov_block`.

---

### Task 1: Return every Ritz point without another HVP

**Files:**
- Modify: `tests/unit/test_block_krylov.py`
- Modify: `pamssw/krylov.py`

- [ ] **Step 1: Write the failing analytic spectrum test**

Add a test with a diagonal total Hessian, a distinct diagonal true Hessian,
and a normalized reference vector:

```python
def test_full_ritz_spectrum_reuses_hvps_and_preserves_selected_pair():
    total = np.diag([1.0, 3.0])
    true = np.diag([4.0, 9.0])
    reference = np.array([1.0, 1.0]) / np.sqrt(2.0)
    calls = 0

    def hvp(vector):
        nonlocal calls
        calls += 1
        return total @ vector, true @ vector

    result = solve_krylov_block(
        IntentBlock(np.eye(2)),
        hvp,
        depth=1,
        reference_direction=reference,
    )

    assert calls == result.hvp_count == 2
    assert len(result.ritz_points) == 2
    assert [point.curvature for point in result.ritz_points] == pytest.approx(
        [1.0, 3.0]
    )
    assert [point.true_curvature for point in result.ritz_points] == (
        pytest.approx([4.0, 9.0])
    )
    assert [point.reference_abs_overlap for point in result.ritz_points] == (
        pytest.approx([1.0 / np.sqrt(2.0)] * 2)
    )
    np.testing.assert_array_equal(
        result.direction,
        result.ritz_points[0].direction,
    )
```

Also assert that every returned direction is read-only and that omitting the
reference produces `reference_abs_overlap is None`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_block_krylov.py::test_full_ritz_spectrum_reuses_hvps_and_preserves_selected_pair
```

Expected: failure because `reference_direction` and `ritz_points` do not yet
exist.

- [ ] **Step 3: Add the immutable point record**

Add:

```python
@dataclass(frozen=True, eq=False)
class KrylovRitzPoint:
    direction: np.ndarray
    curvature: float
    true_curvature: float
    residual_norm: float
    initial_span_overlap: float
    reference_abs_overlap: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            _readonly_float_copy(self.direction),
        )
```

Add:

```python
ritz_points: tuple[KrylovRitzPoint, ...]
```

to `KrylovResult`.

- [ ] **Step 4: Construct all points from stored products**

Extend the solver signature without changing existing positional arguments:

```python
def solve_krylov_block(
    intent: IntentBlock,
    hvp: Hvp,
    depth: int,
    reference_direction: np.ndarray | None = None,
) -> KrylovResult:
```

Normalize and validate the optional reference once.  After the existing
eigendecomposition, loop over the already sorted eigenvector columns:

```python
ritz_points = []
for index in range(eigenvectors.shape[1]):
    coefficients = eigenvectors[:, index]
    direction = q @ coefficients
    norm = float(np.linalg.norm(direction))
    coefficients = coefficients / norm
    direction = direction / norm

    overlaps = initial_q.T @ direction
    if overlaps.size and overlaps[np.argmax(np.abs(overlaps))] < 0.0:
        coefficients = -coefficients
        direction = -direction

    total_product = total_hq @ coefficients
    true_product = true_hq @ coefficients
    curvature = float(np.dot(direction, total_product))
    ritz_points.append(
        KrylovRitzPoint(
            direction=direction,
            curvature=curvature,
            true_curvature=float(np.dot(direction, true_product)),
            residual_norm=float(
                np.linalg.norm(total_product - curvature * direction)
            ),
            initial_span_overlap=float(
                np.linalg.norm(initial_q.T @ direction)
            ),
            reference_abs_overlap=(
                None
                if reference is None
                else abs(float(np.dot(reference, direction)))
            ),
        )
    )
```

Populate every existing selected-pair scalar from `ritz_points[0]`.  Do not
call `hvp` inside or after this loop.

- [ ] **Step 5: Run the complete Krylov tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_block_krylov.py
```

Require zero failures and unchanged call-count assertions.

- [ ] **Step 6: Commit**

```bash
git add pamssw/krylov.py tests/unit/test_block_krylov.py
git commit -m "feat: expose zero-cost Krylov Ritz spectrum"
```

### Task 2: Serialize the spectrum without changing selection

**Files:**
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `pamssw/walker.py`

- [ ] **Step 1: Write the failing walker diagnostic test**

Extend the existing block-Krylov oracle test to require:

```python
spectrum = choice.diagnostics["krylov_ritz_spectrum"]
assert len(spectrum) == choice.diagnostics["krylov_dimensions"][0]
assert sum(point["executed"] for point in spectrum) == 1
assert spectrum[0]["executed"] is True
assert [point["curvature"] for point in spectrum] == sorted(
    point["curvature"] for point in spectrum
)
assert all(0.0 <= point["anchor_abs_overlap"] <= 1.0 for point in spectrum)
assert all(point["participation_ratio"] >= 1.0 for point in spectrum)
assert oracle.calculator.force_evaluations == expected_force_evaluations
```

Retain an exact copy of the previously expected selected direction,
curvatures, residual, and HVP count.

- [ ] **Step 2: Run the test and verify RED**

Run the named block-Krylov test in `tests/unit/test_walker_policy.py`.
Expected: failure because `krylov_ritz_spectrum` is absent.

- [ ] **Step 3: Pass the common anchor into the solver**

Change the internal seam to:

```python
return self._choose_block_krylov_direction(
    state,
    proposal,
    krylov_intents,
    anchor_direction,
)
```

and call:

```python
solve_krylov_block(
    intent,
    directional_hvps,
    depth=self.block_krylov_depth,
    reference_direction=anchor_direction,
)
```

This reference is diagnostic only.

- [ ] **Step 4: Build a flat JSON-safe spectrum**

For every block and point, calculate participation ratio with the existing
movable-atom definition:

```python
atom_squared = np.sum(
    np.square(
        point.direction.reshape(state.n_atoms, 3)[state.movable_mask]
    ),
    axis=1,
)
participation_ratio = 1.0 / float(np.dot(atom_squared, atom_squared))
```

Append:

```python
{
    "block_index": int(block_index),
    "ritz_index": int(ritz_index),
    "executed": bool(
        block_index == selected_block and ritz_index == 0
    ),
    "curvature": float(point.curvature),
    "true_curvature": float(point.true_curvature),
    "residual_norm": float(point.residual_norm),
    "initial_span_overlap": float(point.initial_span_overlap),
    "anchor_abs_overlap": float(point.reference_abs_overlap),
    "participation_ratio": float(participation_ratio),
}
```

under `krylov_ritz_spectrum`.  Preserve every existing scalar diagnostic.

- [ ] **Step 5: Run focused and surrounding tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_block_krylov.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_anchor_consistent_direction_ablation.py
```

Require zero failures and unchanged direction-budget tests.

- [ ] **Step 6: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: log Krylov curvature-overlap frontier"
```

### Task 3: Build the 12-case proposal replay

**Files:**
- Create: `tests/unit/test_krylov_curvature_overlap_frontier.py`
- Create:
  `runs/20260729-krylov-curvature-overlap-frontier/run_audit.py`

- [ ] **Step 1: Write failing cohort and Pareto tests**

Require:

```python
assert len(case_matrix()) == 12
assert set(ARMS) == {"detached_ritz", "anchor_lanczos"}
assert EXPECTED_HVP_PER_SELECTION == {
    "detached_ritz": 12,
    "anchor_lanczos": 12,
}
```

For points with `(curvature, overlap)` equal to:

```python
[(1.0, 0.1), (2.0, 0.4), (3.0, 0.3), (4.0, 0.8)]
```

require frontier indices `[0, 1, 3]`, maximum-overlap index `3`, and curvature
difference `3.0`.

- [ ] **Step 2: Write failing evidence-contract tests**

Construct synthetic replay rows and require `build_evidence` to reject:

- a missing case;
- a nonmatching escape hash;
- a nonmatching selected trace;
- any nonzero landing-quench, bootstrap, validation, or unattributed count;
- a spectrum with no executed point, multiple executed points, unsorted
  curvature, or nonfinite values.

Require the evidence to keep replay FE separate from the referenced prior
terminal FE.

- [ ] **Step 3: Run runner tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_krylov_curvature_overlap_frontier.py
```

Expected: failure because the runner does not exist.

- [ ] **Step 4: Implement exact replay loading**

Load:

```text
runs/20260729-anchor-consistent-direction-ablation/output/evidence.json
```

Verify its SHA-256 and execution commit at runner startup.  Index only the 12
refined-arm prior cases by `(state_id, seed, arm)`.

For every replay, reuse the prior module's locked runtime and arm
configuration.  Execute through `_proposal_pool`, evaluate and write the
escape state, but do not call `relax_true_minimum`.

- [ ] **Step 5: Enforce the replay proof**

For each case:

```python
new_selected_trace = [
    {
        key: value
        for key, value in row.items()
        if key != "krylov_ritz_spectrum"
    }
    for row in direction_rows
]
```

Require exact equality with the prior direction trace and require the replay
escape SHA-256 to equal `prior_case["escape_sha256"]`.

Require the new purpose ledger to equal the prior counts for:

```python
{
    "direction_oracle",
    "biased_proposal_relax",
    "escape_true_pes_check",
}
```

and require all other purposes to be zero.

- [ ] **Step 6: Implement parameter-free frontier summaries**

Sort each spectrum by curvature.  A point is on the frontier when:

```python
point["anchor_abs_overlap"] > best_overlap_so_far
```

Record exact frontier indices, executed and maximum-overlap point metrics,
and their total/true curvature differences.  Aggregate medians by arm and
starter without introducing success thresholds.

- [ ] **Step 7: Write atomic raw and evidence files**

Write per-case summaries plus:

```text
raw.json
evidence.json
```

Record source/model/state/prior-evidence hashes, exact execution commit,
purpose counts, measured proposal wall time, spectrum points, frontier
summaries, and referenced prior terminal fields.

- [ ] **Step 8: Run runner tests and surrounding tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_krylov_curvature_overlap_frontier.py \
  tests/unit/test_anchor_consistent_direction_ablation.py
```

Require zero failures.

- [ ] **Step 9: Commit**

```bash
git add \
  runs/20260729-krylov-curvature-overlap-frontier/run_audit.py \
  tests/unit/test_krylov_curvature_overlap_frontier.py
git commit -m "exp: add Krylov frontier replay audit"
```

### Task 4: Execute, validate, and report

**Files:**
- Create:
  `runs/20260729-krylov-curvature-overlap-frontier/output/`
- Create:
  `runs/20260729-krylov-curvature-overlap-frontier/conclusion.md`

- [ ] **Step 1: Freeze the execution commit**

Require a clean tracked worktree and pass the exact `git rev-parse HEAD`
through `--expected-git-commit`.

- [ ] **Step 2: Run all 12 CUDA proposal replays**

Use `/root/miniforge3/envs/mace_les/bin/python`,
`CUDA_VISIBLE_DEVICES=0`, unbuffered output, and a `/tmp` Matplotlib cache.
Do not run a terminal quench and do not start a second cohort while the first
is active.

- [ ] **Step 3: Independently rebuild the evidence**

Reload `raw.json`, call `build_evidence`, verify every written escape hash,
prior-evidence hash, exact case set, direction trace, purpose ledger, and:

```text
completed_replays == 12
escape_hash_matches == 12
selected_trace_matches == 12
landing_true_quench == 0
bootstrap_true_quench == 0
unattributed == 0
```

- [ ] **Step 4: Write the conclusion**

Separate:

1. unchanged implementation and budget facts;
2. measured spectrum/frontier geometry;
3. relation to the inherited terminal outcomes;
4. counterfactual limitations;
5. whether a constrained selection experiment is justified.

Report full replay FE and wall time.  Do not claim that an unexecuted Ritz
point would have succeeded.

- [ ] **Step 5: Run final verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_block_krylov.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_anchor_consistent_direction_ablation.py \
  tests/unit/test_krylov_curvature_overlap_frontier.py
python -m py_compile \
  pamssw/krylov.py \
  pamssw/walker.py \
  runs/20260729-krylov-curvature-overlap-frontier/run_audit.py
git diff --check
```

Require zero failures.

- [ ] **Step 6: Commit, push, and update PR 10**

Commit the evidence and conclusion, push
`feature/posterior-terminal-outcome-validation`, and post the exact diagnostic
result and claim boundary to PR 10.
