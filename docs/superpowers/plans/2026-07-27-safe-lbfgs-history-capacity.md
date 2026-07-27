# Safe L-BFGS History-Capacity Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the existing history-enabled L-BFGS mechanism, comprising adaptive inverse scaling and two-loop secant corrections, reduces force-evaluation cost on the frozen C60/PdO biased-proposal matrix.

**Architecture:** Add one private, fail-closed injection to the existing safe total-gradient relaxation kernel; do not add a new configured optimizer. A run-local CUDA driver replays the same immutable tasks with history capacity 10 and 0 through the same evaluator, observer, and accounting path, then a deterministic analyzer reports paired evidence without a composite score.

**Tech Stack:** Python, NumPy, ASE, MACE/PyTorch CUDA, pytest, Matplotlib SVG, existing `EvalCounter`, `ProposalRelaxationTask`, and zero-call trace recorder.

---

### Task 1: Private history-capacity injection

**Files:**
- Modify: `pamssw/relax.py`
- Modify: `tests/unit/test_relax.py`

- [ ] **Step 1: Write failing validation tests**

Add tests which use a counting evaluator and assert that invalid experimental
values fail before the first PES call:

```python
@pytest.mark.parametrize("invalid", [True, False, -1, 1, 9, 11, 1.0, "0"])
def test_safe_history_limit_rejects_values_outside_zero_or_existing_capacity(invalid):
    calls = 0
    # evaluator increments calls
    with pytest.raises((TypeError, ValueError), match="history"):
        Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
            state,
            fmax=1e-8,
            maxiter=2,
            _safe_lbfgs_history_limit=invalid,
        )
    assert calls == 0
```

Also parameterize `ase-fire`, `ase-lbfgs`, `scipy-lbfgsb`, and
`bias-separated-lbfgs` to reject an explicit limit before evaluation.

- [ ] **Step 2: Run the validation tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_relax.py -k "history_limit"
```

Expected: failures because `Relaxer.relax` does not accept the keyword.

- [ ] **Step 3: Write failing mechanism-isolation tests**

Add deterministic analytic tests:

1. default `None` and explicit `10` produce identical calls, trajectory,
   telemetry, and endpoint;
2. `0` and `10` are identical for `maxiter=1`;
3. spy on `_lbfgs_inverse_product` and assert every observed history length is
   zero for limit 0;
4. an anisotropic quadratic produces a difference only after the first
   accepted secant;
5. accepted/rejected secant diagnostics and line-search accounting are still
   populated for limit 0.

Use the real `Relaxer`; do not mock the evaluator or line search.

- [ ] **Step 4: Run the mechanism tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_relax.py -k "safe_history"
```

Expected: failures at the missing injection point.

- [ ] **Step 5: Implement the minimal private injection**

Add a resolver which accepts only `None`, `0`, or the existing constant 10 and
rejects booleans explicitly:

```python
def _resolve_safe_lbfgs_history_limit(optimizer: str, value: int | None) -> int:
    if value is None:
        return _SAFE_LBFGS_MEMORY
    if optimizer != "safe-lbfgs-total":
        raise ValueError("safe L-BFGS history limit requires safe-lbfgs-total")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("safe L-BFGS history limit must be integer 0 or 10")
    if value not in {0, _SAFE_LBFGS_MEMORY}:
        raise ValueError("safe L-BFGS history limit must be 0 or 10")
    return value
```

Resolve the value at the start of `Relaxer.relax`, pass it to
`_relax_with_safe_lbfgs`, and replace only the fixed memory truncation:

```python
history.append((s.copy(), y.copy(), 1.0 / curvature))
while len(history) > history_limit:
    history.pop(0)
```

Do not change secant acceptance, telemetry, search directions, line search, or
any configured optimizer surface.

- [ ] **Step 6: Verify Task 1**

Run:

```bash
pytest -q tests/unit/test_relax.py
pytest -q tests/unit/test_proposal_energy_trace.py
git diff --check
```

Expected: all pass and no whitespace errors.

- [ ] **Step 7: Commit Task 1**

```bash
git add pamssw/relax.py tests/unit/test_relax.py
git commit -m "experiment: inject safe L-BFGS history capacity"
```

### Task 2: Fail-closed fixed-task CUDA runner

**Files:**
- Create: `runs/20260727-safe-history-capacity-ablation/plan.md`
- Create: `runs/20260727-safe-history-capacity-ablation/run_gpu_ablation.py`
- Create: `runs/20260727-safe-history-capacity-ablation/output/.gitignore`
- Create: `tests/unit/test_safe_history_capacity_ablation.py`

- [ ] **Step 1: Write failing runner-contract tests**

Load the run-local module with `importlib.util` and test:

- arms are exactly history 10 and 0, both with
  `optimizer_kernel="safe-lbfgs-total"`;
- all 16 source tasks are accepted only from the pinned source-summary SHA;
- canonical task hashes change if any task payload value changes;
- kernel metadata contains the existing inverse scale, atomic-step cap, Armijo,
  backtrack, line-trial, minimum-alpha, and curvature-gate constants;
- malformed/missing arm, task, model, input, or CUDA provenance fails closed;
- the atomic publisher leaves no partial final directory on injected failure.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_safe_history_capacity_ablation.py
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement traced replay for one arm**

Reuse the prior trace recorder by file path. Construct `EvalCounter`,
`RecordingProposalPotential`, and `Relaxer(...,
optimizer="safe-lbfgs-total", component_evaluator=proposal.evaluate_parts)`.
Call:

```python
result = relaxer.relax(
    task.initial_state,
    fmax=task.fmax,
    maxiter=400,
    coordinate_trust_radius=task.coordinate_trust_radius,
    trajectory_callback=record_callback_state,
    _safe_lbfgs_history_limit=history_limit,
)
```

Hard-fail unless:

```python
len(trace_records) == counter.snapshot().total
len(trace_records) == result.telemetry.evaluator_calls
```

- [ ] **Step 4: Implement the complete paired ledger**

For each frozen task, execute both arms with independent calculators and record:

```json
{
  "optimizer_kernel": "safe-lbfgs-total",
  "arm_id": "safe-total-gradient-history0",
  "safe_lbfgs_history_limit": 0,
  "task_sha256": "...",
  "force_evaluations": 0,
  "certificate_satisfied": false,
  "termination_reason": "...",
  "telemetry": {},
  "trace_records": []
}
```

Hash the source summary, task payloads, model, input structures, and kernel
descriptor before calculator creation. Require CUDA. Write all files to a
same-parent staging directory and publish with one directory rename only after
the complete 32-row ledger validates.

- [ ] **Step 5: Verify Task 2 without GPU**

Run:

```bash
pytest -q tests/unit/test_safe_history_capacity_ablation.py
pytest -q tests/unit/test_proposal_energy_trace.py tests/unit/test_relax.py
git diff --check
```

Expected: all pass. No GPU execution belongs in this step.

- [ ] **Step 6: Commit the reviewed runner before physical calls**

```bash
git add runs/20260727-safe-history-capacity-ablation tests/unit/test_safe_history_capacity_ablation.py
git commit -m "experiment: define fixed safe-history GPU ablation"
```

- [ ] **Step 7: Execute the real CUDA matrix once**

Run in `mace_les` with the pinned source summary and model:

```bash
python runs/20260727-safe-history-capacity-ablation/run_gpu_ablation.py \
  --source-summary /tmp/SSW-worktrees/fixed-proposal-replay/runs/20260727-023234-fixed-proposal-replay-gpu/output/summary.json \
  --output-dir runs/20260727-safe-history-capacity-ablation/output
```

Expected: exit 0, 32 rows, exact per-row ledger closure, finite values, and an
atomically published output directory. Do not retry or retune a scientifically
valid negative result.

### Task 3: Deterministic paired analysis

**Files:**
- Create: `runs/20260727-safe-history-capacity-ablation/analyze_ablation.py`
- Create: `runs/20260727-safe-history-capacity-ablation/evidence.json`
- Create: `runs/20260727-safe-history-capacity-ablation/conclusion.md`
- Create: `runs/20260727-safe-history-capacity-ablation/history_capacity_curves.svg`
- Create: `tests/unit/test_safe_history_capacity_analysis.py`

- [ ] **Step 1: Write failing analysis-contract tests**

Tests must reject anything other than the exact 16-task × 2-arm matrix, malformed
numeric/boolean/provenance fields, nonfinite curves, broken ledgers, duplicate
tasks, and incorrect task/kernel hashes.

Test that two independent analyses of the same input produce byte-identical
JSON, Markdown, and SVG. The SVG semantic legend must distinguish all exact
evaluations, callback-observed states, callback-nonaccepted evaluations, and
explicit finalization rechecks.

- [ ] **Step 2: Run analysis tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_safe_history_capacity_analysis.py
```

Expected: import failure because the analyzer does not exist.

- [ ] **Step 3: Implement fail-closed aggregation**

Compute task-level and system-level evidence for:

- certificate and termination reason before cost;
- force evaluations and wall time;
- line-search evaluations and nonaccepted observer labels;
- accepted/rejected secants and MIC resets;
- initial-to-final total energy and accepted-step monotonicity;
- history-0 minus history-10 endpoint energy and MIC displacement.

Do not compute a weighted score, p-value, or automatic promotion decision.

- [ ] **Step 4: Implement deterministic artifacts**

Use stable task/arm sorting, canonical JSON, fixed Matplotlib `Date` metadata and
`svg.hashsalt`, and explicit trace-semantics legends. The conclusion must state
that endpoint differences prevent same-basin speed claims and that one GPU
matrix is not statistical generalization.

- [ ] **Step 5: Generate evidence from the real ledger**

Run:

```bash
python runs/20260727-safe-history-capacity-ablation/analyze_ablation.py \
  --summary runs/20260727-safe-history-capacity-ablation/output/summary.json \
  --output-dir runs/20260727-safe-history-capacity-ablation
```

Expected: deterministic evidence, conclusion, and one SVG.

- [ ] **Step 6: Verify Task 3 and the entire branch**

Run:

```bash
pytest -q tests/unit/test_safe_history_capacity_analysis.py
pytest -q
git diff a26fd50...HEAD --check
git status --short
```

Expected: all tests pass, no whitespace errors, and only intended generated
artifacts are tracked.

- [ ] **Step 7: Commit Task 3**

```bash
git add runs/20260727-safe-history-capacity-ablation tests/unit/test_safe_history_capacity_analysis.py
git commit -m "experiment: analyze safe L-BFGS history capacity"
```

### Task 4: Final independent review and stacked delivery

**Files:**
- Review all changes from `a26fd50...HEAD`

- [ ] **Step 1: Run independent specification review**

Verify every design invariant, fixed mechanism, manifest field, analysis metric,
and claim ceiling against the implementation and raw ledger. Any finding must
be fixed and re-reviewed.

- [ ] **Step 2: Run independent code-quality review**

Check default-path equivalence, validation-before-evaluation, accounting,
atomic publication, deterministic artifacts, unnecessary parameter surfaces,
and run-local code maintainability. Any finding must be fixed and re-reviewed.

- [ ] **Step 3: Perform final verification**

Run fresh:

```bash
pytest -q
git diff a26fd50...HEAD --check
git status --short
```

- [ ] **Step 4: Push and create a stacked PR**

Push `experiment/safe-history-capacity-ablation` and create a PR with base
`experiment/proposal-energy-traces`. Report exact tests, CUDA provenance,
evidence boundaries, and whether the combined history-enabled mechanism was
positive, neutral, or negative on each fixed system without promoting a new
default.
