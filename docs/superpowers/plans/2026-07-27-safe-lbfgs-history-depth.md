# Safe L-BFGS History-Depth Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate whether the newest L-BFGS correction alone explains the
history-enabled contribution, or whether older retained secants add value on
the frozen C60/PdO proposal-relaxation tasks.

**Architecture:** Extend only the existing private history-capacity seam to
accept one retained pair. Run a fail-closed two-arm GPU matrix using the same
frozen tasks and safe-total kernel, then derive deterministic evidence from
the closed raw ledger. Do not expose a new public parameter or modify any
production default.

**Tech Stack:** Python 3.12, NumPy, pytest, ASE, SciPy, MACE/PyTorch CUDA,
existing `EvalCounter`, proposal replay, trace recorder, and safe-L-BFGS
telemetry.

---

## File map

- Modify `pamssw/relax.py`: permit private history capacity 1.
- Modify `tests/unit/test_relax.py`: prove validation and deterministic
  one-versus-ten kernel semantics.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py`:
  immutable 32-row CUDA runner and ledger publisher.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/.gitignore`: keep raw
  GPU output untracked.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/plan.md`: run-local
  execution contract.
- Create `tests/unit/test_safe_lbfgs_history_depth.py`: runner, schema,
  provenance, accounting, and publication tests.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/analyze_ablation.py`:
  strict raw-ledger validator and evidence generator.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/plot_ablation.py`:
  deterministic exact-evaluation trace plot.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/evidence.json`:
  reviewed derived evidence and raw hashes.
- Create `runs/20260727-safe-lbfgs-history-depth-ablation/conclusion.md`:
  certificate-first fixed-matrix conclusion.
- Create
  `runs/20260727-safe-lbfgs-history-depth-ablation/history_depth_curves.svg`:
  deterministic trace visualization.
- Create `tests/unit/test_safe_lbfgs_history_depth_analysis.py`: analysis and
  plot fail-closed tests.

### Task 1: Private history-one kernel seam

**Files:**

- Modify: `tests/unit/test_relax.py`
- Modify: `pamssw/relax.py:262-278`

- [ ] **Step 1: write the validation test first**

Add a test that calls the real `Relaxer` and proves history 1 is accepted only
for `safe-lbfgs-total`, while booleans, 2, negative values, and every other
optimizer fail before the evaluator is called:

```python
@pytest.mark.parametrize("history_limit", [True, False, -1, 2, 9, 11, 1.0, "1"])
def test_safe_lbfgs_history_one_rejects_noncontract_values_before_evaluation(
    history_limit,
):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions).copy())
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    with pytest.raises((TypeError, ValueError), match="history"):
        Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=history_limit,
        )
    assert calls == []


@pytest.mark.parametrize(
    "optimizer",
    ["scipy-lbfgsb", "ase-fire", "ase-fire2", "ase-lbfgs",
     "bias-separated-lbfgs"],
)
def test_history_one_is_private_to_safe_total_before_evaluation(optimizer):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions).copy())
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="safe-lbfgs-total"):
        Relaxer(evaluator, optimizer=optimizer).relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=1,
        )
    assert calls == []
```

- [ ] **Step 2: run RED validation**

Run:

```bash
pytest -q tests/unit/test_relax.py -k "history_one"
```

Expected: the safe-total history-one case fails because the resolver currently
accepts only 0 and 10.

- [ ] **Step 3: write deterministic kernel-path tests**

Use the existing deterministic anisotropic quadratic helper and spy on
`_lbfgs_inverse_product`. Record copies of each history pair and each evaluated
coordinate. Require:

```python
history1, calls1, _ = _run_safe_lbfgs_history_limit(1, maxiter=2)
history10, calls10, _ = _run_safe_lbfgs_history_limit(10, maxiter=2)
np.testing.assert_array_equal(calls1, calls10)
np.testing.assert_array_equal(history1.state.positions, history10.state.positions)
```

Then run three or more iterations and assert that every history-one inverse
product call receives `len(history) <= 1`, while history ten receives a
history length greater than one after the second accepted secant. Add a MIC
branch test proving the existing reset clears history in both arms.

- [ ] **Step 4: run RED kernel test**

Run:

```bash
pytest -q tests/unit/test_relax.py -k "history_one or history_depth"
```

Expected: FAIL only because `_safe_lbfgs_history_limit=1` is rejected.

- [ ] **Step 5: implement the minimal resolver change**

Change only the accepted private values and error text:

```python
if (
    isinstance(history_limit, bool)
    or not isinstance(history_limit, int)
    or history_limit not in {0, 1, _SAFE_LBFGS_MEMORY}
):
    raise ValueError("_safe_lbfgs_history_limit must be None, 0, 1, or 10")
```

Do not change `_lbfgs_inverse_product`, the safe loop, line search, scale
selection, MIC reset, or telemetry. The existing append-and-trim logic already
implements history 1.

- [ ] **Step 6: run GREEN and regression tests**

Run:

```bash
pytest -q tests/unit/test_relax.py
pytest -q tests/unit/test_proposal_energy_trace.py
git diff --check
```

Expected: all pass and no whitespace errors.

- [ ] **Step 7: commit Task 1**

```bash
git add pamssw/relax.py tests/unit/test_relax.py
git commit -m "experiment: admit one-pair safe L-BFGS history"
```

### Task 2: Thin 32-row runner

**Files:**

- Replace: `runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py`
- Replace: `tests/unit/test_safe_lbfgs_history_depth.py`
- Modify: `runs/20260727-safe-lbfgs-history-depth-ablation/.gitignore`
- Modify: `runs/20260727-safe-lbfgs-history-depth-ablation/plan.md`

- [ ] **Step 1: replace defensive tests with behavior tests**

Keep only tests for:

1. exact history1/history10 arm order and the 32-cell task matrix;
2. the relaxation call changing only `_safe_lbfgs_history_limit`;
3. task/source/model/input/pamssw/helper hash preflight before calculator
   creation;
4. one calculator per `(system, arm)` and no sharing across arms;
5. row accounting closure and finite result/trace values;
6. `output.partial` completion and refusal of pre-existing output paths;
7. preflight-only returning before calculator construction;
8. CLI requiring the expected Git commit.

Delete tests for arbitrary unknown nested keys, operating-system schemas,
`renameat2`, concurrent races, `ctypes`, and duplicated summary certificates.

- [ ] **Step 2: run RED against the defensive runner**

The replacement tests must initially fail because the existing runner stores
derived certificate/summary fields and uses the deleted defensive publication
and provenance abstractions.

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth.py
```

- [ ] **Step 3: implement the thin arm and row model**

Use one frozen arm type:

```python
@dataclass(frozen=True)
class Arm:
    arm_id: str
    history_limit: int
```

The row contains raw facts only:

```python
{
    "system": system,
    "seed": seed,
    "task_id": task_id,
    "task_sha256": task_sha256,
    "arm_id": arm.arm_id,
    "history_limit": arm.history_limit,
    "fmax_eV_per_A": task.fmax,
    "final_total_biased_energy_eV": result.energy,
    "final_active_max_force_eV_per_A": result.gradient_norm,
    "iterations": result.n_iter,
    "termination_reason": result.telemetry.termination_reason,
    "displacement_rms_A": result.displacement_rms,
    "displacement_max_A": result.displacement_max,
    "outcome_class": result.outcome_class.value,
    "force_evaluations": counts.total,
    "purpose_counts": counts.as_dict(),
    "telemetry": asdict(result.telemetry),
    "trace": trace_records,
    "final_positions": result.state.positions.tolist(),
    "final_positions_sha256": position_hash(result.state),
    "wall_time_s": wall_time_s,
}
```

Do not store `certificate_satisfied`, certificate counts, termination counts,
or duplicated resolved scale-policy strings.

- [ ] **Step 4: implement minimal preflight**

Before calculator construction verify:

- required expected commit equals `HEAD` and tracked worktree is clean;
- source summary and all 16 canonical task hashes;
- pamssw source-bundle hash and the single external fixed-replay helper hash;
- model and two input hashes;
- CUDA availability, CUDA version, and device name.

Record Python, NumPy, SciPy, ASE, PyTorch, and MACE versions from the selected
GPU runtime. Do not inventory imported symbols or record OS/platform fields.

- [ ] **Step 5: implement only scientific row checks**

Require:

```python
len(row["trace"]) \
    == row["force_evaluations"] \
    == row["telemetry"]["backend_evaluations"] \
    == row["purpose_counts"]["biased_proposal_relax"]
row["purpose_counts"]["unattributed"] == 0
```

Require finite energy, force, displacement, wall time, coordinates, and trace
values. Require task/arm membership in the frozen matrix. The runner does not
derive a certificate; Task 4 does.

- [ ] **Step 6: implement simple single-process completion**

Refuse existing `output` or `output.partial`. Write the two system files into
`output.partial`, verify 32 unique rows, then write `summary.json` last as the
completion marker and rename the directory to `output` with ordinary
`Path.rename`.

The run-local `.gitignore` is:

```text
output/
output.partial/
```

No concurrency or race guarantee is claimed.

- [ ] **Step 7: run GREEN and enforce the size reduction**

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth.py
pytest -q tests/unit/test_relax.py
pytest -q tests/unit/test_proposal_energy_trace.py
git diff --check
wc -l \
  runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py \
  tests/unit/test_safe_lbfgs_history_depth.py
```

The final files must be materially smaller than the rejected 1075-line runner
and 689-line test file. Size is a maintainability check, not an algorithm
parameter.

- [ ] **Step 8: commit the simplification**

```bash
git add \
  runs/20260727-safe-lbfgs-history-depth-ablation/.gitignore \
  runs/20260727-safe-lbfgs-history-depth-ablation/plan.md \
  runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py \
  tests/unit/test_safe_lbfgs_history_depth.py
git commit -m "refactor: keep history-depth runner experiment focused"
```

Pass the resulting commit through required `--expected-git-commit`; do not
embed a self-referential commit constant.

### Task 3: One approved GPU execution

**Files:**

- Generate, ignored:
  `runs/20260727-safe-lbfgs-history-depth-ablation/output/`

- [ ] **Step 1: run preflight without a calculator**

Run:

```bash
pytest -q \
  tests/unit/test_relax.py \
  tests/unit/test_safe_lbfgs_history_depth.py \
  tests/unit/test_proposal_energy_trace.py
git status --short
ssw_history_depth_commit="$(git rev-parse HEAD)"
python runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py \
  --preflight-only \
  --expected-git-commit "$ssw_history_depth_commit"
```

Expected: tests pass, tracked worktree is clean, all hashes match, and CUDA
reports the pinned device/model before any calculator is constructed.

- [ ] **Step 2: execute exactly once**

Run:

```bash
ssw_history_depth_commit="$(git rev-parse HEAD)"
python runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py \
  --expected-git-commit "$ssw_history_depth_commit"
```

If sandbox CUDA visibility fails before calculator construction, rerun the
same command with approved elevated GPU access. Do not retry after any physical
evaluation has occurred.

- [ ] **Step 3: validate raw output**

Run:

```bash
python -c "
import json
from pathlib import Path
p=Path('runs/20260727-safe-lbfgs-history-depth-ablation/output/summary.json')
d=json.loads(p.read_text())
assert d['row_count'] == 32
assert d['task_count'] == 16
assert d['systems'] == ['c60', 'pdo']
print(d['execution_commit'])
print(d['runtime_versions'])
"
git check-ignore -v \
  runs/20260727-safe-lbfgs-history-depth-ablation/output/summary.json
```

Expected: complete 32-row atomic ledger and ignored raw output.

### Task 4: Deterministic analysis and trace figure

**Files:**

- Create:
  `runs/20260727-safe-lbfgs-history-depth-ablation/analyze_ablation.py`
- Create:
  `runs/20260727-safe-lbfgs-history-depth-ablation/plot_ablation.py`
- Create:
  `runs/20260727-safe-lbfgs-history-depth-ablation/evidence.json`
- Create:
  `runs/20260727-safe-lbfgs-history-depth-ablation/conclusion.md`
- Create:
  `runs/20260727-safe-lbfgs-history-depth-ablation/history_depth_curves.svg`
- Create: `tests/unit/test_safe_lbfgs_history_depth_analysis.py`

- [ ] **Step 1: write failing strict-analysis tests**

Require the 32-row matrix, completion marker, raw-file hashes,
source/task/model/input/execution anchors, finite result/trace values,
endpoint hash, and the accounting equality. Test only scientifically meaningful
failures:

```text
missing row, duplicate row, nonfinite result, task hash mismatch,
trace/accounting mismatch, endpoint hash mismatch, missing completion marker
```

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth_analysis.py
```

Expected: FAIL because the analyzer does not exist.

- [ ] **Step 2: implement strict ledger validation**

The analyzer derives `certificate_satisfied` from
`final_active_max_force_eV_per_A <= fmax_eV_per_A`, derives termination counts
from rows, and computes paired history1-minus-history10 fields:

```python
{
    "force_evaluations_delta": h1_calls - h10_calls,
    "wall_time_s_delta": h1_wall - h10_wall,
    "iterations_delta": h1_iter - h10_iter,
    "final_force_delta": h1_force - h10_force,
    "final_total_biased_energy_delta": h1_energy - h10_energy,
    "exact_final_position_hash_equal": h1_hash == h10_hash,
    "mic_endpoint_max_displacement_A": ...,
    "mic_endpoint_rms_displacement_A": ...,
}
```

Use the existing `mic_displacement` implementation and source-task cell/PBC
state. Do not introduce an RMSD or basin-equivalence threshold.

- [ ] **Step 3: implement certificate-first conclusion generation**

The conclusion must follow this order:

1. complete-matrix certificate counts;
2. termination reasons;
3. all-row evaluator calls and wall time;
4. paired diagnostics;
5. one of the three spec interpretations: history1 sufficient on this matrix,
   older corrections positive candidate, or mixed;
6. full claim ceiling.

Do not compute a weighted score or discard incomplete finite rows.

- [ ] **Step 4: write plot tests before the plotter**

Require a deterministic 2-by-2 SVG:

- rows: C60 and PdO;
- columns: total biased energy and active maximum total force;
- task identity: color;
- history 1: dashed;
- history 10: solid;
- markers distinguish all exact evaluations, callback-observed evaluations,
  callback-nonobserved evaluations, and explicit finalizations.

The SVG must not contain a `rejected-line-search` label. Test that two output
directories produce byte-identical SVG files and that render failures close
the Matplotlib figure.

- [ ] **Step 5: implement the plotter with an isolated analyzer import**

Load the analyzer by explicit file path and a run-specific module name:

```python
spec = importlib.util.spec_from_file_location(
    "safe_lbfgs_history_depth_analyzer",
    RUN_ROOT / "analyze_ablation.py",
)
```

Do not import a global module named only `analyze_ablation`, because stacked
test runs contain multiple files with that basename.

- [ ] **Step 6: generate and verify artifacts**

Run:

```bash
python runs/20260727-safe-lbfgs-history-depth-ablation/analyze_ablation.py \
  --ledger-dir runs/20260727-safe-lbfgs-history-depth-ablation/output \
  --output-dir runs/20260727-safe-lbfgs-history-depth-ablation
python runs/20260727-safe-lbfgs-history-depth-ablation/plot_ablation.py \
  --ledger-dir runs/20260727-safe-lbfgs-history-depth-ablation/output \
  --output runs/20260727-safe-lbfgs-history-depth-ablation/history_depth_curves.svg
pytest -q tests/unit/test_safe_lbfgs_history_depth_analysis.py
```

Regenerate into two temporary directories and require byte equality for
`evidence.json`, `conclusion.md`, and the SVG.

- [ ] **Step 7: commit Task 4**

```bash
git add \
  runs/20260727-safe-lbfgs-history-depth-ablation/analyze_ablation.py \
  runs/20260727-safe-lbfgs-history-depth-ablation/plot_ablation.py \
  runs/20260727-safe-lbfgs-history-depth-ablation/evidence.json \
  runs/20260727-safe-lbfgs-history-depth-ablation/conclusion.md \
  runs/20260727-safe-lbfgs-history-depth-ablation/history_depth_curves.svg \
  tests/unit/test_safe_lbfgs_history_depth_analysis.py
git commit -m "experiment: analyze safe L-BFGS history depth"
```

### Task 5: Final verification and stacked PR

**Files:**

- Verify all files changed from base `31fd257`.

- [ ] **Step 1: run fresh full verification**

Run:

```bash
pytest -q
git diff 31fd257...HEAD --check
git status --short
```

Expected: all tests pass, no diff errors, and no tracked changes.

- [ ] **Step 2: request independent final review**

The reviewer must inspect:

- history 1 changes only retained correction capacity;
- no public parameter or default changes;
- runner has exactly 32 cells and calculators are arm-isolated;
- provenance and accounting close before publication;
- evidence and figure regenerate byte-for-byte;
- conclusion obeys the fixed-matrix claim ceiling.

Any blocking issue must be fixed and re-reviewed before push.

- [ ] **Step 3: push and create the stacked PR**

Push:

```bash
git push \
  https://github.com/Phorbol/pam-ssw.git \
  experiment/safe-lbfgs-history-depth-ablation
```

Create a PR with:

```text
base: experiment/safe-lbfgs-scale-decomposition
head: experiment/safe-lbfgs-history-depth-ablation
```

The PR body must report exact certificate counts, all-row force evaluations,
wall time, termination reasons, raw-ledger availability boundary, verification
results, and the claim ceiling. Do not recommend a default change.
