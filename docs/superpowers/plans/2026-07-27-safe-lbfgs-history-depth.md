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

### Task 2: Fail-closed 32-row runner

**Files:**

- Create: `runs/20260727-safe-lbfgs-history-depth-ablation/.gitignore`
- Create: `runs/20260727-safe-lbfgs-history-depth-ablation/plan.md`
- Create: `runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py`
- Create: `tests/unit/test_safe_lbfgs_history_depth.py`

- [ ] **Step 1: write failing runner-contract tests**

The first tests must require the exact arm contract:

```python
assert tuple(
    (arm.arm_id, arm.kernel, arm.history_limit, arm.scale_policy)
    for arm in runner.ARMS
) == (
    (
        "adaptive-scale-history1",
        "safe-lbfgs-total",
        1,
        "latest-history-pair-gamma-plus-one-two-loop-correction",
    ),
    (
        "adaptive-scale-history10",
        "safe-lbfgs-total",
        10,
        "latest-history-pair-gamma-plus-up-to-ten-two-loop-corrections",
    ),
)
```

Also require:

```python
assert runner.SYSTEMS == ("c60", "pdo")
assert runner.SEEDS == tuple(range(42, 50))
assert runner.MAXITER == 400
assert runner.EXPECTED_SOURCE_SUMMARY_SHA256 == (
    "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
)
```

The runner must not exist before this test is written.

- [ ] **Step 2: run RED runner test**

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth.py
```

Expected: FAIL because the run directory and runner do not exist.

- [ ] **Step 3: implement immutable runner constants and arm dispatch**

Create a frozen `Arm` dataclass:

```python
@dataclass(frozen=True)
class Arm:
    arm_id: str
    kernel: str
    history_limit: int
    scale_policy: str
```

The relaxation call must be exactly:

```python
result = Relaxer(
    counter.evaluate_flat,
    optimizer="safe-lbfgs-total",
    component_evaluator=proposal.evaluate_parts,
).relax(
    task.initial_state,
    fmax=task.fmax,
    maxiter=400,
    coordinate_trust_radius=task.coordinate_trust_radius,
    trajectory_callback=trace_recorder,
    trajectory_stride=1,
    _safe_lbfgs_history_limit=arm.history_limit,
)
```

Do not pass the scale-only private flag. Assert it remains `False`.

- [ ] **Step 4: add provenance tests before provenance implementation**

Require exact schemas:

```python
assert set(summary["runtime_versions"]) == {
    "python", "python_implementation", "numpy", "scipy",
    "ase", "torch", "mace",
}
assert set(summary["platform_provenance"]) == {
    "sys_platform", "system", "release", "machine",
}
assert set(summary["git_provenance"]) == {
    "expected_git_commit", "actual_git_commit", "repo_root", "worktree_clean",
}
```

Require declared/measured hashes for the model, both input structures, source
summary, source bundle, kernel descriptor, and every imported helper. Tampered
hashes, a dirty worktree, a non-CUDA device, or an import outside the pinned
source/helper roots must fail before calculator construction.

- [ ] **Step 5: implement fail-closed provenance**

Use `importlib.metadata.version()` for package versions, with the package names
explicitly mapped:

```python
{
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}
```

Use `platform.system()`, `platform.release()`, `platform.machine()`,
`sys.platform`, `sys.version`, and `platform.python_implementation()`. Do not
silently substitute `"unknown"`; a missing required version is fatal.

- [ ] **Step 6: write accounting and atomic-publication tests**

Using the deterministic quadratic calculator, require for each row:

```python
assert len(row["trace_records"]) == row["force_evaluations"]
assert row["force_evaluations"] == row["telemetry"]["backend_evaluations"]
assert row["force_evaluations"] == row["purpose_counts"]["biased_proposal_relax"]
assert row["unattributed_calls"] == 0
```

Test exactly 32 unique `(system, task_id, arm_id)` cells. Test that an existing
output directory, partial row, duplicate row, nonfinite value, open
`EvalCounter` scope, or calculator shared across arms prevents publication.
Test same-parent temporary-directory rename and non-overwrite behavior.
Test that `--preflight-only --expected-git-commit <sha>` completes all
provenance and CUDA gates and returns before the calculator factory is called.

- [ ] **Step 7: implement row validation and one-shot publication**

Create all calculators only after every source, runtime, hash, Git, and CUDA
gate passes. Instantiate one calculator per `(system, arm)` and reuse it only
for that arm's eight sequential tasks. Write `c60.json`, `pdo.json`, and
`summary.json` into a same-parent staging directory; validate all three and
atomically rename the staging directory to `output`.

`.gitignore` must contain exactly:

```text
output/
```

- [ ] **Step 8: run GREEN runner tests**

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth.py
pytest -q tests/unit/test_relax.py
git diff --check
```

Expected: all pass.

- [ ] **Step 9: commit Task 2**

```bash
git add \
  runs/20260727-safe-lbfgs-history-depth-ablation/.gitignore \
  runs/20260727-safe-lbfgs-history-depth-ablation/plan.md \
  runs/20260727-safe-lbfgs-history-depth-ablation/run_gpu_ablation.py \
  tests/unit/test_safe_lbfgs_history_depth.py
git commit -m "experiment: define safe L-BFGS history-depth GPU ablation"
```

Record this commit as the only permitted GPU execution commit. The runner
accepts it through a required `--expected-git-commit` CLI argument and verifies
it against `git rev-parse HEAD`; do not embed a self-referential commit hash in
tracked source. The deterministic source-bundle and helper hashes are pinned
in the runner before this commit, and no tracked change may remain at
execution.

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
print(d['termination_reason_counts'])
print(d['wall_time_total_s'])
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

Require exact 32-row matrix validation, exact schema/type checking, raw-file
hashes, source/kernel/execution anchors, endpoint-to-last-trace closure,
certificate/termination consistency, and the three-way accounting equality.

Parameterize corruptions for:

```text
missing row, duplicate row, unknown field, bool in integer field,
numeric string, nonfinite energy, negative wall time, task hash,
source hash, source bundle, kernel descriptor, execution commit,
trace length, endpoint position hash, endpoint energy, endpoint force,
purpose count, unattributed call, telemetry count, incomplete publication
```

Run:

```bash
pytest -q tests/unit/test_safe_lbfgs_history_depth_analysis.py
```

Expected: FAIL because the analyzer does not exist.

- [ ] **Step 2: implement strict ledger validation**

The analyzer must compute paired history1-minus-history10 fields:

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
