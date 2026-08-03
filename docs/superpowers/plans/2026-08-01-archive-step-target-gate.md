# Archive Step Target Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the un-attributed archive-energy-scale macro uphill target contributes positively relative to the configured fixed 0.8 eV reference.

**Architecture:** Reconstruct the exact target history from the completed S-CR1 accepted-minimum logs without PES calls. If that audit closes, build a self-contained shared-bootstrap runner that replaces only `walker.step_target_controller.target()` in the fixed arm; no `pamssw` production code or default changes.

**Tech Stack:** Python, NumPy, pytest, ASE, MACE CUDA, existing `BudgetedCalculator` purpose ledger and S-CR1 resource loaders.

---

### Task 1: U-T0 pure target-history reconstruction

**Files:**
- Create: `runs/20260801-archive-step-target-audit/PLAN.md`
- Create: `runs/20260801-archive-step-target-audit/analyze.py`
- Test: `tests/unit/test_archive_step_target_audit.py`

- [x] **Step 1: Write failing pure tests**

Test a synthetic bootstrap plus accepted-minimum sequence against the exact
`StepTargetController._archive_target()` formula, including one duplicate
trial and the post-cohort next-attempt target. Test fail-closed handling for
unordered trial indices, inconsistent archive size, non-finite energy and a
source hash mismatch.

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_archive_step_target_audit.py
```

Expected before implementation: collection or import failure.

- [x] **Step 2: Implement the pure analyzer**

Expose these functions without importing MACE or `pamssw`:

```python
def archive_scaled_target(energies: Sequence[float], fallback: float) -> float: ...

def reconstruct_targets(
    *,
    bootstrap_energy_eV: float,
    completed_trials: int,
    accepted_rows: Sequence[Mapping[str, Any]],
    fallback_target_eV: float,
) -> dict[str, Any]: ...

def analyze_case(raw_case: Mapping[str, Any], accepted_log: Path) -> dict[str, Any]: ...
```

The result must include the completed-trial target vector and the next-attempt
target. Require the latter to equal `stats.adaptive_step_target` to 1e-12.

- [x] **Step 3: Run the focused tests**

Expected: all tests in `test_archive_step_target_audit.py` pass.

- [x] **Step 4: Commit the analyzer and tests**

```bash
git add runs/20260801-archive-step-target-audit tests/unit/test_archive_step_target_audit.py
git commit -m "Audit archive scaled uphill targets"
```

### Task 2: Execute and close U-T0

**Files:**
- Create: `runs/20260801-archive-step-target-audit/evidence.json`
- Create: `runs/20260801-archive-step-target-audit/conclusion.md`

- [x] **Step 1: Analyze all twelve S-CR1 cases**

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python \
  runs/20260801-archive-step-target-audit/analyze.py \
  --source runs/20260801-paired-continuation-restart-gate/output \
  --output runs/20260801-archive-step-target-audit/evidence.json
```

Require twelve cases, exact accepted-log hashes, exact final-target closure,
and zero new force evaluations.

- [x] **Step 2: Apply the preregistered U-T0 gate**

Admit U-T1 only when reconstruction closes and at least two systems spend more
than 75% of completed actions at a target different from 0.8 eV. Record target
distributions by system and selector without correlating them post hoc with a
scalar success score.

- [x] **Step 3: Write the evidence-bounded conclusion**

State whether the block is active, not whether fixed is superior. Record the
raw S-CR1 evidence SHA-256 and all twelve accepted-log SHA-256 values.

- [x] **Step 4: Verify and commit U-T0 evidence**

```bash
python -m json.tool runs/20260801-archive-step-target-audit/evidence.json >/dev/null
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_archive_step_target_audit.py
git diff --check
git add runs/20260801-archive-step-target-audit
git commit -m "Conclude archive step target audit"
```

### Task 3: U-T1 protocol and fixed target injection

**Files:**
- Create: `runs/20260801-fixed-step-target-gate/.gitignore`
- Create: `runs/20260801-fixed-step-target-gate/PLAN.md`
- Create: `runs/20260801-fixed-step-target-gate/protocol.py`
- Create: `runs/20260801-fixed-step-target-gate/run_gate.py`
- Test: `tests/unit/test_fixed_step_target_gate.py`

- [x] **Step 1: Write failing protocol tests**

Test the exact six-case matrix, gain-AUC integration, two-of-three admission
rule, partial-cohort non-decision, shared-bootstrap equality, 20,000-FE purpose
closure and `unattributed=0`.

- [x] **Step 2: Write a failing fixed-controller test**

Use a fake mutable controller and prove the run-local replacement:

```python
class FixedReferenceTarget:
    def __init__(self, wrapped, target_eV: float): ...
    def target(self, archive=None) -> float: return self.target_eV
    def record_trial(self, **kwargs) -> None: self.wrapped.record_trial(**kwargs)
    def stats(self) -> dict[str, float | int]: ...
```

The wrapper must retain trial bookkeeping while reporting the exact fixed
target and `step_target_mode="fixed_reference"`. The scaled arm uses the
unmodified controller and reports `step_target_mode="archive_scaled"`.

- [x] **Step 3: Implement the minimal protocol and runner**

Load the existing S-CR1/source resource modules with `importlib`. Reuse their
C60/PdO/CuO calculators, structures, production configurations, shared
bootstrap routine, accepted-row projection and energy-trace representation.
Copy only the `_run_case` seam needed to inject the run-local target wrapper.

Use:

```python
TARGET_MODES = ("archive_scaled", "fixed_reference")
STARTER_MODE = "metropolis_chain"
DEFAULT_FORCE_BUDGET = 20_000
FIXED_REFERENCE_EV = 0.8
```

Persist the actual per-trial target history reconstructed from the accepted
archive log in each case summary. Do not add a `pamssw` config option.

- [x] **Step 4: Run focused tests and an excluded CUDA smoke**

The smoke uses C60 only, both arms and 1,000 FE per arm. Its decision must be
`NOT_EVALUATED_PARTIAL_COHORT` and its output directory remains ignored.

- [x] **Step 5: Commit the runner before formal execution**

```bash
git add runs/20260801-fixed-step-target-gate tests/unit/test_fixed_step_target_gate.py
git commit -m "Add fixed step target GPU gate"
```

### Task 4: Execute U-T1 seed-46 gate

**Files:**
- Create after execution: `runs/20260801-fixed-step-target-gate/evidence.json`
- Create after execution: `runs/20260801-fixed-step-target-gate/conclusion.md`

- [x] **Step 1: Run the formal six-case matrix**

```bash
/root/miniforge3/envs/mace_les/bin/python \
  runs/20260801-fixed-step-target-gate/run_gate.py \
  --output runs/20260801-fixed-step-target-gate/output \
  --expected-commit "$(git rev-parse HEAD)" \
  --systems c60 pdo cuo \
  --seeds 46 \
  --target-modes archive_scaled fixed_reference \
  --force-budget 20000
```

- [x] **Step 2: Run the mechanical checker**

Require six cases, three shared-bootstrap blocks, at most 120,000 FE, exact
purpose closure, no unattributed work and exact execution commit. A residual
is permitted only when it is smaller than one indivisible direction batch and
the case reports budget exhaustion; the observed matrix used 119,998 FE.

- [x] **Step 3: Apply the two-of-three decision**

Advance only if fixed-reference gain AUC is strictly greater in at least two
systems. Report endpoint, time-to-best, archive coverage, failures and
component cost separately.

- [x] **Step 4: Close or conditionally extend**

If U-T1 fails, write `DO_NOT_ADMIT_U_T2`, retain current defaults and do not
tune any target parameter. If it passes, extend the same runner only to seeds
47--48 and apply the preregistered six-of-nine plus positive-median rule.

- [x] **Step 5: Commit the formal evidence independently**

```bash
git add runs/20260801-fixed-step-target-gate/evidence.json \
  runs/20260801-fixed-step-target-gate/conclusion.md
git commit -m "Close fixed step target gate"
```

### Task 5: Reconcile, verify and publish

**Files:**
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [x] **Step 1: Reconcile without reopening closed mechanisms**

Add one roadmap section distinguishing the macro archive target from the
already closed local sigma/weight feedback, Gaussian shape, history, relax
capacity and horizon questions.

- [x] **Step 2: Run verification**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_archive_step_target_audit.py \
  tests/unit/test_fixed_step_target_gate.py \
  tests/unit/test_config.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py
git diff --check
```

- [x] **Step 3: Push and update PR #14**

Publish the decision, raw evidence hash, exact FE breakdown, test result and
claim ceiling. Do not present an unpromoted experimental wrapper as a user
configuration.
