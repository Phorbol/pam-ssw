# True-PES Descent Early-Stop Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a research-only offline audit of first true-PES energy descent across all accepted outer SSW micro steps in the frozen 24-path C60/PdO corpus.

**Architecture:** A pure protocol reconstructs the accepted endpoint sequence, selects the first strict descent crossing, and classifies overshoot versus deeper-terminal trade-offs.  A separate runner validates a tracked 82-endpoint manifest, reuses existing energy/quench rows, evaluates only missing true energies, and quenches only unreused first-crossing or terminal endpoints.  No `pamssw` production code or source action is replayed.

**Tech Stack:** Python 3.12, NumPy, ASE extxyz I/O, existing `pamssw` budgeted MACE calculator and true-quench classifier, pytest.

---

## File structure

- Create `runs/20260801-true-energy-descent-early-stop-gate/protocol.py`: pure cohort reconstruction, crossing selection and outcome classification.
- Create `runs/20260801-true-energy-descent-early-stop-gate/manifest.json`: exact 24-case, 82-accepted-endpoint path and SHA256 manifest.
- Create `runs/20260801-true-energy-descent-early-stop-gate/run_gate.py`: source validation, zero-cost reuse, missing energy evaluation, minimal true quench and ledger closure.
- Create `runs/20260801-true-energy-descent-early-stop-gate/PLAN.md`: frozen scientific question, budgets and exclusions.
- Create `runs/20260801-true-energy-descent-early-stop-gate/.gitignore`: ignore raw `output/` and smoke artifacts only.
- Create `tests/unit/test_true_energy_descent_early_stop_gate.py`: pure protocol, manifest and runner contracts.
- Modify `docs/research/2026-07-31-review-reconciled-roadmap.md`: add only the closed G-E0 result.

### Task 1: Freeze the pure stopping protocol

**Files:**
- Create: `tests/unit/test_true_energy_descent_early_stop_gate.py`
- Create: `runs/20260801-true-energy-descent-early-stop-gate/protocol.py`
- Create: `runs/20260801-true-energy-descent-early-stop-gate/PLAN.md`
- Create: `runs/20260801-true-energy-descent-early-stop-gate/.gitignore`

- [ ] **Step 1: Write RED tests for exact accepted endpoints and first crossing**

```python
def test_accepted_steps_exclude_rejected_attempts():
    case = {"reached_macro_steps": 2, "attempted_macro_steps": 3}
    assert protocol.accepted_step_indices(case) == (1, 2)


def test_first_crossing_is_strict_and_does_not_look_ahead():
    rows = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 2, "checkpoint_delta_eV": -0.001},
        {"step": 3, "checkpoint_delta_eV": -0.4},
    ]
    assert protocol.first_descent_crossing(rows, tolerance=0.001) == rows[2]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/unit/test_true_energy_descent_early_stop_gate.py`

Expected: FAIL because `protocol.py` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

```python
def accepted_step_indices(case):
    reached = int(case["reached_macro_steps"])
    attempted = int(case["attempted_macro_steps"])
    if reached < 0 or attempted < reached:
        raise ValueError("invalid reached/attempted macro-step counts")
    return tuple(range(1, reached + 1))


def first_descent_crossing(rows, *, tolerance):
    for expected, row in enumerate(rows, start=1):
        if int(row["step"]) != expected:
            raise ValueError("accepted endpoint steps must be consecutive")
        delta = row.get("checkpoint_delta_eV")
        if delta is not None and float(delta) < -float(tolerance):
            return dict(row)
    return None
```

- [ ] **Step 4: Add RED tests and implementation for separate outcome classes**

```python
def test_outcome_keeps_overshoot_and_deeper_terminal_separate():
    assert protocol.classify_tradeoff(-7.0, 0.2, 0.001) == "AVOIDED_OVERSHOOT"
    assert protocol.classify_tradeoff(-7.0, -9.0, 0.001) == "FORGONE_DEEPER_TERMINAL"
    assert protocol.classify_tradeoff(-7.0, -7.0, 0.001) == "ENERGY_EQUIVALENT"
```

Implement only four exhaustive values: `AVOIDED_OVERSHOOT`,
`FORGONE_DEEPER_TERMINAL`, `ENERGY_EQUIVALENT`, and `UNLEARNABLE`.

- [ ] **Step 5: Verify protocol tests and commit**

Run: `pytest -q tests/unit/test_true_energy_descent_early_stop_gate.py`

Expected: PASS.

```bash
git add tests/unit/test_true_energy_descent_early_stop_gate.py \
  runs/20260801-true-energy-descent-early-stop-gate/protocol.py \
  runs/20260801-true-energy-descent-early-stop-gate/PLAN.md \
  runs/20260801-true-energy-descent-early-stop-gate/.gitignore
git commit -m "Add true-energy descent gate protocol"
```

### Task 2: Freeze the 82-endpoint manifest

**Files:**
- Create: `runs/20260801-true-energy-descent-early-stop-gate/manifest.json`
- Modify: `tests/unit/test_true_energy_descent_early_stop_gate.py`

- [ ] **Step 1: Add a pure manifest projection and RED test**

```python
def test_manifest_uses_only_reached_steps(tmp_path):
    checkpoint_dir = tmp_path / "macro_checkpoints"
    checkpoint_dir.mkdir()
    for step in (1, 2, 3):
        (checkpoint_dir / f"step{step:03d}_checkpoint.xyz").write_text(
            f"frame {step}\n"
        )
    source_case = {
        "system": "c60",
        "state_id": "plateau_accepted",
        "seed": 42,
        "arm": "D0_exact_anchor",
        "reached_macro_steps": 2,
        "attempted_macro_steps": 3,
    }
    rows = protocol.project_manifest_case(source_case, tmp_path)
    assert [row["step"] for row in rows] == [1, 2]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in rows)
```

`project_manifest_case` resolves exactly
`macro_checkpoints/step{step:03d}_checkpoint.xyz` for `1..reached_macro_steps`,
computes SHA256, and never includes `reached+1..attempted`.

- [ ] **Step 2: Implement the projection and verify GREEN**

Run: `pytest -q tests/unit/test_true_energy_descent_early_stop_gate.py`

Expected: PASS.

- [ ] **Step 3: Generate the manifest read-only, then add its exact output with `apply_patch`**

Run this read-only command, capture stdout, and add the exact JSON as
`manifest.json` using `apply_patch`; do not write it from the helper command:

```bash
python - <<'PY'
import importlib.util
import json
from pathlib import Path

repo = Path.cwd()
run_root = repo / "runs/20260801-true-energy-descent-early-stop-gate"
spec = importlib.util.spec_from_file_location("g_e0_protocol", run_root / "protocol.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raw = json.loads((repo / "runs/20260731-current-action-first-passage/output-v3/evidence.json").read_text())
cases = []
for case in raw["cases"]:
    case_dir = repo / "runs/20260731-current-action-first-passage/output-v3/cases" / case["system"] / case["state_id"] / f"seed-{int(case['seed']):08d}" / case["arm"]
    cases.append({
        "key": [case[name] for name in ("system", "state_id", "seed", "arm")],
        "reached_macro_steps": case["reached_macro_steps"],
        "attempted_macro_steps": case["attempted_macro_steps"],
        "endpoints": module.project_manifest_case(case, case_dir),
    })
payload = {
    "schema_version": 1,
    "case_count": len(cases),
    "accepted_endpoint_count": sum(len(case["endpoints"]) for case in cases),
    "attempted_endpoint_count": sum(int(case["attempted_macro_steps"]) for case in cases),
    "cases": cases,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

The manifest must assert:

```json
{
  "case_count": 24,
  "accepted_endpoint_count": 82,
  "attempted_endpoint_count": 93
}
```

- [ ] **Step 4: Add a manifest closure test and commit**

```python
def test_frozen_manifest_closes_exact_cohort():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["case_count"] == 24
    assert manifest["accepted_endpoint_count"] == 82
    assert manifest["attempted_endpoint_count"] == 93
    assert len(manifest["cases"]) == 24
```

```bash
git add runs/20260801-true-energy-descent-early-stop-gate/manifest.json \
  runs/20260801-true-energy-descent-early-stop-gate/protocol.py \
  tests/unit/test_true_energy_descent_early_stop_gate.py
git commit -m "Freeze true-energy descent endpoint manifest"
```

### Task 3: Implement immutable reuse and missing-energy evaluation

**Files:**
- Create: `runs/20260801-true-energy-descent-early-stop-gate/run_gate.py`
- Modify: `tests/unit/test_true_energy_descent_early_stop_gate.py`

- [ ] **Step 1: Write RED tests for source-row reuse**

```python
def test_existing_checkpoint_energy_reuses_zero_force_evaluations():
    row = runner.reused_energy_row({
        "horizon": 2,
        "checkpoint_energy_eV": -10.5,
        "checkpoint_delta_eV": -0.5,
    })
    assert row["step"] == 2
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_first_passage"
```

- [ ] **Step 2: Implement source/hash validation and reuse**

The runner must validate the tracked source compact evidence, its raw SHA256,
the manifest cohort, every starter hash, and every endpoint hash before loading
MACE.  It builds a source checkpoint map by `horizon` and reuses valid
`checkpoint_energy_eV` values at zero new FE.

- [ ] **Step 3: Write RED tests for one missing-energy evaluation ledger**

```python
class FakeCalculator:
    def __init__(self):
        self.counts = {name: 0 for name in runner.PURPOSE_NAMES}

    @contextmanager
    def purpose(self, purpose):
        self.active = purpose.value
        yield

    def evaluate(self, _state):
        self.counts[self.active] += 1
        return SimpleNamespace(energy=-10.5)


def test_missing_energy_costs_one_true_pes_check(fake_state):
    calculator = FakeCalculator()
    row = runner.evaluate_missing_energy(
        fake_state,
        starter_energy=-10.0,
        calculator=calculator,
    )
    assert row["new_force_evaluations"] == 1
    assert row["purpose_counts"]["escape_true_pes_check"] == 1
    assert row["purpose_counts"]["direction_oracle"] == 0
    assert row["purpose_counts"]["biased_proposal_relax"] == 0
    assert row["purpose_counts"]["unattributed"] == 0
```

- [ ] **Step 4: Implement exact missing-energy evaluation**

Read the endpoint with the starter fixed mask, validate geometry, evaluate it
inside `EvaluationPurpose.ESCAPE_TRUE_PES_CHECK`, and record either a finite
energy delta or an explicit unlearnable failure.  Do not quench during this
pass.

- [ ] **Step 5: Verify runner unit tests and commit**

Run: `pytest -q tests/unit/test_true_energy_descent_early_stop_gate.py`

Expected: PASS.

```bash
git add runs/20260801-true-energy-descent-early-stop-gate/run_gate.py \
  tests/unit/test_true_energy_descent_early_stop_gate.py
git commit -m "Add true-energy descent audit runner"
```

### Task 4: Add first-crossing and terminal quench reuse

**Files:**
- Modify: `runs/20260801-true-energy-descent-early-stop-gate/run_gate.py`
- Modify: `tests/unit/test_true_energy_descent_early_stop_gate.py`

- [ ] **Step 1: Write RED tests for quench reuse and minimal new work**

```python
def test_existing_first_crossing_quench_is_reused():
    source_checkpoint = {
        "horizon": 2,
        "label": "ESCAPED_CERTIFIED",
        "landing_energy_eV": -17.0,
        "landing_delta_eV": -7.0,
        "landing_path": "landing.xyz",
        "landing_sha256": "a" * 64,
    }
    row = runner.reused_quench_row(source_checkpoint)
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_first_passage"


def test_only_crossing_and_terminal_can_request_new_quench():
    requested = runner.required_quench_steps(first_crossing=3, terminal=8, reusable={3})
    assert requested == (8,)
```

- [ ] **Step 2: Compose the existing frozen `_quench_checkpoint` implementation**

For an unreused crossing or terminal, call the source first-passage runner's
`_quench_checkpoint` with the original starter, starter energy, effective
config and endpoint path.  The returned ledger must contain only
`escape_true_pes_check`, `landing_true_quench`, and
`post_relax_validation`.

- [ ] **Step 3: Build exact per-path and aggregate evidence**

Use the pure protocol to classify the crossing-versus-terminal energy outcome.
Keep basin identity, energy trade-off, saved micro steps, and costs as separate
fields.  Do not construct a scalar score.

- [ ] **Step 4: Run unit tests and a one-path GPU smoke**

Run: `pytest -q tests/unit/test_true_energy_descent_early_stop_gate.py`

Run the runner with one preregistered case into a fresh `/tmp/g-e0-smoke-*`
directory.  Expected: evidence closes, calculator is loaded only after all
hash checks, and direction/bias/unattributed FE are zero.

- [ ] **Step 5: Commit the complete runner**

```bash
git add runs/20260801-true-energy-descent-early-stop-gate/run_gate.py \
  tests/unit/test_true_energy_descent_early_stop_gate.py
git commit -m "Complete true-energy descent gate runner"
```

### Task 5: Execute G-E0 and close the research decision

**Files:**
- Create (ignored): `runs/20260801-true-energy-descent-early-stop-gate/output/evidence.json`
- Create: `runs/20260801-true-energy-descent-early-stop-gate/evidence.json`
- Create: `runs/20260801-true-energy-descent-early-stop-gate/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Commit-clean execution and provenance gate**

Run the complete 24-path gate with exact `--expected-commit`, maximum 10,000
new FE and maximum 300 seconds GPU kernel wall.  Stop on hash drift or forbidden
FE before interpreting outcomes.

- [ ] **Step 2: Validate evidence mechanically**

The runner's `--check-evidence` command must verify 24 cases, 82 accepted
endpoints, consecutive steps, exact manifest hashes, purpose-ledger closure,
and recomputed aggregates.

- [ ] **Step 3: Write compact evidence and the physical conclusion**

Record the raw evidence path and SHA256, trigger precision and coverage,
overshoot/deeper-terminal counts and energy magnitudes, saved micro steps, new
FE, wall time, decision about admitting G-E1, production-default status, and
claim ceiling.

- [ ] **Step 4: Run focused and broad relevant tests**

Run:

```bash
pytest -q \
  tests/unit/test_true_energy_descent_early_stop_gate.py \
  tests/unit/test_current_action_first_passage.py \
  tests/unit/test_uphill_relax_counterfactual_gate.py \
  tests/unit/test_uphill_relax_first_passage_gate.py \
  tests/unit/test_walker_policy.py
```

Expected: PASS.  Run `git diff --check` and verify both tracked and raw evidence
JSON with `python -m json.tool`.

- [ ] **Step 5: Commit, push and update PR #14**

```bash
git add runs/20260801-true-energy-descent-early-stop-gate/evidence.json \
  runs/20260801-true-energy-descent-early-stop-gate/conclusion.md \
  docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Conclude true-energy descent early-stop gate"
git push pam feature/direction-continuation-ablation
```

Update PR #14 with the exact FE ledger, physical result, claim ceiling and
whether a fresh online G-E1 gate is admitted.  Do not advertise a production
early-stop option.
