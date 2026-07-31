# G-E1 Online First-Descent Paired Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the complete-action FE and landing trade-off of an actual online first-true-energy-descent stop on 24 fresh paired C60/PdO D0/K4 actions.

**Architecture:** Add one inert protected stop hook to the existing walk loop, expose it only through the current-action research runner's local audit subclass, and build a separate G-E1 protocol/runner.  Reference and stop arms execute the same walker code with seeds 45--47; exact prefix hashes and purpose ledgers are verified before outcomes are compared.

**Tech Stack:** Python 3.12, NumPy, ASE extxyz I/O, existing `pamssw` MACE calculator, `SurfaceWalker`, first-passage quench classifier, pytest.

---

## File structure

- Modify `pamssw/walker.py`: inert protected post-step stop hook and call site.
- Modify `tests/unit/test_walker_policy.py`: default-inert and override-stop contracts.
- Modify `runs/20260731-current-action-first-passage/run_gate.py`: optional research observer on its local audit walker; no change when absent.
- Modify `tests/unit/test_current_action_first_passage.py`: observer row and reason forwarding contract.
- Create `runs/20260801-online-first-descent-paired-gate/protocol.py`: fresh case matrix, prefix comparison, outcome and decision aggregation.
- Create `runs/20260801-online-first-descent-paired-gate/run_gate.py`: paired GPU action execution, terminal quench, ledger and evidence validation.
- Create `runs/20260801-online-first-descent-paired-gate/PLAN.md`: fixed question, cohort, budgets and exclusions.
- Create `runs/20260801-online-first-descent-paired-gate/.gitignore`: ignore raw `output/` and smoke output.
- Create `tests/unit/test_online_first_descent_paired_gate.py`: pure G-E1 protocol and runner contracts.
- Create after execution `runs/20260801-online-first-descent-paired-gate/evidence.json` and `conclusion.md`.
- Modify after execution `docs/research/2026-07-31-review-reconciled-roadmap.md` with only the closed result.

### Task 1: Add the inert walker stop seam

**Files:**
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `pamssw/walker.py`

- [ ] **Step 1: Write a RED override-stop test**

Add a `StopAfterFirstWalker` subclass beside the existing micro-step rebuilding test.  Its hook records the step, state and true energy, then returns `"unit_test_stop"`.  Use the same one-atom quadratic setup with `max_steps_per_walk=3` and assert:

```python
assert len(observed) == 1
assert observed[0]["step"] == 1
assert walker._walk_termination_last_reason == "unit_test_stop"
assert walker.calculator.snapshot().count(
    EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
) == 2
```

The two true checks are the already-existing before/after evaluations for the
first outer step; the hook must not add a third.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py -k walk_early_stop_hook
```

Expected: FAIL because `_walk_early_stop_reason` is not called.

- [ ] **Step 3: Implement the minimal base hook and call site**

Add:

```python
def _walk_early_stop_reason(
    self,
    *,
    step_index: int,
    walk_reference: State,
    current: State,
    true_energy: float,
) -> str | None:
    return None
```

After `current = current_candidate` and the existing clipped termination,
call the hook.  If it returns a nonempty reason, assign
`termination_reason = reason` and break.  Do not add a config field.

- [ ] **Step 4: Verify default behavior and the override**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py -k 'walk_early_stop_hook or walk_rebuilds_local_softening'
```

Expected: 2 PASS; the existing rebuilding test still observes all three
steps, proving the base hook is inert.

- [ ] **Step 5: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "Add inert post-step walk stop hook"
```

### Task 2: Expose the hook through the frozen action runner

**Files:**
- Modify: `tests/unit/test_current_action_first_passage.py`
- Modify: `runs/20260731-current-action-first-passage/run_gate.py`

- [ ] **Step 1: Write RED tests for observer semantics**

Add a top-level pure helper contract:

```python
def test_walk_step_observer_is_optional_and_forwards_reason():
    runner = _runner()
    record = {"step": 1, "true_energy_eV": -11.0}
    assert runner.notify_walk_step(None, record) is None
    seen = []
    reason = runner.notify_walk_step(
        lambda row: seen.append(row) or "true_energy_descent",
        record,
    )
    assert reason == "true_energy_descent"
    assert seen == [record]
    assert seen[0] is not record
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/unit/test_current_action_first_passage.py -k walk_step_observer
```

Expected: FAIL because `notify_walk_step` does not exist.

- [ ] **Step 3: Implement observer forwarding and local audit integration**

Add:

```python
def notify_walk_step(observer, record):
    return None if observer is None else observer(dict(record))
```

Give `_generate_action_path` an optional `walk_step_observer=None`.  Its local
`AnchorAuditWalker` overrides `_walk_early_stop_reason`, appends a row with
one-based step, true energy, starter delta and cumulative purpose counts, and
returns `notify_walk_step(...)`.  Set `walker.gate_starter_energy` immediately
after the existing starter evaluation and return `walk_step_trace` in the
generation record.  The absent-observer path returns `None` on every step.

- [ ] **Step 4: Verify source-runner tests**

Run:

```bash
pytest -q tests/unit/test_current_action_first_passage.py tests/unit/test_walker_policy.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add runs/20260731-current-action-first-passage/run_gate.py \
  tests/unit/test_current_action_first_passage.py
git commit -m "Expose research walk-step observer"
```

### Task 3: Freeze the G-E1 pure protocol

**Files:**
- Create: `runs/20260801-online-first-descent-paired-gate/protocol.py`
- Create: `runs/20260801-online-first-descent-paired-gate/PLAN.md`
- Create: `runs/20260801-online-first-descent-paired-gate/.gitignore`
- Create: `tests/unit/test_online_first_descent_paired_gate.py`

- [ ] **Step 1: Write RED tests for the fresh cohort and first-descent rule**

```python
def test_case_matrix_is_fresh_twenty_four_pair_design():
    cases = protocol.case_matrix()
    assert len(cases) == 24
    assert {row["seed"] for row in cases} == {45, 46, 47}
    assert {row["system"] for row in cases} == {"c60", "pdo"}
    assert {row["arm"] for row in cases} == {
        "D0_exact_anchor", "K4_discrete"
    }


def test_first_descent_is_strict_at_existing_tolerance():
    observer = protocol.FirstDescentObserver(
        starter_energy_eV=-10.0,
        tolerance_eV=0.001,
    )
    assert observer({"step": 1, "true_energy_eV": -10.001}) is None
    assert observer({"step": 2, "true_energy_eV": -10.002}) == (
        "true_energy_descent"
    )
    assert observer.trigger_step == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_online_first_descent_paired_gate.py
```

Expected: FAIL because the protocol does not exist.

- [ ] **Step 3: Implement only the frozen matrix, observer and trade-off class**

Implement Cartesian-product ordering over systems, states, seeds and arms;
store every observer row; latch only the first strict crossing; and reuse the
four G-E0 trade-off labels without a scalar score.

- [ ] **Step 4: Add RED/GREEN prefix integrity tests**

Test exact equality through the shorter trace for:

```python
("selected_direction_sha256", "executed_step_scale",
 "uphill_final_bias_weight", "true_energy_eV", "state_sha256")
```

`compare_prefix(reference, early)` returns a structured mismatch list and
`prefix_valid`; it never silently relaxes a mismatch tolerance.

- [ ] **Step 5: Add RED/GREEN decision tests**

`build_decision(pairs)` admits G-E2 only for 24 valid pairs, at least two
triggers, all triggered early landings certified and below starter, and
strictly positive summed complete-action FE savings.  Test each failed
condition independently.

- [ ] **Step 6: Verify and commit**

```bash
pytest -q tests/unit/test_online_first_descent_paired_gate.py
git add runs/20260801-online-first-descent-paired-gate \
  tests/unit/test_online_first_descent_paired_gate.py
git commit -m "Add online first-descent gate protocol"
```

### Task 4: Implement the paired action runner

**Files:**
- Create: `runs/20260801-online-first-descent-paired-gate/run_gate.py`
- Modify: `tests/unit/test_online_first_descent_paired_gate.py`

- [ ] **Step 1: Write RED tests for algorithmic versus validation cost**

Test that an untriggered, hash-identical early arm may reuse the reference
landing with zero *new validation FE* while its `complete_action_fe` still
includes the copied reference quench cost.  Test that a triggered arm uses its
own executed quench counts.

- [ ] **Step 2: Implement immutable runtime and pair execution**

Reuse `_load_starter`, `_base_config`, `_generate_action_path` and
`_quench_checkpoint` from the frozen runners.  For each pair, create fresh
reference and early directories, instantiate a fresh first-descent observer
only for the early arm, true-quench terminal endpoints, compute state SHA256
with the fixed audit helper, and compare prefixes using the pure protocol.

- [ ] **Step 3: Implement exact ledger and evidence checks**

Track two separate values:

```text
new_validation_fe = force calls actually executed by G-E1
complete_action_fe = calls the arm would pay online, including reused quench
```

Reject any pair with unclosed purpose counts or unattributed calls.  The
`--check-evidence` command recomputes the 24-pair cohort, prefix validity,
trigger count, summed complete-action savings, landing certificates and the
decision from raw pair rows.

- [ ] **Step 4: Run unit tests and a one-pair GPU smoke**

```bash
pytest -q tests/unit/test_online_first_descent_paired_gate.py \
  tests/unit/test_current_action_first_passage.py \
  tests/unit/test_walker_policy.py
```

Then run one preregistered pair in a fresh `/tmp/g-e1-smoke-*` directory with
`--limit-pairs 1`.  Require exact prefix closure, no unattributed FE and a
mechanically valid evidence file.

- [ ] **Step 5: Commit the complete runner**

```bash
git add runs/20260801-online-first-descent-paired-gate/run_gate.py \
  tests/unit/test_online_first_descent_paired_gate.py
git commit -m "Add online first-descent paired runner"
```

### Task 5: Execute and close G-E1

**Files:**
- Create (ignored): `runs/20260801-online-first-descent-paired-gate/output/evidence.json`
- Create: `runs/20260801-online-first-descent-paired-gate/evidence.json`
- Create: `runs/20260801-online-first-descent-paired-gate/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Run the clean 24-pair GPU gate**

Execute with exact `--expected-commit`, maximum 25,000 new FE and 300 seconds
GPU kernel wall.  Stop before interpretation on prefix, hash, geometry,
budget or purpose-ledger failure.

- [ ] **Step 2: Validate raw evidence mechanically**

Run `--check-evidence` and verify all 24 pairs, fresh seeds 45--47, exact
prefixes, trigger rule, complete-action cost reconstruction and aggregate
decision.

- [ ] **Step 3: Write compact evidence and physical conclusion**

Report trigger frequency by system/state/arm, summed and per-trigger FE
savings, direction/proposal/quench cost shifts, landing energy trade-offs,
basin relations, failure taxonomy, G-E2 admission and the claim ceiling.

- [ ] **Step 4: Run focused verification**

```bash
pytest -q \
  tests/unit/test_online_first_descent_paired_gate.py \
  tests/unit/test_true_energy_descent_early_stop_gate.py \
  tests/unit/test_current_action_first_passage.py \
  tests/unit/test_uphill_relax_counterfactual_gate.py \
  tests/unit/test_uphill_relax_first_passage_gate.py \
  tests/unit/test_walker_policy.py
python -m json.tool runs/20260801-online-first-descent-paired-gate/evidence.json
git diff --check
```

- [ ] **Step 5: Commit, push and update PR #14**

```bash
git add runs/20260801-online-first-descent-paired-gate/evidence.json \
  runs/20260801-online-first-descent-paired-gate/conclusion.md \
  docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Conclude online first-descent paired gate"
git push pam feature/direction-continuation-ablation
```

Update PR #14 with the exact executed-FE ledger, reconstructed complete-action
cost, energy/basin trade-off and whether G-E2 is admitted.  Do not expose a
production early-stop parameter.
