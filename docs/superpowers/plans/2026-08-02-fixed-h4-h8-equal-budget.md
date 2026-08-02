# Fixed H4 versus H8 Equal-Budget Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a run-local, shared-bootstrap C60/PdO/CuO gate that changes only the fixed uphill micro-step horizon from H8 to H4 at equal total force-evaluation budget.

**Architecture:** Reuse the validated U-O1 action serializer/analyzer and the shared-bootstrap production runner. A pure protocol module owns the case matrix, gain-AUC calculation, configuration-difference allowlist, and promotion rule; a run-local runner owns GPU execution and evidence serialization. Production configuration and `pamssw` behavior remain unchanged.

**Tech Stack:** Python 3.11, dataclasses, NumPy, ASE/MACE calculators, pytest, JSON/JSONL evidence.

---

### Task 1: Freeze the pure horizon protocol

**Files:**
- Create: `runs/20260802-fixed-h4-h8-equal-budget/protocol.py`
- Test: `tests/unit/test_fixed_h4_h8_equal_budget.py`

- [ ] **Step 1: Write failing tests for the case matrix, gain AUC, config allowlist, and decision rule**

The tests must require the ordered matrix
`c60/pdo/cuo × seed49 × H4/H8`, reject any non-output configuration difference
other than `max_steps_per_walk`, integrate a piecewise-constant best-energy
trace over 20,000 FE, and advance H4 only for at least two system wins with a
positive median delta.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_fixed_h4_h8_equal_budget.py
```

Expected: import failure because the protocol module does not exist.

- [ ] **Step 3: Implement the pure protocol**

Implement `case_matrix`, `gain_auc`, `scientific_config_differences`, and
`cohort_decision`. Keep the decision categorical and preserve raw paired
deltas in the returned result.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Expected: all tests in `test_fixed_h4_h8_equal_budget.py` pass.

### Task 2: Build the run-local shared-bootstrap runner

**Files:**
- Create: `runs/20260802-fixed-h4-h8-equal-budget/run_gate.py`
- Create: `runs/20260802-fixed-h4-h8-equal-budget/.gitignore`
- Modify: `tests/unit/test_fixed_h4_h8_equal_budget.py`

- [ ] **Step 1: Add failing runner-contract tests**

Require `build_config(..., horizon=4)` and `build_config(..., horizon=8)` to
differ scientifically only in `max_steps_per_walk`; require
`seed_selection_mode="metropolis_chain"`, `direction_selection_mode="discrete"`,
`direction_ranking_mode="static_score"`, and unchanged action-history support.

- [ ] **Step 2: Run the focused test and confirm RED**

Expected: runner module or `build_config` import failure.

- [ ] **Step 3: Implement the minimal runner**

Reuse the U-O1 serializer and shared-bootstrap/model helpers. Serialize each
arm's summary, accepted energy trace, best structure, and action history. The
evidence validator must check ordered matrix closure, shared bootstrap identity,
exact purpose ledger, zero unattributed FE, residual below one HVP batch,
configuration allowlist, and recomputed decision equality.

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_fixed_h4_h8_equal_budget.py \
  tests/unit/test_uphill_action_observability.py \
  tests/unit/test_action_history.py
```

Expected: all pass.

### Task 3: Execute and close the GPU gate

**Files:**
- Create: `runs/20260802-fixed-h4-h8-equal-budget/evidence.json`
- Create: `runs/20260802-fixed-h4-h8-equal-budget/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Run a 1,000-FE C60 H4/H8 smoke**

Require both arms, shared bootstrap, `unattributed=0`, action histories, and
evidence validation. Remove no historical output; smoke output stays ignored.

- [ ] **Step 2: Run the preregistered six-case CUDA cohort**

Run C60/PdO/CuO seed49, H4/H8, 20,000 FE per arm. Capture the execution commit,
GPU/model provenance, hashes, purpose counts, and wall time.

- [ ] **Step 3: Recompute evidence independently and write the conclusion**

Apply the frozen decision rule without tuning. Explain the physical trade-off
between action continuation and action throughput, cross-system sign, numerical
sensitivity, and the precise claim ceiling.

- [ ] **Step 4: Run final verification**

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_fixed_h4_h8_equal_budget.py \
  tests/unit/test_action_history.py \
  tests/unit/test_uphill_action_observability.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py
git diff --check
```

Expected: zero test failures and no whitespace errors.
