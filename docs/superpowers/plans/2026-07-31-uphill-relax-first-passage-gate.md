# Uphill Relax First-Passage Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the earliest Safe-LBFGS accepted step that stably reaches the final biased-relax landing basin, then test the resulting single fixed cutoff on untouched informative trajectories.

**Architecture:** Exhaustively true-quench every recorded accepted frame in four preregistered repeated-causal trajectories. Derive one absolute accepted-step cutoff from the final-basin suffix, then evaluate only that cutoff on four holdout trajectories. Existing frame-zero and final-frame quenches are reused by hash; no HVP, biased relaxation, selector, direction, or production code is executed.

**Tech Stack:** Python 3.12, NumPy, ASE extxyz I/O, existing `pamssw` MACE evaluator/quench/matcher, pytest.

---

## File structure

- Create `runs/20260731-uphill-relax-first-passage-gate/protocol.py`: frozen discovery/holdout keys, exact suffix analysis, cutoff derivation, and strict promotion rule.
- Create `runs/20260731-uphill-relax-first-passage-gate/run_gate.py`: immutable source validation, endpoint reuse, intermediate-frame true quenches, and evidence writing.
- Create `runs/20260731-uphill-relax-first-passage-gate/PLAN.md`: physical question, budget and stopping conditions.
- Create `runs/20260731-uphill-relax-first-passage-gate/.gitignore`: generated output only.
- Create `tests/unit/test_uphill_relax_first_passage_gate.py`: protocol and frame extraction tests.
- Modify `docs/research/2026-07-31-review-reconciled-roadmap.md`: add only the closed result.

### Task 1: Freeze the two-stage protocol

**Files:**
- Create: `tests/unit/test_uphill_relax_first_passage_gate.py`
- Create: `runs/20260731-uphill-relax-first-passage-gate/protocol.py`
- Create: `runs/20260731-uphill-relax-first-passage-gate/PLAN.md`

- [ ] **Step 1: Write RED tests for the exact four-plus-four split**

```python
def test_discovery_is_only_the_four_repeated_causal_trajectories():
    assert protocol.DISCOVERY_KEYS == (
        ("plateau_accepted", 42, "D0_exact_anchor", 1),
        ("plateau_accepted", 43, "D0_exact_anchor", 1),
        ("plateau_accepted", 43, "K4_discrete", 2),
        ("plateau_accepted", 44, "K4_discrete", 2),
    )


def test_holdout_contains_the_other_four_mechanism_informative_pairs():
    assert len(protocol.HOLDOUT_KEYS) == 4
    assert set(protocol.DISCOVERY_KEYS).isdisjoint(protocol.HOLDOUT_KEYS)
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/unit/test_uphill_relax_first_passage_gate.py`

Expected: FAIL because the protocol module does not exist.

- [ ] **Step 3: Add RED tests for stable final-basin arrival**

```python
def test_stable_final_basin_step_uses_the_complete_final_suffix():
    rows = frames("RETURN", "FINAL", "OTHER", "FINAL", "FINAL")
    result = protocol.summarize_trajectory(rows)
    assert result["first_escape_step"] == 1
    assert result["stable_final_basin_step"] == 3


def test_cutoff_is_the_maximum_stable_final_arrival_without_weights():
    summaries = [{"stable_final_basin_step": value} for value in (7, 11, 4, 9)]
    assert protocol.derive_cutoff(summaries) == 11
```

- [ ] **Step 4: Implement the minimal pure protocol and verify GREEN**

Run: `pytest -q tests/unit/test_uphill_relax_first_passage_gate.py`

Expected: PASS.

- [ ] **Step 5: Commit the preregistered protocol**

```bash
git add docs/superpowers/plans/2026-07-31-uphill-relax-first-passage-gate.md \
  runs/20260731-uphill-relax-first-passage-gate/PLAN.md \
  runs/20260731-uphill-relax-first-passage-gate/protocol.py \
  runs/20260731-uphill-relax-first-passage-gate/.gitignore \
  tests/unit/test_uphill_relax_first_passage_gate.py
git commit -m "Add uphill relax first-passage protocol"
```

### Task 2: Implement immutable frame replay

**Files:**
- Create: `runs/20260731-uphill-relax-first-passage-gate/run_gate.py`
- Modify: `tests/unit/test_uphill_relax_first_passage_gate.py`

- [ ] **Step 1: Write RED tests for exact frame extraction and endpoint reuse**

```python
def test_extract_frame_preserves_requested_accepted_step(tmp_path):
    write_three_frame_trace(tmp_path / "trace.xyz")
    state = runner.extract_frame(tmp_path / "trace.xyz", 1, template)
    np.testing.assert_allclose(state.positions, expected_second_positions)


def test_endpoint_rows_cost_zero_new_force_evaluations():
    row = runner.reused_endpoint_row(source_row, frame_index=0)
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_g_up0"
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/unit/test_uphill_relax_first_passage_gate.py`

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement replay by composing existing G-UP0 quench code**

The runner must:

1. verify G-UP0 and first-passage raw evidence SHA256;
2. verify every optimizer trajectory SHA256;
3. require `safe-lbfgs-total`, trajectory stride 1 and no clipped checkpoint;
4. interpret frame index as accepted optimizer step;
5. reuse frame 0 and final endpoint landings at zero new FE;
6. quench every intermediate discovery frame with the frozen true-PES protocol;
7. stop before exceeding 32,000 new FE or 900 seconds of gate kernel wall;
8. record zero direction, biased-relax and unattributed FE.

- [ ] **Step 4: Run unit tests and one intermediate-frame GPU smoke**

Run: `pytest -q tests/unit/test_uphill_relax_first-passage-gate.py`

Expected: PASS.

Run: `python runs/20260731-uphill-relax-first-passage-gate/run_gate.py --smoke-key plateau_accepted,42,D0_exact_anchor,1 --smoke-frame 1 --output-dir /tmp/g-up1-smoke`

Expected: one new true quench, exact closed ledger, no HVP or biased-relax FE.

- [ ] **Step 5: Commit the runner**

```bash
git add runs/20260731-uphill-relax-first-passage-gate/run_gate.py \
  tests/unit/test_uphill_relax_first_passage_gate.py
git commit -m "Add uphill relax first-passage runner"
```

### Task 3: Execute discovery and derive the fixed cutoff

**Files:**
- Create (ignored): `runs/20260731-uphill-relax-first-passage-gate/output/discovery.json`

- [ ] **Step 1: Run every accepted frame in the four discovery trajectories**

Run: `python runs/20260731-uphill-relax-first-passage-gate/run_gate.py --stage discovery --output-dir runs/20260731-uphill-relax-first-passage-gate/output`

Expected: 238 total frames, eight reused endpoints, all intermediate rows classified, at most 32,000 new FE.

- [ ] **Step 2: Validate exact first-passage summaries**

Run: `python runs/20260731-uphill-relax-first-passage-gate/run_gate.py --check-discovery runs/20260731-uphill-relax-first-passage-gate/output/discovery.json`

Expected: one stable-final-basin step per trajectory and one cutoff equal to their maximum.

- [ ] **Step 3: Stop if no shorter cutoff exists**

If the cutoff is not strictly below every discovery final step, record `NO_COMMON_SHORTER_CUTOFF` and do not execute holdout.

### Task 4: Execute the untouched holdout and conclude

**Files:**
- Create (ignored): `runs/20260731-uphill-relax-first-passage-gate/output/evidence.json`
- Create: `runs/20260731-uphill-relax-first-passage-gate/evidence.json`
- Create: `runs/20260731-uphill-relax-first-passage-gate/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Evaluate the frozen cutoff once on each holdout trajectory**

Run: `python runs/20260731-uphill-relax-first-passage-gate/run_gate.py --stage holdout --output-dir runs/20260731-uphill-relax-first-passage-gate/output`

Expected: no cutoff refit; each holdout reports final-basin reproduction or no headroom.

- [ ] **Step 2: Apply the strict promotion rule**

Open a fixed-cutoff full-action gate only when every holdout has positive headroom, every cutoff landing matches its final basin, all labels are certified, and the total ledger closes. Otherwise retain the current length without adding an adaptive rule.

- [ ] **Step 3: Verify evidence and tests**

Run: `python runs/20260731-uphill-relax-first-passage-gate/run_gate.py --check-evidence runs/20260731-uphill-relax-first-passage-gate/output/evidence.json`

Run: `pytest -q tests/unit/test_uphill_relax_first_passage_gate.py tests/unit/test_uphill_relax_counterfactual_gate.py tests/unit/test_current_action_first_passage.py tests/unit/test_walker_policy.py`

Expected: evidence valid and tests PASS.

- [ ] **Step 4: Commit and push the existing research branch**

```bash
git add runs/20260731-uphill-relax-first-passage-gate/evidence.json \
  runs/20260731-uphill-relax-first-passage-gate/conclusion.md \
  docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Conclude uphill relax first-passage gate"
git push pam feature/direction-continuation-ablation
```

