# Uphill Relax Counterfactual Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether biased-PES proposal relaxation changes the basin reached beyond the already chosen explicit SSW displacement, without changing the starter, direction, step scale, true quench, or production code.

**Architecture:** Replay the first frame of each recorded C60 proposal-relax trajectory as the exact pre-relax explicit displacement. Quench that state on the true PES and compare it with the already recorded post-relax checkpoint landing. The source trajectories and relaxed-arm evidence are immutable inputs verified by SHA256; only the explicit-arm quenches consume new force evaluations.

**Tech Stack:** Python 3.12, NumPy, ASE extxyz I/O, existing `pamssw` evaluator/quench/matcher code, pytest.

---

## File structure

- Create `runs/20260731-uphill-relax-counterfactual-gate/protocol.py`: frozen cohort, causal labels, ledger validation, and repeated-context decision logic.
- Create `runs/20260731-uphill-relax-counterfactual-gate/run_gate.py`: source-evidence validation, frame-zero replay, true-PES quench, landing comparison, and evidence writing.
- Create `runs/20260731-uphill-relax-counterfactual-gate/PLAN.md`: preregistered physical question, cohort, budget, and stopping rules.
- Create `runs/20260731-uphill-relax-counterfactual-gate/.gitignore`: ignore generated output only.
- Create `tests/unit/test_uphill_relax_counterfactual_gate.py`: pure protocol and replay-contract tests.
- Modify `docs/research/2026-07-31-review-reconciled-roadmap.md`: add the result only after the evidence closes.

### Task 1: Freeze the causal protocol

**Files:**
- Create: `tests/unit/test_uphill_relax_counterfactual_gate.py`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/protocol.py`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/PLAN.md`

- [ ] **Step 1: Write failing tests for the exact cohort and labels**

```python
def test_pair_specs_use_only_c60_horizons_1_2_4_with_recorded_relaxation():
    specs = protocol.pair_specs(source_cases)
    assert len(specs) == 34
    assert {spec["system"] for spec in specs} == {"c60"}
    assert {spec["horizon"] for spec in specs} <= {1, 2, 4}


def test_repeated_relaxed_only_escape_retains_relaxation():
    result = protocol.decide(rows_with_two_of_three_relaxed_only)
    assert result["decision"] == "RETAIN_RELAXATION_CAUSAL_SIGNAL"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py`

Expected: FAIL because `protocol.py` does not exist.

- [ ] **Step 3: Implement only the frozen cohort, labels, and decision rules**

```python
SYSTEMS = ("c60",)
HORIZONS = (1, 2, 4)
SEEDS = (42, 43, 44)


def decide(rows):
    # Exact redundancy requires every learnable pair to reproduce the same
    # landing. A repeated 2/3 relaxed-only escape in one frozen context is a
    # causal retention signal. All other mixtures change no default.
    learnable = [row for row in rows if row["causal_outcome"] != "UNLEARNABLE"]
    repeated = repeated_contexts(
        learnable,
        outcome="RELAXED_ONLY_ESCAPE",
        required_seed_count=2,
        required_context_size=3,
    )
    if repeated:
        return {
            "decision": "RETAIN_RELAXATION_CAUSAL_SIGNAL",
            "repeated_relaxed_only_contexts": repeated,
        }
    if len(learnable) == len(rows) and all(
        row["causal_outcome"]
        in {"BOTH_RETURN_STARTER", "SAME_ESCAPED_LANDING"}
        for row in learnable
    ):
        return {
            "decision": "EXACT_REDUNDANCY_SIGNAL",
            "repeated_relaxed_only_contexts": [],
        }
    return {
        "decision": "MIXED_NO_DEFAULT_CHANGE",
        "repeated_relaxed_only_contexts": [],
    }
```

- [ ] **Step 4: Run the protocol tests and verify GREEN**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py`

Expected: PASS.

- [ ] **Step 5: Commit the preregistered protocol**

```bash
git add docs/superpowers/plans/2026-07-31-uphill-relax-counterfactual-gate.md \
  runs/20260731-uphill-relax-counterfactual-gate/PLAN.md \
  runs/20260731-uphill-relax-counterfactual-gate/protocol.py \
  tests/unit/test_uphill_relax_counterfactual_gate.py
git commit -m "Add uphill relax counterfactual protocol"
```

### Task 2: Implement the immutable replay runner

**Files:**
- Modify: `tests/unit/test_uphill_relax_counterfactual_gate.py`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/run_gate.py`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/.gitignore`

- [ ] **Step 1: Write failing tests for source hashes and frame-zero extraction**

```python
def test_extract_explicit_state_returns_first_optimizer_frame(tmp_path):
    write_two_frame_xyz(tmp_path / "trace.xyz", first, second)
    replayed = runner.extract_explicit_state(tmp_path / "trace.xyz", template)
    np.testing.assert_allclose(replayed.positions, first.positions)


def test_source_checkpoint_hash_drift_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="SHA256"):
        runner.verify_source_file(tmp_path / "trace.xyz", "wrong")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py`

Expected: FAIL because `run_gate.py` does not exist.

- [ ] **Step 3: Implement replay without changing `pamssw/`**

```python
def extract_explicit_state(path, template):
    # Every Relaxer backend records task.initial_state before its first
    # optimization step; frame zero is therefore x_k + sigma_k u_k.
    return state_from_atoms(read(path, index=0), template)
```

The runner must validate the source evidence hash, every checkpoint/raw-trajectory hash, the fixed C60 cohort, zero unattributed evaluations, and the 5,000-FE additional budget. It must use the source run's effective quench configuration and existing structure matcher.

- [ ] **Step 4: Run unit tests and a one-pair smoke**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py`

Expected: PASS.

Run: `python runs/20260731-uphill-relax-counterfactual-gate/run_gate.py --limit 1 --output-dir /tmp/g-up0-smoke`

Expected: one completed pair, a closed purpose ledger, and zero unattributed force evaluations.

- [ ] **Step 5: Commit the runner**

```bash
git add runs/20260731-uphill-relax-counterfactual-gate tests/unit/test_uphill_relax_counterfactual_gate.py
git commit -m "Add uphill relax counterfactual runner"
```

### Task 3: Run the fixed GPU gate and interpret only registered outcomes

**Files:**
- Create (ignored): `runs/20260731-uphill-relax-counterfactual-gate/output/evidence.json`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/evidence.json`
- Create: `runs/20260731-uphill-relax-counterfactual-gate/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Run all 34 replay pairs**

Run: `python runs/20260731-uphill-relax-counterfactual-gate/run_gate.py --output-dir runs/20260731-uphill-relax-counterfactual-gate/output`

Expected: 34 completed pairs, at most 5,000 new force evaluations, zero unattributed evaluations.

- [ ] **Step 2: Verify evidence integrity**

Run: `python runs/20260731-uphill-relax-counterfactual-gate/run_gate.py --check-evidence runs/20260731-uphill-relax-counterfactual-gate/output/evidence.json`

Expected: source hashes, cohort, labels, and ledger all close.

- [ ] **Step 3: Write the narrow conclusion**

The conclusion must report exact counts for same landing, relaxed-only escape, explicit-only escape, different escaped landings, invalidity, new FE, reused source FE, and wall time. It must not promote an online kick policy because later explicit-only states are counterfactual branches from the original relaxed path.

- [ ] **Step 4: Run focused and broader regression tests**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py tests/unit/test_current_action_first_passage.py tests/unit/test_walker_policy.py`

Expected: PASS.

- [ ] **Step 5: Commit evidence and roadmap update**

```bash
git add runs/20260731-uphill-relax-counterfactual-gate/evidence.json \
  runs/20260731-uphill-relax-counterfactual-gate/conclusion.md \
  docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Conclude uphill relax counterfactual gate"
```

### Task 4: Verify and publish the branch update

**Files:**
- No new files.

- [ ] **Step 1: Verify tracked-tree scope**

Run: `git status --short && git diff --check`

Expected: only preserved ignored/untracked historical outputs remain; no whitespace errors.

- [ ] **Step 2: Run the final focused suite**

Run: `pytest -q tests/unit/test_uphill_relax_counterfactual_gate.py tests/unit/test_current_action_first_passage.py tests/unit/test_ls_four_operator_gate.py`

Expected: PASS.

- [ ] **Step 3: Push the existing research branch**

Run: `git push pam feature/direction-continuation-ablation`

Expected: branch fast-forwards and PR #14 is updated.
