# Step-1 Direction Counterfactual Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and execute a run-local causal gate that forces every production K4 candidate at micro-step 1, giving momentum, bond, and random directions repeatable full-H8 landing labels from an identical biased prefix.

**Architecture:** Reuse the existing shared-step-0 pool and production bootstrap/config helpers. A run-local controller intercepts the first normal oracle call, which is micro-step 1 because micro-step 0 is supplied through `initial_direction_choice`; the reference arm evaluates and records one shared pool, while forced arms regenerate only the zero-FE candidate vectors, verify the prefix, restore the post-pool RNG state, and execute a stored choice. No `pamssw` production file or configuration field changes.

**Tech Stack:** Python 3.12, NumPy, ASE/MACE calculators, dataclasses, pytest, JSON/JSONL evidence, CUDA.

---

### Task 1: Freeze the pure group and repeat protocol

**Files:**
- Create: `runs/20260802-step1-direction-counterfactual/protocol.py`
- Create: `tests/unit/test_step1_direction_counterfactual.py`

- [x] **Step 1: Write failing protocol tests**

Require the ordered matrix and conservative repeat decisions:

```python
def test_group_matrix_is_three_system_three_seed_order():
    assert protocol.group_matrix() == [
        {"system": system, "seed": seed}
        for system in ("c60", "pdo", "cuo")
        for seed in (52, 53, 54)
    ]


def test_repeat_summary_rejects_unstable_best_identity():
    result = protocol.summarize_repeats(
        synthetic_rows(best_by_repeat=("momentum", "bond"))
    )
    assert result["decision"] == "STOP_NON_IDENTIFIABLE_DIRECTION_LABELS"
    assert result["unstable_best_pools"] == ["c60:52"]


def test_static_bottleneck_requires_every_system_to_pass():
    result = protocol.summarize_repeats(
        synthetic_complete_rows(
            stable_misses={"c60": 3, "pdo": 2, "cuo": 1}
        )
    )
    assert result["static_continuation_bottleneck"] is False
    assert result["posterior_gate_allowed"] is False
```

- [x] **Step 2: Run the focused test and confirm RED**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_step1_direction_counterfactual.py
```

Expected: import failure because `protocol.py` does not exist.

- [x] **Step 3: Implement the minimal pure protocol**

Define `SYSTEMS = ("c60", "pdo", "cuo")`, `SEEDS = (52, 53, 54)`, and
`REPEATS = (0, 1)`. Implement these exact public call signatures:

- `group_matrix(systems: Sequence[str] = SYSTEMS, seeds: Sequence[int] = SEEDS) -> list[dict[str, object]]`
- `family_terminal_medians(rows: Sequence[Mapping[str, object]]) -> dict[str, float]`
- `summarize_repeats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]`

`summarize_repeats` must require two identity-stable rows for every candidate,
exclude uncertified/invalid/fragmented/prefix-invalid pools, determine
repeat-stable best candidates, compute static-winner and momentum regrets, and
apply the exact source/static/posterior stop rules from the design. It must not
fit weights.

- [x] **Step 4: Run focused tests and confirm GREEN**

Expected: every test in `test_step1_direction_counterfactual.py` passes.

- [x] **Step 5: Commit the protocol**

```bash
git add runs/20260802-step1-direction-counterfactual/protocol.py \
  tests/unit/test_step1_direction_counterfactual.py
git commit -m "Add step-one direction gate protocol"
```

### Task 2: Implement shared pool evaluation and prefix certificates

**Files:**
- Create: `runs/20260802-step1-direction-counterfactual/run_gate.py`
- Create: `runs/20260802-step1-direction-counterfactual/.gitignore`
- Modify: `tests/unit/test_step1_direction_counterfactual.py`

- [x] **Step 1: Add failing hash and replay tests**

```python
def test_prefix_certificate_changes_with_positions_or_bias():
    left = runner.prefix_certificate(state, proposal, previous_direction)
    moved = runner.prefix_certificate(moved_state, proposal, previous_direction)
    changed = runner.prefix_certificate(state, changed_proposal, previous_direction)
    assert left["positions_sha256"] != moved["positions_sha256"]
    assert left["biases_sha256"] != changed["biases_sha256"]


def test_replayed_pool_requires_exact_identity_and_one_momentum():
    reference = synthetic_pool(("momentum", "bond", "bond", "random"))
    runner.validate_replayed_pool(reference, deepcopy(reference))
    with pytest.raises(RuntimeError, match="candidate identity"):
        runner.validate_replayed_pool(reference, reversed_pool(reference))
```

- [x] **Step 2: Run tests and confirm RED**

Expected: runner module or helper import failure.

- [x] **Step 3: Implement deterministic hashes and prefix validation**

Implement `direction_sha256`, `bias_sha256`, `prefix_certificate`, and
`validate_replayed_pool`. Hash normalized directions and little-endian float64
arrays. Bias hashes include center, direction, sigma, and weight in list order.
Validation requires four candidates, exactly one momentum, at least one
non-momentum, identical kind/index/hash/static-rank, identical bias hashes, and
maximum absolute position difference no larger than `1e-10 Å`.

- [x] **Step 4: Implement one native pool evaluator**

```python
def evaluate_native_pool(
    *, walker, state, proposal, previous_direction, anchor_direction,
    archive, history_gradient, continuity_weight, n_bond_pairs,
    score_sigma, score_sigma_fn, step_scale_fn,
) -> dict[str, object]:
    candidates = walker.oracle.generator.generate(
        state,
        previous_direction,
        anchor_direction=anchor_direction,
        anchor_mixing_alpha=walker.config.anchor_mixing_alpha,
        n_bond_pairs=n_bond_pairs,
    )
    # Use the existing batched central-FD HVP helper once.
    # Return DirectionChoice objects, records, ranks, prefix, and post-pool RNG.
```

Four candidates must cost exactly eight shared direction-oracle force
evaluations. Do not add a second HVP implementation.

- [x] **Step 5: Run focused and neighboring direction tests**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_step1_direction_counterfactual.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

- [x] **Step 6: Commit pool helpers**

```bash
git add runs/20260802-step1-direction-counterfactual \
  tests/unit/test_step1_direction_counterfactual.py
git commit -m "Add shared step-one direction pool runner"
```

### Task 3: Implement reference capture and forced replay

**Files:**
- Modify: `runs/20260802-step1-direction-counterfactual/run_gate.py`
- Modify: `tests/unit/test_step1_direction_counterfactual.py`

- [x] **Step 1: Add failing controller tests**

```python
def test_controller_branches_only_at_step_one():
    controller = runner.StepOneController(
        walker=fake_walker,
        original_choose=original_choose,
        reference_pool=pool,
        forced_index=2,
    )
    forced = controller(first_normal_oracle_call_args)
    later = controller(second_normal_oracle_call_args)
    assert forced.diagnostics["shared_step1_direction"] is True
    assert later is original_choice


def test_forced_controller_charges_no_step1_selection_hvp():
    controller = runner.StepOneController(reference_pool=pool, forced_index=2)
    before = counter.snapshot()
    controller(first_normal_oracle_call_args)
    after = counter.snapshot()
    assert after.direction_oracle - before.direction_oracle == 0
```

- [x] **Step 2: Run tests and confirm RED**

Expected: `StepOneController` is missing.

- [x] **Step 3: Implement `StepOneController`**

The first normal oracle call is step 1 because step 0 uses
`initial_direction_choice`. The reference controller calls
`evaluate_native_pool`; replays regenerate candidate vectors without HVPs,
validate identity, restore `post_pool_rng_state`, and return a deep copy of the
stored forced choice with `shared_step1_direction=True`. All later calls
delegate to the original production chooser.

- [x] **Step 4: Implement post-step-0 RNG replay**

Subclass `SurfaceWalker._initialize_walk_direction_context` using the existing
counterfactual pattern: regenerate/check the anchor, then restore the shared
post-step-0-pool RNG state before executing the shared step-0 choice.

- [x] **Step 5: Implement terminal outcome capture**

For every candidate/repeat, create an independent walker/archive, install the
controller, execute `_walk_candidate_from_seed` with the shared step-0 choice,
true-quench, and serialize landing energy, certificate, geometry,
fragmentation, basin identity, termination, trace, purpose ledger, and wall
time. Require `unattributed=0` and exact ledger closure. Count the reference
static-winner trajectory as repeat 0, so each pool executes exactly eight
terminal arms.

- [x] **Step 6: Run all focused neighboring tests**

Expected: step-one, direction-budget, walker-policy, action-history, and
accounting tests all pass.

- [x] **Step 7: Commit forced replay**

```bash
git add runs/20260802-step1-direction-counterfactual/run_gate.py \
  tests/unit/test_step1_direction_counterfactual.py
git commit -m "Add forced step-one direction replay"
```

### Task 4: Execute and close the bounded CUDA gate

**Files:**
- Create: `runs/20260802-step1-direction-counterfactual/evidence.json`
- Create: `runs/20260802-step1-direction-counterfactual/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [x] **Step 1: Run CUDA preflight and one C60 seed52 smoke**

Require current commit, clean tracked worktree, RTX 3060, exact input/model
hashes, H8, K4, static ranker, direction-type UCB disabled, one momentum in the
step-1 pool, four stable candidate identities, eight terminal arms,
`unattributed=0`, and exact ledger closure. Smoke output remains ignored.

- [x] **Step 2: Run the fixed production matrix**

Execute C60/PdO/CuO seeds 52--54. Stop if cumulative new FE exceeds 60,000 or
single-GPU wall time exceeds 35 minutes. Record but do not replace a pool that
terminates before step 1.

- [x] **Step 3: Derive compact evidence with zero new PES calls**

Run `protocol.summarize_repeats`; preserve raw evidence SHA256, FE purposes,
prefix validity, every per-pool metric, and the categorical decision. Explain
C60, PdO, and CuO separately.

- [x] **Step 4: Apply stop rules without tuning**

A source-dominance pass permits only one later source-only prospective gate. A
repeat-stable static-bottleneck pass permits only a separately designed
family-context posterior gate. Mixed evidence stops without source quotas,
static-weight tuning, UCB, TS, or descriptor changes.

- [x] **Step 5: Run final verification**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_step1_direction_counterfactual.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_action_history.py \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py
python -m json.tool \
  runs/20260802-step1-direction-counterfactual/evidence.json >/dev/null
git diff --check
```

Expected: zero failures, valid JSON, exact purpose closure, and no production
default change.

- [x] **Step 6: Commit, push, and update PR #14**

Commit only the run-local gate, tests, compact evidence, conclusion, and
roadmap; keep raw trajectories ignored. Push
`feature/direction-continuation-ablation` and report the categorical decision
and claim ceiling in PR #14.
