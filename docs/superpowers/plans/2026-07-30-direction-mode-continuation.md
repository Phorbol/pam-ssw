# Direction-Mode Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two research-only direction-continuation modes and run the preregistered C60 terminal ablation against the unchanged detached-Ritz control.

**Architecture:** Keep the existing `SurfaceWalker` and `SoftModeOracle` structure. Store the preceding selected mode separately from the relaxed displacement, choose the common detached-Ritz direction at step zero, and switch only the later-step direction input for the two experimental arms. Reuse the existing exact-direction HVP and block-Krylov implementations, and keep all reporting in a dedicated run directory.

**Tech Stack:** Python 3.12, NumPy, pytest, ASE, MACE CUDA, existing `BudgetedEvaluator` purpose ledger.

---

## File map

- Modify `pamssw/config.py`: validate the two explicit research modes.
- Modify `pamssw/walker.py`: add direction kinds, exact transported-direction evaluation, continuation Krylov input, separate selected-mode state, and zero-cost continuation diagnostics.
- Modify `tests/unit/test_config.py`: prove public config acceptance and unchanged defaults.
- Modify `tests/unit/test_walker_policy.py`: prove common step zero, HVP costs, projection, sign alignment, and state separation.
- Create `runs/20260730-direction-mode-continuation/run_ablation.py`: execute the locked C60 cohort and build closed evidence.
- Create `runs/20260730-direction-mode-continuation/analyze_ablation.py`: independently reload cases, enforce the survivor gate, and write the conclusion.
- Create `tests/unit/test_direction_mode_continuation_ablation.py`: lock the 18-case protocol and evidence semantics.
- Create `runs/20260730-direction-mode-continuation/conclusion.md`: generated scientific conclusion after C1.

### Task 1: Expose only the two approved research modes

**Files:**
- Modify: `pamssw/config.py:323-352`
- Modify: `tests/unit/test_config.py:376-440`

- [ ] **Step 1: Write failing config tests**

Add:

```python
def test_config_accepts_direction_continuation_modes_without_changing_default():
    assert SSWConfig().direction_selection_mode == "discrete"
    assert (
        SSWConfig(
            direction_selection_mode="transported_direction"
        ).direction_selection_mode
        == "transported_direction"
    )
    assert (
        SSWConfig(
            direction_selection_mode="continuation_krylov",
            block_krylov_depth=12,
        ).direction_selection_mode
        == "continuation_krylov"
    )


@pytest.mark.parametrize(
    "mode",
    ["transported_direction", "continuation_krylov"],
)
def test_continuation_modes_reject_regularized_ritz_synthesis(mode):
    with pytest.raises(ValueError, match="explicit direction_selection_mode"):
        SSWConfig(
            direction_selection_mode=mode,
            direction_synthesis_mode="regularized_ritz",
        )
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q \
  tests/unit/test_config.py::test_config_accepts_direction_continuation_modes_without_changing_default \
  tests/unit/test_config.py::test_continuation_modes_reject_regularized_ritz_synthesis
```

Expected: FAIL because both modes are rejected by `SSWConfig`.

- [ ] **Step 3: Extend the existing validation sets**

Add both strings to `direction_selection_modes` and to the explicit-mode set:

```python
direction_selection_modes = {
    "discrete",
    "rayleigh_ritz",
    "block_krylov",
    "exact_anchor",
    "anchor_krylov",
    "energy_bounded_anchor",
    "transported_direction",
    "continuation_krylov",
}
```

Update the error text to enumerate the new values. Add no configuration
fields, weights, thresholds, or fallback modes.

- [ ] **Step 4: Run config tests**

Run: `pytest -q tests/unit/test_config.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pamssw/config.py tests/unit/test_config.py
git commit -m "feat: expose direction continuation research modes"
```

### Task 2: Implement exact transported-direction evaluation

**Files:**
- Modify: `pamssw/walker.py:480-510`
- Modify: `pamssw/walker.py:1799-1835`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing oracle tests**

Add a test using a two-atom diagonal quadratic:

```python
def test_transport_direction_projects_aligns_and_spends_one_hvp():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_selection_mode="transported_direction",
            n_bond_pairs=0,
        ),
        softening_enabled=False,
    )
    previous = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    proposal = ProposalPotential(walker.calculator)
    before = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )
    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        choice = walker.oracle.choose_transported_direction(
            state, proposal, -previous, previous
        )
    after = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )
    assert after - before == 2
    assert np.dot(choice.direction, previous) > 0.0
    assert choice.diagnostics["direction_hvp_count"] == 1
    assert choice.diagnostics["continuation_source"] == "selected_mode"
```

Add a fixed-mask/degeneracy test that expects
`ContinuationDirectionDegenerate` when projection removes every component.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py \
  -k "transport_direction or continuation_projection"
```

Expected: FAIL because the exception, kind, and helper do not exist.

- [ ] **Step 3: Add the minimal direction representation**

Add:

```python
class ContinuationDirectionDegenerate(RuntimeError):
    pass


class DirectionCandidateKind(str, Enum):
    # existing values remain unchanged
    TRANSPORTED = "transported"
    CONTINUATION_RITZ = "continuation_ritz"
```

Add a `SoftModeOracle` helper:

```python
def choose_transported_direction(
    self,
    state: State,
    proposal: ProposalPotential,
    direction: np.ndarray,
    reference: np.ndarray,
) -> DirectionChoice:
    projected = project_out_rigid_body_modes(state, direction)
    projected.reshape(state.n_atoms, 3)[state.fixed_mask] = 0.0
    normalized = self._normalized_or_none(projected)
    if normalized is None:
        raise ContinuationDirectionDegenerate(
            "continuation direction vanished after projection"
        )
    if float(np.dot(normalized, reference)) < 0.0:
        normalized = -normalized
    total_hvp, true_hvp = self._candidate_directional_hvps(
        state, proposal, normalized
    )
    return DirectionChoice(
        direction=normalized,
        curvature=float(np.dot(normalized, total_hvp)),
        kind=DirectionCandidateKind.TRANSPORTED,
        candidate_count=1,
        true_curvature=float(np.dot(normalized, true_hvp)),
        diagnostics={
            "direction_hvp_count": 1,
            "continuation_source": "selected_mode",
        },
    )
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py \
  -k "transport_direction or continuation_projection"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: evaluate transported continuation directions"
```

### Task 3: Integrate selected-mode continuation into the biased walk

**Files:**
- Modify: `pamssw/walker.py:3120-3450`
- Modify: `tests/unit/test_walker_policy.py:760-930`

- [ ] **Step 1: Write a failing common-step-zero test**

Instrument `oracle.choose_direction` and `_relax_proposal_task` for three
walkers with paired RNG seeds. Assert:

```python
assert np.allclose(
    choices["fixed_intent_ritz"][0],
    choices["transported_direction"][0],
)
assert np.allclose(
    choices["fixed_intent_ritz"][0],
    choices["continuation_krylov"][0],
)
assert choice_hashes["fixed_intent_ritz"][0] == (
    choice_hashes["transported_direction"][0]
)
```

For step one assert:

```python
assert transported_hvps == 1
assert continuation_krylov_hvps == 12
assert continuation_initial_columns == [1]
assert np.allclose(
    continuation_initial_basis[:, 0],
    preceding_selected_direction,
)
```

Also assert that the normalized relaxed displacement can differ from the
stored preceding selected mode without changing the continuation basis.

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py \
  -k "continuation_walk_common_first_step or selected_mode_is_not_relaxed_displacement"
```

Expected: FAIL because the walk has no separate selected-mode state.

- [ ] **Step 3: Add separate local state and mode dispatch**

At walk initialization add:

```python
previous_relaxed_displacement: np.ndarray | None = None
previous_selected_direction: np.ndarray | None = None
```

Preserve the current variable only where discrete scoring needs relaxed
displacement continuity. Before `choose_direction`, dispatch:

```python
continuation_mode = self.config.direction_selection_mode in {
    "transported_direction",
    "continuation_krylov",
}
effective_mode = (
    "block_krylov"
    if continuation_mode and step_index == 0
    else self.config.direction_selection_mode
)
```

At step zero, pass the unchanged detached intents. At later steps:

```python
if effective_mode == "transported_direction":
    choice = self.oracle.choose_transported_direction(
        current,
        scoring_proposal,
        previous_selected_direction,
        previous_selected_direction,
    )
elif effective_mode == "continuation_krylov":
    continuation_intents = (
        IntentBlock(
            basis=self._project_continuation_direction(
                current, previous_selected_direction
            )[:, None]
        ),
    )
    choice = self.oracle._choose_block_krylov_direction(
        current,
        scoring_proposal,
        continuation_intents,
        previous_selected_direction,
        None,
        None,
    )
    choice.kind = DirectionCandidateKind.CONTINUATION_RITZ
    choice.diagnostics["continuation_source"] = "selected_mode"
else:
    # unchanged current call
```

After selection, copy the oriented selected mode:

```python
previous_selected_direction = np.asarray(
    choice.direction, dtype=float
).copy()
```

After relaxation, update only `previous_relaxed_displacement`. Do not overwrite
`previous_selected_direction`.

Handle `ContinuationDirectionDegenerate` by recording
`continuation_projection_degenerate` and terminating the current walk without
fallback.

- [ ] **Step 4: Add zero-cost continuation diagnostics**

Before recording diagnostics, populate:

```python
choice.diagnostics.update(
    self._continuation_diagnostics(
        choice.direction,
        previous_selected_direction,
        previous_relaxed_displacement,
        anchor_direction,
    )
)
```

The helper returns signed/absolute cosines and a SHA-256 hash over normalized
`<f8` vector bytes. It performs no calculator call.

- [ ] **Step 5: Run walker tests**

Run:

```bash
pytest -q \
  tests/unit/test_walker_policy.py \
  tests/unit/test_config.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: follow selected modes across biased steps"
```

### Task 4: Build the locked C60 runner and evidence contract

**Files:**
- Create: `runs/20260730-direction-mode-continuation/run_ablation.py`
- Create: `runs/20260730-direction-mode-continuation/analyze_ablation.py`
- Create: `tests/unit/test_direction_mode_continuation_ablation.py`

- [ ] **Step 1: Write failing protocol tests**

Lock:

```python
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = {
    "fixed_intent_ritz": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "transported_direction": {
        "direction_selection_mode": "transported_direction",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "continuation_lanczos": {
        "direction_selection_mode": "continuation_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}
```

Assert 18 unique cases, zero bootstrap cost, zero unattributed evaluations,
strict certificates, identical recorded step-zero hashes within every
starter-seed group, and variable post-step-zero HVP costs:

```python
assert row0["direction_hvp_count"] == 12
assert transported_later["direction_hvp_count"] == 1
assert control_later["direction_hvp_count"] == 12
assert continuation_later["direction_hvp_count"] == 12
```

- [ ] **Step 2: Run protocol tests and verify failure**

Run:

```bash
pytest -q tests/unit/test_direction_mode_continuation_ablation.py
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the runner by narrowing the existing pattern**

Copy only the reusable locked-state loading, hashing, strict-quench,
purpose-ledger, and JSON-atomic-write functions from
`runs/20260729-anchor-consistent-direction-ablation/run_ablation.py`.

The new `_validate_direction_trace()` must inspect each step rather than assume
constant HVP cost:

```python
expected_hvp = (
    12
    if index == 0 or arm != "transported_direction"
    else 1
)
if row["oracle_selection_force_evaluations_delta"] != 2 * expected_hvp:
    raise RuntimeError("direction selection cost violates arm contract")
```

Reject a cohort whose summed force evaluations exceed 10,000.

- [ ] **Step 4: Implement independent analysis and the survivor rule**

`analyze_ablation.py` reloads every case JSON and writes `evidence.json` plus
`conclusion.md`. It reports:

- strict meaningful outcomes;
- landing-energy deltas;
- new basins;
- median absolute consecutive-mode cosine excluding step zero;
- purpose-resolved force evaluations;
- invalidity, fragmentation, fallback, and certificate counts.

It emits exactly one of:

```text
no_survivor
transported_direction_survives
continuation_lanczos_survives
multiple_survivors
```

A survivor must satisfy all three preregistered conditions in the spec. The
analysis must not rank arms with a weighted scalar.

- [ ] **Step 5: Run protocol tests**

Run:

```bash
pytest -q tests/unit/test_direction_mode_continuation_ablation.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  runs/20260730-direction-mode-continuation \
  tests/unit/test_direction_mode_continuation_ablation.py
git commit -m "exp: add direction continuation C60 protocol"
```

### Task 5: Verify C0 before GPU use

**Files:**
- Modify only if a failing test exposes a scoped defect.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_direction_mode_continuation_ablation.py
```

Expected: all pass.

- [ ] **Step 2: Run the core suite**

Run:

```bash
pytest -q tests/unit \
  --ignore=tests/unit/test_gpu_production_scripts.py
```

Expected: all pass with zero failures and errors.

- [ ] **Step 3: Check source integrity**

Run:

```bash
python -m compileall -q pamssw \
  runs/20260730-direction-mode-continuation
git diff --check 8c5dd1c...HEAD
git status --short
```

Expected: compile succeeds, diff check is clean, and only intended generated
run artifacts are untracked after execution.

### Task 6: Execute and conclude Stage C1

**Files:**
- Generate: `runs/20260730-direction-mode-continuation/output/**`
- Generate: `runs/20260730-direction-mode-continuation/evidence.json`
- Generate: `runs/20260730-direction-mode-continuation/conclusion.md`

- [ ] **Step 1: Record the execution commit**

Commit any final source corrections before GPU execution. The runner must
refuse a tracked-dirty worktree.

- [ ] **Step 2: Run the 18-case CUDA cohort**

Run in the `mace_les` environment:

```bash
python runs/20260730-direction-mode-continuation/run_ablation.py \
  --output runs/20260730-direction-mode-continuation/output
```

Expected: 18/18 completed, 18 strict certificates, zero unattributed force
evaluations, and at most 10,000 total accounted evaluations.

- [ ] **Step 3: Rebuild evidence independently**

Run:

```bash
python runs/20260730-direction-mode-continuation/analyze_ablation.py \
  --output runs/20260730-direction-mode-continuation/output
```

Expected: `evidence.json` and `conclusion.md` are written and agree with every
case ledger.

- [ ] **Step 4: Apply the stopping gate**

If the decision is `no_survivor`, stop the continuation route and report that
the next scientific task is new physical event content.

If one or more arms survive, create the exact-repeat matrix containing only
the control and survivor. Do not change the algorithm or parameters.

- [ ] **Step 5: Commit the conclusion**

```bash
git add \
  runs/20260730-direction-mode-continuation/evidence.json \
  runs/20260730-direction-mode-continuation/conclusion.md
git commit -m "exp: conclude direction continuation C60 ablation"
```

### Task 7: Conditional repeat and PdO transfer

**Files:**
- Create only when Stage C1 has a survivor:
  `runs/20260730-direction-mode-continuation/repeat-output/**`
- Create only after a stable repeat:
  `runs/20260730-direction-mode-continuation/pdo-output/**`

- [ ] **Step 1: Repeat only control and survivor**

Execute the same two C60 starters and seeds with no source or configuration
change. Rebuild the evidence and require stable meaningful classification by
paired starter-seed.

- [ ] **Step 2: Stop on reversal**

If the repeat reverses the terminal advantage, write `repeat_unstable` and
stop. Do not add another seed, weight, threshold, or mixture to rescue it.

- [ ] **Step 3: Run locked PdO transfer only for a stable survivor**

Use the same two-arm protocol on two locked PdO accepted states and seeds
42/43/44. Treat PdO only as transfer validation; do not tune on it.

- [ ] **Step 4: Write the next research decision**

The final report must distinguish:

1. code and accounting facts;
2. measured C60/PdO outcomes;
3. physical interpretation;
4. unproven generalization;
5. the next bounded experiment.

Do not begin posterior learning unless two actions have stable meaningful
outcomes under matched contexts.
