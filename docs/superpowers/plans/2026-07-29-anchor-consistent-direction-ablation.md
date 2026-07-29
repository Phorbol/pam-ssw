# Anchor-Consistent Direction Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the current detached block-Ritz direction with the exact
random-plus-bond anchor and a 12-HVP Lanczos solve seeded by that same anchor,
then produce a complete paired C60 GPU evidence report.

**Architecture:** Add two explicit, default-off direction-selection modes at
the existing `SoftModeOracle` seam. Generate the physical anchor before any
arm-specific intent, reuse the existing Krylov solver, and keep the entire
uphiller/quench path unchanged. A dedicated runner reuses the locked-state and
strict-evidence contracts from the completed fixed-starter experiments.

**Tech Stack:** Python, NumPy, pytest, ASE, MACE/CUDA, existing
`BudgetedCalculator`, `SurfaceWalker`, and `solve_krylov_block`.

---

### Task 1: Lock the direction-mode contracts

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`

- [ ] **Step 1: Write failing config and exact-anchor tests**

Add tests that require:

```python
assert SSWConfig(
    direction_selection_mode="exact_anchor"
).direction_selection_mode == "exact_anchor"
assert SSWConfig(
    direction_selection_mode="anchor_krylov"
).direction_selection_mode == "anchor_krylov"
```

At the oracle seam, pass a normalized projected anchor and a diagonal
calculator. Require `exact_anchor` to return that exact direction, its total
and true Rayleigh curvatures, kind `ANCHOR`, and exactly one HVP call.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_config.py::test_config_validates_direction_selection_mode \
  tests/unit/test_walker_policy.py::test_exact_anchor_mode_uses_one_curvature_hvp
```

Expected: failure because the modes and branch do not exist.

- [ ] **Step 3: Implement the minimal modes**

Extend the validated mode set and the regularized-Ritz incompatibility set.
In `SoftModeOracle.choose_direction`, route `exact_anchor` through a focused
helper that:

```python
total_hvp, true_hvp = self._candidate_directional_hvps(
    state, proposal, anchor
)
return DirectionChoice(
    direction=anchor,
    curvature=float(anchor @ total_hvp),
    true_curvature=float(anchor @ true_hvp),
    kind=DirectionCandidateKind.ANCHOR,
    candidate_count=1,
    score=None,
    diagnostics={"direction_hvp_count": 1},
)
```

Use the existing projection/normalization rules; reject a missing or unusable
anchor.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the two tests above and require both to pass.

- [ ] **Step 5: Commit**

```bash
git add pamssw/config.py pamssw/walker.py \
  tests/unit/test_config.py tests/unit/test_walker_policy.py
git commit -m "feat: add exact anchor direction mode"
```

### Task 2: Generate one common anchor and seed Lanczos from it

**Files:**
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `pamssw/walker.py`

- [ ] **Step 1: Write failing paired-anchor tests**

Add a test with a recording generator/oracle that runs the proposal-walk
bootstrap for `block_krylov`, `exact_anchor`, and `anchor_krylov`. Require:

```python
np.testing.assert_array_equal(
    observed_anchor["block_krylov"],
    observed_anchor["exact_anchor"],
)
np.testing.assert_array_equal(
    observed_anchor["exact_anchor"],
    observed_anchor["anchor_krylov"],
)
```

Also require the anchor-Krylov intent to have exactly one column equal to that
anchor and to consume `block_krylov_depth` HVPs.

- [ ] **Step 2: Run focused tests and verify RED**

Run the new tests. Expected failure: current block intents consume RNG before
anchor generation and `anchor_krylov` is not routed.

- [ ] **Step 3: Move common-anchor creation to walk bootstrap**

In `_walk_candidate_from_seed`, generate `anchor_direction` once before
arm-specific intents:

```python
anchor_direction = self.oracle.generator.generate_initial_direction(...)
```

Then construct:

```python
if mode == "block_krylov":
    krylov_intents = generator.generate_krylov_intents(...)
elif mode == "anchor_krylov":
    krylov_intents = (
        IntentBlock(basis=anchor_direction[:, None]),
    )
else:
    krylov_intents = None
```

Route both Krylov modes through the existing block solver. Do not change
softening, Gaussian bias, relaxation, or terminal-quench logic.

- [ ] **Step 4: Add explicit diagnostics**

Record `direction_hvp_count=1` for exact anchor and retain existing
`krylov_hvp_consumed`, initial-span overlap, residual, and participation ratio
for anchor-Krylov. The existing `anchor_cosine` remains the executable
alignment metric.

- [ ] **Step 5: Run focused and surrounding tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_block_krylov.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

Require zero failures.

- [ ] **Step 6: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: seed direction refinement from common anchor"
```

### Task 3: Build the exact 18-case evidence runner

**Files:**
- Create: `runs/20260729-anchor-consistent-direction-ablation/run_ablation.py`
- Create: `tests/unit/test_anchor_consistent_direction_ablation.py`

- [ ] **Step 1: Write failing runner-contract tests**

Require the exact case matrix:

```python
2 states * 3 seeds * 3 arms == 18 cases
```

Require arm configs:

```python
detached_ritz = block_krylov, blocks=1, depth=6
exact_anchor = exact_anchor
anchor_lanczos = anchor_krylov, depth=12
```

Require exact expected direction FE per selection: 24, 2, and 24.
Require complete status, strict certificate, exact starter, closed purpose
ledger, zero bootstrap/unattributed counts, and no selector/posterior fields.

- [ ] **Step 2: Run runner tests and verify RED**

Expected failure because the runner does not exist.

- [ ] **Step 3: Implement the runner**

Reuse the frozen runtime, locked C60 state registry, strict quench checks, and
source-hash verification from:

```text
runs/20260728-block-krylov-fixed-starter-escape/run_ablation.py
runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py
```

Run one terminal proposal per case. Write atomic per-case summaries,
`raw.json`, and `evidence.json`. Keep the MACE calculator shared sequentially
on CUDA; no thread scheduler is introduced in this mechanistic experiment.

- [ ] **Step 4: Verify runner contracts**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit/test_anchor_consistent_direction_ablation.py
```

Require all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add runs/20260729-anchor-consistent-direction-ablation/run_ablation.py \
  tests/unit/test_anchor_consistent_direction_ablation.py
git commit -m "exp: add anchor-consistent direction runner"
```

### Task 4: Run GPU cohort and report the mechanism

**Files:**
- Create:
  `runs/20260729-anchor-consistent-direction-ablation/output/`
- Create:
  `runs/20260729-anchor-consistent-direction-ablation/conclusion.md`

- [ ] **Step 1: Freeze and record the execution commit**

Require a clean tracked worktree and pass the exact `git rev-parse HEAD` value
through `--expected-git-commit`.

- [ ] **Step 2: Run all 18 CUDA cases**

Run with `/root/miniforge3/envs/mace_les/bin/python`,
`CUDA_VISIBLE_DEVICES=0`, unbuffered output, and a `/tmp` Matplotlib cache.
Do not start a second run while the first is active.

- [ ] **Step 3: Independently validate evidence**

Reload `raw.json`, rebuild evidence, verify all structure/output hashes, close
every purpose ledger, and check:

```text
completed_cases == 18
bootstrap_force_evaluations == 0
unattributed == 0
direction FE/selection == {24, 2, 24}
```

- [ ] **Step 4: Write the conclusion**

Report by state/arm:

- meaningful terminal outcomes;
- landing-energy changes;
- new-basin and certificate counts;
- total and purpose-resolved FE;
- wall sections;
- anchor cosine, curvature, participation ratio, and residual summaries.

Separate:

1. verified implementation facts;
2. measured three-seed outcomes;
3. physical interpretation;
4. what remains unproven.

Do not promote a default from three seeds.

- [ ] **Step 5: Run final verification**

Run focused direction/runner tests, `py_compile`, `git diff --check`, and
independent evidence validation.

- [ ] **Step 6: Commit, push, and update PR 10**

Commit the complete evidence and conclusion, push
`feature/posterior-terminal-outcome-validation`, and post the exact results and
execution caveats to PR 10.

