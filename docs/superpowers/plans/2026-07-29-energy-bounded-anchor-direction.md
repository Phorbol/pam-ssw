# Energy-Bounded Anchor Direction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one opt-in, zero-extra-HVP anchor reconstruction constrained by the walker's existing local quadratic energy budget, then test it against anchor-Lanczos on the locked C60 terminal cohort.

**Architecture:** Put the projected secular solve in `pamssw/krylov.py`, route one new explicit direction mode through the existing anchor-seeded Krylov seam, and compute the exact `per_atom_rms/all_atoms` execution step before selection. Keep the existing lowest-Ritz path and every production default unchanged. Use a dedicated self-contained run ledger for the paired GPU experiment.

**Tech Stack:** Python 3.12, NumPy, SciPy root solving, pytest, existing budgeted MACE calculator and C60 terminal-audit runner.

---

### Task 1: Projected energy-bounded anchor algebra

**Files:**
- Modify: `pamssw/krylov.py`
- Create: `tests/unit/test_energy_bounded_anchor.py`

- [ ] **Step 1: Write failing tests for inactive, active, and infeasible constraints**

Create analytic diagonal-Hessian cases proving:

```python
selection = select_energy_bounded_anchor(
    basis=np.eye(3),
    true_products=np.diag([1.0, 4.0, 9.0]),
    anchor=np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    curvature_limit=...,
)
```

Require:

- an already-feasible anchor is returned exactly;
- an active constraint returns unit norm, positive anchor overlap, and
  curvature equal to the requested limit;
- an infeasible limit below the smallest projected eigenvalue returns the
  lowest direction and `feasible=False`;
- the active solution has no lower anchor overlap than any dense,
  independently enumerated feasible unit direction on a three-dimensional
  sphere, within the enumeration resolution.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_energy_bounded_anchor.py
```

Expected: collection fails because `select_energy_bounded_anchor` does not
exist.

- [ ] **Step 3: Implement the minimal immutable result and secular solve**

Add:

```python
@dataclass(frozen=True, eq=False)
class EnergyBoundedAnchorResult:
    direction: np.ndarray
    feasible: bool
    active: bool
    overlap: float
    true_curvature: float
    exact_anchor_curvature: float
    curvature_limit: float


def select_energy_bounded_anchor(
    basis: np.ndarray,
    true_products: np.ndarray,
    anchor: np.ndarray,
    curvature_limit: float,
) -> EnergyBoundedAnchorResult:
    ...
```

The implementation must symmetrize the projected true Hessian, solve only the
one-dimensional secular equation, return read-only owned arrays, and contain
no scientific tuning constant.  Numerical tolerances are module-level
floating-point safeguards only.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest -q tests/unit/test_energy_bounded_anchor.py
```

Expected: all tests pass.

- [ ] **Step 5: Run existing Krylov regression tests**

Run:

```bash
pytest -q \
  tests/unit/test_block_krylov.py \
  tests/unit/test_krylov_curvature_overlap_frontier.py
```

Expected: all tests pass with unchanged lowest-Ritz results and HVP counts.

- [ ] **Step 6: Commit the algebra**

```bash
git add pamssw/krylov.py tests/unit/test_energy_bounded_anchor.py
git commit -m "feat: add energy-bounded anchor projection"
```

### Task 2: Opt-in walker integration with exact budget semantics

**Files:**
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_direction_candidate_budget.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing config and walker contract tests**

Require:

```python
SSWConfig(
    direction_selection_mode="energy_bounded_anchor",
    step_length_mode="per_atom_rms",
    step_rms_scope="all_atoms",
)
```

to validate, and require the new mode to reject any other step-length or RMS
scope.  At the walker seam, use an analytic quadratic calculator and require:

- one exact random-plus-bond anchor block;
- exactly the configured 12 HVPs and 24 direction force evaluations;
- no extra HVP for the projected selection;
- the direction satisfies the quadratic energy bound;
- the recorded requested and actual execution steps agree when feasible;
- an infeasible requested step is analytically shortened so that the executed
  quadratic energy still satisfies the same bound;
- the control `anchor_krylov` direction and diagnostics remain unchanged.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

Expected: failures identify the unknown mode and missing diagnostics.

- [ ] **Step 3: Add the minimal config mode and validation**

Extend only `direction_selection_mode` with
`"energy_bounded_anchor"`.  Validate that this mode requires
`step_length_mode="per_atom_rms"` and `step_rms_scope="all_atoms"`.  Do not add
configuration fields.

- [ ] **Step 4: Route the mode through the existing anchor Krylov context**

Treat `energy_bounded_anchor` exactly like `anchor_krylov` when constructing
the common random-plus-bond anchor and one-column Krylov intent.  Do not change
random draw order.

- [ ] **Step 5: Pass the existing physical budget into direction selection**

At each walk step compute:

```python
target_rms = min(
    config.target_step_rms * sigma_scale,
    config.max_step_rms,
)
step_scale = target_rms * np.sqrt(current.n_atoms)
energy_limit = (
    config.target_uphill_energy
    if step_target is None
    else step_target
)
```

Pass those values only for the new mode.  In the oracle, invoke
`select_energy_bounded_anchor` using the already stored Krylov basis and true
Hessian products.  If the returned direction is infeasible at the requested
step, cap the execution step using the selected true curvature and the same
energy target.  Record requested and executed step diagnostics specified by
the design.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
pytest -q \
  tests/unit/test_energy_bounded_anchor.py \
  tests/unit/test_config.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

Expected: all tests pass.

- [ ] **Step 7: Run the complete direction and proposal regression slice**

Run:

```bash
pytest -q \
  tests/unit/test_bias.py \
  tests/unit/test_relax.py \
  tests/unit/test_block_krylov.py \
  tests/unit/test_krylov_curvature_overlap_frontier.py \
  tests/unit/test_anchor_consistent_direction_ablation.py \
  tests/unit/test_block_krylov_gpu_ablation.py \
  tests/unit/test_config.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit the opt-in integration**

```bash
git add pamssw/config.py pamssw/walker.py tests/unit
git commit -m "feat: add energy-bounded anchor direction mode"
```

### Task 3: Self-contained paired C60 terminal experiment

**Files:**
- Create: `runs/20260729-energy-bounded-anchor-c60/run_ablation.py`
- Create: `runs/20260729-energy-bounded-anchor-c60/output/raw.json`
- Create: `runs/20260729-energy-bounded-anchor-c60/output/evidence.json`
- Create: `runs/20260729-energy-bounded-anchor-c60/conclusion.md`
- Create: `tests/unit/test_energy_bounded_anchor_c60_ablation.py`

- [ ] **Step 1: Write the failing ledger contract**

Require the runner and evidence to contain exactly:

- two locked starter IDs;
- seeds 42, 43, and 44;
- arms `anchor_lanczos` and `energy_bounded_anchor`;
- 12 completed cases;
- one fresh strict-quench certificate per case;
- zero bootstrap, starter-quench, and unattributed evaluations;
- exact 12-HVP cost per completed direction selection;
- no production-default change.

- [ ] **Step 2: Run the ledger test and verify RED**

Run:

```bash
pytest -q tests/unit/test_energy_bounded_anchor_c60_ablation.py
```

Expected: failure because the runner and evidence do not exist.

- [ ] **Step 3: Implement the runner by reusing locked loaders**

Reuse the state/model loader, strict terminal-quench certificate, structure
hashing, and purpose accounting from:

```text
runs/20260729-krylov-curvature-overlap-frontier/run_audit.py
```

Change only the two arm definitions and add validation for the new diagnostics.
Do not copy or inherit a prior landing outcome.

- [ ] **Step 4: Run non-GPU runner validation**

Run:

```bash
python -m py_compile \
  runs/20260729-energy-bounded-anchor-c60/run_ablation.py
pytest -q tests/unit/test_energy_bounded_anchor_c60_ablation.py
```

Expected: the static runner contract passes while execution-dependent evidence
assertions remain explicitly skipped until the locked run is present.

- [ ] **Step 5: Commit and lock the execution code**

```bash
git add \
  runs/20260729-energy-bounded-anchor-c60/run_ablation.py \
  tests/unit/test_energy_bounded_anchor_c60_ablation.py
git commit -m "exp: preregister energy-bounded anchor C60 ablation"
```

Record this exact commit as `execution_commit`.

- [ ] **Step 6: Execute all 12 GPU cases**

Run in the MACE environment:

```bash
python runs/20260729-energy-bounded-anchor-c60/run_ablation.py
```

Expected: process exit code 0, 12/12 cases complete, 12 fresh strict-quench
certificates, and no fallback or unaccounted evaluations unless reported as a
failed preregistered outcome.

- [ ] **Step 7: Independently validate evidence**

Recompute:

- case-key uniqueness;
- structure SHA-256 values;
- terminal force certificates;
- per-purpose and total force sums;
- arm outcome counts;
- HVP costs;
- overlap, curvature, and quadratic-energy distributions.

Require the recomputed object to equal the stored evidence.

- [ ] **Step 8: Write the evidence-limited conclusion**

State separately:

- execution integrity;
- terminal outcomes;
- direction geometry;
- force and wall-time cost;
- numerical or scientific failures;
- promotion/rejection decision.

Do not infer superiority from overlap alone and do not tune the energy target
after seeing outcomes.

- [ ] **Step 9: Run full verification and commit evidence**

Run:

```bash
pytest -q
python -m py_compile pamssw/krylov.py pamssw/config.py pamssw/walker.py \
  runs/20260729-energy-bounded-anchor-c60/run_ablation.py
git diff --check
```

Then:

```bash
git add \
  docs/superpowers/specs/2026-07-29-energy-bounded-anchor-direction-design.md \
  docs/superpowers/plans/2026-07-29-energy-bounded-anchor-direction.md \
  runs/20260729-energy-bounded-anchor-c60 \
  tests/unit/test_energy_bounded_anchor_c60_ablation.py
git commit -m "exp: record energy-bounded anchor C60 evidence"
```

### Task 4: PR update without changing defaults

**Files:**
- Modify: PR #10 description or add one evidence comment

- [ ] **Step 1: Verify branch and remote state**

Run:

```bash
git status --short --branch
git log -5 --oneline
git diff pam/feature/posterior-terminal-outcome-validation...HEAD --stat
```

- [ ] **Step 2: Push the exact verified branch**

```bash
git push pam feature/posterior-terminal-outcome-validation
```

- [ ] **Step 3: Add one concise PR evidence update**

Report the execution commit, case count, terminal outcomes, force accounting,
and the keep/reject decision.  Explicitly state that no production default,
UCB/TS selector, or reaction-network logic changed.
