# LS Four-Operator Causal Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate fixed-geometry LS-operator curvature, prestrain-induced true-PES curvature, and their interaction on the existing nine frozen C60 tasks before permitting another LS production search.

**Architecture:** Add a run-local, pure NumPy protocol for analytic pair-potential Hessian actions and rigid-frame direction transport. A separate GPU runner will reuse the existing paper-ordered C60 cohort, freeze one candidate pool per state/seed, evaluate the four operators with only two true-PES HVP stencils, and persist compact evidence. Production `SurfaceWalker` behavior and defaults remain unchanged.

**Tech Stack:** Python 3.11, NumPy, ASE/MACE through existing `pamssw` evaluators, pytest, existing purpose-resolved `EvalCounter`.

---

## Scope and file map

- Create `runs/20260731-ls-four-operator-gate/protocol.py`: pure analytic pair Hessian, Kabsch transport, operator-row validation, and compact paired evidence.
- Create `runs/20260731-ls-four-operator-gate/preregistration.md`: frozen cohort, operators, metrics, FE ceiling, and stopping rules.
- Create `runs/20260731-ls-four-operator-gate/run_gate.py`: C60 GPU execution and resume-safe artifact writing.
- Create `runs/20260731-ls-four-operator-gate/conclusion.md`: generated only after a complete gate; no conclusion is written from partial rows.
- Create `tests/unit/test_ls_four_operator_gate.py`: formula, transport, cohort, accounting, and evidence tests.
- Modify `docs/research/2026-07-31-review-reconciled-roadmap.md`: replace direct long `oracle` versus `none` with the completed or pending four-operator gate based on actual results.

No production config, package default, `walker.py`, selector, optimizer, pair rule, or LS strength law changes in this plan.

### Task 1: Analytic pair-Hessian action

**Files:**
- Create: `runs/20260731-ls-four-operator-gate/protocol.py`
- Test: `tests/unit/test_ls_four_operator_gate.py`

- [ ] **Step 1: Write the failing radial/transverse decomposition test**

Create a two-atom `State`, a non-adaptive exponential `LocalSofteningModel`, and a normalized direction with both radial and transverse relative displacement. Assert that the protocol returns

```python
expected_radial = second_derivative * radial_projection**2
expected_transverse = first_derivative / distance * transverse_norm_sq
```

and that `total == radial + transverse`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_ls_four_operator_gate.py::test_pair_hessian_splits_radial_and_transverse_curvature
```

Expected: import or missing-function failure for `pair_operator_action`.

- [ ] **Step 3: Implement the minimal analytic action**

Implement:

```python
@dataclass(frozen=True)
class PairOperatorAction:
    hvp: np.ndarray
    radial_curvature: float
    transverse_curvature: float
    pair_expressivity_numerator: float


def pair_operator_action(
    model: LocalSofteningModel,
    state: State,
    direction: np.ndarray,
) -> PairOperatorAction:
    ...
```

For each pair use

```python
relative = u_j - u_i
radial = dot(n, relative)
transverse = relative - radial * n
h_relative = v_second * radial * n + (v_first / distance) * transverse
```

and accumulate `-h_relative` on atom `i`, `+h_relative` on atom `j`. Support the existing non-adaptive `buckingham_repulsive` and `gaussian_well` formulas; reject `adaptive_strength=True` because its nonsmooth derivative is outside this gate.

- [ ] **Step 4: Verify GREEN**

Run the focused test and expect PASS.

- [ ] **Step 5: Add a failing finite-difference HVP test**

Compare the analytic `hvp` against the central finite difference of `LocalSofteningModel.evaluate(...)[1]` for exponential and Gaussian terms at a displaced geometry.

- [ ] **Step 6: Run RED, implement missing formula coverage, then run GREEN**

Run:

```bash
pytest -q tests/unit/test_ls_four_operator_gate.py -k pair_hessian
```

Expected final result: all pair-Hessian tests PASS with absolute and relative tolerance `1e-6`.

- [ ] **Step 7: Commit**

```bash
git add runs/20260731-ls-four-operator-gate/protocol.py tests/unit/test_ls_four_operator_gate.py
git commit -m "Add analytic LS operator decomposition"
```

### Task 2: Direction transport and four-operator rows

**Files:**
- Modify: `runs/20260731-ls-four-operator-gate/protocol.py`
- Modify: `tests/unit/test_ls_four_operator_gate.py`

- [ ] **Step 1: Write a failing rigid-rotation transport test**

Generate a non-collinear reference geometry, rotate and translate it, and assert:

```python
alignment = align_prestrained_to_reference(reference, transformed)
np.testing.assert_allclose(alignment.positions_in_reference_frame, reference.positions)
np.testing.assert_allclose(
    alignment.direction_to_prestrained(direction),
    direction.reshape(-1, 3) @ known_rotation.T,
)
```

- [ ] **Step 2: Verify RED**

Run the single test and expect a missing-function failure.

- [ ] **Step 3: Implement proper-rotation Kabsch transport**

Implement one SVD, correct reflections by flipping the final singular vector, and expose both reference-to-prestrained and prestrained-to-reference direction transforms. Preserve fixed atom indices; do not introduce permutation matching.

- [ ] **Step 4: Verify GREEN**

Run the transport test and expect PASS.

- [ ] **Step 5: Write failing four-operator invariant tests**

Define a row constructor accepting true HVPs at `x0` and `xR`, analytic LS actions at both geometries, and one candidate direction. Assert:

```python
kappa_a == dot(u0, true_hvp_x0)
kappa_b == kappa_a + dot(u0, ls_hvp_x0)
kappa_c == dot(u_r, true_hvp_xr)
kappa_d == kappa_c + dot(u_r, ls_hvp_xr)
```

and `operator_effect == B-A`, `prestrain_effect == C-A`, `interaction == D-C-B+A`.

- [ ] **Step 6: Implement `build_operator_row` and verify GREEN**

The row must include radial/transverse LS curvature, pair expressivity, direction norm, and finite-value validation. It must not contain a scalar search reward.

- [ ] **Step 7: Commit**

```bash
git add runs/20260731-ls-four-operator-gate/protocol.py tests/unit/test_ls_four_operator_gate.py
git commit -m "Add LS four-operator evidence rows"
```

### Task 3: Preregister and test the fixed cohort

**Files:**
- Create: `runs/20260731-ls-four-operator-gate/preregistration.md`
- Modify: `runs/20260731-ls-four-operator-gate/protocol.py`
- Modify: `tests/unit/test_ls_four_operator_gate.py`

- [ ] **Step 1: Write the preregistration**

Freeze:

```text
system: C60
states: bootstrap, mid, late
seeds: 42, 43, 44
operators: A, B, C, D
paper initial strength: 0.1083 eV
paper xi: 0.2 * pair reference distance
candidate pool: generated once at x0, then rigid-frame transported
proposal-side LS: forbidden
```

Primary metrics are selected identity/angle, radial and transverse curvature effects, pair expressivity, first-passage class, terminal landing, and per-purpose FE. Lower curvature alone is explicitly insufficient for promotion.

- [ ] **Step 2: Write a failing evidence-cohort test**

Construct synthetic rows and require exactly `3 * 3 * K` candidate blocks with all A/B/C/D values, one shared candidate-pool hash per state/seed, closed purpose accounting, and zero unattributed FE.

- [ ] **Step 3: Verify RED, implement `build_evidence`, verify GREEN**

The evidence must report paired operator, prestrain, and interaction effects without fitting weights or declaring a winner from incomplete rows.

- [ ] **Step 4: Commit**

```bash
git add runs/20260731-ls-four-operator-gate/preregistration.md runs/20260731-ls-four-operator-gate/protocol.py tests/unit/test_ls_four_operator_gate.py
git commit -m "Preregister LS four-operator C60 gate"
```

### Task 4: Minimal GPU runner

**Files:**
- Create: `runs/20260731-ls-four-operator-gate/run_gate.py`
- Modify: `tests/unit/test_ls_four_operator_gate.py`

- [ ] **Step 1: Write failing runner protocol tests**

Test that the runner:

- reuses `STATE_FILES`, C60 model provenance, and `PAPER_INITIAL_STRENGTH_EV` from the existing paper-ordered gate;
- generates one frozen native candidate pool at `x0` per state/seed;
- uses one true-PES HVP stencil per candidate at `x0` and one at `xR`;
- adds analytic LS HVP locally for B and D;
- never puts LS into proposal relaxation;
- refuses a dirty tracked worktree, wrong execution commit, missing CUDA, incomplete cohort, or unattributed FE.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/unit/test_ls_four_operator_gate.py -k runner
```

Expected: missing runner/protocol failure.

- [ ] **Step 3: Implement `preflight`, `run`, and `analyze`**

The runner writes one summary per state/seed, plus compact `evidence.json`. It supports resume by accepting an existing case only after hashes, cohort identity, force-accounting closure, and certificate fields validate.

- [ ] **Step 4: Verify GREEN and run the focused suite**

```bash
pytest -q tests/unit/test_ls_four_operator_gate.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add runs/20260731-ls-four-operator-gate/run_gate.py tests/unit/test_ls_four_operator_gate.py
git commit -m "Add C60 LS four-operator runner"
```

### Task 5: Execute G-LS0A and decide whether G-LS0B opens

**Files:**
- Create after complete execution: `runs/20260731-ls-four-operator-gate/conclusion.md`
- Modify after complete execution: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Run preflight on the exact committed tree**

```bash
python runs/20260731-ls-four-operator-gate/run_gate.py preflight --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: CUDA/model/state/hash checks pass and tracked worktree is clean.

- [ ] **Step 2: Execute G-LS0A**

```bash
python runs/20260731-ls-four-operator-gate/run_gate.py run --expected-git-commit "$(git rev-parse HEAD)" --output runs/20260731-ls-four-operator-gate/output
```

Stop immediately if the FE ledger does not close or if any A/B or C/D pair uses different true-PES stencils.

- [ ] **Step 3: Analyze without post-hoc thresholds**

```bash
python runs/20260731-ls-four-operator-gate/run_gate.py analyze --output runs/20260731-ls-four-operator-gate/output
```

Open G-LS0B only if B, C, or D changes candidate ranking/selected identity in a repeatable direction. If only scalar curvature changes, close LS without a full action gate.

- [ ] **Step 4: Write the evidence-bounded conclusion and roadmap update**

State separately: verified formula, observed operator effect, observed prestrain effect, action-ranking effect, FE cost, and unproven terminal-search effect. Do not call paper-ordered LS production-positive.

- [ ] **Step 5: Run verification**

```bash
pytest -q tests/unit/test_ls_four_operator_gate.py tests/unit/test_softening.py tests/unit/test_walker_policy.py -k "soften or four_operator or paper_ordered"
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add runs/20260731-ls-four-operator-gate/conclusion.md docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Conclude LS four-operator mechanism gate"
```

## Stop boundary

This plan ends after G-LS0A and its decision. It does not automatically implement faithful `P_LS=Y`, CB-ZFLS, local compliance sampling, posterior allocation, a new starter selector, or a production default. Those require the mechanism result produced here.
