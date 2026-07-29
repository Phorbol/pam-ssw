# Block Krylov Direction Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, exactly accounted block Krylov--Ritz direction oracle and determine whether fixed HVP budget is better spent on direction breadth or refinement depth.

**Architecture:** Put the small-matrix Krylov/Ritz algebra in one independent NumPy module. Keep random/pair intent generation and the existing `DirectionChoice` integration in `walker.py`, with the current discrete path untouched. Validate the algebra and HVP ledger before running fixed-state and paired GPU experiments.

**Tech Stack:** Python 3.11, NumPy, pytest, existing PAM-SSW analytic calculators, MACE calculator, existing `EvalCounter` purpose ledger.

---

## Scope and file map

This plan modifies only:

- Create `pamssw/krylov.py`: immutable intent/result records and pure block
  Krylov--Ritz linear algebra.
- Modify `pamssw/config.py`: opt-in mode and two integer allocation controls.
- Modify `pamssw/walker.py`: uniform pair-intent generation, oracle dispatch,
  `DirectionChoice` diagnostics, and JSONL diagnostics.
- Create `tests/unit/test_block_krylov.py`: exact quadratic algebra tests.
- Modify `tests/unit/test_config.py`: configuration validation.
- Modify `tests/unit/test_walker_policy.py`: oracle integration and no-scorer
  contract.
- Modify `tests/unit/test_direction_candidate_budget.py`: exact central-HVP
  accounting.
- Create `runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py`:
  analytic and stored-state direction-only evidence.
- Create `runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py`:
  fail-closed evidence projection.
- Create `runs/20260728-block-krylov-direction-audit/README.md`: exact commands
  and claim ceiling.
- Create `runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py`:
  paired fixed-total-FE C60 survivor gate and conditional PdO transfer gate.
- Create `runs/20260728-block-krylov-direction-gpu-ablation/analyze_evidence.py`:
  configuration-diff, budget, and survivor assertions.

Do not modify the starter-policy, posterior, optimizer, archive, accounting,
executor, or calculator-adapter modules.

## Task 1: Pure block Krylov--Ritz algebra

**Files:**

- Create: `pamssw/krylov.py`
- Create: `tests/unit/test_block_krylov.py`

- [ ] **Step 1: Write the failing exact-eigenpair test**

```python
# tests/unit/test_block_krylov.py
import numpy as np
import pytest

from pamssw.krylov import IntentBlock, solve_krylov_block


def test_depth_two_recovers_lowest_mode_in_seeded_krylov_space():
    hessian = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    calls = []

    def hvp(vector):
        calls.append(vector.copy())
        product = hessian @ vector
        return product, product

    intent = IntentBlock(
        basis=np.array([[1.0], [0.0], [0.0]]),
        pair=None,
    )
    result = solve_krylov_block(intent, hvp, depth=2)

    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    assert abs(float(result.direction @ expected)) == pytest.approx(1.0)
    assert result.curvature == pytest.approx(1.0)
    assert result.true_curvature == pytest.approx(1.0)
    assert result.residual_norm == pytest.approx(0.0, abs=1e-12)
    assert result.hvp_count == 2
    assert len(calls) == 2
```

- [ ] **Step 2: Run the test and verify the module is absent**

Run:

```bash
pytest -q tests/unit/test_block_krylov.py::test_depth_two_recovers_lowest_mode_in_seeded_krylov_space
```

Expected: failure during collection with
`ModuleNotFoundError: No module named 'pamssw.krylov'`.

- [ ] **Step 3: Add immutable records and orthogonalization helpers**

```python
# pamssw/krylov.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


Hvp = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, eq=False)
class IntentBlock:
    basis: np.ndarray
    pair: tuple[int, int] | None

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        if basis.ndim != 2 or basis.shape[1] < 1:
            raise ValueError("basis must be a non-empty 2D column matrix")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis must be finite")
        basis = basis.copy()
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        if self.pair is not None:
            atom_i, atom_j = self.pair
            if atom_i < 0 or atom_j < 0 or atom_i == atom_j:
                raise ValueError("pair must contain two distinct non-negative indices")


@dataclass(frozen=True, eq=False)
class KrylovResult:
    direction: np.ndarray
    curvature: float
    true_curvature: float
    residual_norm: float
    initial_span_overlap: float
    antisymmetry: float
    dimension: int
    initial_rank: int
    hvp_count: int
    termination_reason: str

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=float)
        if direction.ndim != 1 or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite vector")
        direction = direction.copy()
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)


def _orthonormal_columns(
    vectors: np.ndarray,
    *,
    against: np.ndarray | None = None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    accepted: list[np.ndarray] = []
    fixed = None if against is None else np.asarray(against, dtype=float)
    for column in vectors.T:
        candidate = column.copy()
        for _ in range(2):
            if fixed is not None and fixed.size:
                candidate -= fixed @ (fixed.T @ candidate)
            if accepted:
                local = np.column_stack(accepted)
                candidate -= local @ (local.T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > tolerance:
            accepted.append(candidate / norm)
    if not accepted:
        return np.empty((vectors.shape[0], 0), dtype=float)
    return np.column_stack(accepted)
```

- [ ] **Step 4: Implement the fixed-depth Krylov solve**

```python
# append to pamssw/krylov.py
def solve_krylov_block(
    intent: IntentBlock,
    hvp: Hvp,
    *,
    depth: int,
) -> KrylovResult:
    if depth <= 0:
        raise ValueError("depth must be positive")
    initial = _orthonormal_columns(intent.basis)
    if initial.shape[1] == 0:
        raise ValueError("intent basis has zero numerical rank")

    basis = initial.copy()
    frontier = initial
    total_products: list[np.ndarray] = []
    true_products: list[np.ndarray] = []
    evaluated_columns = 0
    termination = "depth_reached"

    for level in range(depth):
        frontier_total: list[np.ndarray] = []
        for column in frontier.T:
            total_hvp, true_hvp = hvp(column)
            total_hvp = np.asarray(total_hvp, dtype=float)
            true_hvp = np.asarray(true_hvp, dtype=float)
            if total_hvp.shape != column.shape or true_hvp.shape != column.shape:
                raise ValueError("HVP outputs must match the input vector shape")
            if not np.all(np.isfinite(total_hvp)) or not np.all(np.isfinite(true_hvp)):
                raise ValueError("HVP outputs must be finite")
            total_products.append(total_hvp)
            true_products.append(true_hvp)
            frontier_total.append(total_hvp)
            evaluated_columns += 1

        if level + 1 == depth:
            break
        next_frontier = _orthonormal_columns(
            np.column_stack(frontier_total),
            against=basis,
        )
        if next_frontier.shape[1] == 0:
            termination = "krylov_breakdown"
            break
        basis = np.column_stack([basis, next_frontier])
        frontier = next_frontier

    total_matrix = np.column_stack(total_products)
    true_matrix = np.column_stack(true_products)
    if total_matrix.shape[1] != basis.shape[1]:
        raise RuntimeError("every retained Krylov basis vector must have one HVP")

    projected_raw = basis.T @ total_matrix
    denominator = max(1.0, float(np.linalg.norm(projected_raw)))
    antisymmetry = float(np.linalg.norm(projected_raw - projected_raw.T) / denominator)
    projected = 0.5 * (projected_raw + projected_raw.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    index = int(np.argmin(eigenvalues))
    coefficients = eigenvectors[:, index]
    direction = basis @ coefficients
    direction /= np.linalg.norm(direction)

    overlaps = initial.T @ direction
    pivot = int(np.argmax(np.abs(overlaps)))
    if overlaps[pivot] < 0.0:
        direction = -direction
        coefficients = -coefficients

    curvature = float(direction @ (total_matrix @ coefficients))
    true_curvature = float(direction @ (true_matrix @ coefficients))
    residual = total_matrix @ coefficients - curvature * direction
    initial_overlap = float(np.linalg.norm(initial.T @ direction))

    return KrylovResult(
        direction=direction,
        curvature=curvature,
        true_curvature=true_curvature,
        residual_norm=float(np.linalg.norm(residual)),
        initial_span_overlap=initial_overlap,
        antisymmetry=antisymmetry,
        dimension=int(basis.shape[1]),
        initial_rank=int(initial.shape[1]),
        hvp_count=evaluated_columns,
        termination_reason=termination,
    )
```

- [ ] **Step 5: Run the exact test**

Run:

```bash
pytest -q tests/unit/test_block_krylov.py::test_depth_two_recovers_lowest_mode_in_seeded_krylov_space
```

Expected: `1 passed`.

- [ ] **Step 6: Add breakdown, rank-deficiency, true-HVP, and monotonicity tests**

```python
# append to tests/unit/test_block_krylov.py
def test_rank_deficient_initial_block_is_reduced_without_resampling():
    hessian = np.diag([1.0, 2.0, 3.0])
    intent = IntentBlock(
        basis=np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
        pair=(0, 1),
    )
    result = solve_krylov_block(
        intent,
        lambda vector: (hessian @ vector, hessian @ vector),
        depth=3,
    )
    assert result.initial_rank == 1
    assert result.hvp_count == 1
    assert result.termination_reason == "krylov_breakdown"


def test_true_curvature_uses_true_hvps_without_an_extra_call():
    total = np.diag([1.0, 3.0])
    true = np.diag([4.0, 9.0])
    calls = 0

    def hvp(vector):
        nonlocal calls
        calls += 1
        return total @ vector, true @ vector

    result = solve_krylov_block(
        IntentBlock(np.eye(2), pair=(0, 1)),
        hvp,
        depth=1,
    )
    assert result.curvature == pytest.approx(1.0)
    assert result.true_curvature == pytest.approx(4.0)
    assert result.hvp_count == calls == 2


def test_ritz_curvature_is_nonincreasing_with_depth():
    hessian = np.array(
        [[3.0, -1.0, 0.2], [-1.0, 2.0, -0.5], [0.2, -0.5, 1.0]]
    )
    intent = IntentBlock(np.array([[1.0], [0.0], [0.0]]), pair=None)
    curvatures = [
        solve_krylov_block(
            intent,
            lambda vector: (hessian @ vector, hessian @ vector),
            depth=depth,
        ).curvature
        for depth in (1, 2, 3)
    ]
    assert curvatures[1] <= curvatures[0] + 1e-12
    assert curvatures[2] <= curvatures[1] + 1e-12
```

- [ ] **Step 7: Run the algebra suite**

Run:

```bash
pytest -q tests/unit/test_block_krylov.py
```

Expected: `4 passed`.

- [ ] **Step 8: Commit the pure algebra**

```bash
git add pamssw/krylov.py tests/unit/test_block_krylov.py
git commit -m "feat: add budgeted block Krylov Ritz solver"
```

## Task 2: Intent-block generation without a mixing weight

**Files:**

- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write the failing intent-generation test**

```python
# append to tests/unit/test_walker_policy.py
def test_krylov_intents_keep_random_and_pair_axes_as_separate_columns():
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]]
        ),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(7),
        n_random=0,
    )

    intents = generator.generate_krylov_intents(state, n_blocks=2)

    assert len(intents) == 2
    assert all(intent.basis.shape == (9, 2) for intent in intents)
    assert all(intent.pair is not None for intent in intents)
    for intent in intents:
        gram = intent.basis.T @ intent.basis
        assert np.allclose(gram, np.eye(2), atol=1e-12)
```

- [ ] **Step 2: Run it and verify the method is absent**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py::test_krylov_intents_keep_random_and_pair_axes_as_separate_columns
```

Expected: failure with
`AttributeError: 'CandidateDirectionGenerator' object has no attribute 'generate_krylov_intents'`.

- [ ] **Step 3: Import the intent type and implement uniform movable-pair blocks**

```python
# pamssw/walker.py imports
from .krylov import IntentBlock, KrylovResult, solve_krylov_block
```

```python
# CandidateDirectionGenerator
def generate_krylov_intents(
    self,
    state: State,
    *,
    n_blocks: int,
) -> tuple[IntentBlock, ...]:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    coordinates = CartesianCoordinates.from_state(state)
    movable = np.flatnonzero(state.movable_mask)
    all_pairs = [
        (int(movable[left]), int(movable[right]))
        for left in range(len(movable))
        for right in range(left + 1, len(movable))
    ]
    pair_order = (
        self.rng.permutation(len(all_pairs)).tolist()
        if all_pairs
        else []
    )
    intents: list[IntentBlock] = []
    for block_index in range(n_blocks):
        active = self._random_active_direction(state, coordinates)
        random_full = coordinates.full_tangent_from_active(active).values
        random_axis = self._candidate(
            state,
            DirectionCandidateKind.RANDOM,
            random_full,
        ).direction
        columns = [random_axis]
        pair = None
        if block_index < len(pair_order):
            pair = all_pairs[int(pair_order[block_index])]
            pair_axis = self._pair_direction(
                state,
                pair[0],
                pair[1],
                sign=1.0,
            )
            if pair_axis is not None:
                pair_axis = self._candidate(
                    state,
                    DirectionCandidateKind.BOND,
                    pair_axis,
                ).direction
                pair_axis -= random_axis * float(random_axis @ pair_axis)
                pair_norm = float(np.linalg.norm(pair_axis))
                if pair_norm > 1e-12:
                    columns.append(pair_axis / pair_norm)
                else:
                    pair = None
        intents.append(IntentBlock(np.column_stack(columns), pair=pair))
    return tuple(intents)
```

The pair is sampled uniformly from all unordered movable-atom pairs. Do not
call `_random_non_neighbor_pairs`; its threshold and closest-pair fallback are
not part of this experiment.

- [ ] **Step 4: Run the intent test**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py::test_krylov_intents_keep_random_and_pair_axes_as_separate_columns
```

Expected: `1 passed`.

- [ ] **Step 5: Add the rank-one fallback test**

```python
def test_krylov_intent_is_rank_one_when_no_movable_pair_exists():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        fixed_mask=np.array([False, True]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(3),
        n_random=0,
    )

    [intent] = generator.generate_krylov_intents(state, n_blocks=1)

    assert intent.pair is None
    assert intent.basis.shape == (6, 1)
    assert np.linalg.norm(intent.basis[:, 0]) == pytest.approx(1.0)
```

- [ ] **Step 6: Run both intent tests**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py -k "krylov_intent"
```

Expected: `2 passed`.

- [ ] **Step 7: Commit intent generation**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: generate separate random pair Krylov intents"
```

## Task 3: Opt-in oracle configuration and dispatch

**Files:**

- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `tests/unit/test_direction_candidate_budget.py`

- [ ] **Step 1: Write failing configuration tests**

```python
# append to tests/unit/test_config.py
def test_block_krylov_config_is_opt_in_and_positive():
    default = SSWConfig()
    assert default.direction_selection_mode == "discrete"
    assert default.block_krylov_blocks == 2
    assert default.block_krylov_depth == 3

    configured = SSWConfig(
        direction_selection_mode="block_krylov",
        block_krylov_blocks=3,
        block_krylov_depth=2,
    )
    assert configured.block_krylov_blocks == 3
    assert configured.block_krylov_depth == 2

    with pytest.raises(ValueError, match="block_krylov_blocks"):
        SSWConfig(block_krylov_blocks=0)
    with pytest.raises(ValueError, match="block_krylov_depth"):
        SSWConfig(block_krylov_depth=0)


def test_block_krylov_rejects_regularized_ritz_synthesis():
    with pytest.raises(ValueError, match="cannot be combined"):
        SSWConfig(
            direction_selection_mode="block_krylov",
            direction_synthesis_mode="regularized_ritz",
        )
```

- [ ] **Step 2: Run the configuration tests and verify failure**

Run:

```bash
pytest -q tests/unit/test_config.py -k "block_krylov"
```

Expected: failures because the fields and mode are not defined.

- [ ] **Step 3: Add two inactive allocation controls**

```python
# SSWConfig fields, next to direction_selection_mode
direction_selection_mode: str = "discrete"
block_krylov_blocks: int = 2
block_krylov_depth: int = 3
```

```python
# SSWConfig.__post_init__
if self.direction_selection_mode not in {
    "discrete",
    "rayleigh_ritz",
    "block_krylov",
}:
    raise ValueError(
        "direction_selection_mode must be discrete, rayleigh_ritz, or block_krylov"
    )
for name in ("block_krylov_blocks", "block_krylov_depth"):
    value = getattr(self, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
if (
    self.direction_selection_mode in {"rayleigh_ritz", "block_krylov"}
    and self.direction_synthesis_mode == "regularized_ritz"
):
    raise ValueError(
        f"direction_selection_mode {self.direction_selection_mode} "
        "cannot be combined with regularized_ritz synthesis"
    )
```

- [ ] **Step 4: Run the configuration tests**

Run:

```bash
pytest -q tests/unit/test_config.py -k "direction_selection_mode or block_krylov"
```

Expected: all selected tests pass.

- [ ] **Step 5: Write a failing no-scorer oracle test**

```python
# append to tests/unit/test_walker_policy.py
def test_block_krylov_selects_lowest_ritz_block_without_direction_scorer(monkeypatch):
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    calculator = AnalyticCalculator(Quadratic())
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
        block_krylov_depth=1,
    )
    intents = (
        IntentBlock(np.array([[1.0], [0.0], [0.0]]), pair=None),
        IntentBlock(np.array([[0.0], [1.0], [0.0]]), pair=None),
    )
    monkeypatch.setattr(
        oracle.scorer,
        "score_candidate",
        lambda *args, **kwargs: pytest.fail("block Krylov must bypass DirectionScorer"),
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        krylov_intents=intents,
    )

    assert choice.kind is DirectionCandidateKind.BLOCK_RITZ
    assert choice.score is None
    assert choice.candidate_count == 2
```

- [ ] **Step 6: Extend the public records and oracle constructor**

```python
# DirectionCandidateKind
BLOCK_RITZ = "block_ritz"
```

```python
# DirectionChoice
diagnostics: dict[str, object] = field(default_factory=dict)
```

```python
# SoftModeOracle.__init__ arguments
block_krylov_depth: int = 3,
```

```python
# SoftModeOracle.__init__ body
self.block_krylov_depth = block_krylov_depth
```

```python
# SoftModeOracle.choose_direction argument
krylov_intents: tuple[IntentBlock, ...] | None = None,
```

- [ ] **Step 7: Implement the early block-Krylov dispatch**

Add this at the start of `SoftModeOracle.choose_direction`, before native
candidate generation:

```python
if self.direction_selection_mode == "block_krylov":
    if not krylov_intents:
        raise ValueError("block_krylov requires at least one intent block")
    block_results: list[KrylovResult] = []
    for intent in krylov_intents:
        block_results.append(
            solve_krylov_block(
                intent,
                lambda direction: self._candidate_directional_hvps(
                    state,
                    proposal,
                    direction,
                ),
                depth=self.block_krylov_depth,
            )
        )
    selected_index = min(
        range(len(block_results)),
        key=lambda index: block_results[index].curvature,
    )
    selected = block_results[selected_index]
    atom_amplitudes = np.sum(selected.direction.reshape(-1, 3) ** 2, axis=1)
    atom_amplitudes /= max(float(np.sum(atom_amplitudes)), 1e-30)
    participation_ratio = 1.0 / max(
        float(np.sum(atom_amplitudes**2)),
        1e-30,
    )
    return DirectionChoice(
        direction=selected.direction,
        curvature=selected.curvature,
        true_curvature=selected.true_curvature,
        kind=DirectionCandidateKind.BLOCK_RITZ,
        candidate_count=len(block_results),
        score=None,
        diagnostics={
            "krylov_blocks": len(block_results),
            "krylov_selected_block": selected_index,
            "krylov_hvp_count": sum(result.hvp_count for result in block_results),
            "krylov_dimensions": [result.dimension for result in block_results],
            "krylov_initial_ranks": [result.initial_rank for result in block_results],
            "krylov_residual_norm": selected.residual_norm,
            "krylov_initial_span_overlap": selected.initial_span_overlap,
            "krylov_antisymmetry": selected.antisymmetry,
            "krylov_termination": selected.termination_reason,
            "direction_participation_ratio": participation_ratio,
        },
    )
```

- [ ] **Step 8: Pass the new constructor setting from `SurfaceWalker`**

In the existing `SoftModeOracle(...)` construction, add:

```python
block_krylov_depth=config.block_krylov_depth,
```

- [ ] **Step 9: Run the no-scorer test**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py::test_block_krylov_selects_lowest_ritz_block_without_direction_scorer
```

Expected: `1 passed`.

- [ ] **Step 10: Add the exact central-HVP budget test**

```python
# append to tests/unit/test_direction_candidate_budget.py
def test_block_krylov_two_blocks_depth_three_costs_twelve_central_force_calls():
    class CoupledQuadratic:
        hessian = np.array(
            [
                [4.0, 1.0, 0.2],
                [1.0, 3.0, 0.4],
                [0.2, 0.4, 2.0],
            ]
        )

        def energy_gradient(self, flat_positions, state):
            positions = np.asarray(flat_positions, dtype=float)
            gradient = self.hessian @ positions
            return 0.5 * float(positions @ gradient), gradient

    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    calculator = EvalCounter(AnalyticCalculator(CoupledQuadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
        block_krylov_depth=3,
    )
    intents = (
        IntentBlock(np.array([[1.0], [0.0], [0.0]]), pair=None),
        IntentBlock(np.array([[0.0], [1.0], [0.0]]), pair=None),
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        krylov_intents=intents,
    )

    assert choice.diagnostics["krylov_hvp_count"] == 6
    assert calculator.force_evaluations == 12
```

- [ ] **Step 11: Run focused integration tests**

Run:

```bash
pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_block_krylov.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py -k "block_krylov or krylov_intent"
```

Expected: all selected tests pass.

- [ ] **Step 12: Commit opt-in oracle dispatch**

```bash
git add pamssw/config.py pamssw/walker.py \
  tests/unit/test_config.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_direction_candidate_budget.py
git commit -m "feat: add opt-in block Krylov direction mode"
```

## Task 4: Hold intent blocks fixed for one walk and persist diagnostics

**Files:**

- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write a failing walk-lifetime test**

```python
def test_block_krylov_intents_are_generated_once_per_walk(monkeypatch):
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_selection_mode="block_krylov",
            block_krylov_blocks=2,
            block_krylov_depth=1,
            max_steps_per_walk=2,
            proposal_relax_steps=0,
            n_bond_pairs=0,
            rng_seed=0,
        ),
        softening_enabled=False,
    )
    generated = []
    intents = (
        IntentBlock(np.array([[1.0], [0.0], [0.0]]), pair=None),
        IntentBlock(np.array([[0.0], [1.0], [0.0]]), pair=None),
    )

    def generate(*args, **kwargs):
        generated.append(1)
        return intents

    monkeypatch.setattr(walker.oracle.generator, "generate_krylov_intents", generate)
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_initial_direction",
        lambda *args, **kwargs: np.array([1.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            true_curvature=1.0,
            kind=DirectionCandidateKind.BLOCK_RITZ,
            candidate_count=2,
        ),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(
            task.initial_state,
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        ),
    )

    walker._walk_candidate_from_seed(state)

    assert len(generated) == 1
```

- [ ] **Step 2: Introduce one walk-local variable**

At the start of `_walk_candidate_from_seed`:

```python
krylov_intents: tuple[IntentBlock, ...] | None = None
```

After the legacy `anchor_direction` is generated:

```python
if (
    self.config.direction_selection_mode == "block_krylov"
    and krylov_intents is None
):
    krylov_intents = self.oracle.generator.generate_krylov_intents(
        current,
        n_blocks=self.config.block_krylov_blocks,
    )
```

Pass to `choose_direction`:

```python
krylov_intents=krylov_intents,
```

The existing legacy anchor remains responsible for current softening and
downstream walk semantics. The block oracle does not use its mixing weight or
scorer.

- [ ] **Step 3: Run the lifetime test**

Run:

```bash
pytest -q tests/unit/test_walker_policy.py::test_block_krylov_intents_are_generated_once_per_walk
```

Expected: `1 passed`.

- [ ] **Step 4: Extend JSONL diagnostics without changing existing keys**

In `_record_direction_diagnostics`:

```python
payload.update(choice.diagnostics)
```

All values in the block diagnostics are JSON scalar/list values. Do not write
vectors, coordinates, force arrays, or model data to this JSONL.

- [ ] **Step 5: Add a diagnostics persistence test**

```python
def test_direction_diagnostics_persist_block_krylov_metrics(tmp_path):
    path = tmp_path / "direction_trace.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(path),
        ),
        softening_enabled=False,
    )
    choice = DirectionChoice(
        direction=np.array([1.0, 0.0, 0.0]),
        curvature=-0.5,
        true_curvature=-0.25,
        kind=DirectionCandidateKind.BLOCK_RITZ,
        candidate_count=2,
        diagnostics={
            "krylov_hvp_count": 12,
            "krylov_residual_norm": 0.01,
            "krylov_initial_span_overlap": 0.8,
        },
    )

    walker._record_direction_diagnostics(
        trial_index=0,
        proposal_index=0,
        step_index=0,
        choice=choice,
        anchor_direction=None,
    )

    [row] = [
        json.loads(line)
        for line in path.read_text().splitlines()
    ]
    assert row["selected_kind"] == "block_ritz"
    assert row["krylov_hvp_count"] == 12
    assert row["krylov_residual_norm"] == pytest.approx(0.01)
    assert row["krylov_initial_span_overlap"] == pytest.approx(0.8)
```

- [ ] **Step 6: Run all changed unit suites**

Run:

```bash
pytest -q \
  tests/unit/test_block_krylov.py \
  tests/unit/test_config.py \
  tests/unit/test_direction_candidate_budget.py \
  tests/unit/test_walker_policy.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit walk integration and diagnostics**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py
git commit -m "feat: retain Krylov intents and log direction diagnostics"
```

## Task 5: Regression verification before experiments

**Files:**

- No source changes expected.

- [ ] **Step 1: Run formatting/static import checks used by this repository**

Run:

```bash
python -m compileall -q pamssw tests
```

Expected: exit code `0`.

- [ ] **Step 2: Run the complete unit suite**

Run:

```bash
pytest -q tests/unit
```

Expected: all tests pass with no xfail introduced for block Krylov.

- [ ] **Step 3: Run integration tests**

Run:

```bash
pytest -q tests/integration
```

Expected: all tests pass.

- [ ] **Step 4: Verify the production default is unchanged**

Run:

```bash
python - <<'PY'
from pamssw.config import SSWConfig

config = SSWConfig()
assert config.direction_selection_mode == "discrete"
assert config.direction_synthesis_mode == "none"
print("default-direction-path=discrete")
PY
```

Expected: `default-direction-path=discrete`.

- [ ] **Step 5: Record the verified test counts in the next commit message**

No file edit is required. If any unrelated baseline test fails, stop and
diagnose it; do not weaken the test or add a skip.

## Task 6: Analytic and fixed-state direction audit

**Files:**

- Create: `runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py`
- Create: `runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py`
- Create: `runs/20260728-block-krylov-direction-audit/README.md`
- Test: `tests/unit/test_block_krylov.py`

- [ ] **Step 1: Add a direct symmetry/residual evidence projection test**

```python
# append to tests/unit/test_block_krylov.py
def test_krylov_result_exposes_unsymmetrized_operator_defect():
    operator = np.array([[1.0, 0.2], [0.0, 2.0]])
    intent = IntentBlock(np.eye(2), pair=(0, 1))
    result = solve_krylov_block(
        intent,
        lambda vector: (operator @ vector, operator @ vector),
        depth=1,
    )
    expected = np.linalg.norm(operator - operator.T) / max(
        1.0,
        np.linalg.norm(operator),
    )
    assert result.antisymmetry == pytest.approx(expected)
```

- [ ] **Step 2: Create the fixed-state runner with preregistered arms**

The runner defines exactly:

```python
ALLOCATIONS = {
    "variational_breadth": {"block_krylov_blocks": 6, "block_krylov_depth": 1},
    "shallow_refinement": {"block_krylov_blocks": 3, "block_krylov_depth": 2},
    "balanced_refinement": {"block_krylov_blocks": 2, "block_krylov_depth": 3},
    "deep_refinement": {"block_krylov_blocks": 1, "block_krylov_depth": 6},
}
HVP_EPSILON = 1.0e-3
MAX_HVPS = 12
```

For each analytic/fixed state:

1. reconstruct one `ProposalPotential`;
2. reset an RNG to the same per-state intent seed for every allocation;
3. generate the requested walk-local intents;
4. call the real `SoftModeOracle` under
   `EvaluationPurpose.DIRECTION_ORACLE`;
5. independently parse the exact `EvalCounter.snapshot()`;
6. record `DirectionChoice.diagnostics`;
7. assert `force_evaluations == 2 * krylov_hvp_count`;
8. assert `krylov_hvp_count <= MAX_HVPS`;
9. record structure/model checksum, commit, precision, device, and wall time.

The analytic cases are:

- diagonal quadratic with known eigenpairs;
- coupled quadratic with a seed that requires expansion;
- repeated/near-degenerate lowest eigenvalue;
- rank-one initial block.

The C60/PdO states are selected before looking at arm results:

- one initial/bootstrap minimum;
- one intermediate accepted minimum;
- one plateau-state minimum.

If a required stored state is absent or cannot be tied to its originating run
and model, fail rather than substituting another state.

- [ ] **Step 3: Implement the fail-closed analyzer**

The analyzer reads runner JSON and raises on:

```python
REQUIRED_DIAGNOSTICS = {
    "krylov_blocks",
    "krylov_selected_block",
    "krylov_hvp_count",
    "krylov_dimensions",
    "krylov_initial_ranks",
    "krylov_residual_norm",
    "krylov_initial_span_overlap",
    "krylov_antisymmetry",
    "krylov_termination",
    "direction_participation_ratio",
}
```

It writes `evidence.json` containing:

```text
schema_version
git_commit
dirty
operator
hvp_epsilon
allocations
analytic_cases
c60_states
pdo_states
accounting_invariants
claim_ceiling
```

Required assertions:

```python
assert not raw["dirty"]
assert raw["operator"] == "total_proposal_central_fd"
assert raw["hvp_epsilon"] == 1.0e-3
assert set(raw["allocations"]) == set(ALLOCATIONS)
assert all(case["force_evaluations"] == 2 * case["krylov_hvp_count"] for case in rows)
assert all(case["krylov_hvp_count"] <= 12 for case in rows)
assert all(REQUIRED_DIAGNOSTICS <= case["diagnostics"].keys() for case in rows)
```

Do not rank allocations by terminal energy in this direction-only audit.

- [ ] **Step 4: Document exact commands and claim ceiling**

`README.md` must include:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py \
  --output runs/20260728-block-krylov-direction-audit/raw.json
python runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py \
  --input runs/20260728-block-krylov-direction-audit/raw.json \
  --output runs/20260728-block-krylov-direction-audit/evidence.json
```

The claim ceiling states that this audit validates direction algebra, operator
diagnostics, and force accounting only.

- [ ] **Step 5: Run analytic audit first**

Run:

```bash
python runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py \
  --analytic-only \
  --output /tmp/block-krylov-analytic.json
python runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py \
  --input /tmp/block-krylov-analytic.json \
  --output /tmp/block-krylov-analytic-evidence.json
```

Expected: both exit `0`, every analytic invariant passes, no GPU is required.

- [ ] **Step 6: Commit the reproducible audit package**

```bash
git add \
  pamssw/krylov.py \
  tests/unit/test_block_krylov.py \
  runs/20260728-block-krylov-direction-audit
git commit -m "test: add block Krylov direction audit"
```

## Task 7: Paired fixed-budget C60 survivor gate

**Files:**

- Create: `runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py`
- Create: `runs/20260728-block-krylov-direction-gpu-ablation/analyze_evidence.py`
- Create after execution:
  `runs/20260728-block-krylov-direction-gpu-ablation/evidence.json`
- Create after execution:
  `runs/20260728-block-krylov-direction-gpu-ablation/conclusion.md`
- Test: `tests/unit/test_block_krylov_gpu_ablation.py`

- [ ] **Step 1: Write runner-contract tests before the runner**

```python
# tests/unit/test_block_krylov_gpu_ablation.py
from dataclasses import asdict

from pamssw.config import SSWConfig


def test_gpu_arms_freeze_every_config_except_direction_allocation():
    shared = dict(
        max_force_evals=6000,
        direction_synthesis_mode="none",
        direction_diagnostics_enabled=True,
    )
    baseline = SSWConfig(**shared, direction_selection_mode="discrete")
    balanced = SSWConfig(
        **shared,
        direction_selection_mode="block_krylov",
        block_krylov_blocks=2,
        block_krylov_depth=3,
    )
    diff = {
        key
        for key, value in asdict(baseline).items()
        if value != asdict(balanced)[key]
    }
    assert diff == {
        "direction_selection_mode",
        "block_krylov_blocks",
        "block_krylov_depth",
    }
```

- [ ] **Step 2: Define exactly three Stage-2 arms**

```python
ARMS = {
    "discrete": {
        "direction_selection_mode": "discrete",
    },
    "variational_breadth": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 6,
        "block_krylov_depth": 1,
    },
    "balanced_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    },
}
SEEDS = (42, 43, 44)
TOTAL_FORCE_BUDGET = 6000
```

The optional fourth allocation from Stage 1 is not added automatically. Adding
it requires a preregistered amendment committed before any Stage-2 terminal
energy is inspected.

- [ ] **Step 3: Adapt the proven paired Ritz runner contract**

Create the runner from the checked-in
`runs/20260728-direction-selection-ritz-gpu-ablation/run_ablation.py` so its
existing MACE construction, structure loading, force-budget loop, artifact
schema, and recovery behavior remain byte-for-byte unchanged. Apply only these
named changes:

```text
PAIR_DIFF
  -> ALLOWED_DIRECTION_DIFF with the three block fields below
two-arm discrete/Ritz construction
  -> ARMS matrix declared in Step 2
system/seed loop
  -> iterate requested ARMS in stable insertion order
plain_ritz_hvp_contract
  -> block_krylov_hvp_contract
direction trace audit
  -> require the block diagnostic keys declared in Task 6
```

The new runner must assert before each run:

```python
allowed_changes = {
    "direction_selection_mode",
    "block_krylov_blocks",
    "block_krylov_depth",
    "rng_seed",
    "direction_diagnostics_path",
}
assert actual_config_diff <= allowed_changes
assert effective.max_force_evals == TOTAL_FORCE_BUDGET
assert effective.direction_synthesis_mode == "none"
assert not effective.direction_type_ucb_enabled
assert not effective.plateau_evolution_enabled
assert not effective.archive_escape_momentum_enabled
```

The runner stores:

- best-energy trace indexed by cumulative force evaluations;
- final best energy and energy drop from bootstrap;
- archive size and duplicate fraction;
- terminal failure counts;
- complete purpose ledger;
- direction JSONL diagnostics;
- total wall time;
- exact config, commit, structure checksum, model checksum, device, precision.

- [ ] **Step 4: Implement the evidence analyzer and survivor rule**

For every run:

```python
assert purpose_sum == total_force_evaluations
assert total_force_evaluations <= TOTAL_FORCE_BUDGET
assert unattributed == 0
assert "direction_oracle" in purpose_counts
```

For every block arm:

```python
assert all(row["selected_kind"] == "block_ritz" for row in direction_rows)
assert all(row["krylov_hvp_count"] <= 12 for row in direction_rows)
assert all(row["krylov_blocks"] == expected_blocks for row in direction_rows)
```

Paired improvement is:

```python
delta = block_best_energy - discrete_best_energy
improved = delta < 0.0
```

Coverage guard:

```python
coverage_ratio = block_archive_size / max(discrete_archive_size, 1)
coverage_ok = coverage_ratio >= 0.8
```

An arm survives C60 only when:

```python
survives = (
    sum(row["improved"] for row in paired_rows) >= 2
    and all(row["coverage_ok"] for row in paired_rows)
    and all(row["budget_closed"] for row in paired_rows)
)
```

The 0.8 coverage rule is a preregistered experimental guard, not a production
algorithm parameter.

- [ ] **Step 5: Run runner/analyzer unit tests**

Run:

```bash
pytest -q tests/unit/test_block_krylov_gpu_ablation.py
```

Expected: all tests pass without loading MACE or requiring a GPU.

- [ ] **Step 6: Commit the runner before inspecting GPU outcomes**

```bash
git add \
  runs/20260728-block-krylov-direction-gpu-ablation \
  tests/unit/test_block_krylov_gpu_ablation.py
git commit -m "test: preregister block Krylov GPU ablation"
```

- [ ] **Step 7: Execute C60 paired runs**

Run:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py \
  --system c60 \
  --seeds 42 43 44 \
  --force-budget 6000 \
  --device cuda
```

Expected: nine completed run artifacts, each with total FE at or below 6000,
zero unattributed evaluations, and a direction diagnostics JSONL.

- [ ] **Step 8: Analyze C60 without changing the preregistered rule**

Run:

```bash
python runs/20260728-block-krylov-direction-gpu-ablation/analyze_evidence.py \
  --system c60
```

Expected: exit `0` and `evidence.json` contains paired deltas, AUC, coverage,
purpose costs, wall time, and the computed survivor flags.

- [ ] **Step 9: Execute PdO only for a surviving arm**

If no block arm survives, skip this step and record `pdo_status:
not_run_no_c60_survivor`.

For exactly one survivor selected by the preregistered rule:

```bash
python runs/20260728-block-krylov-direction-gpu-ablation/run_ablation.py \
  --system pdo \
  --seeds 42 43 44 \
  --force-budget 6000 \
  --device cuda \
  --arms discrete balanced_refinement
```

The command above is valid when `balanced_refinement` is the analyzer-produced
survivor. If `variational_breadth` is the survivor, replace only the final
token with `variational_breadth`. Do not choose by manual inspection.

- [ ] **Step 10: Write the evidence-backed conclusion**

`conclusion.md` must state:

- exact commit and whether the worktree was clean;
- per-seed best-energy deltas;
- energy AUC;
- unique minima and duplicate fraction;
- total and direction force evaluations;
- wall times;
- C60 survivor decision;
- PdO transfer decision or explicit reason it was not run;
- that three seeds are a survivor gate, not significance;
- that production default remains `discrete`.

- [ ] **Step 11: Commit reviewed evidence**

```bash
git add runs/20260728-block-krylov-direction-gpu-ablation
git commit -m "results: record block Krylov direction ablation"
```

## Task 8: Final regression, review, and PR update

**Files:**

- Modify only if required by review findings.

- [ ] **Step 1: Run the full verification suite**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Confirm no default or unrelated config drift**

Run:

```bash
git diff 6492356 -- pamssw/config.py pamssw/walker.py pamssw/krylov.py
```

Review must confirm:

- default `direction_selection_mode` remains `discrete`;
- no starter selector, posterior, optimizer, archive, or accounting equation
  changed;
- no new weighted score term exists;
- block mode uses only HVP budget through blocks and depth;
- selected Ritz direction reuses stored true HVP projections.

- [ ] **Step 3: Request code review**

Use the `requesting-code-review` skill against the complete diff and require
separate checks for:

- mathematical correctness;
- exact FE accounting;
- ablation isolation;
- evidence/claim ceiling.

- [ ] **Step 4: Address only concrete review findings**

For every accepted finding:

1. add or expose a failing test;
2. run it and confirm failure;
3. apply the minimal fix;
4. rerun focused and full tests;
5. commit the fix separately.

- [ ] **Step 5: Push and update the existing pull request**

```bash
git push pam feature/posterior-terminal-outcome-validation
```

Update PR #10 with:

- implementation summary;
- exact tests and counts;
- analytic/fixed-state audit;
- C60/PdO survivor evidence;
- unchanged default statement;
- known untested questions.

Do not state that block Krylov is superior unless it passed the preregistered
C60 and PdO gates.

## Self-review checklist

- Spec coverage: pure algebra, separate random/pair axes, fixed HVP
  breadth--depth, total/true HVP reuse, diagnostics, exact accounting, C60
  survivor gate, conditional PdO gate, posterior deferral, and claim ceiling
  all map to explicit tasks.
- Deliberate deferral: one-sided HVP reuse and standard Dimer comparison remain
  separate future one-seam experiments after central-HVP block Krylov evidence.
- Type consistency: `IntentBlock`, `KrylovResult`, `solve_krylov_block`,
  `DirectionCandidateKind.BLOCK_RITZ`, `block_krylov_blocks`,
  `block_krylov_depth`, and `DirectionChoice.diagnostics` use the same names
  throughout.
- Default safety: every new behavior is behind
  `direction_selection_mode="block_krylov"`.
- No new production weighting parameter is introduced.
- Experiment-only allocation and coverage thresholds are preregistered and do
  not enter the search algorithm.
