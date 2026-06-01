# Reference Dimer Component Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in reference-SSW dimer soft-mode direction engine to PAM-SSW and prepare a traceable k=3 component ablation against `/tmp/ssw-reference`.

**Architecture:** Port only the reference direction pieces (`sample_mixed_mode` and biased dimer rotation) into a focused `pamssw/reference_dimer.py` module. Route `SurfaceWalker._walk_candidate_from_seed()` through either the existing scored direction pool or the new reference dimer engine via `SSWConfig.direction_engine`, while preserving existing bias+relax, Direct-QP, archive, UCB, and Metropolis-chain machinery for controlled hybrids.

**Tech Stack:** Python 3.12, NumPy, ASE/MACE calculators, pytest, existing PAM-SSW `State`, `ProposalPotential`, `DirectionChoice`, and benchmark runner patterns.

---

## File Structure

- Create `pamssw/reference_dimer.py`
  - Owns reference-compatible mixed-mode sampling and biased dimer rotation.
  - Does not depend on `/tmp/ssw-reference` at runtime.
  - Accepts PAM-SSW calculator interfaces through a tiny adapter callback.

- Modify `pamssw/config.py`
  - Adds `direction_engine` and reference dimer controls.
  - Validates fields and keeps `scored_pool` as the default.

- Modify `pamssw/walker.py`
  - Adds `DirectionCandidateKind.REFERENCE_DIMER`.
  - Adds `SurfaceWalker._choose_walk_direction()` to isolate direction-engine branching.
  - Adds reference dimer diagnostics counters and stats.
  - Uses dimer curvature without a redundant HVP in Direct-QP mode.

- Modify `tests/unit/test_config.py`
  - Covers config defaults and validation.

- Modify `tests/unit/test_walker_policy.py`
  - Covers reference dimer sampling, walker routing, diagnostics, and Direct-QP curvature reuse behavior.

- Modify `runs/20260530-direct-qp-ssw-c60-cuo-pdo-benchmark/run_matrix.py` or create a new runner under `runs/20260601-reference-dimer-component-ablation/`
  - Prefer a new runner and ledger directory for the approved ablation.
  - Adds variants for the Phase 1 and Phase 2 matrices.

---

## Task 1: Reference Dimer Module

**Files:**
- Create: `pamssw/reference_dimer.py`
- Test: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing tests for mixed-mode sampling**

Add these tests near existing direction-generator tests in `tests/unit/test_walker_policy.py`:

```python
from pamssw.reference_dimer import sample_mixed_mode


def test_reference_dimer_sample_mixed_mode_returns_normalized_direction():
    rng = np.random.default_rng(7)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    direction, info = sample_mixed_mode(
        positions,
        rng=rng,
        lam=0.5,
        min_distance=3.0,
    )

    assert direction.shape == positions.shape
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert info["lambda"] == pytest.approx(0.5)
    assert info["pair"] in {(0, 1), (0, 2), (1, 2)}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_sample_mixed_mode_returns_normalized_direction -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pamssw.reference_dimer'`.

- [ ] **Step 3: Implement mixed-mode sampling**

Create `pamssw/reference_dimer.py` with this initial content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLikeForceEvaluator = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class ReferenceDimerResult:
    direction: np.ndarray
    curvature: float
    rotations: int
    dot_initial: float
    converged: bool
    lambda_value: float
    local_pair: tuple[int, int]


def sample_global_mode(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    positions = np.asarray(positions, dtype=float)
    if masses is None:
        mode = rng.normal(size=positions.shape)
    else:
        masses = np.asarray(masses, dtype=float)
        mode = rng.normal(size=positions.shape) * np.sqrt(1.0 / masses)[:, None]
    return _normalized_matrix(mode, rng)


def sample_local_bond_mode(
    positions: np.ndarray,
    min_distance: float = 3.0,
    cell: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    rng = np.random.default_rng() if rng is None else rng
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    if n_atoms < 2:
        return sample_global_mode(positions, rng=rng), (0, 0)

    valid_pairs: list[tuple[int, int]] = []
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            delta = _mic(positions[atom_i] - positions[atom_j], cell, pbc)
            if float(np.linalg.norm(delta)) > min_distance:
                valid_pairs.append((atom_i, atom_j))

    if valid_pairs:
        pair = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
    else:
        atom_i = int(rng.integers(0, n_atoms))
        atom_j = atom_i
        while atom_j == atom_i:
            atom_j = int(rng.integers(0, n_atoms))
        pair = (atom_i, atom_j)

    atom_i, atom_j = pair
    mode = np.zeros_like(positions)
    delta = _mic(positions[atom_j] - positions[atom_i], cell, pbc)
    mode[atom_i] = delta
    mode[atom_j] = -delta
    return _normalized_matrix(mode, rng), pair


def sample_mixed_mode(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    min_distance: float = 3.0,
    cell: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    lam: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng() if rng is None else rng
    lambda_value = float(rng.uniform(0.1, 1.5) if lam is None else lam)
    global_mode = sample_global_mode(positions, masses=masses, rng=rng)
    local_mode, pair = sample_local_bond_mode(
        positions,
        min_distance=min_distance,
        cell=cell,
        pbc=pbc,
        rng=rng,
    )
    mixed = _normalized_matrix(global_mode + lambda_value * local_mode, rng)
    return mixed, {
        "lambda": lambda_value,
        "pair": pair,
        "N_global_norm": float(np.linalg.norm(global_mode)),
        "N_local_norm": float(np.linalg.norm(local_mode)),
    }


def _normalized_matrix(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix))
    if norm > 1e-15 and np.all(np.isfinite(matrix)):
        return matrix / norm
    fallback = rng.normal(size=matrix.shape)
    return fallback / (float(np.linalg.norm(fallback)) + 1e-12)


def _mic(
    delta: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool] | np.ndarray | None,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    if cell is None or pbc is None or not np.any(pbc):
        return delta.copy()
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) < 1e-12:
        return delta.copy()
    pbc_arr = np.asarray(pbc, dtype=bool)
    if pbc_arr.shape == ():
        pbc_arr = np.repeat(bool(pbc_arr), 3)
    if pbc_arr.shape != (3,):
        return delta.copy()
    fractional = delta @ np.linalg.inv(cell)
    fractional[..., pbc_arr] -= np.round(fractional[..., pbc_arr])
    return fractional @ cell
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_sample_mixed_mode_returns_normalized_direction -q
```

Expected: PASS.

- [ ] **Step 5: Write failing tests for biased dimer rotation**

Add this test:

```python
from pamssw.reference_dimer import ReferenceDimerRotator


def test_reference_dimer_rotator_returns_direction_and_curvature():
    class HarmonicSurface:
        def __init__(self):
            self.calls = 0
            self.hessian = np.diag([2.0, 6.0, 10.0, 2.0, 6.0, 10.0])

        def evaluate(self, flat_positions):
            self.calls += 1
            gradient = self.hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            force = -gradient.reshape(2, 3)
            return energy, force

    surface = HarmonicSurface()
    rotator = ReferenceDimerRotator(
        delta=1e-3,
        bias_strength=10.0,
        max_steps=4,
        rotation_tol=1e-12,
        angular_step=0.05,
    )
    positions = np.zeros((2, 3), dtype=float)
    initial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    initial /= np.linalg.norm(initial)

    result = rotator.rotate(positions, initial, surface.evaluate)

    assert result.direction.shape == positions.shape
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    assert np.isfinite(result.curvature)
    assert result.rotations >= 1
    assert surface.calls >= 2
```

- [ ] **Step 6: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_rotator_returns_direction_and_curvature -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `ReferenceDimerRotator`.

- [ ] **Step 7: Implement biased dimer rotation**

Append this class to `pamssw/reference_dimer.py`:

```python
class ReferenceDimerRotator:
    def __init__(
        self,
        delta: float = 0.005,
        bias_strength: float = 500.0,
        max_steps: int = 15,
        rotation_tol: float = 0.03,
        angular_step: float = 0.05,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        if bias_strength <= 0:
            raise ValueError("bias_strength must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if rotation_tol <= 0:
            raise ValueError("rotation_tol must be positive")
        if angular_step <= 0:
            raise ValueError("angular_step must be positive")
        self.delta = float(delta)
        self.bias_strength = float(bias_strength)
        self.max_steps = int(max_steps)
        self.rotation_tol = float(rotation_tol)
        self.angular_step = float(angular_step)

    def rotate(
        self,
        positions: np.ndarray,
        initial_direction: np.ndarray,
        evaluate_forces: ArrayLikeForceEvaluator,
        *,
        initial_forces: np.ndarray | None = None,
        lambda_value: float = 0.0,
        local_pair: tuple[int, int] = (0, 0),
    ) -> ReferenceDimerResult:
        positions = np.asarray(positions, dtype=float)
        initial = self._normalized(initial_direction)
        direction = initial.copy()
        if initial_forces is None:
            _, force0 = evaluate_forces(positions)
        else:
            force0 = np.asarray(initial_forces, dtype=float)

        converged = False
        previous = direction.copy()
        rotations = 0
        for rotations in range(1, self.max_steps + 1):
            endpoint = positions + self.delta * direction
            _, force1_real = evaluate_forces(endpoint)
            force1 = force1_real + self._bias_force(endpoint, positions, initial)

            force0_perp = force0 - float(np.sum(force0 * direction)) * direction
            force1_perp = force1 - float(np.sum(force1 * direction)) * direction
            rotational_force = force1_perp - force0_perp
            rotational_norm = float(np.linalg.norm(rotational_force))
            if rotational_norm / self.delta < self.rotation_tol:
                converged = True
                break
            rotation_axis = rotational_force / (rotational_norm + 1e-15)
            candidate = direction * np.cos(self.angular_step) + rotation_axis * np.sin(self.angular_step)
            direction = self._normalized(candidate)
            angle = float(np.arccos(np.clip(np.sum(direction * previous), -1.0, 1.0)))
            if angle < self.rotation_tol:
                converged = True
                break
            previous = direction.copy()

        endpoint = positions + self.delta * direction
        _, force1_real = evaluate_forces(endpoint)
        force1 = force1_real + self._bias_force(endpoint, positions, initial)
        curvature = float(np.sum((force1 - force0) * direction) / self.delta)
        return ReferenceDimerResult(
            direction=direction,
            curvature=curvature,
            rotations=rotations,
            dot_initial=float(np.sum(direction * initial)),
            converged=converged,
            lambda_value=float(lambda_value),
            local_pair=(int(local_pair[0]), int(local_pair[1])),
        )

    def _bias_force(self, endpoint: np.ndarray, center: np.ndarray, initial: np.ndarray) -> np.ndarray:
        displacement = endpoint - center
        scalar = float(np.sum(displacement * initial))
        return 4.0 * self.bias_strength * scalar * initial

    @staticmethod
    def _normalized(direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-15 or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be finite and nonzero")
        return direction / norm
```

- [ ] **Step 8: Run reference dimer unit tests**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_sample_mixed_mode_returns_normalized_direction tests/unit/test_walker_policy.py::test_reference_dimer_rotator_returns_direction_and_curvature -q
```

Expected: PASS.

---

## Task 2: Config Surface

**Files:**
- Modify: `pamssw/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing config default test**

Add to `tests/unit/test_config.py`:

```python
def test_config_accepts_reference_dimer_direction_engine_defaults():
    config = SSWConfig(direction_engine="reference_dimer")

    assert config.direction_engine == "reference_dimer"
    assert config.reference_dimer_delta == pytest.approx(0.005)
    assert config.reference_dimer_bias_strength == pytest.approx(500.0)
    assert config.reference_dimer_max_steps == 15
    assert config.reference_dimer_rotation_tol == pytest.approx(0.03)
    assert config.reference_dimer_angular_step == pytest.approx(0.05)
    assert config.reference_dimer_lambda_min == pytest.approx(0.1)
    assert config.reference_dimer_lambda_max == pytest.approx(1.5)
    assert config.reference_dimer_min_pair_distance == pytest.approx(3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_config.py::test_config_accepts_reference_dimer_direction_engine_defaults -q
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'direction_engine'`.

- [ ] **Step 3: Add config fields**

Modify `SSWConfig` in `pamssw/config.py` near existing direction fields:

```python
direction_engine: str = "scored_pool"
reference_dimer_delta: float = 0.005
reference_dimer_bias_strength: float = 500.0
reference_dimer_max_steps: int = 15
reference_dimer_rotation_tol: float = 0.03
reference_dimer_angular_step: float = 0.05
reference_dimer_lambda_min: float = 0.1
reference_dimer_lambda_max: float = 1.5
reference_dimer_min_pair_distance: float = 3.0
```

- [ ] **Step 4: Add validation tests**

Add:

```python
def test_config_rejects_invalid_reference_dimer_controls():
    with pytest.raises(ValueError, match="direction_engine"):
        SSWConfig(direction_engine="unknown")
    with pytest.raises(ValueError, match="reference_dimer_delta"):
        SSWConfig(reference_dimer_delta=0.0)
    with pytest.raises(ValueError, match="reference_dimer_bias_strength"):
        SSWConfig(reference_dimer_bias_strength=0.0)
    with pytest.raises(ValueError, match="reference_dimer_max_steps"):
        SSWConfig(reference_dimer_max_steps=0)
    with pytest.raises(ValueError, match="reference_dimer_rotation_tol"):
        SSWConfig(reference_dimer_rotation_tol=0.0)
    with pytest.raises(ValueError, match="reference_dimer_angular_step"):
        SSWConfig(reference_dimer_angular_step=0.0)
    with pytest.raises(ValueError, match="reference_dimer_lambda_min"):
        SSWConfig(reference_dimer_lambda_min=-0.1)
    with pytest.raises(ValueError, match="reference_dimer_lambda"):
        SSWConfig(reference_dimer_lambda_min=2.0, reference_dimer_lambda_max=1.0)
    with pytest.raises(ValueError, match="reference_dimer_min_pair_distance"):
        SSWConfig(reference_dimer_min_pair_distance=0.0)
```

- [ ] **Step 5: Run validation test to verify it fails**

Run:

```bash
pytest tests/unit/test_config.py::test_config_rejects_invalid_reference_dimer_controls -q
```

Expected: FAIL because validation has not been added.

- [ ] **Step 6: Implement validation**

Add to `SSWConfig.__post_init__()` in `pamssw/config.py`:

```python
if self.direction_engine not in {"scored_pool", "reference_dimer"}:
    raise ValueError("direction_engine must be scored_pool or reference_dimer")
for name in (
    "reference_dimer_delta",
    "reference_dimer_bias_strength",
    "reference_dimer_rotation_tol",
    "reference_dimer_angular_step",
    "reference_dimer_min_pair_distance",
):
    value = getattr(self, name)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
if isinstance(self.reference_dimer_max_steps, bool) or not isinstance(self.reference_dimer_max_steps, int):
    raise ValueError("reference_dimer_max_steps must be a positive integer")
if self.reference_dimer_max_steps <= 0:
    raise ValueError("reference_dimer_max_steps must be a positive integer")
if not isfinite(self.reference_dimer_lambda_min) or self.reference_dimer_lambda_min < 0:
    raise ValueError("reference_dimer_lambda_min must be finite and non-negative")
if not isfinite(self.reference_dimer_lambda_max) or self.reference_dimer_lambda_max < 0:
    raise ValueError("reference_dimer_lambda_max must be finite and non-negative")
if self.reference_dimer_lambda_min > self.reference_dimer_lambda_max:
    raise ValueError("reference_dimer_lambda_min cannot exceed reference_dimer_lambda_max")
```

- [ ] **Step 7: Run config tests**

Run:

```bash
pytest tests/unit/test_config.py::test_config_accepts_reference_dimer_direction_engine_defaults tests/unit/test_config.py::test_config_rejects_invalid_reference_dimer_controls -q
```

Expected: PASS.

---

## Task 3: Walker Routing and Diagnostics

**Files:**
- Modify: `pamssw/walker.py`
- Test: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing test for direction kind**

Add:

```python
def test_direction_candidate_kind_includes_reference_dimer():
    assert DirectionCandidateKind.REFERENCE_DIMER.value == "reference_dimer"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_direction_candidate_kind_includes_reference_dimer -q
```

Expected: FAIL with `AttributeError: REFERENCE_DIMER`.

- [ ] **Step 3: Add enum value**

Modify `DirectionCandidateKind` in `pamssw/walker.py`:

```python
REFERENCE_DIMER = "reference_dimer"
```

- [ ] **Step 4: Run enum test**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_direction_candidate_kind_includes_reference_dimer -q
```

Expected: PASS.

- [ ] **Step 5: Write failing walker routing test**

Add:

```python
def test_reference_dimer_direction_engine_bypasses_scored_pool(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        max_steps_per_walk=1,
        proposal_step_mode="bias_relax",
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=QuadraticCalculator(), config=config, softening_enabled=False)

    def forbidden_choose_direction(*args, **kwargs):
        raise AssertionError("scored pool should not be called")

    monkeypatch.setattr(walker.oracle, "choose_direction", forbidden_choose_direction)
    choice = walker._choose_walk_direction(
        current=state,
        proposal=ProposalPotential(walker.calculator),
        scoring_proposal=ProposalPotential(walker.calculator),
        previous_direction=None,
        anchor_direction=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0),
        archive=None,
        step_target=None,
        sigma_scale=1.0,
        previous_relax_outcome=None,
        trial_index=0,
        proposal_index=0,
        seed_entry_id=0,
        step_index=0,
        plateau_evolution_active=False,
    )

    assert choice.kind == DirectionCandidateKind.REFERENCE_DIMER
    assert np.linalg.norm(choice.direction) == pytest.approx(1.0)
    assert np.isfinite(choice.curvature)
```

If `QuadraticCalculator` is not already available in the file, use the existing local test calculator class closest to other `SurfaceWalker` tests, or add:

```python
class QuadraticCalculator:
    def __init__(self):
        self.counter = EvalCounter()

    def evaluate(self, state):
        flat = state.flatten_positions()
        energy = 0.5 * float(np.dot(flat, flat))
        gradient = flat.copy()
        self.counter.record_force()
        return SimpleNamespace(energy=energy, gradient=gradient.reshape(state.positions.shape))

    def evaluate_flat(self, flat_positions, template):
        energy = 0.5 * float(np.dot(flat_positions, flat_positions))
        gradient = flat_positions.copy()
        self.counter.record_force()
        return energy, gradient

    def exhausted(self):
        return False
```

- [ ] **Step 6: Run routing test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_direction_engine_bypasses_scored_pool -q
```

Expected: FAIL because `_choose_walk_direction` does not exist.

- [ ] **Step 7: Add walker helper and reference dimer counters**

Modify imports in `pamssw/walker.py`:

```python
from .reference_dimer import ReferenceDimerRotator, ReferenceDimerResult, sample_mixed_mode
```

Add reset fields in `_reset_direct_qp_stats()` or a new `_reset_reference_dimer_stats()` called from `search()` initialization:

```python
self._reference_dimer_steps = 0
self._reference_dimer_rotation_sum = 0.0
self._reference_dimer_converged = 0
self._reference_dimer_curvature_sum = 0.0
self._reference_dimer_abs_dot_sum = 0.0
```

Add helper methods to `SurfaceWalker`:

```python
def _choose_walk_direction(
    self,
    *,
    current: State,
    proposal: ProposalPotential,
    scoring_proposal: ProposalPotential,
    previous_direction: np.ndarray | None,
    anchor_direction: np.ndarray | None,
    archive,
    step_target: float | None,
    sigma_scale: float,
    previous_relax_outcome: RelaxOutcomeClass | None,
    trial_index: int | None,
    proposal_index: int | None,
    seed_entry_id: int | None,
    step_index: int,
    plateau_evolution_active: bool,
) -> DirectionChoice:
    if self.config.direction_engine == "reference_dimer":
        return self._choose_reference_dimer_direction(current)
    score_sigma_fn = self._direction_score_sigma_fn(sigma_scale, step_target=step_target)
    return self.oracle.choose_direction(
        current,
        scoring_proposal,
        previous_direction,
        anchor_direction=anchor_direction,
        step_scale_fn=lambda curvature: self._scaled_step_scale(
            curvature,
            sigma_scale,
            step_target=step_target,
        ),
        archive=archive,
        history_gradient=self._history_bias_gradient(current, proposal.biases),
        continuity_weight=self._continuity_weight_for_outcome(previous_relax_outcome),
        n_bond_pairs=self._n_bond_pairs_for_outcome(previous_relax_outcome),
        score_sigma=(
            None
            if score_sigma_fn is not None
            else self._direction_score_sigma(sigma_scale, step_target=step_target)
        ),
        score_sigma_fn=score_sigma_fn,
        direction_type_bonus_fn=(
            self.direction_type_memory.bonus if self.config.direction_type_ucb_enabled else None
        ),
        plateau_evolution_active=plateau_evolution_active,
        plateau_history=(
            self.successful_records(
                seed_entry_id=seed_entry_id,
                limit=self.config.plateau_evolution_history_limit,
            )
            if plateau_evolution_active
            else []
        ),
        plateau_evolution_children=self.config.plateau_evolution_children,
        plateau_evolution_crossover_pairs=self.config.plateau_evolution_crossover_pairs,
        plateau_evolution_mutation_count=self.config.plateau_evolution_mutation_count,
        archive_momentum_history=self._archive_momentum_history_for_seed(seed_entry_id),
        archive_momentum_limit=self.config.archive_escape_momentum_limit,
    )

def _choose_reference_dimer_direction(self, current: State) -> DirectionChoice:
    lam = float(
        self.rng.uniform(
            self.config.reference_dimer_lambda_min,
            self.config.reference_dimer_lambda_max,
        )
    )
    initial, info = sample_mixed_mode(
        current.positions,
        min_distance=self.config.reference_dimer_min_pair_distance,
        cell=current.cell,
        pbc=current.pbc,
        rng=self.rng,
        lam=lam,
    )

    def evaluate_forces(positions: np.ndarray) -> tuple[float, np.ndarray]:
        trial = replace(current, positions=np.asarray(positions, dtype=float))
        result = self.calculator.evaluate(trial)
        return result.energy, -np.asarray(result.gradient, dtype=float)

    rotator = ReferenceDimerRotator(
        delta=self.config.reference_dimer_delta,
        bias_strength=self.config.reference_dimer_bias_strength,
        max_steps=self.config.reference_dimer_max_steps,
        rotation_tol=self.config.reference_dimer_rotation_tol,
        angular_step=self.config.reference_dimer_angular_step,
    )
    result = rotator.rotate(
        current.positions,
        initial,
        evaluate_forces,
        lambda_value=float(info["lambda"]),
        local_pair=info["pair"],
    )
    self._record_reference_dimer_result(result)
    direction = result.direction.reshape(-1)
    direction = direction / (float(np.linalg.norm(direction)) + 1e-12)
    return DirectionChoice(
        direction=direction,
        curvature=result.curvature,
        kind=DirectionCandidateKind.REFERENCE_DIMER,
        candidate_count=1,
        score=None,
    )

def _record_reference_dimer_result(self, result: ReferenceDimerResult) -> None:
    self._reference_dimer_steps += 1
    self._reference_dimer_rotation_sum += float(result.rotations)
    self._reference_dimer_converged += int(result.converged)
    self._reference_dimer_curvature_sum += float(result.curvature)
    self._reference_dimer_abs_dot_sum += abs(float(result.dot_initial))
```

Use `dataclasses.replace`; if not already imported, add it to the existing dataclass import:

```python
from dataclasses import dataclass, field, replace
```

- [ ] **Step 8: Replace inline `oracle.choose_direction` call**

In `_walk_candidate_from_seed()`, replace the current `choice = self.oracle.choose_direction(...)` block with:

```python
choice = self._choose_walk_direction(
    current=current,
    proposal=proposal,
    scoring_proposal=scoring_proposal,
    previous_direction=previous_direction,
    anchor_direction=anchor_direction,
    archive=archive,
    step_target=step_target,
    sigma_scale=sigma_scale,
    previous_relax_outcome=previous_relax_outcome,
    trial_index=trial_index,
    proposal_index=proposal_index,
    seed_entry_id=seed_entry_id,
    step_index=step_index,
    plateau_evolution_active=plateau_evolution_active,
)
```

- [ ] **Step 9: Add stats summary fields**

In `_direction_stats_summary()` or `_stats_summary()` where direction counts are emitted, include the new kind:

```python
"direction_selected_reference_dimer": self._direction_selected[DirectionCandidateKind.REFERENCE_DIMER],
```

Add a helper:

```python
def _reference_dimer_stats_summary(self) -> dict[str, StatsValue]:
    steps = max(1, self._reference_dimer_steps)
    return {
        "reference_dimer_steps": self._reference_dimer_steps,
        "reference_dimer_mean_rotations": self._reference_dimer_rotation_sum / steps,
        "reference_dimer_converged_fraction": self._reference_dimer_converged / steps,
        "reference_dimer_mean_curvature": self._reference_dimer_curvature_sum / steps,
        "reference_dimer_mean_abs_dot_initial": self._reference_dimer_abs_dot_sum / steps,
    }
```

Merge it into the main stats dictionary:

```python
**self._reference_dimer_stats_summary(),
```

- [ ] **Step 10: Run routing test**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_direction_engine_bypasses_scored_pool -q
```

Expected: PASS.

---

## Task 4: Direct-QP Curvature Reuse

**Files:**
- Modify: `pamssw/walker.py`
- Test: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing test that reference dimer Direct-QP skips redundant true HVP**

Add:

```python
def test_reference_dimer_direct_qp_uses_choice_curvature_without_true_hvp(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        proposal_step_mode="direct_qp",
        direct_qp_hessian="rank1",
        direct_qp_kappa=240.0,
        max_steps_per_walk=1,
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=QuadraticCalculator(), config=config, softening_enabled=False)

    def fail_true_hvp(*args, **kwargs):
        raise AssertionError("reference dimer curvature should be reused for Direct-QP")

    monkeypatch.setattr(walker, "_true_directional_curvature", fail_true_hvp)
    result = walker._walk_candidate_from_seed(state)

    assert isinstance(result, State)
    assert walker._reference_dimer_steps == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_direct_qp_uses_choice_curvature_without_true_hvp -q
```

Expected: FAIL because `_walk_candidate_from_seed()` still calls `_true_directional_curvature`.

- [ ] **Step 3: Implement curvature reuse**

In `_walk_candidate_from_seed()`, replace:

```python
true_curvature = self._true_directional_curvature(current, choice.direction)
```

with:

```python
true_curvature = (
    float(choice.curvature)
    if self.config.direction_engine == "reference_dimer"
    else self._true_directional_curvature(current, choice.direction)
)
```

Keep `inner_curvature` as `choice.curvature` for `reference_dimer` unless explicit verification is later added:

```python
inner_curvature = (
    choice.curvature
    if self.config.direction_engine == "reference_dimer"
    else (
        choice.curvature
        if self.config.direction_curvature_source == "inner" and not rebuild_softening_for_choice
        else self.oracle._directional_curvature(current, proposal, choice.direction)
    )
)
```

- [ ] **Step 4: Run Direct-QP reuse test**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_reference_dimer_direct_qp_uses_choice_curvature_without_true_hvp -q
```

Expected: PASS.

---

## Task 5: Direction Pool Ablation Controls

**Files:**
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Write failing config test for disabling momentum directions**

Add:

```python
def test_config_accepts_direction_pool_disable_momentum():
    config = SSWConfig(direction_pool_disable_momentum=True)

    assert config.direction_pool_disable_momentum is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_config.py::test_config_accepts_direction_pool_disable_momentum -q
```

Expected: FAIL with unexpected keyword.

- [ ] **Step 3: Add config field and validation**

Add to `SSWConfig`:

```python
direction_pool_disable_momentum: bool = False
```

Add to validation:

```python
if not isinstance(self.direction_pool_disable_momentum, bool):
    raise ValueError("direction_pool_disable_momentum must be a boolean")
```

- [ ] **Step 4: Write failing walker test**

Add:

```python
def test_direction_pool_disable_momentum_filters_momentum_candidates():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(direction_pool_disable_momentum=True)
    walker = SurfaceWalker(calculator=QuadraticCalculator(), config=config, softening_enabled=False)
    previous = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    previous /= np.linalg.norm(previous)

    candidates = walker.oracle.generator.generate(state, previous)
    filtered = walker._filter_direction_candidates(candidates)

    assert DirectionCandidateKind.MOMENTUM in {candidate.kind for candidate in candidates}
    assert DirectionCandidateKind.MOMENTUM not in {candidate.kind for candidate in filtered}
```

- [ ] **Step 5: Run walker test to verify it fails**

Run:

```bash
pytest tests/unit/test_walker_policy.py::test_direction_pool_disable_momentum_filters_momentum_candidates -q
```

Expected: FAIL because `_filter_direction_candidates` does not exist.

- [ ] **Step 6: Implement filter hook**

Add to `SurfaceWalker`:

```python
def _filter_direction_candidates(self, candidates: list[DirectionCandidate]) -> list[DirectionCandidate]:
    if not self.config.direction_pool_disable_momentum:
        return candidates
    filtered = [
        candidate
        for candidate in candidates
        if candidate.kind not in {DirectionCandidateKind.MOMENTUM, DirectionCandidateKind.ARCHIVE_MOMENTUM}
    ]
    return filtered if filtered else candidates
```

Modify `SoftModeOracle.choose_direction()` to accept:

```python
candidate_filter: Callable[[list[DirectionCandidate]], list[DirectionCandidate]] | None = None,
```

After candidates and archive momentum candidates are assembled:

```python
if candidate_filter is not None:
    candidates = candidate_filter(candidates)
```

Pass it from `_choose_walk_direction()`:

```python
candidate_filter=self._filter_direction_candidates,
```

- [ ] **Step 7: Run filter tests**

Run:

```bash
pytest tests/unit/test_config.py::test_config_accepts_direction_pool_disable_momentum tests/unit/test_walker_policy.py::test_direction_pool_disable_momentum_filters_momentum_candidates -q
```

Expected: PASS.

---

## Task 6: Benchmark Runner and Ledger Setup

**Files:**
- Create: `runs/20260601-reference-dimer-component-ablation/run_matrix.py`
- Create: `runs/20260601-reference-dimer-component-ablation/plan.md`
- Create: `runs/20260601-reference-dimer-component-ablation/questions.md`
- Create: `runs/20260601-reference-dimer-component-ablation/run_manifest.yaml`
- Create: `runs/20260601-reference-dimer-component-ablation/event_log.jsonl`
- Create: `runs/20260601-reference-dimer-component-ablation/artifacts.json`
- Create: `runs/20260601-reference-dimer-component-ablation/status.md`
- Create: `runs/20260601-reference-dimer-component-ablation/summary.md`

- [ ] **Step 1: Create run ledger files**

Create the directory and files with `apply_patch`. Initial `run_manifest.yaml` content:

```yaml
task: reference_dimer_component_ablation
date: 2026-06-01
approved_scope: design approved; implementation and benchmark plan pending explicit execution approval
repo_worktree: /tmp/SSW-worktrees/direct-qp-ssw
reference_repo: /tmp/ssw-reference
systems:
  phase1: [c60]
  phase2: [cuo, pdo]
seeds: [0, 1, 2]
trials_per_seed: 40
device: cuda
model_default: /root/.cache/mace/mace-omat-0-small.model
```

Initial `questions.md` content:

```markdown
# Questions

- User approved the design plan with "Approve design plan".
- Implementation scope still requires approval before code execution beyond this plan.
- Benchmark execution requires explicit approval after implementation tests pass.
```

Initial `plan.md` content should reference this implementation plan path and list the Phase 1 and Phase 2 matrices from the spec.

Initial `event_log.jsonl` first line:

```json
{"event_id":"0001","phase":"planning","step":"ledger_created","kind":"decision","status":"completed","reason":"Create provenance files before implementation and benchmark execution.","operator":"codex"}
```

Initial `artifacts.json`:

```json
{"artifacts":[]}
```

Initial `status.md`:

```markdown
# Status

Current phase: implementation planning.

No benchmark commands have been executed.
```

Initial `summary.md`:

```markdown
# Summary

No results yet. This file will be completed after approved benchmark execution.
```

- [ ] **Step 2: Create benchmark runner by copying existing runner shape**

Create `runs/20260601-reference-dimer-component-ablation/run_matrix.py` from the current Direct-QP runner structure, with these variants:

```python
VARIANTS = {
    "paw_current_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "scored_pool",
    },
    "paw_metropolis_pool_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "metropolis_chain",
        "direction_engine": "scored_pool",
    },
    "paw_ucb_reference_dimer_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "reference_dimer",
        "reference_dimer_max_steps": 15,
        "reference_dimer_rotation_tol": 0.03,
    },
    "paw_ucb_pool_no_momentum_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "scored_pool",
        "direction_pool_disable_momentum": True,
    },
    "paw_ucb_reference_dimer_direct_qp_adaptive50": {
        "proposal_step_mode": "direct_qp",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "reference_dimer",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
    },
}
```

Add external reference support as a separately logged path:

```python
REFERENCE_VARIANT = "reference_original"
REFERENCE_ROOT = Path("/tmp/ssw-reference")
```

The runner must execute the C60 external reference baseline through:

```bash
python /tmp/ssw-reference/c60/run_c60_global_opt.py \
  --methods ssw \
  --input /mnt/d/Download/trae-research-code/SSW/runs/20260428-c60-mace-production/prerelaxed_c60.xyz \
  --model /root/.cache/mace/mace-omat-0-small.model \
  --outdir runs/20260601-reference-dimer-component-ablation/output/c60_reference_original_seed0_trials40_cuda \
  --seed 0 \
  --device cuda \
  --default-dtype float32 \
  --ssw-steps 40 \
  --ssw-dimer-steps 15 \
  --ssw-dimer-tol 0.03 \
  --ssw-optimizer fire
```

For seeds 1 and 2, replace `--seed 0` and the output directory suffix. The runner should parse `/tmp/ssw-reference` `summary.json` and normalize it to the same result row keys used by PAM-SSW rows.

- [ ] **Step 3: Run runner help or dry parse**

Run:

```bash
python runs/20260601-reference-dimer-component-ablation/run_matrix.py --help
```

Expected: exit code 0 and CLI options for systems, variants, seeds, trials, and device.

---

## Task 7: Verification

**Files:**
- All files touched above.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest \
  tests/unit/test_config.py::test_config_accepts_reference_dimer_direction_engine_defaults \
  tests/unit/test_config.py::test_config_rejects_invalid_reference_dimer_controls \
  tests/unit/test_config.py::test_config_accepts_direction_pool_disable_momentum \
  tests/unit/test_walker_policy.py::test_reference_dimer_sample_mixed_mode_returns_normalized_direction \
  tests/unit/test_walker_policy.py::test_reference_dimer_rotator_returns_direction_and_curvature \
  tests/unit/test_walker_policy.py::test_direction_candidate_kind_includes_reference_dimer \
  tests/unit/test_walker_policy.py::test_reference_dimer_direction_engine_bypasses_scored_pool \
  tests/unit/test_walker_policy.py::test_reference_dimer_direct_qp_uses_choice_curvature_without_true_hvp \
  tests/unit/test_walker_policy.py::test_direction_pool_disable_momentum_filters_momentum_candidates \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full unit suite**

Run:

```bash
pytest tests/unit -q
```

Expected: PASS. The previous known baseline was 393 passing unit tests before this plan.

- [ ] **Step 3: Run one CPU smoke**

Run:

```bash
python runs/20260601-reference-dimer-component-ablation/run_matrix.py \
  --systems c60 \
  --variants paw_ucb_reference_dimer_bias_relax \
  --seeds 0 \
  --trials 1 \
  --device cpu
```

Expected:

- A summary JSON is written under `runs/20260601-reference-dimer-component-ablation/output/`.
- `stats.reference_dimer_steps` is greater than zero.
- `stats.direction_selected_reference_dimer` is greater than zero.
- The run exits without geometry or finite-value failures.

- [ ] **Step 4: Update ledger after smoke**

Append command start/completion events to `runs/20260601-reference-dimer-component-ablation/event_log.jsonl`.

Update `artifacts.json` with the smoke summary path:

```json
{
  "artifact_id": "c60_reference_dimer_smoke_summary",
  "type": "summary_json",
  "path": "runs/20260601-reference-dimer-component-ablation/output/<case>/ssw_summary.json",
  "producer_step": "cpu_smoke",
  "description": "One-trial CPU smoke for UCB + reference dimer + bias relax"
}
```

Update `status.md` with the smoke result and whether CUDA benchmark execution is ready.

---

## Task 8: Approved Benchmark Execution Plan

**Files:**
- Modify: `runs/20260601-reference-dimer-component-ablation/plan.md`
- Modify: `runs/20260601-reference-dimer-component-ablation/status.md`
- Modify during execution: `event_log.jsonl`, `artifacts.json`, `summary.md`

- [ ] **Step 1: Ask for explicit benchmark approval**

After implementation tests and CPU smoke pass, ask the user to approve this command group:

```bash
python runs/20260601-reference-dimer-component-ablation/run_matrix.py \
  --systems c60 \
  --variants reference_original,paw_current_bias_relax,paw_metropolis_pool_bias_relax,paw_ucb_reference_dimer_bias_relax,paw_ucb_pool_no_momentum_bias_relax,paw_ucb_reference_dimer_direct_qp_adaptive50 \
  --seeds 0,1,2 \
  --trials 40 \
  --device cuda
```

Expected runtime depends on dimer convergence and true quench cost. Treat this as a long CUDA benchmark.

- [ ] **Step 2: Run Phase 1 only after approval**

Use escalated execution because CUDA access and long-running benchmark execution require it in this environment.

Expected artifacts:

- `results.csv`
- `results.json`
- `summary.md`
- per-case `ssw_summary.json`
- per-case `energy_trace.json`
- per-case `archive_minima.xyz`
- per-case `best_minimum.xyz`

- [ ] **Step 3: Analyze Phase 1 before Phase 2**

Compute and report:

- mean and best `best_energy` by variant,
- mean `n_minima`,
- mean `duplicate_rate`,
- mean `force_evaluations`,
- `best_energy_per_1k_force`,
- direction productivity for reference dimer and momentum-like directions,
- C60 geometry sanity.

Proceed to CuO/PdO only if Phase 1 implementation is stable and C60 results are interpretable.

- [ ] **Step 4: Ask for Phase 2 approval**

Ask before running:

```bash
python runs/20260601-reference-dimer-component-ablation/run_matrix.py \
  --systems cuo,pdo \
  --variants reference_original,paw_current_bias_relax,paw_ucb_reference_dimer_bias_relax \
  --seeds 0,1,2 \
  --trials 40 \
  --device cuda
```

- [ ] **Step 5: Finalize summary**

Write `runs/20260601-reference-dimer-component-ablation/summary.md` with:

- exact command list,
- artifact paths,
- result table,
- component-level conclusions,
- whether to promote `reference_dimer`,
- whether to keep UCB selector,
- whether to disable momentum-like directions for C60,
- whether Direct-QP remains a C60 production candidate.

---

## Self-Review

Spec coverage:

- External `/tmp/ssw-reference` baseline is covered by Task 6 and Task 8.
- Opt-in reference dimer direction engine is covered by Tasks 1-4.
- Direction pool no-momentum ablation is covered by Task 5.
- k=3 C60 matrix and slab sanity matrix are covered by Task 8.
- Run ledger requirements are covered by Task 6 and Task 8.
- Diagnostics and decision criteria are covered by Tasks 3, 6, 7, and 8.

Every task has concrete files, commands, and expected outcomes.
