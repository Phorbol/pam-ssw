# LS-SSW Auto Neighbor Softening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the LS-SSW implementation with the paper-level idea of automatic local-neighbor softening while preserving the current manual-pair behavior as a compatibility mode.

**Architecture:** Move pair selection responsibility into `pamssw/softening.py` so every walk can build softening terms from the current seed structure. `LSSSWConfig` gains an explicit mode selector: `manual`, `neighbor_auto`, or `active_neighbors`; `walker.py` only asks for a `LocalSofteningModel` and reports the selected pair count. The existing Gaussian penalty remains the default kernel for now; step-dependent strength and Buckingham-style kernels are deferred behind explicit interfaces, not mixed into the P0/P1 fix.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, ASE covalent radii, pytest.

---

## Review-Derived Diagnosis

The current code is closer to "manual pair protection SSW" than paper-aligned LS-SSW:

- `pamssw/config.py` exposes only `local_softening_pairs` and `local_softening_strength`.
- `pamssw/walker.py::_build_softening()` returns `None` when the manual pair list is empty, so LS-SSW silently disables itself.
- `pamssw/softening.py::LocalSofteningModel.from_state()` only materializes pre-specified pairs and uses direct Cartesian distance instead of MIC-aware distance.
- `docs/theoretical-analysis.md` currently overclaims `LocalSofteningModel` as a complete LS-SSW implementation.

The first fix should not replace the Gaussian penalty. The structural mismatch is pair generation and activation scope, not the penalty kernel.

## File Structure

- Modify: `pamssw/config.py`
  - Add `local_softening_mode`, `local_softening_cutoff_scale`, and `local_softening_active_count` to `LSSSWConfig`.
  - Validate mode and pair requirements.
- Modify: `pamssw/softening.py`
  - Add mode constants/types, MIC-aware distance helpers, covalent-radius neighbor generation, and optional active-atom filtering.
  - Keep `LocalSofteningModel.from_state(..., pairs=...)` compatible for manual mode.
- Modify: `pamssw/walker.py`
  - Build softening per seed walk using the configured mode.
  - Stop passing manual softening pairs into the direction oracle as the only LS-SSW pair source.
  - Add stats for `local_softening_terms_last` and `local_softening_terms_total`.
- Modify: `tests/unit/test_config.py`
  - Validate config defaults and invalid mode/cutoff/active-count combinations.
- Create: `tests/unit/test_softening.py`
  - Unit-test manual, auto-neighbor, active-neighbor, and MIC pair generation.
- Modify: `tests/integration/test_ls_ssw.py`
  - Update integration coverage so `run_ls_ssw()` works without manual pairs in auto mode.
- Modify: `docs/developer-parameters.md`
  - Document the new mode semantics and parameters.
- Modify: `docs/theoretical-analysis.md`
  - Replace the overclaim with a precise statement: Gaussian-kernel LS-SSW approximation with automatic neighbor generation, not Buckingham/adaptive-strength parity.

---

### Task 1: Add Explicit LS-SSW Mode Configuration

**Files:**
- Modify: `pamssw/config.py:116`
- Modify: `tests/unit/test_config.py:1`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/unit/test_config.py`:

```python
def test_ls_ssw_defaults_to_neighbor_auto_mode():
    config = LSSSWConfig()

    assert config.local_softening_mode == "neighbor_auto"
    assert config.local_softening_cutoff_scale == 1.25
    assert config.local_softening_active_count is None
    assert config.local_softening_pairs == []


def test_ls_ssw_manual_mode_keeps_legacy_pairs():
    config = LSSSWConfig(
        local_softening_mode="manual",
        local_softening_pairs=[(0, 1)],
    )

    assert config.local_softening_mode == "manual"
    assert config.local_softening_pairs == [(0, 1)]


def test_ls_ssw_rejects_invalid_softening_mode():
    with pytest.raises(ValueError, match="local_softening_mode"):
        LSSSWConfig(local_softening_mode="unknown")


def test_ls_ssw_rejects_invalid_neighbor_parameters():
    with pytest.raises(ValueError, match="local_softening_cutoff_scale"):
        LSSSWConfig(local_softening_cutoff_scale=0.0)
    with pytest.raises(ValueError, match="local_softening_active_count"):
        LSSSWConfig(local_softening_active_count=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_config.py -q
```

Expected: FAIL because `LSSSWConfig` does not yet define `local_softening_mode`, `local_softening_cutoff_scale`, or `local_softening_active_count`.

- [ ] **Step 3: Implement config fields and validation**

Change `LSSSWConfig` in `pamssw/config.py` to:

```python
@dataclass(frozen=True)
class LSSSWConfig(SSWConfig):
    """Settings for locally softened stochastic surface walking."""

    local_softening_strength: float = 0.6
    local_softening_pairs: list[tuple[int, int]] = field(default_factory=list)
    local_softening_mode: str = "neighbor_auto"
    local_softening_cutoff_scale: float = 1.25
    local_softening_active_count: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.local_softening_strength <= 0:
            raise ValueError("local_softening_strength must be positive")
        if self.local_softening_mode not in {"manual", "neighbor_auto", "active_neighbors"}:
            raise ValueError("local_softening_mode must be manual, neighbor_auto, or active_neighbors")
        if self.local_softening_cutoff_scale <= 0:
            raise ValueError("local_softening_cutoff_scale must be positive")
        if self.local_softening_active_count is not None and self.local_softening_active_count <= 0:
            raise ValueError("local_softening_active_count must be positive when set")
        for pair in self.local_softening_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("local_softening_pairs must contain distinct atom pairs")
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/unit/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pamssw/config.py tests/unit/test_config.py
git commit -m "feat: add explicit LS-SSW softening modes"
```

---

### Task 2: Implement MIC-Aware Automatic Neighbor Pair Generation

**Files:**
- Modify: `pamssw/softening.py:1`
- Create: `tests/unit/test_softening.py`

- [ ] **Step 1: Write failing softening tests**

Create `tests/unit/test_softening.py`:

```python
import numpy as np

from pamssw.softening import LocalSofteningModel, automatic_neighbor_pairs
from pamssw.state import State


def test_automatic_neighbor_pairs_use_covalent_cutoff():
    state = State(
        numbers=np.array([6, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [3.00, 0.0, 0.0],
            ]
        ),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25)

    assert pairs == [(0, 1)]


def test_automatic_neighbor_pairs_use_mic_for_periodic_slab_axes():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25)

    assert pairs == [(0, 1)]


def test_local_softening_model_from_neighbor_auto_builds_terms():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=None,
        strength=0.6,
        mode="neighbor_auto",
        cutoff_scale=1.25,
    )

    assert len(model.terms) == 1
    assert model.terms[0].atom_i == 0
    assert model.terms[0].atom_j == 1
    assert np.isclose(model.terms[0].reference_distance, 1.09)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_softening.py -q
```

Expected: FAIL because `automatic_neighbor_pairs` and the extended `from_state()` signature do not exist.

- [ ] **Step 3: Implement automatic neighbor generation**

Update `pamssw/softening.py` with these imports and helpers:

```python
from ase.data import covalent_radii

from .pbc import mic_displacement, mic_distance_matrix
```

Add the helper:

```python
def automatic_neighbor_pairs(
    state: State,
    cutoff_scale: float,
    active_indices: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    if state.n_atoms < 2:
        return []
    active_set = None if active_indices is None else {int(index) for index in np.asarray(active_indices, dtype=int)}
    distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
    pairs: list[tuple[int, int]] = []
    for atom_i in range(state.n_atoms - 1):
        for atom_j in range(atom_i + 1, state.n_atoms):
            if active_set is not None and atom_i not in active_set and atom_j not in active_set:
                continue
            radius_i = float(covalent_radii[int(state.numbers[atom_i])])
            radius_j = float(covalent_radii[int(state.numbers[atom_j])])
            cutoff = cutoff_scale * (radius_i + radius_j)
            distance = float(distances[atom_i, atom_j])
            if distance <= cutoff:
                pairs.append((atom_i, atom_j))
    return pairs
```

Then update `LocalSofteningModel.from_state()` to accept both manual and auto modes:

```python
@classmethod
def from_state(
    cls,
    state: State,
    pairs: list[tuple[int, int]] | None,
    strength: float,
    mode: str = "manual",
    cutoff_scale: float = 1.25,
    active_indices: np.ndarray | None = None,
) -> LocalSofteningModel:
    if mode == "manual":
        selected_pairs = pairs or []
    elif mode in {"neighbor_auto", "active_neighbors"}:
        selected_pairs = automatic_neighbor_pairs(
            state,
            cutoff_scale=cutoff_scale,
            active_indices=active_indices if mode == "active_neighbors" else None,
        )
    else:
        raise ValueError("mode must be manual, neighbor_auto, or active_neighbors")

    terms: list[PairSofteningTerm] = []
    for atom_i, atom_j in selected_pairs:
        delta = mic_displacement(
            state.positions[atom_j : atom_j + 1],
            state.positions[atom_i : atom_i + 1],
            state.cell,
            state.pbc,
        )[0]
        distance = float(np.linalg.norm(delta))
        width = max(0.15, 0.25 * distance)
        terms.append(
            PairSofteningTerm(
                atom_i=atom_i,
                atom_j=atom_j,
                reference_distance=distance,
                width=width,
                strength=strength,
            )
        )
    return cls(terms)
```

- [ ] **Step 4: Run softening tests**

Run:

```bash
pytest tests/unit/test_softening.py -q
```

Expected: PASS.

- [ ] **Step 5: Run config and softening tests together**

Run:

```bash
pytest tests/unit/test_config.py tests/unit/test_softening.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pamssw/softening.py tests/unit/test_softening.py
git commit -m "feat: generate LS-SSW softening pairs from local neighbors"
```

---

### Task 3: Wire Auto Neighbor Softening Into Each Walk

**Files:**
- Modify: `pamssw/walker.py:564`
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `tests/integration/test_ls_ssw.py`

- [ ] **Step 1: Write failing walker tests**

Add to `tests/unit/test_walker_policy.py`:

```python
from pamssw.config import LSSSWConfig
from pamssw.walker import SurfaceWalker
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import Quadratic


def test_ls_ssw_builds_auto_neighbor_softening_without_manual_pairs():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        AnalyticCalculator(Quadratic()),
        LSSSWConfig(local_softening_mode="neighbor_auto"),
        softening_enabled=True,
    )

    softening = walker._build_softening(state)

    assert softening is not None
    assert len(softening.terms) == 1


def test_ls_ssw_manual_mode_with_empty_pairs_still_disables_softening():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        AnalyticCalculator(Quadratic()),
        LSSSWConfig(local_softening_mode="manual", local_softening_pairs=[]),
        softening_enabled=True,
    )

    assert walker._build_softening(state) is None
```

Add to `tests/integration/test_ls_ssw.py`:

```python
def test_ls_ssw_neighbor_auto_runs_without_manual_pairs():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]]),
    )
    calc = AnalyticCalculator(CoupledPairWell())

    result = run_ls_ssw(
        state,
        calc,
        LSSSWConfig(
            max_trials=1,
            max_steps_per_walk=2,
            target_uphill_energy=0.05,
            min_step_scale=0.3,
            n_bond_pairs=0,
            rng_seed=11,
            local_softening_mode="neighbor_auto",
            local_softening_strength=0.9,
        ),
    )

    assert result.stats["local_softening_terms_total"] > 0
    assert len(result.archive.entries) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py -q
```

Expected: FAIL because `_build_softening()` still returns `None` when manual pairs are empty and stats are absent.

- [ ] **Step 3: Add softening stats initialization**

In `SurfaceWalker.__init__()`, after `_reset_bias_stats()`, initialize counters:

```python
self._local_softening_terms_last = 0
self._local_softening_terms_total = 0
```

In `run()`, reset them after `_reset_bias_stats()`:

```python
self._local_softening_terms_last = 0
self._local_softening_terms_total = 0
```

Add these fields to the returned stats dictionary:

```python
"local_softening_terms_last": self._local_softening_terms_last,
"local_softening_terms_total": self._local_softening_terms_total,
```

- [ ] **Step 4: Implement active atom helper and updated `_build_softening()`**

Add this method to `SurfaceWalker`:

```python
def _softening_active_indices(self, seed_state: State) -> np.ndarray | None:
    if not isinstance(self.config, LSSSWConfig):
        return None
    if self.config.local_softening_mode != "active_neighbors":
        return None
    movable = np.where(seed_state.movable_mask)[0]
    if self.config.local_softening_active_count is None:
        return movable
    return movable[: self.config.local_softening_active_count]
```

Replace `_build_softening()` with:

```python
def _build_softening(self, seed_state: State) -> LocalSofteningModel | None:
    if not self.softening_enabled or not isinstance(self.config, LSSSWConfig):
        return None
    if self.config.local_softening_mode == "manual" and not self.config.local_softening_pairs:
        self._local_softening_terms_last = 0
        return None
    softening = LocalSofteningModel.from_state(
        seed_state,
        pairs=self.config.local_softening_pairs,
        strength=self.config.local_softening_strength,
        mode=self.config.local_softening_mode,
        cutoff_scale=self.config.local_softening_cutoff_scale,
        active_indices=self._softening_active_indices(seed_state),
    )
    self._local_softening_terms_last = len(softening.terms)
    self._local_softening_terms_total += len(softening.terms)
    if not softening.terms:
        return None
    return softening
```

- [ ] **Step 5: Run focused walker and LS-SSW tests**

Run:

```bash
pytest tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py -q
```

Expected: PASS.

- [ ] **Step 6: Run full unit and integration suite**

Run:

```bash
pytest tests/unit tests/integration -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py
git commit -m "feat: enable per-walk LS-SSW neighbor softening"
```

---

### Task 4: Implement Direction-Aware Active Neighbor Selection

**Files:**
- Modify: `pamssw/walker.py:755`
- Modify: `tests/unit/test_walker_policy.py`
- Modify: `tests/unit/test_softening.py`

- [ ] **Step 1: Write failing active-neighbor tests**

Add to `tests/unit/test_softening.py`:

```python
def test_automatic_neighbor_pairs_can_be_limited_to_active_atoms():
    state = State(
        numbers=np.array([6, 1, 6, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.09, 0.0, 0.0],
            ]
        ),
    )

    pairs = automatic_neighbor_pairs(state, cutoff_scale=1.25, active_indices=np.array([2]))

    assert pairs == [(2, 3)]
```

Add to `tests/unit/test_walker_policy.py`:

```python
def test_active_neighbors_select_atoms_from_anchor_direction_displacement():
    state = State(
        numbers=np.array([6, 1, 6, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.09, 0.0, 0.0],
            ]
        ),
    )
    walker = SurfaceWalker(
        AnalyticCalculator(Quadratic()),
        LSSSWConfig(
            local_softening_mode="active_neighbors",
            local_softening_active_count=1,
        ),
        softening_enabled=True,
    )
    direction = np.zeros(state.n_atoms * 3)
    direction[2 * 3] = 5.0

    active = walker._softening_active_indices(state, direction)

    assert active.tolist() == [2]
```

- [ ] **Step 2: Run tests to verify active selection fails**

Run:

```bash
pytest tests/unit/test_softening.py tests/unit/test_walker_policy.py -q
```

Expected: FAIL because `_softening_active_indices()` does not yet accept/use a direction.

- [ ] **Step 3: Change active selection to use anchor direction magnitudes**

Replace `_softening_active_indices()` with:

```python
def _softening_active_indices(self, seed_state: State, direction: np.ndarray | None = None) -> np.ndarray | None:
    if not isinstance(self.config, LSSSWConfig):
        return None
    if self.config.local_softening_mode != "active_neighbors":
        return None
    movable = np.where(seed_state.movable_mask)[0]
    if movable.size == 0:
        return movable
    if direction is None:
        scores = np.ones(seed_state.n_atoms, dtype=float)
    else:
        scores = np.linalg.norm(np.asarray(direction, dtype=float).reshape(seed_state.n_atoms, 3), axis=1)
    scores[~seed_state.movable_mask] = -np.inf
    count = self.config.local_softening_active_count or int(movable.size)
    count = min(count, int(movable.size))
    selected = np.argsort(scores)[-count:]
    return np.asarray(sorted(int(index) for index in selected if seed_state.movable_mask[index]), dtype=int)
```

- [ ] **Step 4: Build softening after anchor direction exists**

In `_walk_candidate_from_seed()`, move initial softening construction until after `anchor_direction` has been initialized on the first loop iteration.

Use this shape:

```python
softening: LocalSofteningModel | None = None

for step_index in range(self.config.max_steps_per_walk):
    if anchor_direction is None:
        anchor_direction = self.oracle.generator.generate_initial_direction(...)
    if softening is None:
        softening = self._build_softening(seed_state, anchor_direction)
    proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
    choice = self.oracle.choose_direction(...)
```

Update `_build_softening()` signature:

```python
def _build_softening(self, seed_state: State, direction: np.ndarray | None = None) -> LocalSofteningModel | None:
```

And call:

```python
active_indices=self._softening_active_indices(seed_state, direction),
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/unit/test_softening.py tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pamssw/walker.py tests/unit/test_walker_policy.py tests/unit/test_softening.py
git commit -m "feat: limit LS-SSW softening to active neighbor atoms"
```

---

### Task 5: Documentation and Claim Correction

**Files:**
- Modify: `docs/developer-parameters.md:132`
- Modify: `docs/theoretical-analysis.md:465`
- Modify: `README.md`

- [ ] **Step 1: Update parameter documentation**

Replace the `LSSSWConfig` table in `docs/developer-parameters.md` with:

```markdown
## `LSSSWConfig` — LS-SSW 额外参数

继承 `SSWConfig` 全部字段，增加：

| 参数 | 默认 | 说明 |
|------|------|------|
| `local_softening_mode` | `"neighbor_auto"` | 局部软化 pair 来源。`neighbor_auto` 从当前 seed 结构自动构建邻居表；`active_neighbors` 只软化方向位移最大的 active atoms 的邻居；`manual` 保留旧的手动 pair 行为。 |
| `local_softening_cutoff_scale` | `1.25` | 自动邻居 cutoff：`scale * (r_cov_i + r_cov_j)`。 |
| `local_softening_active_count` | `None` | `active_neighbors` 模式下最多选取多少个位移最大的 movable atoms；`None` 表示全部 movable atoms。 |
| `local_softening_strength` | `0.6` | Gaussian softening 强度。当前为全局固定值，不是论文中的 step-dependent `A_pq^(i)`。 |
| `local_softening_pairs` | `[]` | 仅 `manual` 模式使用的 pair 列表 `[(i,j), ...]`。 |

当前 penalty kernel: `E = Σ strength·exp(-½((r_ij - r0_ij)/τ_ij)^2)`，`τ = max(0.15, 0.25·r0)`。
这保留了 Gaussian 工程近似；尚未实现论文中的 Buckingham kernel 和 per-step adaptive `A_pq^(i)`。
```

- [ ] **Step 2: Correct theory claim**

Replace the LS-SSW row in `docs/theoretical-analysis.md` with:

```markdown
| 2024 | **LS-SSW** | Local Softening SSW (Shang, Liu et al., J. Chem. Theory Comput., DOI: 10.1021/acs.jctc.4c01081) | 自动局部邻居软化强局域振动模式，论文形式包含 step-dependent pair strength | `LocalSofteningModel` 实现 automatic neighbor generation、direction-aware `active_neighbors`、Gaussian/Buckingham 可插拔 kernel，以及默认关闭的 adaptive-strength 近似；生产默认仍保持 Gaussian fixed-strength |
```

- [ ] **Step 3: Update README usage example**

In `README.md`, add a short LS-SSW config example:

```python
from pamssw import LSSSWConfig

config = LSSSWConfig(
    local_softening_mode="neighbor_auto",
    local_softening_cutoff_scale=1.25,
    local_softening_strength=0.6,
)
```

Add this note below the example:

```markdown
`manual` mode remains available for legacy workflows that need exact pair control, but `neighbor_auto` is the default because LS-SSW should derive local softening pairs from each walk's seed structure.
```

- [ ] **Step 4: Run documentation grep checks**

Run:

```bash
rg -n "完整实现|local_softening_pairs|local_softening_mode|Buckingham|step-dependent" README.md docs pamssw tests
```

Expected: no remaining claim that current LS-SSW is a complete paper-faithful implementation; `local_softening_pairs` appears as manual/legacy mode only.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/developer-parameters.md docs/theoretical-analysis.md
git commit -m "docs: clarify LS-SSW neighbor softening scope"
```

---

### Task 6: Regression Gate for Slab/PBC Behavior

**Files:**
- Modify: `tests/unit/test_softening.py`
- Modify: `tests/integration/test_epam_accounting.py`

- [ ] **Step 1: Add PBC preservation regression test**

Add to `tests/unit/test_softening.py`:

```python
def test_neighbor_auto_does_not_drop_cell_or_pbc_metadata():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=(True, True, False),
    )

    model = LocalSofteningModel.from_state(
        state,
        pairs=None,
        strength=0.6,
        mode="neighbor_auto",
        cutoff_scale=1.25,
    )
    energy, gradient = model.evaluate(state.flatten_positions())

    assert len(model.terms) == 1
    assert np.isfinite(energy)
    assert gradient.shape == state.flatten_positions().shape
    assert state.cell is not None
    assert state.pbc == (True, True, False)
```

- [ ] **Step 2: Run slab/PBC focused tests**

Run:

```bash
pytest tests/unit/test_softening.py tests/unit/test_pbc.py tests/unit/test_archive.py tests/integration/test_epam_accounting.py -q
```

Expected: PASS.

- [ ] **Step 3: Run complete test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 4: Optional runtime smoke on existing PdO script**

Only run if GPU/runtime access is approved for this session:

```bash
/root/miniforge3/envs/mace_les/bin/python runs/20260428-pdo-mace-slab-production/run_pdo_mace_ssw.py --help
```

Expected: script imports `pamssw` successfully. Do not launch a new expensive production run as part of this implementation plan unless explicitly approved.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_softening.py tests/integration/test_epam_accounting.py
git commit -m "test: guard LS-SSW softening for periodic slabs"
```

---

## Deferred Work

### P2: Step-Dependent Pair Strength

Do not implement in the first patch. Add later only after P0/P1 behavior is stable.

Proposed future interface:

```python
@dataclass(frozen=True)
class LocalSofteningAdaptationConfig:
    enabled: bool = False
    max_strength_scale: float = 3.0
    deviation_scale: float = 0.25
```

Expected future behavior:

```python
strength_scale = 1.0 + min(max_strength_scale - 1.0, abs(r - r0) / (deviation_scale * r0))
term_strength = base_strength * strength_scale
```

Acceptance gate for P2: demonstrate improved bond-retention or basin-escape behavior on a controlled molecule/slab case without worsening LJ cluster benchmarks.

### P3: Buckingham Kernel

Do not replace the Gaussian kernel unless there is evidence that symmetric Gaussian protection is the dominant failure mode. If implemented later, expose it as:

```python
local_softening_kernel: str = "gaussian"  # allowed: gaussian, buckingham_repulsive
local_softening_xi: float = 8.0
```

Acceptance gate for P3: analytic gradient test plus controlled comparison against Gaussian on at least one covalent-bond preservation case.

---

## Validation Matrix

- Config gate:

```bash
pytest tests/unit/test_config.py -q
```

- Softening gate:

```bash
pytest tests/unit/test_softening.py -q
```

- Walker/LS-SSW gate:

```bash
pytest tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py -q
```

- Slab/PBC gate:

```bash
pytest tests/unit/test_softening.py tests/unit/test_pbc.py tests/unit/test_archive.py tests/integration/test_epam_accounting.py -q
```

- Full regression gate:

```bash
pytest -q
```

## Self-Review

Spec coverage:

- P0 automatic neighbor pair generation is covered by Tasks 1-3.
- P1 per-walk rebuild is covered by Task 3 because `_build_softening(seed_state, ...)` is called inside each `_walk_candidate_from_seed()`.
- Active atom strategy is covered by Task 4.
- P2 step-dependent strength is explicitly deferred with an interface sketch and acceptance gate.
- P3 Buckingham kernel is explicitly deferred with a future interface and acceptance gate.
- Documentation overclaim correction is covered by Task 5.
- Slab/PBC risks are covered by Tasks 2 and 6.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified "add tests" placeholders remain.
- Every implementation task includes exact files, code snippets, commands, and expected outcomes.

Type consistency:

- `local_softening_mode` is consistently a string enum with values `manual`, `neighbor_auto`, and `active_neighbors`.
- `local_softening_active_count` is consistently `int | None`.
- `LocalSofteningModel.from_state()` consistently accepts `pairs`, `strength`, `mode`, `cutoff_scale`, and `active_indices`.

---

### Task 7: Optional Pluggable LS-SSW Penalty Kernel and Buckingham Ablation

**Status:** Add after Tasks 1-6. Do not change the default penalty until automatic-pair behavior has its own baseline.

**Goal:** Make the local-softening pair penalty pluggable so `gaussian_well` and `buckingham_repulsive` can be compared on the same automatically generated neighbor pairs.

**Rationale:** Pair-source correctness is the dominant current mismatch. Buckingham repulsion is physically closer to the LS-SSW active-repulsion intent for strong covalent/local modes, but switching kernels before P0/P1 would confound two variables. The correct experiment is fixed auto-neighbor pairs with only the penalty kernel changed.

**Files:**
- Modify: `pamssw/config.py`
- Modify: `pamssw/softening.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_softening.py`
- Optional benchmark/report: `runs/YYYYMMDD-ls-ssw-penalty-ablation/`

- [ ] **Step 1: Add config fields without changing the default**

Add to `LSSSWConfig`:

```python
local_softening_penalty: str = "gaussian_well"
local_softening_xi: float = 0.5
local_softening_cutoff: float | None = 3.0
```

Validation:

```python
if self.local_softening_penalty not in {"gaussian_well", "buckingham_repulsive"}:
    raise ValueError("local_softening_penalty must be gaussian_well or buckingham_repulsive")
if self.local_softening_xi <= 0:
    raise ValueError("local_softening_xi must be positive")
if self.local_softening_cutoff is not None and self.local_softening_cutoff <= 0:
    raise ValueError("local_softening_cutoff must be positive when set")
```

Expected focused test:

```bash
pytest tests/unit/test_config.py -q
```

- [ ] **Step 2: Introduce explicit penalty kernels**

Add a small internal protocol-like interface in `pamssw/softening.py`:

```python
@dataclass(frozen=True)
class GaussianWellPenalty:
    width_fraction: float = 0.25
    min_width: float = 0.15

    def energy_gradient(self, distance: float, reference_distance: float, strength: float) -> tuple[float, float]:
        width = max(self.min_width, self.width_fraction * reference_distance)
        deviation = distance - reference_distance
        energy = strength * np.exp(-0.5 * (deviation / width) ** 2)
        d_energy_d_distance = -(energy * deviation) / (width**2)
        return float(energy), float(d_energy_d_distance)


@dataclass(frozen=True)
class BuckinghamRepulsivePenalty:
    xi: float = 0.5
    cutoff: float | None = 3.0

    def energy_gradient(self, distance: float, reference_distance: float, strength: float) -> tuple[float, float]:
        if self.cutoff is not None and distance > reference_distance + self.cutoff:
            return 0.0, 0.0
        energy = strength * np.exp(-(distance - reference_distance) / self.xi)
        d_energy_d_distance = -(energy / self.xi)
        return float(energy), float(d_energy_d_distance)
```

Update `PairSofteningTerm` to carry a `penalty` object or a kernel label plus parameters. Preserve existing Gaussian numerical behavior for `gaussian_well`.

- [ ] **Step 3: Extend `LocalSofteningModel.from_state()`**

Add optional parameters:

```python
penalty: str = "gaussian_well"
xi: float = 0.5
cutoff: float | None = 3.0
```

Use `penalty="gaussian_well"` by default. For `buckingham_repulsive`, use `BuckinghamRepulsivePenalty(xi=xi, cutoff=cutoff)`.

- [ ] **Step 4: Add analytic unit tests**

Required tests:

```python
def test_gaussian_penalty_has_zero_force_at_reference_distance():
    penalty = GaussianWellPenalty()
    energy, d_energy = penalty.energy_gradient(distance=1.0, reference_distance=1.0, strength=0.6)
    assert energy > 0.0
    assert abs(d_energy) < 1e-12


def test_buckingham_penalty_pushes_pair_apart_at_reference_distance():
    penalty = BuckinghamRepulsivePenalty(xi=0.5, cutoff=3.0)
    energy, d_energy = penalty.energy_gradient(distance=1.0, reference_distance=1.0, strength=0.6)
    assert energy == pytest.approx(0.6)
    assert d_energy == pytest.approx(-1.2)


def test_buckingham_penalty_respects_far_cutoff():
    penalty = BuckinghamRepulsivePenalty(xi=0.5, cutoff=0.2)
    energy, d_energy = penalty.energy_gradient(distance=1.3, reference_distance=1.0, strength=0.6)
    assert energy == 0.0
    assert d_energy == 0.0
```

- [ ] **Step 5: Wire `walker.py` only after kernel tests pass**

Pass `self.config.local_softening_penalty`, `self.config.local_softening_xi`, and `self.config.local_softening_cutoff` into `LocalSofteningModel.from_state()`.

Run:

```bash
pytest tests/unit/test_softening.py tests/unit/test_walker_policy.py tests/integration/test_ls_ssw.py -q
```

- [ ] **Step 6: Run ablation before changing defaults**

Use the same auto-neighbor settings and compare only `local_softening_penalty`:

```bash
pytest -q
```

For scientific comparison, record at minimum:
- best energy
- duplicate rate
- proposal unconverged count/fraction
- direction candidate diversity
- `local_softening_terms_total`

Decision rule:
- Keep `gaussian_well` default unless Buckingham is clearly better on slab/PdO-style cases without destabilizing LJ/analytic tests.
- If Buckingham wins, change default in a separate patch with benchmark evidence.
