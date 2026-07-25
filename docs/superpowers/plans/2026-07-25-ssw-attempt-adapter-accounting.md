# SSW Attempt Adapter and Exact Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect one posterior-exploration starter action to one isolated, reproducible, single-trial SSW run with exact analytic-calculator call accounting under a per-action budget.

**Architecture:** Keep `ExplorationController` unchanged and add a callable `SSWAttemptWorker` in `pamssw.exploration`. Each call creates a fresh calculator and `SurfaceWalker`, overrides only action-owned execution identity fields, extracts the actual landing from the one-trial walk history, and returns an `AttemptResult`. First make `EvalCounter` the only calculator gateway, then validate the worker alone, and finally validate it through a real `ThreadPoolExecutor` batch.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, `concurrent.futures.ThreadPoolExecutor`, pytest, existing analytic potentials and PAM-SSW kernel.

---

## Scope and File Map

Create:

```text
pamssw/exploration/ssw_worker.py
tests/unit/test_accounting.py
tests/unit/test_ssw_attempt_worker.py
tests/integration/test_ssw_attempt_worker.py
```

Modify:

```text
pamssw/accounting.py
pamssw/walker.py
pamssw/exploration/__init__.py
tests/integration/test_epam_accounting.py
tests/integration/test_exploration_controller.py
README.md
```

Do not modify:

```text
pamssw/acquisition.py
pamssw/exploration/policies.py
pamssw/exploration/posterior.py
pamssw/exploration/controller.py
pamssw/runner.py
pamssw/config.py
```

No new configurable weights, selection rules, direction sources, purpose-cost
weights, process/GPU backends, or top-level package exports are part of this
plan.

### Task 1: Count Every Started Calculator Call Exactly Once

**Files:**
- Create: `tests/unit/test_accounting.py`
- Modify: `pamssw/accounting.py`

- [ ] **Step 1: Write failing counter tests**

Create `tests/unit/test_accounting.py`:

```python
import numpy as np
import pytest

from pamssw.accounting import BudgetExceeded, EvalCounter
from pamssw.calculators import EnergyResult
from pamssw.state import State


def _state() -> State:
    return State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )


class RecordingCalculator:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def _record(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic calculator failure")

    def evaluate(self, state: State) -> EnergyResult:
        self._record()
        return EnergyResult(energy=0.0, gradient=np.zeros_like(state.positions))

    def evaluate_flat(self, flat_positions: np.ndarray, template: State):
        self._record()
        return 0.0, np.zeros_like(flat_positions)


def test_counter_counts_a_started_call_that_raises():
    calculator = RecordingCalculator(fail_on_call=1)
    counter = EvalCounter(calculator, max_force_evals=2)

    with pytest.raises(RuntimeError, match="synthetic calculator failure"):
        counter.evaluate(_state())

    assert calculator.calls == 1
    assert counter.force_evaluations == 1
    assert counter.energy_evaluations == 1


def test_counter_shares_one_budget_across_both_interfaces():
    calculator = RecordingCalculator()
    counter = EvalCounter(calculator, max_force_evals=2)
    state = _state()

    counter.evaluate(state)
    counter.evaluate_flat(state.flatten_positions(), state)

    with pytest.raises(BudgetExceeded):
        counter.evaluate(state)

    assert calculator.calls == 2
    assert counter.force_evaluations == 2
    assert counter.energy_evaluations == 2
    assert counter.exhausted()


def test_budget_rejection_does_not_call_or_increment():
    calculator = RecordingCalculator()
    counter = EvalCounter(calculator, max_force_evals=1)
    state = _state()
    counter.evaluate(state)

    with pytest.raises(BudgetExceeded):
        counter.evaluate_flat(state.flatten_positions(), state)

    assert calculator.calls == 1
    assert counter.force_evaluations == 1
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_accounting.py
```

Expected: `test_counter_counts_a_started_call_that_raises` fails because the
counter currently increments only after the delegated call succeeds.

- [ ] **Step 3: Move increments before delegation**

Change `EvalCounter.evaluate` and `evaluate_flat` in
`pamssw/accounting.py` to:

```python
def evaluate(self, state: State):
    self._reserve()
    self.force_evaluations += 1
    self.energy_evaluations += 1
    return self.calculator.evaluate(state)

def evaluate_flat(
    self,
    flat_positions: np.ndarray,
    template: State,
) -> tuple[float, np.ndarray]:
    self._reserve()
    self.force_evaluations += 1
    self.energy_evaluations += 1
    return self.calculator.evaluate_flat(flat_positions, template)
```

Do not change `_reserve` or add retry logic.

- [ ] **Step 4: Run focused and existing accounting tests**

Run:

```bash
pytest -q \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add pamssw/accounting.py tests/unit/test_accounting.py
git commit -m "Count started calculator evaluations"
```

### Task 2: Route Every Oracle Evaluation Through the Counter

**Files:**
- Modify: `pamssw/walker.py:1528-1538`
- Modify: `pamssw/walker.py:1737-1780`
- Modify: `tests/integration/test_epam_accounting.py`

- [ ] **Step 1: Add an instrumented analytic calculator**

Append to `tests/integration/test_epam_accounting.py`:

```python
class CountingAnalyticCalculator:
    def __init__(self) -> None:
        self.inner = AnalyticCalculator(DoubleWell2D())
        self.calls = 0

    def evaluate(self, state):
        self.calls += 1
        return self.inner.evaluate(state)

    def evaluate_flat(self, flat_positions, template):
        self.calls += 1
        return self.inner.evaluate_flat(flat_positions, template)


def test_direction_probe_calls_share_the_walker_counter():
    calculator = CountingAnalyticCalculator()
    state = State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    walker = SurfaceWalker(
        calculator=calculator,
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=3,
            direction_probe_enabled=True,
            direction_probe_top_k=2,
            rng_seed=7,
        ),
        softening_enabled=False,
    )

    result = walker.run(state)

    assert walker.oracle.calculator is walker.calculator
    assert result.stats["force_evaluations"] == calculator.calls
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
pytest -q \
  tests/integration/test_epam_accounting.py::test_direction_probe_calls_share_the_walker_counter
```

Expected: the identity assertion fails, and the underlying calculator count is
greater than the walker count when the direct probe path runs.

- [ ] **Step 3: Pass the wrapped counter to `SoftModeOracle`**

In `SurfaceWalker.__init__`, replace:

```python
self.oracle = SoftModeOracle(
    calculator,
```

with:

```python
self.oracle = SoftModeOracle(
    self.calculator,
```

Do not otherwise change `SoftModeOracle`, `ProposalPotential`, direction
scoring, or HVP formulas.

- [ ] **Step 3a: Preserve budget exhaustion as a control signal**

The direct-probe loop currently catches every `Exception`. Once it uses the
shared counter, add this ordering:

```python
try:
    probe_energy = self.calculator.evaluate(trial_state).energy
except BudgetExceeded:
    raise
except Exception:
    continue
```

Add a regression that exhausts the counter during probe refinement and proves:

- no direction choice is recorded after exhaustion;
- raw calculator calls equal counter calls;
- both equal the configured budget;
- ordinary non-budget probe failures are still skipped.

This changes termination timing only. It does not change direction formulas,
candidate scores, or the treatment of ordinary calculator failures.

- [ ] **Step 4: Run accounting, walker, and SSW integration tests**

Run:

```bash
pytest -q \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py \
  tests/integration/test_ls_ssw.py \
  tests/unit/test_walker_policy.py
```

Expected: all selected tests pass. Budgeted test counts may become larger or
stop earlier because previously bypassed calls are now included; no count may
exceed `max_force_evals`.

- [ ] **Step 5: Commit**

```bash
git add pamssw/walker.py tests/integration/test_epam_accounting.py
git commit -m "Route oracle evaluations through accounting"
```

### Task 3: Implement the Isolated Single-Trial SSW Worker

**Files:**
- Create: `pamssw/exploration/ssw_worker.py`
- Create: `tests/unit/test_ssw_attempt_worker.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing constructor and mapping tests**

Create `tests/unit/test_ssw_attempt_worker.py`:

```python
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw import SSWConfig
from pamssw.archive import MinimaArchive
from pamssw.calculators import AnalyticCalculator
from pamssw.exploration import AttemptStatus, StarterAction
from pamssw.exploration.ssw_worker import SSWAttemptWorker
from pamssw.potentials import DoubleWell2D
from pamssw.result import SearchResult, WalkRecord
from pamssw.state import State


def _state(x: float) -> State:
    return State(
        numbers=np.array([1]),
        positions=np.array([[x, 0.0, 0.0]]),
    )


def _action(*, budget: int | None = 20) -> StarterAction:
    return StarterAction(
        action_id="batch-00000000-slot-0000",
        batch_id=0,
        slot_id=0,
        policy_name="uniform",
        policy_version=0,
        archive_version=0,
        starter_id=0,
        selection_probability=1.0,
        random_seed=17,
        force_budget=budget,
    )


def _search_result(
    *,
    landing: bool,
    budget_exhausted: bool = False,
    fragments: int = 0,
    force_evaluations: int = 7,
) -> SearchResult:
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.05)
    starter = archive.add(_state(-1.0), -2.0, parent_id=None)
    history = []
    if landing:
        discovered = archive.add(_state(1.0), -1.0, parent_id=starter.entry_id)
        history.append(
            WalkRecord(
                seed_entry_id=starter.entry_id,
                discovered_entry_id=discovered.entry_id,
                energy=discovered.energy,
                accepted_new_basin=True,
            )
        )
    return SearchResult(
        best_state=starter.state,
        best_energy=starter.energy,
        archive=archive,
        walk_history=history,
        stats={
            "force_evaluations": force_evaluations,
            "budget_exhausted": int(budget_exhausted),
            "fragment_rejections": fragments,
        },
    )


def test_worker_rejects_hidden_internal_proposal_competition():
    with pytest.raises(ValueError, match="proposal_pool_size"):
        SSWAttemptWorker(
            lambda: AnalyticCalculator(DoubleWell2D()),
            SSWConfig(proposal_pool_size=2),
        )

    with pytest.raises(ValueError, match="proposal_duplicate_rescue_optimizer"):
        SSWAttemptWorker(
            lambda: AnalyticCalculator(DoubleWell2D()),
            SSWConfig(proposal_duplicate_rescue_optimizer="ase-fire"),
        )


@pytest.mark.parametrize(
    "config",
    [
        SSWConfig(accepted_structures_log="shared.jsonl"),
        SSWConfig(accepted_structures_dir="shared"),
        SSWConfig(write_proposal_minima=True, proposal_minima_dir="shared"),
        SSWConfig(
            write_relaxation_trajectories=True,
            relaxation_trajectory_dir="shared",
        ),
        SSWConfig(
            direction_diagnostics_enabled=True,
            direction_diagnostics_path="shared.jsonl",
        ),
        SSWConfig(
            direction_archive_enabled=True,
            direction_archive_path="shared.jsonl",
        ),
    ],
)
def test_worker_rejects_shared_filesystem_outputs(config):
    with pytest.raises(ValueError, match="filesystem output"):
        SSWAttemptWorker(
            lambda: AnalyticCalculator(DoubleWell2D()),
            config,
        )


def test_worker_uses_landing_not_global_best(monkeypatch):
    captured = {}

    class FakeWalker:
        def __init__(self, calculator, config, softening_enabled):
            captured["config"] = config
            captured["softening_enabled"] = softening_enabled
            self.calculator = SimpleNamespace(
                force_evaluations=7,
                exhausted=lambda: False,
            )

        def run(self, starter_state):
            return _search_result(landing=True)

    monkeypatch.setattr(
        "pamssw.exploration.ssw_worker.SurfaceWalker",
        FakeWalker,
    )
    worker = SSWAttemptWorker(
        lambda: AnalyticCalculator(DoubleWell2D()),
        SSWConfig(max_trials=9, rng_seed=99, max_force_evals=999),
    )

    result = worker(_action(budget=20), _state(-1.0))

    assert result.status is AttemptStatus.COMPLETED
    assert result.landing_energy == pytest.approx(-1.0)
    assert result.landing_state.positions[0, 0] == pytest.approx(1.0)
    assert captured["config"].max_trials == 1
    assert captured["config"].rng_seed == 17
    assert captured["config"].max_force_evals == 20


def test_valid_landing_precedes_exact_budget_exhaustion(monkeypatch):
    class FakeWalker:
        def __init__(self, calculator, config, softening_enabled):
            self.calculator = SimpleNamespace(
                force_evaluations=7,
                exhausted=lambda: True,
            )

        def run(self, starter_state):
            return _search_result(
                landing=True,
                budget_exhausted=True,
                force_evaluations=7,
            )

    monkeypatch.setattr(
        "pamssw.exploration.ssw_worker.SurfaceWalker",
        FakeWalker,
    )
    result = SSWAttemptWorker(
        lambda: AnalyticCalculator(DoubleWell2D()),
        SSWConfig(),
    )(_action(budget=7), _state(-1.0))

    assert result.status is AttemptStatus.COMPLETED


@pytest.mark.parametrize(
    ("budget_exhausted", "fragments", "expected"),
    [
        (True, 0, AttemptStatus.BUDGET_EXHAUSTED),
        (False, 1, AttemptStatus.FRAGMENTED),
        (False, 0, AttemptStatus.INVALID),
    ],
)
def test_no_landing_maps_explicit_terminal_status(
    monkeypatch,
    budget_exhausted,
    fragments,
    expected,
):
    class FakeWalker:
        def __init__(self, calculator, config, softening_enabled):
            self.calculator = SimpleNamespace(
                force_evaluations=5,
                exhausted=lambda: budget_exhausted,
            )

        def run(self, starter_state):
            return _search_result(
                landing=False,
                budget_exhausted=budget_exhausted,
                fragments=fragments,
                force_evaluations=5,
            )

    monkeypatch.setattr(
        "pamssw.exploration.ssw_worker.SurfaceWalker",
        FakeWalker,
    )
    result = SSWAttemptWorker(
        lambda: AnalyticCalculator(DoubleWell2D()),
        SSWConfig(),
    )(_action(budget=10), _state(-1.0))

    assert result.status is expected
    assert result.landing_state is None
    assert result.force_evaluations == 5
```

- [ ] **Step 2: Run the unit tests and verify import RED**

Run:

```bash
pytest -q tests/unit/test_ssw_attempt_worker.py
```

Expected: collection fails because
`pamssw.exploration.ssw_worker.SSWAttemptWorker` does not exist.

- [ ] **Step 3: Implement constructor validation and result mapping**

Create `pamssw/exploration/ssw_worker.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from numbers import Integral

from ..accounting import BudgetExceeded
from ..config import LSSSWConfig, SSWConfig
from ..result import SearchResult
from ..state import State
from ..walker import GeometryValidator, SurfaceWalker
from .actions import AttemptResult, AttemptStatus, StarterAction


class SSWAttemptWorker:
    """Run one isolated single-trial SSW attempt for one starter action."""

    def __init__(
        self,
        calculator_factory: Callable[[], object],
        config: SSWConfig,
        *,
        softening_enabled: bool = False,
    ) -> None:
        if not callable(calculator_factory):
            raise ValueError("calculator_factory must be callable")
        if not isinstance(config, SSWConfig):
            raise ValueError("config must be an SSWConfig")
        if not isinstance(softening_enabled, bool):
            raise ValueError("softening_enabled must be a boolean")
        if softening_enabled and not isinstance(config, LSSSWConfig):
            raise ValueError("softening_enabled requires LSSSWConfig")
        if config.proposal_pool_size != 1:
            raise ValueError("proposal_pool_size must equal one")
        if config.proposal_duplicate_rescue_optimizer is not None:
            raise ValueError(
                "proposal_duplicate_rescue_optimizer must be disabled"
            )
        if _filesystem_output_enabled(config):
            raise ValueError(
                "filesystem output must be disabled for threaded attempts"
            )
        self._calculator_factory = calculator_factory
        self._config = config
        self._softening_enabled = softening_enabled

    def __call__(
        self,
        action: StarterAction,
        starter_state: State,
    ) -> AttemptResult:
        if not isinstance(action, StarterAction):
            raise ValueError("action must be a StarterAction")
        if not isinstance(starter_state, State):
            raise ValueError("starter_state must be a State")
        if not GeometryValidator().is_valid_state(starter_state):
            return _failure(
                action,
                AttemptStatus.INVALID,
                0,
                "invalid starter geometry",
            )

        try:
            calculator = self._calculator_factory()
            if not callable(getattr(calculator, "evaluate", None)):
                raise ValueError(
                    "calculator_factory must return an evaluate-capable calculator"
                )
            if not callable(getattr(calculator, "evaluate_flat", None)):
                raise ValueError(
                    "calculator_factory must return an evaluate_flat-capable calculator"
                )
            attempt_config = replace(
                self._config,
                max_trials=1,
                rng_seed=action.random_seed,
                max_force_evals=action.force_budget,
            )
            walker = SurfaceWalker(
                calculator=calculator,
                config=attempt_config,
                softening_enabled=self._softening_enabled,
            )
        except Exception as exc:
            return _failure(
                action,
                AttemptStatus.WORKER_ERROR,
                0,
                f"{type(exc).__name__}: {exc}",
            )

        try:
            result = walker.run(deepcopy(starter_state))
            return _from_search_result(action, result)
        except BudgetExceeded as exc:
            count = walker.calculator.force_evaluations
            status = (
                AttemptStatus.BUDGET_EXHAUSTED
                if walker.calculator.exhausted()
                else AttemptStatus.INVALID
            )
            return _failure(
                action,
                status,
                count,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return _failure(
                action,
                AttemptStatus.WORKER_ERROR,
                walker.calculator.force_evaluations,
                f"{type(exc).__name__}: {exc}",
            )


def _filesystem_output_enabled(config: SSWConfig) -> bool:
    return bool(
        config.accepted_structures_log is not None
        or config.accepted_structures_dir is not None
        or config.write_proposal_minima
        or config.write_relaxation_trajectories
        or config.direction_diagnostics_enabled
        or (
            config.direction_archive_enabled
            and config.direction_archive_path is not None
        )
    )


def _from_search_result(
    action: StarterAction,
    result: SearchResult,
) -> AttemptResult:
    if not isinstance(result, SearchResult):
        raise ValueError("SurfaceWalker.run must return SearchResult")
    count = _stat_nonnegative_int(
        result,
        "force_evaluations",
    )
    if len(result.walk_history) > 1:
        raise ValueError("single-trial worker returned multiple walk records")
    if result.walk_history:
        discovered_id = result.walk_history[0].discovered_entry_id
        matches = [
            entry
            for entry in result.archive.entries
            if entry.entry_id == discovered_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "walk history discovered_entry_id is not unique in archive"
            )
        landing = matches[0]
        if not isfinite(landing.energy):
            raise ValueError("landing energy must be finite")
        return AttemptResult(
            action=action,
            landing_state=deepcopy(landing.state),
            landing_energy=landing.energy,
            force_evaluations=count,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )

    if bool(_stat_nonnegative_int(result, "budget_exhausted")):
        return _failure(
            action,
            AttemptStatus.BUDGET_EXHAUSTED,
            count,
            "force-evaluation budget exhausted before landing",
        )
    if _stat_nonnegative_int(result, "fragment_rejections") > 0:
        return _failure(
            action,
            AttemptStatus.FRAGMENTED,
            count,
            "single proposal rejected as fragmented",
        )
    return _failure(
        action,
        AttemptStatus.INVALID,
        count,
        "no_landing_minimum",
    )


def _stat_nonnegative_int(result: SearchResult, name: str) -> int:
    value = result.stats.get(name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"SearchResult.stats[{name!r}] must be a non-negative integer")
    return int(value)


def _failure(
    action: StarterAction,
    status: AttemptStatus,
    count: int,
    reason: str,
) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=None,
        landing_energy=None,
        force_evaluations=count,
        status=status,
        failure_reason=reason,
    )


__all__ = ["SSWAttemptWorker"]
```

Export `SSWAttemptWorker` from `pamssw/exploration/__init__.py`. Do not add it
to `pamssw/__init__.py`.

- [ ] **Step 4: Add missing exception and invariant tests**

Add tests that prove:

```python
def test_invalid_starter_is_reported_without_creating_calculator():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return AnalyticCalculator(DoubleWell2D())

    worker = SSWAttemptWorker(factory, SSWConfig())
    invalid = State(
        numbers=np.array([1]),
        positions=np.array([[float("nan"), 0.0, 0.0]]),
    )

    result = worker(_action(), invalid)

    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 0
    assert calls == 0


def test_factory_failure_is_zero_cost_worker_error():
    def factory():
        raise RuntimeError("factory failed")

    result = SSWAttemptWorker(factory, SSWConfig())(
        _action(),
        _state(-1.0),
    )

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 0
    assert result.failure_reason == "RuntimeError: factory failed"
```

Also cover:

- factory returns an object without `evaluate`;
- factory returns an object without `evaluate_flat`;
- fake walker returns more than one history record;
- missing or non-integer required stats;
- discovered entry ID is absent or duplicated;
- `softening_enabled=True` rejects plain `SSWConfig`;
- `softening_enabled=True` accepts `LSSSWConfig`.

- [ ] **Step 5: Run the worker unit suite**

Run:

```bash
pytest -q \
  tests/unit/test_ssw_attempt_worker.py \
  tests/unit/test_exploration_actions.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  pamssw/exploration/ssw_worker.py \
  pamssw/exploration/__init__.py \
  tests/unit/test_ssw_attempt_worker.py
git commit -m "Add isolated SSW attempt worker"
```

### Task 4: Validate Real Analytic Attempts and Failure Costs

**Files:**
- Create: `tests/integration/test_ssw_attempt_worker.py`
- Modify: `pamssw/exploration/ssw_worker.py` only if a failing integration test exposes an adapter defect

- [ ] **Step 1: Add a recording analytic calculator**

Create `tests/integration/test_ssw_attempt_worker.py`:

```python
from dataclasses import dataclass

import numpy as np
import pytest

from pamssw import SSWConfig, State
from pamssw.calculators import AnalyticCalculator
from pamssw.exploration import (
    AttemptStatus,
    SSWAttemptWorker,
    StarterAction,
)
from pamssw.potentials import DoubleWell2D


def _state() -> State:
    return State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def _action(action_id: str, seed: int, budget: int) -> StarterAction:
    return StarterAction(
        action_id=action_id,
        batch_id=0,
        slot_id=0,
        policy_name="uniform",
        policy_version=0,
        archive_version=0,
        starter_id=0,
        selection_probability=1.0,
        random_seed=seed,
        force_budget=budget,
    )


@dataclass
class RecordingAnalyticCalculator:
    fail_on_call: int | None = None

    def __post_init__(self) -> None:
        self.inner = AnalyticCalculator(DoubleWell2D())
        self.calls = 0

    def _record(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic analytic failure")

    def evaluate(self, state):
        self._record()
        return self.inner.evaluate(state)

    def evaluate_flat(self, flat_positions, template):
        self._record()
        return self.inner.evaluate_flat(flat_positions, template)
```

- [ ] **Step 2: Test one real action against its calculator calls**

Add:

```python
def test_real_attempt_reports_exact_underlying_calls():
    calculators = []

    def factory():
        calculator = RecordingAnalyticCalculator()
        calculators.append(calculator)
        return calculator

    worker = SSWAttemptWorker(
        factory,
        SSWConfig(
            max_trials=99,
            max_steps_per_walk=1,
            oracle_candidates=2,
            direction_probe_enabled=True,
            direction_probe_top_k=1,
        ),
    )
    result = worker(
        _action("batch-00000000-slot-0000", seed=11, budget=400),
        _state(),
    )

    assert len(calculators) == 1
    assert result.force_evaluations == calculators[0].calls
    assert result.force_evaluations <= 400
    assert result.status in {
        AttemptStatus.COMPLETED,
        AttemptStatus.INVALID,
        AttemptStatus.FRAGMENTED,
    }
```

- [ ] **Step 3: Test deterministic replay with fresh calculators**

Add:

```python
def test_same_action_seed_replays_status_landing_and_cost():
    worker = SSWAttemptWorker(
        lambda: RecordingAnalyticCalculator(),
        SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=2,
        ),
    )
    action = _action("batch-00000000-slot-0000", seed=19, budget=400)

    first = worker(action, _state())
    second = worker(action, _state())

    assert first.status is second.status
    assert first.force_evaluations == second.force_evaluations
    if first.landing_energy is None:
        assert second.landing_energy is None
    else:
        assert first.landing_energy == pytest.approx(second.landing_energy)
    if first.landing_state is not None:
        np.testing.assert_allclose(
            first.landing_state.positions,
            second.landing_state.positions,
        )
```

- [ ] **Step 4: Test budget and calculator failures**

Add:

```python
def test_small_budget_never_overruns():
    calculators = []

    def factory():
        calculator = RecordingAnalyticCalculator()
        calculators.append(calculator)
        return calculator

    result = SSWAttemptWorker(
        factory,
        SSWConfig(max_steps_per_walk=2, oracle_candidates=3),
    )(
        _action("batch-00000000-slot-0000", seed=23, budget=5),
        _state(),
    )

    assert result.status is AttemptStatus.BUDGET_EXHAUSTED
    assert result.force_evaluations == 5
    assert calculators[0].calls == 5


def test_calculator_failure_preserves_started_call_count():
    calculators = []

    def factory():
        calculator = RecordingAnalyticCalculator(fail_on_call=3)
        calculators.append(calculator)
        return calculator

    result = SSWAttemptWorker(
        factory,
        SSWConfig(max_steps_per_walk=1, oracle_candidates=2),
    )(
        _action("batch-00000000-slot-0000", seed=29, budget=20),
        _state(),
    )

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 3
    assert calculators[0].calls == 3
    assert "synthetic analytic failure" in result.failure_reason
```

- [ ] **Step 5: Run real worker integration tests**

Run:

```bash
pytest -q \
  tests/integration/test_ssw_attempt_worker.py \
  tests/integration/test_epam_accounting.py
```

Expected: all selected tests pass. If the actual deterministic action lands in
a different documented terminal status than the broad allowed set, preserve
the exact status in a focused regression rather than adding status coercion.

- [ ] **Step 6: Commit**

```bash
git add \
  pamssw/exploration/ssw_worker.py \
  tests/integration/test_ssw_attempt_worker.py
git commit -m "Validate analytic SSW attempt accounting"
```

### Task 5: Validate Threaded Controller Integration and Document the Boundary

**Files:**
- Modify: `tests/integration/test_exploration_controller.py`
- Modify: `README.md`

- [ ] **Step 1: Add a real worker/controller integration test**

Append to `tests/integration/test_exploration_controller.py`. Reuse the local
`_archive()` helper and add imports for `SSWConfig`, `SSWAttemptWorker`,
`AnalyticCalculator`, and `DoubleWell2D`.

```python
def test_threaded_ssw_workers_use_distinct_calculators_and_exact_costs(tmp_path):
    import threading

    calculators = []
    outcome_calls = {}
    local = threading.local()
    lock = threading.Lock()

    class RecordingCalculator:
        def __init__(self):
            self.inner = AnalyticCalculator(DoubleWell2D())
            self.calls = 0

        def evaluate(self, state):
            self.calls += 1
            return self.inner.evaluate(state)

        def evaluate_flat(self, flat_positions, template):
            self.calls += 1
            return self.inner.evaluate_flat(flat_positions, template)

    def factory():
        calculator = RecordingCalculator()
        with lock:
            calculators.append(calculator)
        local.calculator = calculator
        return calculator

    isolated_worker = SSWAttemptWorker(
        factory,
        SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=2,
        ),
    )

    def tracked_worker(action, starter_state):
        result = isolated_worker(action, starter_state)
        with lock:
            outcome_calls[action.action_id] = local.calculator.calls
        return result

    controller = ExplorationController(
        archive=_archive(),
        policy_name="uniform",
        master_seed=31,
        event_log=ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = controller.run_batch(
            executor,
            tracked_worker,
            batch_size=3,
            force_budget=400,
        )

    assert len(calculators) == 3
    assert len({id(calculator) for calculator in calculators}) == 3
    assert [outcome.action_id for outcome in outcomes] == [
        "batch-00000000-slot-0000",
        "batch-00000000-slot-0001",
        "batch-00000000-slot-0002",
    ]
    assert all(
        outcome.force_evaluations == outcome_calls[outcome.action_id]
        for outcome in outcomes
    )
    assert all(outcome.force_evaluations <= 400 for outcome in outcomes)
    assert controller.posterior.completed_attempts == 3
    replayed = controller.event_log.reconstruct_posterior()
    assert replayed.completed_attempts == 3
```

- [ ] **Step 2: Add completion-order perturbation**

Wrap the real worker:

```python
def delayed_worker(action, starter_state):
    import time

    time.sleep(0.01 * (2 - min(action.slot_id, 2)))
    return tracked_worker(action, starter_state)
```

Run the batch with `delayed_worker` and keep the same slot-order assertions.
Do not assert wall-clock speedup.

- [ ] **Step 3: Document Phase-2 experimental usage**

Add to the existing README posterior-exploration section:

```markdown
`pamssw.exploration.SSWAttemptWorker` adapts one starter action to one isolated
single-trial SSW run. It requires a calculator factory and a side-effect-free
`SSWConfig`; each action receives a new calculator, walker, RNG, and evaluation
counter. The action seed, one-trial limit, and per-action force budget override
the corresponding base-config values.

The adapter currently supports analytic-calculator validation with
`ThreadPoolExecutor`. It re-quenches every starter and requires
`proposal_pool_size=1` with duplicate rescue disabled. It does not validate
MACE/GPU sharing, process execution, aggregate budgets, or search-performance
improvement.
```

Do not add `SSWAttemptWorker` to root `pamssw.__all__`.

- [ ] **Step 4: Run focused Phase-1 and Phase-2 suites**

Run:

```bash
pytest -q \
  tests/unit/test_accounting.py \
  tests/unit/test_ssw_attempt_worker.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw_attempt_worker.py \
  tests/integration/test_exploration_controller.py \
  tests/unit/test_exploration_actions.py \
  tests/unit/test_exploration_event_log.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Run the complete suite and scope checks**

Run:

```bash
pytest -q
git diff --check
git status --short
git diff --exit-code fbfe674 -- \
  pamssw/acquisition.py \
  pamssw/exploration/policies.py \
  pamssw/exploration/posterior.py \
  pamssw/exploration/controller.py \
  pamssw/runner.py \
  pamssw/config.py
```

Expected:

- the complete suite passes;
- diff checks are clean;
- the named policy, posterior, controller, runner, and config files are
  unchanged from Phase 1.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/integration/test_exploration_controller.py
git commit -m "Document threaded SSW attempt validation"
```

## Final Review and Claim Boundary

- [ ] Request a whole-feature review over `c775e2b..HEAD`.
- [ ] Re-run `pytest -q` in the main agent.
- [ ] Confirm `git diff --check` and a clean worktree.
- [ ] Record the exact underlying-call/accounting regression results.
- [ ] State explicitly that the phase proves analytic ThreadPool isolation and
  total-call accounting only.
- [ ] Do not claim MACE safety, process parallelism, aggregate-budget
  optimality, policy-performance improvement, or removal of starter re-quench.
