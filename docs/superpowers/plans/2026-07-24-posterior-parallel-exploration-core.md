# Posterior Parallel Exploration Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a walker-independent, statistically auditable controller that plans and executes synchronous parallel starter actions using uniform, minimal-UCB, or posterior-proportional selection.

**Architecture:** Add a focused `pamssw.exploration` package containing immutable action contracts, a fixed-prior Beta-Bernoulli posterior, exact policy probability construction, deterministic with-replacement batch planning, dispatch-snapshot credit, append-only batch logging, and a generic executor-driven controller. Reuse the current `MinimaArchive` matching semantics through a small public `find_match()`/`clone()` API; do not modify `SurfaceWalker`, `SSWConfig`, direction selection, or the physical escape kernel in this plan.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, `concurrent.futures`, JSONL, pytest.

---

## Scope Boundary

This is implementation phase 1 of the approved design.

It produces working software with a deterministic fake-worker integration test.
It deliberately does not:

- add `run_parallel_ssw`;
- create calculators inside workers;
- divide the global force budget between SSW attempts;
- modify `SurfaceWalker`;
- add Thompson sampling;
- add reward weights, forgetting, epsilon exploration, or direction learning.

The later SSW-adapter plan may begin only after this plan's full suite and
statistical invariants pass.

## File Map

- Modify: `pamssw/archive.py`
  - Add non-mutating `find_match()` and independent `clone()` operations.
- Create: `pamssw/exploration/__init__.py`
  - Export the intentionally small public exploration API.
- Create: `pamssw/exploration/actions.py`
  - Define immutable snapshot, action, attempt, and credited-outcome contracts.
- Create: `pamssw/exploration/posterior.py`
  - Store fixed-prior Beta-Bernoulli starter productivity counts.
- Create: `pamssw/exploration/policies.py`
  - Construct uniform, minimal-UCB, and posterior-proportional probability vectors.
- Create: `pamssw/exploration/batch.py`
  - Derive deterministic seeds and sample independent batch slots with replacement.
- Create: `pamssw/exploration/event_log.py`
  - Append and validate committed policy-snapshot/attempt batches.
- Create: `pamssw/exploration/controller.py`
  - Execute generic workers concurrently and commit outcomes in slot order.
- Modify: `pamssw/__init__.py`
  - Export only `ExplorationController` and `StarterProductivityPosterior`.
- Create: `tests/unit/test_exploration_actions.py`
- Create: `tests/unit/test_exploration_posterior.py`
- Create: `tests/unit/test_exploration_policies.py`
- Create: `tests/unit/test_exploration_batch.py`
- Create: `tests/unit/test_exploration_event_log.py`
- Modify: `tests/unit/test_archive.py`
- Create: `tests/integration/test_exploration_controller.py`
- Modify: `README.md`
  - Document the experimental core and its claim boundary.

### Task 1: Make Archive Matching Snapshot-Safe

**Files:**
- Modify: `pamssw/archive.py:38-72`
- Modify: `tests/unit/test_archive.py`

- [ ] **Step 1: Write failing tests for non-mutating lookup and cloning**

Append to `tests/unit/test_archive.py`:

```python
def test_archive_find_match_does_not_mutate_visits_or_duplicates():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)
    entry = archive.add(_state(-1.0), -1.0, parent_id=None)

    match = archive.find_match(_state(-1.02), -1.0005)

    assert match is entry
    assert entry.visits == 1
    assert entry.duplicate_hits == 0


def test_archive_clone_is_independent_of_source_mutations():
    archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.05)
    archive.add(_state(-1.0), -1.0, parent_id=None)
    cloned = archive.clone()

    cloned.add(_state(1.0), -0.8, parent_id=0)
    cloned.entries[0].node_trials = 7

    assert len(archive.entries) == 1
    assert archive.entries[0].node_trials == 0
    assert len(cloned.entries) == 2
```

- [ ] **Step 2: Run the archive tests and verify the new tests fail**

Run:

```bash
pytest -q \
  tests/unit/test_archive.py::test_archive_find_match_does_not_mutate_visits_or_duplicates \
  tests/unit/test_archive.py::test_archive_clone_is_independent_of_source_mutations
```

Expected: both tests fail because `find_match` and `clone` do not exist.

- [ ] **Step 3: Implement lookup and cloning, then route `add()` through lookup**

Add `import copy` to `pamssw/archive.py`, then implement:

```python
def find_match(self, state: State, energy: float) -> MinimaEntry | None:
    for entry in self.entries:
        if abs(entry.energy - energy) > self.energy_tol:
            continue
        if self._rmsd(entry.state, state) <= self.rmsd_tol:
            return entry
    return None

def clone(self) -> MinimaArchive:
    return copy.deepcopy(self)
```

Replace the duplicate loop at the start of `add()` with:

```python
match = self.find_match(state, energy)
if match is not None:
    match.visits += 1
    match.duplicate_hits += 1
    return match
```

- [ ] **Step 4: Run the focused archive suite**

Run:

```bash
pytest -q tests/unit/test_archive.py
```

Expected: all archive tests pass.

- [ ] **Step 5: Commit the archive seam**

```bash
git add pamssw/archive.py tests/unit/test_archive.py
git commit -m "Add snapshot-safe archive matching"
```

### Task 2: Define Immutable Exploration Contracts

**Files:**
- Create: `pamssw/exploration/__init__.py`
- Create: `pamssw/exploration/actions.py`
- Create: `tests/unit/test_exploration_actions.py`

- [ ] **Step 1: Write failing contract tests**

Create `tests/unit/test_exploration_actions.py`:

```python
import numpy as np
import pytest

from pamssw.exploration.actions import (
    AttemptResult,
    AttemptStatus,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
)
from pamssw.state import State


def _state(x: float) -> State:
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


def test_policy_snapshot_requires_aligned_normalized_probabilities():
    snapshot = PolicySnapshot(
        version=2,
        archive_version=3,
        policy_name="posterior_proportional",
        eligible_starter_ids=(4, 9),
        probabilities=(0.25, 0.75),
        support_complete=True,
    )

    assert snapshot.probability_for(9) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "probabilities",
    [(0.2,), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 1.0)],
)
def test_policy_snapshot_rejects_invalid_probability_vectors(probabilities):
    with pytest.raises(ValueError):
        PolicySnapshot(
            version=0,
            archive_version=0,
            policy_name="uniform",
            eligible_starter_ids=(0, 1),
            probabilities=probabilities,
            support_complete=True,
        )


def test_policy_snapshot_rejects_false_full_support_claim():
    with pytest.raises(ValueError, match="support_complete"):
        PolicySnapshot(
            version=0,
            archive_version=0,
            policy_name="minimal_ucb",
            eligible_starter_ids=(0, 1),
            probabilities=(1.0, 0.0),
            support_complete=True,
        )


def test_completed_attempt_requires_finite_landing_data():
    action = StarterAction(
        action_id="batch-00000001-slot-0000",
        batch_id=1,
        slot_id=0,
        policy_name="uniform",
        policy_version=0,
        archive_version=0,
        starter_id=0,
        selection_probability=1.0,
        random_seed=12,
        force_budget=None,
    )

    result = AttemptResult(
        action=action,
        landing_state=_state(1.0),
        landing_energy=-1.0,
        force_evaluations=4,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )

    assert result.status is AttemptStatus.COMPLETED


def test_failed_attempt_requires_reason_and_no_landing():
    action = StarterAction(
        action_id="batch-00000001-slot-0000",
        batch_id=1,
        slot_id=0,
        policy_name="uniform",
        policy_version=0,
        archive_version=0,
        starter_id=0,
        selection_probability=1.0,
        random_seed=12,
        force_budget=8,
    )

    with pytest.raises(ValueError, match="failure_reason"):
        AttemptResult(
            action=action,
            landing_state=None,
            landing_energy=None,
            force_evaluations=8,
            status=AttemptStatus.BUDGET_EXHAUSTED,
            failure_reason=None,
        )


def test_credited_outcome_preserves_dispatch_and_commit_facts():
    outcome = CreditedOutcome(
        action_id="a",
        starter_id=2,
        discovered_against_snapshot=True,
        inserted_into_archive=False,
        within_batch_collision=True,
        force_evaluations=7,
        status=AttemptStatus.COMPLETED,
        landing_entry_id=5,
        landing_energy=-3.0,
        failure_reason=None,
    )

    assert outcome.discovered_against_snapshot
    assert outcome.within_batch_collision
```

- [ ] **Step 2: Run the contract tests and verify import failure**

Run:

```bash
pytest -q tests/unit/test_exploration_actions.py
```

Expected: collection fails because `pamssw.exploration.actions` does not exist.

- [ ] **Step 3: Implement the contracts**

Create `pamssw/exploration/actions.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from ..state import State


class AttemptStatus(str, Enum):
    COMPLETED = "completed"
    INVALID = "invalid"
    FRAGMENTED = "fragmented"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WORKER_ERROR = "worker_error"


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    archive_version: int
    policy_name: str
    eligible_starter_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    support_complete: bool

    def __post_init__(self) -> None:
        _nonnegative_int("version", self.version)
        _nonnegative_int("archive_version", self.archive_version)
        if not self.eligible_starter_ids:
            raise ValueError("eligible_starter_ids cannot be empty")
        if len(set(self.eligible_starter_ids)) != len(self.eligible_starter_ids):
            raise ValueError("eligible_starter_ids must be unique")
        if len(self.eligible_starter_ids) != len(self.probabilities):
            raise ValueError("probabilities must align with eligible_starter_ids")
        values = np.asarray(self.probabilities, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("probabilities must sum to one")
        if self.support_complete and np.any(values <= 0.0):
            raise ValueError("support_complete requires positive probability for every starter")

    def probability_for(self, starter_id: int) -> float:
        try:
            index = self.eligible_starter_ids.index(starter_id)
        except ValueError as exc:
            raise KeyError(starter_id) from exc
        return float(self.probabilities[index])


@dataclass(frozen=True)
class StarterAction:
    action_id: str
    batch_id: int
    slot_id: int
    policy_name: str
    policy_version: int
    archive_version: int
    starter_id: int
    selection_probability: float
    random_seed: int
    force_budget: int | None

    def __post_init__(self) -> None:
        for name in ("batch_id", "slot_id", "policy_version", "archive_version", "starter_id", "random_seed"):
            _nonnegative_int(name, getattr(self, name))
        if not self.action_id:
            raise ValueError("action_id cannot be empty")
        if not isfinite(self.selection_probability) or not 0.0 < self.selection_probability <= 1.0:
            raise ValueError("selection_probability must be finite and in (0, 1]")
        if self.force_budget is not None:
            if isinstance(self.force_budget, bool) or not isinstance(self.force_budget, int) or self.force_budget <= 0:
                raise ValueError("force_budget must be a positive integer when set")


@dataclass(frozen=True)
class AttemptResult:
    action: StarterAction
    landing_state: State | None
    landing_energy: float | None
    force_evaluations: int
    status: AttemptStatus
    failure_reason: str | None

    def __post_init__(self) -> None:
        _nonnegative_int("force_evaluations", self.force_evaluations)
        if self.status is AttemptStatus.COMPLETED:
            if self.landing_state is None or self.landing_energy is None or not isfinite(self.landing_energy):
                raise ValueError("completed attempts require finite landing data")
            if self.failure_reason is not None:
                raise ValueError("completed attempts cannot have failure_reason")
        else:
            if self.landing_state is not None or self.landing_energy is not None:
                raise ValueError("failed attempts cannot carry landing data")
            if not self.failure_reason:
                raise ValueError("failed attempts require failure_reason")


@dataclass(frozen=True)
class CreditedOutcome:
    action_id: str
    starter_id: int
    discovered_against_snapshot: bool
    inserted_into_archive: bool
    within_batch_collision: bool
    force_evaluations: int
    status: AttemptStatus
    landing_entry_id: int | None
    landing_energy: float | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        _nonnegative_int("starter_id", self.starter_id)
        _nonnegative_int("force_evaluations", self.force_evaluations)
        if self.landing_entry_id is not None:
            _nonnegative_int("landing_entry_id", self.landing_entry_id)
        if self.landing_energy is not None and not isfinite(self.landing_energy):
            raise ValueError("landing_energy must be finite when set")
        if self.status is AttemptStatus.COMPLETED:
            if self.landing_entry_id is None or self.landing_energy is None:
                raise ValueError("completed outcomes require landing data")
            if self.failure_reason is not None:
                raise ValueError("completed outcomes cannot have failure_reason")
        elif not self.failure_reason:
            raise ValueError("failed outcomes require failure_reason")
```

Create `pamssw/exploration/__init__.py` exporting these five symbols.

- [ ] **Step 4: Run the contract tests**

Run:

```bash
pytest -q tests/unit/test_exploration_actions.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add pamssw/exploration tests/unit/test_exploration_actions.py
git commit -m "Add exploration action contracts"
```

### Task 3: Implement the Fixed-Prior Starter Posterior

**Files:**
- Create: `pamssw/exploration/posterior.py`
- Create: `tests/unit/test_exploration_posterior.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing posterior tests**

Create `tests/unit/test_exploration_posterior.py`:

```python
import pytest

from pamssw.exploration.posterior import StarterProductivityPosterior


def test_new_starter_has_fixed_uniform_beta_prior():
    posterior = StarterProductivityPosterior()

    posterior.ensure((3,))

    assert posterior.counts(3) == (0, 0)
    assert posterior.mean(3) == pytest.approx(0.5)


def test_posterior_updates_successes_and_failures_exactly_once():
    posterior = StarterProductivityPosterior()

    posterior.update(3, discovered=True)
    posterior.update(3, discovered=False)

    assert posterior.counts(3) == (1, 1)
    assert posterior.mean(3) == pytest.approx(0.5)
    assert posterior.completed_attempts == 2


def test_posterior_clone_is_independent():
    posterior = StarterProductivityPosterior()
    posterior.update(1, discovered=True)
    cloned = posterior.clone()

    cloned.update(1, discovered=False)

    assert posterior.counts(1) == (1, 0)
    assert cloned.counts(1) == (1, 1)


def test_posterior_rejects_boolean_starter_ids():
    posterior = StarterProductivityPosterior()

    with pytest.raises(ValueError, match="starter_id"):
        posterior.update(True, discovered=True)
```

- [ ] **Step 2: Run the posterior tests and verify import failure**

Run:

```bash
pytest -q tests/unit/test_exploration_posterior.py
```

Expected: collection fails because `posterior.py` does not exist.

- [ ] **Step 3: Implement the posterior**

Create `pamssw/exploration/posterior.py`:

```python
from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class _Counts:
    successes: int = 0
    failures: int = 0


class StarterProductivityPosterior:
    PRIOR_ALPHA = 1.0
    PRIOR_BETA = 1.0

    def __init__(self) -> None:
        self._counts: dict[int, _Counts] = {}

    @staticmethod
    def _validate_starter_id(starter_id: int) -> None:
        if isinstance(starter_id, bool) or not isinstance(starter_id, int) or starter_id < 0:
            raise ValueError("starter_id must be a non-negative integer")

    def ensure(self, starter_ids: tuple[int, ...] | list[int]) -> None:
        for starter_id in starter_ids:
            self._validate_starter_id(starter_id)
            self._counts.setdefault(starter_id, _Counts())

    def update(self, starter_id: int, discovered: bool) -> None:
        self._validate_starter_id(starter_id)
        if not isinstance(discovered, bool):
            raise ValueError("discovered must be a boolean")
        counts = self._counts.setdefault(starter_id, _Counts())
        if discovered:
            counts.successes += 1
        else:
            counts.failures += 1

    def counts(self, starter_id: int) -> tuple[int, int]:
        self._validate_starter_id(starter_id)
        counts = self._counts.get(starter_id, _Counts())
        return counts.successes, counts.failures

    def mean(self, starter_id: int) -> float:
        successes, failures = self.counts(starter_id)
        return float(
            (self.PRIOR_ALPHA + successes)
            / (self.PRIOR_ALPHA + self.PRIOR_BETA + successes + failures)
        )

    @property
    def completed_attempts(self) -> int:
        return sum(item.successes + item.failures for item in self._counts.values())

    def clone(self) -> StarterProductivityPosterior:
        return copy.deepcopy(self)
```

Export `StarterProductivityPosterior` from `pamssw/exploration/__init__.py`.

- [ ] **Step 4: Run the posterior tests**

Run:

```bash
pytest -q tests/unit/test_exploration_posterior.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the posterior**

```bash
git add pamssw/exploration/posterior.py pamssw/exploration/__init__.py tests/unit/test_exploration_posterior.py
git commit -m "Add starter productivity posterior"
```

### Task 4: Implement Three Explicit Starter Policies

**Files:**
- Create: `pamssw/exploration/policies.py`
- Create: `tests/unit/test_exploration_policies.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/unit/test_exploration_policies.py`:

```python
import pytest

from pamssw.exploration.policies import build_policy_snapshot
from pamssw.exploration.posterior import StarterProductivityPosterior


def test_uniform_policy_has_exact_full_support():
    snapshot = build_policy_snapshot("uniform", (2, 5, 9), StarterProductivityPosterior(), 4, 7)

    assert snapshot.probabilities == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert snapshot.support_complete


def test_posterior_proportional_policy_normalizes_posterior_means():
    posterior = StarterProductivityPosterior()
    posterior.update(2, discovered=True)   # mean = 2/3
    posterior.update(5, discovered=False)  # mean = 1/3

    snapshot = build_policy_snapshot("posterior_proportional", (2, 5), posterior, 1, 1)

    assert snapshot.probabilities == pytest.approx((2 / 3, 1 / 3))
    assert snapshot.support_complete


def test_minimal_ucb_selects_lowest_id_untried_starter_first():
    posterior = StarterProductivityPosterior()
    posterior.update(5, discovered=True)

    snapshot = build_policy_snapshot("minimal_ucb", (5, 2, 9), posterior, 1, 1)

    assert snapshot.eligible_starter_ids == (2, 5, 9)
    assert snapshot.probabilities == (1.0, 0.0, 0.0)
    assert not snapshot.support_complete


def test_minimal_ucb_uses_only_success_mean_and_canonical_confidence():
    posterior = StarterProductivityPosterior()
    posterior.update(1, discovered=True)
    posterior.update(1, discovered=True)
    posterior.update(2, discovered=False)
    posterior.update(2, discovered=False)

    snapshot = build_policy_snapshot("minimal_ucb", (1, 2), posterior, 2, 2)

    assert snapshot.probabilities == (1.0, 0.0)


def test_policy_builder_rejects_legacy_ucb_inside_new_controller():
    with pytest.raises(ValueError, match="legacy_ucb"):
        build_policy_snapshot("legacy_ucb", (0,), StarterProductivityPosterior(), 0, 0)
```

- [ ] **Step 2: Run the policy tests and verify import failure**

Run:

```bash
pytest -q tests/unit/test_exploration_policies.py
```

Expected: collection fails because `policies.py` does not exist.

- [ ] **Step 3: Implement the policy builder**

Create `pamssw/exploration/policies.py`:

```python
from __future__ import annotations

from math import log, sqrt

import numpy as np

from .actions import PolicySnapshot
from .posterior import StarterProductivityPosterior

SUPPORTED_POLICIES = {"uniform", "minimal_ucb", "posterior_proportional"}


def build_policy_snapshot(
    policy_name: str,
    eligible_starter_ids: tuple[int, ...],
    posterior: StarterProductivityPosterior,
    version: int,
    archive_version: int,
) -> PolicySnapshot:
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(
            f"{policy_name!r} is not supported by the auditable controller; "
            "legacy_ucb remains an external comparator"
        )
    starter_ids = tuple(sorted(eligible_starter_ids))
    if not starter_ids:
        raise ValueError("eligible_starter_ids cannot be empty")
    posterior.ensure(starter_ids)

    if policy_name == "uniform":
        probabilities = np.full(len(starter_ids), 1.0 / len(starter_ids))
        support_complete = True
    elif policy_name == "posterior_proportional":
        means = np.asarray([posterior.mean(starter_id) for starter_id in starter_ids], dtype=float)
        probabilities = means / means.sum()
        support_complete = True
    else:
        untried = [
            starter_id
            for starter_id in starter_ids
            if sum(posterior.counts(starter_id)) == 0
        ]
        if untried:
            winner = min(untried)
        else:
            total = max(2, posterior.completed_attempts)
            winner = max(
                starter_ids,
                key=lambda starter_id: (
                    posterior.counts(starter_id)[0] / sum(posterior.counts(starter_id))
                    + sqrt(2.0 * log(total) / sum(posterior.counts(starter_id))),
                    -starter_id,
                ),
            )
        probabilities = np.asarray([1.0 if starter_id == winner else 0.0 for starter_id in starter_ids])
        support_complete = False

    return PolicySnapshot(
        version=version,
        archive_version=archive_version,
        policy_name=policy_name,
        eligible_starter_ids=starter_ids,
        probabilities=tuple(float(value) for value in probabilities),
        support_complete=support_complete,
    )
```

Export `build_policy_snapshot` from `pamssw/exploration/__init__.py`.

- [ ] **Step 4: Run the policy tests**

Run:

```bash
pytest -q tests/unit/test_exploration_policies.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the policies**

```bash
git add pamssw/exploration/policies.py pamssw/exploration/__init__.py tests/unit/test_exploration_policies.py
git commit -m "Add auditable starter policies"
```

### Task 5: Plan Deterministic Parallel Batches

**Files:**
- Create: `pamssw/exploration/batch.py`
- Create: `tests/unit/test_exploration_batch.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing batch-planning tests**

Create `tests/unit/test_exploration_batch.py`:

```python
from pamssw.exploration.actions import PolicySnapshot
from pamssw.exploration.batch import derive_action_seed, plan_batch


def _snapshot() -> PolicySnapshot:
    return PolicySnapshot(
        version=3,
        archive_version=4,
        policy_name="posterior_proportional",
        eligible_starter_ids=(2, 5),
        probabilities=(0.25, 0.75),
        support_complete=True,
    )


def test_action_seed_is_deterministic_and_slot_specific():
    assert derive_action_seed(17, 3, 1) == derive_action_seed(17, 3, 1)
    assert derive_action_seed(17, 3, 1) != derive_action_seed(17, 3, 2)


def test_batch_plan_is_reproducible_and_records_exact_propensities():
    first = plan_batch(_snapshot(), batch_id=8, batch_size=6, master_seed=19, force_budget=30)
    second = plan_batch(_snapshot(), batch_id=8, batch_size=6, master_seed=19, force_budget=30)

    assert first == second
    assert len(first) == 6
    assert [item.slot_id for item in first] == list(range(6))
    assert all(
        item.selection_probability == _snapshot().probability_for(item.starter_id)
        for item in first
    )


def test_batch_sampling_is_with_replacement():
    snapshot = PolicySnapshot(
        version=0,
        archive_version=0,
        policy_name="uniform",
        eligible_starter_ids=(7,),
        probabilities=(1.0,),
        support_complete=True,
    )

    actions = plan_batch(snapshot, batch_id=0, batch_size=4, master_seed=0, force_budget=None)

    assert [item.starter_id for item in actions] == [7, 7, 7, 7]
    assert len({item.action_id for item in actions}) == 4
```

- [ ] **Step 2: Run the batch tests and verify import failure**

Run:

```bash
pytest -q tests/unit/test_exploration_batch.py
```

Expected: collection fails because `batch.py` does not exist.

- [ ] **Step 3: Implement deterministic planning**

Create `pamssw/exploration/batch.py`:

```python
from __future__ import annotations

import numpy as np

from .actions import PolicySnapshot, StarterAction


def derive_action_seed(master_seed: int, batch_id: int, slot_id: int) -> int:
    sequence = np.random.SeedSequence([master_seed, batch_id, slot_id, 0x535357])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def plan_batch(
    snapshot: PolicySnapshot,
    batch_id: int,
    batch_size: int,
    master_seed: int,
    force_budget: int | None,
) -> tuple[StarterAction, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, batch_id, 0x42415443]))
    choices = rng.choice(
        len(snapshot.eligible_starter_ids),
        size=batch_size,
        replace=True,
        p=np.asarray(snapshot.probabilities, dtype=float),
    )
    actions = []
    for slot_id, choice in enumerate(choices.tolist()):
        starter_id = snapshot.eligible_starter_ids[choice]
        actions.append(
            StarterAction(
                action_id=f"batch-{batch_id:08d}-slot-{slot_id:04d}",
                batch_id=batch_id,
                slot_id=slot_id,
                policy_name=snapshot.policy_name,
                policy_version=snapshot.version,
                archive_version=snapshot.archive_version,
                starter_id=starter_id,
                selection_probability=snapshot.probabilities[choice],
                random_seed=derive_action_seed(master_seed, batch_id, slot_id),
                force_budget=force_budget,
            )
        )
    return tuple(actions)
```

Export `derive_action_seed` and `plan_batch`.

- [ ] **Step 4: Run the batch tests**

Run:

```bash
pytest -q tests/unit/test_exploration_batch.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the batch planner**

```bash
git add pamssw/exploration/batch.py pamssw/exploration/__init__.py tests/unit/test_exploration_batch.py
git commit -m "Add deterministic parallel batch planning"
```

### Task 6: Add Append-Only Committed Batch Logging

**Files:**
- Create: `pamssw/exploration/event_log.py`
- Create: `tests/unit/test_exploration_event_log.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing event-log tests**

Create `tests/unit/test_exploration_event_log.py`:

```python
import json

import pytest

from pamssw.exploration.actions import AttemptStatus, CreditedOutcome, PolicySnapshot, StarterAction
from pamssw.exploration.event_log import ExplorationEventLog
from pamssw.exploration.posterior import StarterProductivityPosterior


def _snapshot() -> PolicySnapshot:
    return PolicySnapshot(0, 0, "uniform", (0,), (1.0,), True)


def _action() -> StarterAction:
    return StarterAction("batch-00000000-slot-0000", 0, 0, "uniform", 0, 0, 0, 1.0, 3, None)


def _outcome(discovered: bool) -> CreditedOutcome:
    return CreditedOutcome(
        action_id=_action().action_id,
        starter_id=0,
        discovered_against_snapshot=discovered,
        inserted_into_archive=discovered,
        within_batch_collision=False,
        force_evaluations=5,
        status=AttemptStatus.COMPLETED,
        landing_entry_id=1 if discovered else 0,
        landing_energy=-1.0,
        failure_reason=None,
    )


def test_event_log_writes_snapshot_attempt_and_commit_records(tmp_path):
    path = tmp_path / "attempts.jsonl"
    log = ExplorationEventLog(path)

    log.append_batch(_snapshot(), (_action(),), (_outcome(True),))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["policy_snapshot", "attempt", "batch_commit"]
    assert rows[1]["selection_probability"] == pytest.approx(1.0)
    assert rows[1]["discovered_against_snapshot"] is True


def test_event_log_reconstructs_posterior_from_committed_attempts(tmp_path):
    path = tmp_path / "attempts.jsonl"
    log = ExplorationEventLog(path)
    log.append_batch(_snapshot(), (_action(),), (_outcome(True),))

    posterior = log.reconstruct_posterior()

    assert posterior.counts(0) == (1, 0)


def test_event_log_fails_closed_on_incomplete_batch(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text(json.dumps({"schema_version": 1, "record_type": "policy_snapshot", "batch_id": 0}) + "\n")

    with pytest.raises(ValueError, match="incomplete batch"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_event_log_rejects_action_outcome_mismatch(tmp_path):
    bad = CreditedOutcome(
        action_id="other",
        starter_id=0,
        discovered_against_snapshot=False,
        inserted_into_archive=False,
        within_batch_collision=False,
        force_evaluations=1,
        status=AttemptStatus.WORKER_ERROR,
        landing_entry_id=None,
        landing_energy=None,
        failure_reason="synthetic mismatch",
    )

    with pytest.raises(ValueError, match="action_id"):
        ExplorationEventLog(tmp_path / "attempts.jsonl").append_batch(_snapshot(), (_action(),), (bad,))
```

- [ ] **Step 2: Run the event-log tests and verify import failure**

Run:

```bash
pytest -q tests/unit/test_exploration_event_log.py
```

Expected: collection fails because `event_log.py` does not exist.

- [ ] **Step 3: Implement committed JSONL batches**

Create `pamssw/exploration/event_log.py`. Implement:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from .actions import CreditedOutcome, PolicySnapshot, StarterAction
from .posterior import StarterProductivityPosterior

SCHEMA_VERSION = 1


class ExplorationEventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append_batch(
        self,
        snapshot: PolicySnapshot,
        actions: tuple[StarterAction, ...],
        outcomes: tuple[CreditedOutcome, ...],
    ) -> None:
        if len(actions) != len(outcomes):
            raise ValueError("actions and outcomes must have equal length")
        if any(action.action_id != outcome.action_id for action, outcome in zip(actions, outcomes, strict=True)):
            raise ValueError("action_id mismatch between actions and outcomes")
        snapshot_row = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "policy_snapshot",
            "batch_id": actions[0].batch_id,
            "policy_name": snapshot.policy_name,
            "policy_version": snapshot.version,
            "archive_version": snapshot.archive_version,
            "support_complete": snapshot.support_complete,
            "eligible_starter_ids": list(snapshot.eligible_starter_ids),
            "probabilities": list(snapshot.probabilities),
        }
        rows = [snapshot_row]
        for action, outcome in zip(actions, outcomes, strict=True):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "attempt",
                    **action.__dict__,
                    "status": outcome.status.value,
                    "failure_reason": outcome.failure_reason,
                    "force_evaluations": outcome.force_evaluations,
                    "discovered_against_snapshot": outcome.discovered_against_snapshot,
                    "inserted_into_archive": outcome.inserted_into_archive,
                    "within_batch_collision": outcome.within_batch_collision,
                    "landing_entry_id": outcome.landing_entry_id,
                    "landing_energy": outcome.landing_energy,
                }
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "batch_commit",
                "batch_id": actions[0].batch_id,
                "action_ids": [action.action_id for action in actions],
            }
        )
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def reconstruct_posterior(self) -> StarterProductivityPosterior:
        posterior = StarterProductivityPosterior()
        if not self.path.exists():
            return posterior
        pending_batch: int | None = None
        pending_attempts: list[dict] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version at line {line_number}")
            record_type = row.get("record_type")
            if record_type == "policy_snapshot":
                if pending_batch is not None:
                    raise ValueError("incomplete batch before next policy snapshot")
                pending_batch = int(row["batch_id"])
                pending_attempts = []
            elif record_type == "attempt":
                if pending_batch is None or int(row["batch_id"]) != pending_batch:
                    raise ValueError("attempt outside its policy snapshot")
                pending_attempts.append(row)
            elif record_type == "batch_commit":
                if pending_batch is None or int(row["batch_id"]) != pending_batch:
                    raise ValueError("batch_commit without matching snapshot")
                if row["action_ids"] != [item["action_id"] for item in pending_attempts]:
                    raise ValueError("batch_commit action_ids do not match attempts")
                for item in pending_attempts:
                    posterior.update(int(item["starter_id"]), bool(item["discovered_against_snapshot"]))
                pending_batch = None
                pending_attempts = []
            else:
                raise ValueError(f"unknown record_type at line {line_number}")
        if pending_batch is not None:
            raise ValueError("incomplete batch at end of event log")
        return posterior
```

Export `ExplorationEventLog`.

- [ ] **Step 4: Run the event-log tests**

Run:

```bash
pytest -q tests/unit/test_exploration_event_log.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the event log**

```bash
git add pamssw/exploration/event_log.py pamssw/exploration/__init__.py tests/unit/test_exploration_event_log.py
git commit -m "Add committed exploration event log"
```

### Task 7: Execute and Credit Synchronous Parallel Batches

**Files:**
- Create: `pamssw/exploration/controller.py`
- Create: `tests/integration/test_exploration_controller.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/integration/test_exploration_controller.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pytest

from pamssw.archive import MinimaArchive
from pamssw.exploration.actions import AttemptResult, AttemptStatus
from pamssw.exploration.controller import ExplorationController
from pamssw.exploration.event_log import ExplorationEventLog
from pamssw.state import State


def _state(x: float) -> State:
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


def _archive() -> MinimaArchive:
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.05)
    archive.add(_state(-1.0), -1.0, parent_id=None)
    archive.add(_state(1.0), -0.9, parent_id=None)
    return archive


def test_parallel_controller_commits_in_slot_order_not_completion_order(tmp_path):
    def worker(action, starter_state):
        time.sleep(0.01 * (3 - action.slot_id))
        return AttemptResult(
            action=action,
            landing_state=_state(3.0 + action.slot_id),
            landing_energy=-2.0 - action.slot_id,
            force_evaluations=4,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )

    controller = ExplorationController(
        archive=_archive(),
        policy_name="uniform",
        master_seed=4,
        event_log=ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=3, force_budget=10)

    assert [outcome.action_id for outcome in outcomes] == [
        "batch-00000000-slot-0000",
        "batch-00000000-slot-0001",
        "batch-00000000-slot-0002",
    ]
    assert controller.posterior.completed_attempts == 3
    assert controller.policy_version == 1
    assert controller.archive_version == 1


def test_same_novel_landing_credits_both_actions_and_inserts_once(tmp_path):
    def worker(action, starter_state):
        return AttemptResult(
            action=action,
            landing_state=_state(4.0),
            landing_energy=-3.0,
            force_evaluations=2,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )

    controller = ExplorationController(
        archive=_archive(),
        policy_name="uniform",
        master_seed=8,
        event_log=ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    before = len(controller.archive.entries)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=2, force_budget=None)

    assert [outcome.discovered_against_snapshot for outcome in outcomes] == [True, True]
    assert sum(outcome.inserted_into_archive for outcome in outcomes) == 1
    assert sum(outcome.within_batch_collision for outcome in outcomes) == 1
    assert len(controller.archive.entries) == before + 1


def test_failed_attempt_updates_failure_posterior_once(tmp_path):
    def worker(action, starter_state):
        return AttemptResult(
            action=action,
            landing_state=None,
            landing_energy=None,
            force_evaluations=5,
            status=AttemptStatus.WORKER_ERROR,
            failure_reason="synthetic failure",
        )

    controller = ExplorationController(
        archive=_archive(),
        policy_name="posterior_proportional",
        master_seed=2,
        event_log=ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=1, force_budget=5)

    starter_id = outcomes[0].starter_id
    assert controller.posterior.counts(starter_id) == (0, 1)


def test_log_failure_leaves_controller_state_unchanged(tmp_path):
    class FailingLog:
        def append_batch(self, snapshot, actions, outcomes):
            raise OSError("disk full")

    def worker(action, starter_state):
        return AttemptResult(
            action=action,
            landing_state=_state(5.0),
            landing_energy=-4.0,
            force_evaluations=1,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )

    controller = ExplorationController(_archive(), "uniform", 1, FailingLog())
    original_entries = len(controller.archive.entries)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(OSError, match="disk full"):
            controller.run_batch(executor, worker, batch_size=1, force_budget=None)

    assert len(controller.archive.entries) == original_entries
    assert controller.posterior.completed_attempts == 0
    assert controller.policy_version == 0
    assert controller.archive_version == 0
```

- [ ] **Step 2: Run the integration tests and verify import failure**

Run:

```bash
pytest -q tests/integration/test_exploration_controller.py
```

Expected: collection fails because `controller.py` does not exist.

- [ ] **Step 3: Implement the controller**

Create `pamssw/exploration/controller.py`:

```python
from __future__ import annotations

from concurrent.futures import Executor, as_completed
from typing import Callable, Protocol

from ..archive import MinimaArchive
from ..state import State
from .actions import AttemptResult, AttemptStatus, CreditedOutcome, StarterAction
from .batch import plan_batch
from .policies import build_policy_snapshot
from .posterior import StarterProductivityPosterior


class BatchLog(Protocol):
    def append_batch(self, snapshot, actions, outcomes) -> None: ...


Worker = Callable[[StarterAction, State], AttemptResult]


class ExplorationController:
    def __init__(
        self,
        archive: MinimaArchive,
        policy_name: str,
        master_seed: int,
        event_log: BatchLog,
    ) -> None:
        self.archive = archive.clone()
        self.policy_name = policy_name
        self.master_seed = master_seed
        self.event_log = event_log
        self.posterior = StarterProductivityPosterior()
        self.policy_version = 0
        self.archive_version = 0
        self.batch_id = 0

    def run_batch(
        self,
        executor: Executor,
        worker: Worker,
        batch_size: int,
        force_budget: int | None,
    ) -> tuple[CreditedOutcome, ...]:
        snapshot = build_policy_snapshot(
            self.policy_name,
            tuple(entry.entry_id for entry in self.archive.entries),
            self.posterior,
            self.policy_version,
            self.archive_version,
        )
        actions = plan_batch(snapshot, self.batch_id, batch_size, self.master_seed, force_budget)
        dispatch_archive = self.archive.clone()
        futures = {
            executor.submit(worker, action, dispatch_archive.entries[action.starter_id].state): action
            for action in actions
        }
        result_by_slot: dict[int, AttemptResult] = {}
        for future in as_completed(futures):
            action = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = AttemptResult(
                    action=action,
                    landing_state=None,
                    landing_energy=None,
                    force_evaluations=0,
                    status=AttemptStatus.WORKER_ERROR,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            if result.action != action:
                raise ValueError("worker returned a result for the wrong action")
            result_by_slot[action.slot_id] = result

        ordered_results = tuple(result_by_slot[index] for index in range(len(actions)))
        shadow_archive = self.archive.clone()
        shadow_posterior = self.posterior.clone()
        outcomes: list[CreditedOutcome] = []
        for result in ordered_results:
            discovered = False
            inserted = False
            collision = False
            landing_entry_id = None
            landing_energy = result.landing_energy
            if result.status is AttemptStatus.COMPLETED:
                assert result.landing_state is not None
                assert result.landing_energy is not None
                discovered = dispatch_archive.find_match(result.landing_state, result.landing_energy) is None
                before = len(shadow_archive.entries)
                landing = shadow_archive.add(
                    result.landing_state,
                    result.landing_energy,
                    parent_id=result.action.starter_id,
                )
                inserted = len(shadow_archive.entries) > before
                collision = discovered and not inserted
                landing_entry_id = landing.entry_id
            shadow_posterior.update(result.action.starter_id, discovered=discovered)
            outcomes.append(
                CreditedOutcome(
                    action_id=result.action.action_id,
                    starter_id=result.action.starter_id,
                    discovered_against_snapshot=discovered,
                    inserted_into_archive=inserted,
                    within_batch_collision=collision,
                    force_evaluations=result.force_evaluations,
                    status=result.status,
                    landing_entry_id=landing_entry_id,
                    landing_energy=landing_energy,
                    failure_reason=result.failure_reason,
                )
            )

        finalized = tuple(outcomes)
        self.event_log.append_batch(snapshot, actions, finalized)
        self.archive = shadow_archive
        self.posterior = shadow_posterior
        self.policy_version += 1
        self.archive_version += 1
        self.batch_id += 1
        return finalized
```

Export `ExplorationController`.

- [ ] **Step 4: Run the controller integration tests**

Run:

```bash
pytest -q tests/integration/test_exploration_controller.py
```

Expected: all tests pass.

- [ ] **Step 5: Run all exploration and archive tests together**

Run:

```bash
pytest -q \
  tests/unit/test_archive.py \
  tests/unit/test_exploration_actions.py \
  tests/unit/test_exploration_posterior.py \
  tests/unit/test_exploration_policies.py \
  tests/unit/test_exploration_batch.py \
  tests/unit/test_exploration_event_log.py \
  tests/integration/test_exploration_controller.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the controller**

```bash
git add pamssw/exploration/controller.py pamssw/exploration/__init__.py tests/integration/test_exploration_controller.py
git commit -m "Add synchronous exploration controller"
```

### Task 8: Public API, Documentation, and Full Verification

**Files:**
- Modify: `pamssw/__init__.py`
- Modify: `README.md`
- Modify: `tests/unit/test_exploration_actions.py`

- [ ] **Step 1: Write a failing public-API test**

Append to `tests/unit/test_exploration_actions.py`:

```python
def test_exploration_core_has_small_public_api():
    import pamssw

    assert pamssw.ExplorationController.__name__ == "ExplorationController"
    assert pamssw.StarterProductivityPosterior.__name__ == "StarterProductivityPosterior"
```

- [ ] **Step 2: Run the public-API test and verify failure**

Run:

```bash
pytest -q tests/unit/test_exploration_actions.py::test_exploration_core_has_small_public_api
```

Expected: fail because the two names are not exported from `pamssw`.

- [ ] **Step 3: Export only the two intended top-level symbols**

Add to `pamssw/__init__.py`:

```python
from .exploration import ExplorationController, StarterProductivityPosterior
```

Add both names to `__all__`. Do not export internal action dataclasses or policy
helpers at the package root.

- [ ] **Step 4: Document the experimental boundary**

Append this section to `README.md`:

```markdown
## Experimental posterior-driven exploration core

`pamssw.ExplorationController` provides a walker-independent synchronous batch
controller for `uniform`, `minimal_ucb`, and `posterior_proportional` starter
selection. Uniform and posterior-proportional sampling record exact positive
propensities for every eligible starter. Minimal UCB is a deterministic
comparator and does not provide full action support.

The current core is validated with generic workers. It does not yet provide a
`run_parallel_ssw` entry point, does not alter the default SSW walker, and does
not claim thermodynamic or kinetic sampling. The physical SSW worker adapter
and force-budget integration are a separate validation phase.
```

- [ ] **Step 5: Run the focused exploration suite**

Run:

```bash
pytest -q \
  tests/unit/test_archive.py \
  tests/unit/test_exploration_actions.py \
  tests/unit/test_exploration_posterior.py \
  tests/unit/test_exploration_policies.py \
  tests/unit/test_exploration_batch.py \
  tests/unit/test_exploration_event_log.py \
  tests/integration/test_exploration_controller.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the complete test suite**

Run:

```bash
pytest -q
```

Expected baseline: at least the original `359` tests plus the new exploration
tests, with zero failures.

- [ ] **Step 7: Check formatting and repository scope**

Run:

```bash
git diff --check
git status --short
```

Expected:

- `git diff --check` emits no output;
- status lists only exploration-core implementation, tests, README, and the
  archive seam described by this plan.

- [ ] **Step 8: Commit the public API and documentation**

```bash
git add pamssw/__init__.py README.md tests/unit/test_exploration_actions.py
git commit -m "Document posterior exploration core"
```

- [ ] **Step 9: Record final proof**

Run:

```bash
git log --oneline 630eccc..HEAD
pytest -q
```

Record:

- exact commit list;
- exact passed-test count;
- proof that the original default `run_ssw` path remains unchanged;
- explicit remaining boundary: no real SSW worker adapter or MACE parallel
  runtime has been validated in this phase.
