# Recoverable Budgeted Posterior Exploration Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in analytic ThreadPool posterior SSW runner with exact per-purpose calculator accounting, a fixed campaign force budget, atomic committed-batch recovery, and a force-budgeted multi-seed policy harness.

**Architecture:** Keep the existing serial SSW kernel and Phase-1 policies fixed. First make calculator-purpose accounting, posterior-observation semantics, and terminal action facts explicit, then promote the controller's pending payload into a complete committed-batch contract. Add a pure campaign ledger, an atomic per-batch run store with replay validation, durable session/abort facts, and finally compose those pieces in new posterior SSW/LS-SSW runner functions and a separate benchmark harness.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, `concurrent.futures.ThreadPoolExecutor`, strict JSON, POSIX fsync/atomic rename, pytest.

---

## Scope and file structure

New focused modules:

```text
pamssw/exploration/
  campaign.py       # outer-run configuration, fixed-cap budget ledger, result summary
  committed.py      # complete immutable batch fact contract
  run_store.py      # strict manifest/batch serialization, atomic commit, recovery
  runner.py         # bootstrap and synchronous campaign orchestration

benchmarks/
  posterior_policy_compare.py
```

Existing modules changed in place:

```text
pamssw/accounting.py                 # purpose enum, immutable counts, scoped counter
pamssw/walker.py                     # purpose attribution only; no numerical changes
pamssw/exploration/actions.py        # exact counts/cost-known terminal contracts
pamssw/exploration/ssw_worker.py     # attach action-local counter snapshots
pamssw/exploration/controller.py     # complete commit object, strict-cost mode, replay
pamssw/exploration/event_log.py      # schema v2 counts/cost facts
pamssw/runner.py                     # public opt-in wrappers
pamssw/__init__.py                   # public config/result/runner exports
pamssw/exploration/__init__.py       # experimental contracts and store exports
README.md                            # exact claim and runtime boundaries
```

The implementation must not change `pamssw/exploration/policies.py`,
`pamssw/exploration/posterior.py`, the numerical SSW direction/bias/step
formulas, or any legacy default.

---

### Task 1: Immutable evaluation-purpose ledger

**Files:**
- Modify: `pamssw/accounting.py`
- Modify: `tests/unit/test_accounting.py`

- [ ] **Step 1: Write failing tests for the closed purpose contract**

Add tests that construct canonical zero and nonzero snapshots:

```python
from pamssw.accounting import EvaluationCounts, EvaluationPurpose


def test_evaluation_counts_are_canonical_immutable_and_totaled():
    counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.DIRECTION_ORACLE: 2,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 3,
        }
    )
    assert counts.total == 5
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 2
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert counts.as_dict()["landing_true_quench"] == 3


def test_unattributed_counts_capture_legacy_scalar_cost():
    counts = EvaluationCounts.unattributed(4)
    assert counts.total == 4
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 4
```

Reject booleans, negative counts, unknown keys, wrong tuple length, and mutable
mapping aliasing.

- [ ] **Step 2: Run the new contract tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_accounting.py -k "evaluation_counts or unattributed"
```

Expected: import or attribute failure because the new types do not exist.

- [ ] **Step 3: Implement the enum and immutable snapshot**

Add:

```python
class EvaluationPurpose(str, Enum):
    BOOTSTRAP_TRUE_QUENCH = "bootstrap_true_quench"
    STARTER_TRUE_QUENCH = "starter_true_quench"
    DIRECTION_ORACLE = "direction_oracle"
    ESCAPE_TRUE_PES_CHECK = "escape_true_pes_check"
    BIASED_PROPOSAL_RELAX = "biased_proposal_relax"
    LANDING_TRUE_QUENCH = "landing_true_quench"
    POST_RELAX_VALIDATION = "post_relax_validation"
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class EvaluationCounts:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if len(values) != len(EvaluationPurpose):
            raise ValueError("evaluation counts must cover every purpose")
        canonical = tuple(_nonnegative_int("evaluation count", value) for value in values)
        object.__setattr__(self, "values", canonical)

    @classmethod
    def zero(cls) -> "EvaluationCounts":
        return cls((0,) * len(EvaluationPurpose))

    @classmethod
    def unattributed(cls, total: int) -> "EvaluationCounts":
        counts = [0] * len(EvaluationPurpose)
        counts[list(EvaluationPurpose).index(EvaluationPurpose.UNATTRIBUTED)] = (
            _nonnegative_int("total", total)
        )
        return cls(tuple(counts))

    @classmethod
    def from_mapping(cls, values: Mapping[EvaluationPurpose | str, int]) -> "EvaluationCounts":
        normalized: dict[EvaluationPurpose, int] = {}
        for key, value in values.items():
            try:
                purpose = key if isinstance(key, EvaluationPurpose) else EvaluationPurpose(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown evaluation purpose") from exc
            if purpose in normalized:
                raise ValueError("duplicate evaluation purpose")
            normalized[purpose] = _nonnegative_int(purpose.value, value)
        return cls(tuple(normalized.get(purpose, 0) for purpose in EvaluationPurpose))

    @property
    def total(self) -> int:
        return sum(self.values)

    def count(self, purpose: EvaluationPurpose) -> int:
        return self.values[list(EvaluationPurpose).index(purpose)]

    def as_dict(self) -> dict[str, int]:
        return {
            purpose.value: self.count(purpose)
            for purpose in EvaluationPurpose
        }
```

Canonicalize in enum declaration order. Validate every integer using the same
bool-rejecting nonnegative rule as `EvalCounter`.

- [ ] **Step 4: Write failing scoped-counter tests**

Cover success, started failure, budget rejection, nesting, and exception
restoration:

```python
def test_eval_counter_counts_one_active_purpose_before_failed_delegation():
    raw = RaisingCalculator()
    counter = EvalCounter(raw)
    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        with pytest.raises(RuntimeError):
            counter.evaluate(_state())
    assert counter.force_evaluations == 1
    assert counter.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE) == 1


def test_budget_rejection_counts_no_purpose():
    counter = EvalCounter(RecordingCalculator(), max_force_evals=1)
    with counter.purpose(EvaluationPurpose.LANDING_TRUE_QUENCH):
        counter.evaluate(_state())
        with pytest.raises(BudgetExceeded):
            counter.evaluate(_state())
    assert counter.snapshot().total == 1


def test_nested_purpose_scope_restores_after_exception():
    counter = EvalCounter(RecordingCalculator())
    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        with pytest.raises(RuntimeError):
            with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
                raise RuntimeError("stop")
        counter.evaluate(_state())
    assert counter.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE) == 1
```

- [ ] **Step 5: Run the scoped tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_accounting.py -k "purpose or snapshot"
```

Expected: missing `purpose`/`snapshot`.

- [ ] **Step 6: Implement scoped counting in the sole physical-call authority**

Use `contextlib.contextmanager` and a private stack:

```python
@contextmanager
def purpose(self, purpose: EvaluationPurpose):
    if not isinstance(purpose, EvaluationPurpose):
        raise ValueError("purpose must be an EvaluationPurpose")
    self._purpose_stack.append(purpose)
    try:
        yield self
    finally:
        popped = self._purpose_stack.pop()
        if popped is not purpose:
            raise RuntimeError("evaluation purpose stack corrupted")

def _record_started(self) -> None:
    purpose = self._purpose_stack[-1] if self._purpose_stack else EvaluationPurpose.UNATTRIBUTED
    self.force_evaluations += 1
    self.energy_evaluations += 1
    self._purpose_counts[purpose] += 1
```

Call `_reserve()` before `_record_started()` in both evaluation methods.
`snapshot()` must return a new immutable `EvaluationCounts`.

- [ ] **Step 7: Run focused and existing accounting tests**

Run:

```bash
pytest -q tests/unit/test_accounting.py tests/integration/test_epam_accounting.py
```

Expected: all pass and existing aggregate semantics remain unchanged.

- [ ] **Step 8: Commit**

```bash
git add pamssw/accounting.py tests/unit/test_accounting.py
git commit -m "Add exact evaluation purpose ledger"
```

---

### Task 2: Terminal results carry exact purpose snapshots

**Files:**
- Modify: `pamssw/exploration/actions.py`
- Modify: `pamssw/exploration/controller.py`
- Modify: `pamssw/exploration/event_log.py`
- Modify: `pamssw/exploration/ssw_worker.py`
- Modify: `tests/unit/test_exploration_actions.py`
- Modify: `tests/unit/test_exploration_event_log.py`
- Modify: `tests/unit/test_ssw_attempt_worker.py`
- Modify: `tests/integration/test_exploration_controller.py`

- [ ] **Step 1: Write failing `AttemptResult` and `CreditedOutcome` tests**

Add coverage for exact counts, legacy unattributed conversion, mismatch, and
unknown cost:

```python
def test_attempt_result_defaults_legacy_scalar_to_unattributed():
    result = _failed_attempt(force_evaluations=3)
    assert result.evaluation_counts.total == 3
    assert result.evaluation_counts.count(EvaluationPurpose.UNATTRIBUTED) == 3
    assert result.cost_is_exact is True


def test_attempt_result_rejects_count_total_mismatch():
    with pytest.raises(ValueError, match="evaluation counts"):
        _failed_attempt(
            force_evaluations=3,
            evaluation_counts=EvaluationCounts.unattributed(2),
        )


def test_credited_outcome_preserves_exact_cost_fact():
    outcome = _failed_outcome(
        force_evaluations=1,
        evaluation_counts=EvaluationCounts.from_mapping(
            {EvaluationPurpose.DIRECTION_ORACLE: 1}
        ),
        cost_is_exact=False,
    )
    assert outcome.cost_is_exact is False
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_exploration_actions.py -k "evaluation or exact_cost or unattributed"
```

- [ ] **Step 3: Extend the terminal dataclasses**

Append backward-compatible fields:

```python
evaluation_counts: EvaluationCounts | None = None
cost_is_exact: bool = True
```

In `__post_init__`:

```python
counts = (
    EvaluationCounts.unattributed(force_evaluations)
    if self.evaluation_counts is None
    else self.evaluation_counts
)
if not isinstance(counts, EvaluationCounts):
    raise ValueError("evaluation_counts must be an EvaluationCounts")
if counts.total != force_evaluations:
    raise ValueError("evaluation counts must sum to force_evaluations")
if not isinstance(self.cost_is_exact, bool):
    raise ValueError("cost_is_exact must be a boolean")
object.__setattr__(self, "evaluation_counts", counts)
```

Apply the same invariant to `CreditedOutcome`.

Append `posterior_observed: bool = True` to `CreditedOutcome` and validate it as
a strict boolean. This is a persisted observation decision, not a policy
weight.

- [ ] **Step 4: Write failing controller propagation tests**

Assert that normal results copy their exact counts into credited outcomes and
escaped future exceptions are zero-count but unknown:

```python
def test_credit_preserves_evaluation_counts_and_exact_cost(tmp_path):
    counts = EvaluationCounts.from_mapping(
        {EvaluationPurpose.DIRECTION_ORACLE: 2}
    )
    captured: list[AttemptResult] = []

    def worker(action, starter_state):
        result = _completed(
            action,
            x=3.0,
            energy=-2.0,
            force_evaluations=2,
            evaluation_counts=counts,
        )
        captured.append(result)
        return result

    controller = ExplorationController(
        _archive(),
        "uniform",
        4,
        ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        (outcome,) = controller.run_batch(
            executor, worker, batch_size=1, force_budget=2
        )
    (result,) = captured
    assert outcome.evaluation_counts == result.evaluation_counts
    assert outcome.cost_is_exact is True


def test_escaped_future_exception_is_unknown_cost(tmp_path):
    def worker(action, starter_state):
        raise RuntimeError("escaped")

    controller = ExplorationController(
        _archive(),
        "uniform",
        4,
        ExplorationEventLog(tmp_path / "events.jsonl"),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        (outcome,) = controller.run_batch(
            executor, worker, batch_size=1, force_budget=2
        )
    assert outcome.force_evaluations == 0
    assert outcome.cost_is_exact is False
```

- [ ] **Step 5: Implement controller propagation**

Construct the controller fallback as:

```python
return AttemptResult(
    action=action,
    landing_state=None,
    landing_energy=None,
    force_evaluations=0,
    status=AttemptStatus.WORKER_ERROR,
    failure_reason=f"{type(exc).__name__}: {exc}",
    evaluation_counts=EvaluationCounts.zero(),
    cost_is_exact=False,
)
```

Pass both accounting fields from `AttemptResult` into `CreditedOutcome`.
Centralize the observation rule:

```python
def _posterior_observed(result: AttemptResult) -> bool:
    if result.status is AttemptStatus.WORKER_ERROR:
        return False
    if result.force_evaluations == 0:
        return False
    return result.status in {
        AttemptStatus.COMPLETED,
        AttemptStatus.BUDGET_EXHAUSTED,
        AttemptStatus.FRAGMENTED,
        AttemptStatus.INVALID,
    }
```

Only update `shadow_posterior` when this predicate is true and persist the
predicate result in the outcome. Test that factory, constructor,
escaped-future, and zero-cost invalid failures do not change Beta counts, while
physical completed, budget-exhausted, fragmented, and no-landing outcomes do.

- [ ] **Step 6: Write failing worker snapshot tests**

For completed, budget-exhausted, run-error, invalid-starter, factory-error, and
result-mapping-error paths assert:

```python
assert result.evaluation_counts.total == result.force_evaluations
assert result.cost_is_exact is True
```

For pre-calculator paths assert `EvaluationCounts.zero()`.

- [ ] **Step 7: Attach the action-local counter snapshot in every worker result**

Change result helpers to accept an `EvaluationCounts`. After walker
construction, read only `walker.calculator.snapshot()`. Before construction,
use `EvaluationCounts.zero()`.

Pass the snapshot into `_map_search_result`; do not infer purpose counts from
`SearchResult.stats`.

Use `walker.calculator.snapshot().total` as the authoritative terminal scalar.
Require `SearchResult.stats["force_evaluations"]` to equal it before mapping; a
mismatch becomes a result-mapping error carrying the same exact snapshot.

- [ ] **Step 8: Upgrade the compact event log in the same contract change**

Set `SCHEMA_VERSION = 2`. Serialize `evaluation_counts`, `cost_is_exact`, and
`posterior_observed` for every outcome. Require every purpose key in canonical
order, the purpose sum equal to `force_evaluations`, and strict JSON booleans.
Reject v1 rows, missing/extra purpose keys, sum mismatch, and omitted
observation facts.

Add a post-write acknowledgement retry regression proving the exact purpose
vector round-trips unchanged. This prevents an intermediate commit from
persisting exact terminal data through the lossy v1 event-log boundary.

Change `ExplorationEventLog.reconstruct_posterior()` in this same step to
recompute the observation predicate, validate stored `posterior_observed`, and
skip Beta updates for false flags. Update existing escaped-worker and
partial-submit controller assertions: their zero-cost `WORKER_ERROR` outcomes
remain logged but no longer increment Beta counts. Add one regression comparing
live controller posterior counts to compact-log reconstruction after a mixed
observed/unobserved batch.

- [ ] **Step 9: Run focused tests**

Run:

```bash
pytest -q \
  tests/unit/test_exploration_actions.py \
  tests/unit/test_exploration_event_log.py \
  tests/unit/test_ssw_attempt_worker.py \
  tests/integration/test_ssw_attempt_worker_integration.py \
  tests/integration/test_exploration_controller.py
```

- [ ] **Step 10: Commit**

```bash
git add pamssw/exploration/actions.py pamssw/exploration/controller.py \
  pamssw/exploration/event_log.py pamssw/exploration/ssw_worker.py \
  tests/unit/test_exploration_actions.py tests/unit/test_exploration_event_log.py \
  tests/unit/test_ssw_attempt_worker.py \
  tests/integration/test_exploration_controller.py
git commit -m "Persist exact terminal accounting"
```

---

### Task 3: Attribute the unchanged physical SSW path

**Files:**
- Modify: `pamssw/walker.py`
- Modify: `tests/integration/test_epam_accounting.py`
- Modify: `tests/integration/test_ssw_attempt_worker_integration.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] **Step 1: Add a real-worker route-coverage regression**

Use the existing recording analytic calculator and assert:

```python
result = worker(_action(force_budget=400), _state())
counts = result.evaluation_counts
assert counts.total == calculator.calls == result.force_evaluations
assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
assert counts.count(EvaluationPurpose.STARTER_TRUE_QUENCH) > 0
assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) > 0
assert counts.count(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK) > 0
assert counts.count(EvaluationPurpose.BIASED_PROPOSAL_RELAX) > 0
assert counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH) > 0
assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) > 0
```

Also assert failure and exact-budget paths retain exact purpose totals.

Parameterize the integration matrix over `SSWConfig` and `LSSSWConfig` and the
completed, duplicate/invalid, fragmented, worker-error, and budget-exhausted
terminal routes. For every case assert `counts.total == raw calculator calls`
and zero unattributed calls after walker construction. Pre-calculator failures
must instead have an exact zero vector.

Record deterministic parent-branch fixtures before instrumentation for both
SSW and LS-SSW completed, invalid/duplicate, and budget-exhausted paths. The
post-change parity gate compares landing state/energy, status, and aggregate
call total against those fixtures; purpose attribution must not change the
physical kernel.

- [ ] **Step 2: Run the route test and verify RED**

Run:

```bash
pytest -q tests/integration/test_ssw_attempt_worker_integration.py -k "purpose"
```

Expected: all calls are still unattributed.

- [ ] **Step 3: Add explicit quench and validation scopes**

Change:

```python
def relax_true_minimum(
    self,
    state: State,
    trajectory_name: str | None = None,
    *,
    quench_purpose: EvaluationPurpose = EvaluationPurpose.LANDING_TRUE_QUENCH,
) -> RelaxResult:
```

Wrap the relaxer call in `quench_purpose` and the finite post-check in
`POST_RELAX_VALIDATION`. Pass `STARTER_TRUE_QUENCH` for the initial action-local
quench in `run()`.

- [ ] **Step 4: Attribute direction and uphill evaluation sites**

In `_walk_candidate_from_seed`, scope:

```python
with self.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
    choice = self.oracle.choose_direction(
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
        history_gradient=self._history_bias_gradient(current, biases),
        continuity_weight=self._continuity_weight_for_outcome(previous_relax_outcome),
        n_bond_pairs=self._n_bond_pairs_for_outcome(previous_relax_outcome),
        score_sigma=(
            None
            if score_sigma_fn is not None
            else self._direction_score_sigma(sigma_scale, step_target=step_target)
        ),
        score_sigma_fn=score_sigma_fn,
        direction_type_bonus_fn=(
            self.direction_type_memory.bonus
            if self.config.direction_type_ucb_enabled
            else None
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
        archive_momentum_history=self._archive_momentum_history_for_seed(
            seed_entry_id
        ),
        archive_momentum_limit=self.config.archive_escape_momentum_limit,
    )

with self.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
    true_curvature = self._true_directional_curvature(
        current, choice.direction
    )

with self.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
    inner_curvature = (
        choice.curvature
        if self.config.direction_curvature_source == "inner"
        and not rebuild_softening_for_choice
        else self.oracle._directional_curvature(
            current, proposal, choice.direction
        )
    )

with self.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
    true_before = self.calculator.evaluate(current)

with self.calculator.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
    proposal_relax = Relaxer(
        proposal.evaluate,
        optimizer=proposal_optimizer,
    ).relax(
        trial_state,
        fmax=self.config.proposal_fmax,
        maxiter=self.config.proposal_relax_steps,
        coordinate_trust_radius=self.config.proposal_trust_radius,
        trajectory_callback=self._relaxation_trajectory_callback(
            self._trajectory_name(
                "proposal_relax",
                trial_index=trial_index,
                proposal_index=proposal_index,
                step_index=step_index,
            )
        ),
        trajectory_stride=self.config.relaxation_trajectory_stride,
    )

with self.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
    true_energy_after = self.calculator.evaluate(current_candidate).energy
```

Do not reorder calls, combine HVPs, alter RNG use, or change exception handling.

- [ ] **Step 5: Add a bootstrap-purpose hook without adding the runner**

Permit a caller to pass
`EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH` to `relax_true_minimum`; test this
directly on a fresh walker.

- [ ] **Step 6: Run accounting, walker, and parity tests**

Run:

```bash
pytest -q \
  tests/unit/test_walker_policy.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw_attempt_worker_integration.py \
  tests/integration/test_ssw.py \
  tests/integration/test_ls_ssw.py
```

The completed analytic action's total/raw call count and deterministic landing
must remain identical to its pre-attribution value.

- [ ] **Step 7: Commit**

```bash
git add pamssw/walker.py tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw_attempt_worker_integration.py \
  tests/unit/test_walker_policy.py
git commit -m "Attribute SSW evaluations by physical phase"
```

---

### Task 4: Complete committed-batch contract and strict event schema

**Files:**
- Create: `pamssw/exploration/committed.py`
- Modify: `pamssw/exploration/controller.py`
- Modify: `pamssw/exploration/event_log.py`
- Modify: `pamssw/exploration/__init__.py`
- Modify: `tests/unit/test_exploration_event_log.py`
- Modify: `tests/integration/test_exploration_controller.py`

- [ ] **Step 1: Write failing committed-batch alignment tests**

Define expectations:

```python
batch = CommittedExplorationBatch(snapshot, actions, results, outcomes)
assert batch.batch_id == actions[0].batch_id

with pytest.raises(ValueError, match="slot order"):
    CommittedExplorationBatch(snapshot, actions, tuple(reversed(results)), outcomes)
```

Reject non-tuples, empty batches, unequal lengths, action/result/outcome ID
mismatches, duplicate slots, snapshot mismatch, and non-slot order. For every
slot require:

```python
assert result.force_evaluations == outcome.force_evaluations
assert result.evaluation_counts == outcome.evaluation_counts
assert result.cost_is_exact == outcome.cost_is_exact
```

Include tampering whose scalar totals match but purpose distributions differ.

- [ ] **Step 2: Implement the immutable batch contract**

Create:

```python
@dataclass(frozen=True)
class CommittedExplorationBatch:
    snapshot: PolicySnapshot
    actions: tuple[StarterAction, ...]
    results: tuple[AttemptResult, ...]
    outcomes: tuple[CreditedOutcome, ...]

    @property
    def batch_id(self) -> int:
        return self.actions[0].batch_id
```

Capture/deepcopy result landing states through the already snapshotting
`AttemptResult`; do not duplicate states into `CreditedOutcome`.

- [ ] **Step 3: Refactor `BatchLog` and pending commit**

Change the protocol to:

```python
class BatchLog(Protocol):
    def append_batch(self, batch: CommittedExplorationBatch) -> None:
        """Durably append one complete finalized batch."""
```

Store the complete batch in `_PendingCommit`. Keep shadow archive/posterior and
next versions private. `run_batch` still returns outcomes.

Expose:

```python
def reconcile_pending_commit(self) -> CommittedExplorationBatch:
    """Retry/acknowledge the complete pending batch and install it once."""
```

`run_batch` uses this method after finalizing a batch. A log write failure
leaves `_PendingCommit` intact; calling the method again never samples or
dispatches. On success it installs shadow state, clears pending, and returns the
complete batch. Add read-only `has_pending_commit` for runner control flow.

Add `require_exact_cost: bool = False` to the controller constructor. Before
credit or log mutation:

```python
class UnknownActionCostError(RuntimeError):
    """A dispatched batch contains a result whose physical cost is unknown."""


if self.require_exact_cost and any(not result.cost_is_exact for result in ordered_results):
    raise UnknownActionCostError("strict exploration requires exact worker costs")
```

The Phase-3 runner must construct the controller with
`require_exact_cost=True`; the default exists only for the generic Phase-1
controller.

- [ ] **Step 4: Update controller tests**

Prove:

- generic non-strict mode still commits unknown-cost failures;
- strict mode refuses the whole batch before posterior/archive/log mutation;
- in-process pending write retry reuses the exact complete batch;
- completion order still cannot affect slot-order batch content.

- [ ] **Step 5: Move event-log v2 onto the complete-batch protocol**

Change `append_batch` to accept `CommittedExplorationBatch`; preserve the
schema-v2 accounting and observation fields introduced in Task 2. Serialize
snapshot/actions/outcomes from the complete object. Idempotency compares
canonical serialized facts, not NumPy state equality.

- [ ] **Step 6: Add strict parser regressions**

Reject:

- schema v1 rows;
- missing/extra purpose keys;
- purpose sum mismatch;
- non-boolean exact flag;
- non-boolean posterior-observed flag;
- result/outcome action mismatch at append;
- same-total/different-purpose result/outcome mismatch;
- incomplete final batch.

Round-trip posterior reconstruction must remain exact.

- [ ] **Step 7: Run focused controller/event-log suites**

Run:

```bash
pytest -q \
  tests/unit/test_exploration_event_log.py \
  tests/integration/test_exploration_controller.py
```

- [ ] **Step 8: Commit**

```bash
git add pamssw/exploration/committed.py pamssw/exploration/controller.py \
  pamssw/exploration/event_log.py pamssw/exploration/__init__.py \
  tests/unit/test_exploration_event_log.py \
  tests/integration/test_exploration_controller.py
git commit -m "Make committed exploration batches complete"
```

---

### Task 5: Fixed-fidelity campaign budget and configuration

**Files:**
- Create: `pamssw/exploration/campaign.py`
- Create: `tests/unit/test_exploration_campaign.py`
- Modify: `pamssw/accounting.py`
- Modify: `pamssw/exploration/__init__.py`
- Modify: `tests/unit/test_accounting.py`

- [ ] **Step 1: Write failing configuration tests**

Construct the public frozen configuration:

```python
config = PosteriorExplorationConfig(
    policy_name="uniform",
    batch_size=3,
    max_workers=2,
    action_force_budget=10,
    total_force_budget=101,
    master_seed=7,
    calculator_label="analytic-double-well-v1",
    calculator_fingerprint="coupled-pair-well:v1:a=1.0:b=1.0:coupling=0.25",
    run_directory=tmp_path / "run",
    mode="new",
)
```

Reject unsupported policies, bool-as-int, nonpositive sizes/budgets, workers
larger than batch size, empty label/fingerprint, invalid mode, and non-path run
directory. Do not add policy temperatures or budget weights.

- [ ] **Step 2: Implement `PosteriorExplorationConfig`**

Normalize `run_directory` to `Path` in `__post_init__` and reuse
`SUPPORTED_POLICIES`.

- [ ] **Step 3: Write failing pure-ledger tests**

Cover bootstrap and batch reservation:

```python
ledger = CampaignBudget(total=101, action_force_budget=10)
ledger.record_bootstrap(EvaluationCounts.from_mapping(
    {EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH: 11}
))
assert ledger.next_batch_size(max_batch_size=3) == 3

ledger.commit_batch(
    action_counts=(
        EvaluationCounts.from_mapping(
            {EvaluationPurpose.DIRECTION_ORACLE: 8}
        ),
        EvaluationCounts.from_mapping(
            {EvaluationPurpose.LANDING_TRUE_QUENCH: 10}
        ),
        EvaluationCounts.from_mapping(
            {EvaluationPurpose.BIASED_PROPOSAL_RELAX: 3}
        ),
    ),
)
assert ledger.spent == 32
assert ledger.remaining == 69
```

Final tail:

```python
ledger = CampaignBudget(total=25, action_force_budget=10)
ledger.record_bootstrap(EvaluationCounts.from_mapping(
    {EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH: 6}
))
assert ledger.next_batch_size(3) == 1
ledger.commit_batch(
    (
        EvaluationCounts.from_mapping(
            {EvaluationPurpose.DIRECTION_ORACLE: 7}
        ),
    ),
)
assert ledger.next_batch_size(3) == 1
ledger.commit_batch(
    (
        EvaluationCounts.from_mapping(
            {EvaluationPurpose.LANDING_TRUE_QUENCH: 6}
        ),
    ),
)
assert ledger.next_batch_size(3) == 0
assert ledger.unused == 6
```

Reject overspend, actual cost larger than reservation, wrong reservation width,
double bootstrap, negative/boolean costs, bool/zero/negative batch arguments,
non-`EvaluationCounts` elements, and commit after terminal stop.

- [ ] **Step 4: Implement the ledger**

Use a mutable campaign-owned dataclass with no policy knowledge:

```python
class CampaignStopReason(str, Enum):
    BUDGET_TAIL = "budget_tail"
    ZERO_COST_STALL = "zero_cost_stall"


@dataclass
class CampaignBudget:
    total: int
    action_force_budget: int
    bootstrap_counts: EvaluationCounts = field(default_factory=EvaluationCounts.zero)
    action_counts: EvaluationCounts = field(default_factory=EvaluationCounts.zero)
    committed_batches: int = 0
    committed_attempts: int = 0
    bootstrap_recorded: bool = False
    stop_reason: CampaignStopReason | None = None

    @property
    def spent(self) -> int:
        return self.bootstrap_counts.total + self.action_counts.total

    @property
    def remaining(self) -> int:
        return self.total - self.spent

    def next_batch_size(self, max_batch_size: int) -> int:
        if self.stop_reason is not None:
            return 0
        # strictly validate the batch argument before division
        return min(max_batch_size, self.remaining // self.action_force_budget)

    def record_bootstrap(self, counts: EvaluationCounts) -> None:
        if self.bootstrap_recorded:
            raise RuntimeError("bootstrap cost is already recorded")
        if counts.total > self.total:
            raise ValueError("bootstrap cost exceeds campaign budget")
        self.bootstrap_counts = counts
        self.bootstrap_recorded = True

    @property
    def unused(self) -> int:
        return self.remaining
```

Implement:

```python
def commit_batch(
    self,
    action_counts: tuple[EvaluationCounts, ...],
) -> None:
    if not action_counts:
        raise ValueError("action_counts cannot be empty")
    if len(action_counts) * self.action_force_budget > self.remaining:
        raise ValueError("batch reservation exceeds remaining campaign budget")
    if any(counts.total > self.action_force_budget for counts in action_counts):
        raise ValueError("action cost exceeds its reserved budget")
    merged = EvaluationCounts.sum(action_counts)
    self.action_counts = self.action_counts + merged
    self.committed_batches += 1
    self.committed_attempts += len(action_counts)
```

Add immutable elementwise `EvaluationCounts.__add__` and
`EvaluationCounts.sum`. `commit_batch` consumes actual counts, not reservation.
If merged batch cost is zero, set `stop_reason=ZERO_COST_STALL`; otherwise, if
`remaining < action_force_budget`, set `stop_reason=BUDGET_TAIL`. Bootstrap
performs the same tail check.

The immutable snapshot must persist the fixed action budget and the complete
pure-budget batch history: one positive action count and one merged
`EvaluationCounts` value for every committed batch. Do not use duplicated
aggregate counters or only the final batch spend as recovery authority.
Restore receives the manifest action budget, requires exact equality with the
snapshot, and validates the historical batches sequentially from the recorded
bootstrap state:

1. each batch reservation fitted the remaining budget at its historical
   dispatch boundary;
2. each merged batch cost is no larger than its action count multiplied by the
   fixed action budget;
3. a zero-cost batch is necessarily the final committed batch;
4. aggregate action counts, committed attempts, committed batches, final batch
   spend, and the terminal reason are derived from the validated history.

Both terminal reasons survive restore and make `next_batch_size` return zero.

- [ ] **Step 5: Define the result summary**

Define the complete frozen public contract:

```python
@dataclass(frozen=True)
class PosteriorExplorationResult:
    archive: MinimaArchive
    posterior: StarterProductivityPosterior
    policy_name: str
    completed_batches: int
    completed_attempts: int
    failed_attempts: int
    posterior_observed_attempts: int
    bootstrap_evaluations: int
    action_evaluations: int
    total_evaluations: int
    purpose_counts: EvaluationCounts
    total_force_budget: int
    unused_force_budget: int
    stop_reason: CampaignStopReason
    benchmark_eligible: bool
    benchmark_ineligibility_reasons: tuple[str, ...]
    run_directory: Path
```

Validate policy name, nonnegative/non-boolean counters, attempt partition,
posterior-observation bound, total equations, purpose sum, budget equation,
strict tuple reasons, and archive/posterior types. Clone mutable archive and
posterior inputs in `__post_init__`.

- [ ] **Step 6: Run the campaign unit suite**

Run:

```bash
pytest -q tests/unit/test_exploration_campaign.py
```

- [ ] **Step 7: Commit**

```bash
git add pamssw/exploration/campaign.py \
  pamssw/accounting.py \
  pamssw/exploration/__init__.py \
  tests/unit/test_exploration_campaign.py \
  tests/unit/test_accounting.py
git commit -m "Add fixed-fidelity campaign budget"
```

---

> **Superseded remainder:** Tasks 6–11 below are not part of the active
> implementation scope. They were replaced by
> `docs/superpowers/specs/2026-07-26-minimal-posterior-ablation-runner-design.md`
> and
> `docs/superpowers/plans/2026-07-26-minimal-posterior-ablation-runner.md`.
> Phase-3 completion now means a non-recoverable analytic ThreadPool runner
> and raw three-policy harness; no manifest/resume/replay layer is required.

### Task 6: Strict state and run-manifest serialization

**Files:**
- Create: `pamssw/exploration/run_store.py`
- Create: `tests/unit/test_exploration_run_store.py`

- [ ] **Step 1: Write failing strict state-codec tests**

Round-trip:

```python
payload = encode_state(state)
decoded = decode_state(payload)
np.testing.assert_array_equal(decoded.numbers, state.numbers)
np.testing.assert_array_equal(decoded.positions, state.positions)
np.testing.assert_array_equal(decoded.fixed_mask, state.fixed_mask)
assert decoded.pbc == state.pbc
assert decoded.metadata == state.metadata
```

Reject unknown/missing keys, NaN/Inf, shape mismatch, atom-count mismatch,
non-boolean PBC/mask, duplicate JSON keys, and metadata containing NumPy arrays,
sets, bytes, custom objects, non-string dictionary keys, or nonfinite numbers.

- [ ] **Step 2: Implement canonical state encoding**

Use exact fields:

```python
{
    "numbers": [1, 8],
    "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    "cell": None,
    "pbc": [False, False, False],
    "fixed_mask": [False, False],
    "metadata": {"label": "example"},
}
```

Implement a recursive JSON-value validator. Do not use pickle, `default=str`,
ASE binary serialization, or silent metadata dropping.

- [ ] **Step 3: Write failing manifest tests**

Define a versioned `RunManifest` containing:

```text
run_id
schema_version
created_at_utc
exploration_config
policy_support_semantics
ssw_config_kind
caller_ssw_config
effective_bootstrap_config
effective_action_config
initial_state
bootstrap_state
bootstrap_energy
bootstrap_counts
archive_tolerances
calculator_label
calculator_fingerprint
telemetry_schema_version
run_store_schema_version
environment
repository
```

Test deterministic canonical JSON encoding except for explicitly supplied
`run_id`/timestamp. Test new-mode directory rules and resume compatibility.

- [ ] **Step 4: Implement manifest creation and validation**

Serialize dataclass/enum/path/tuple values through one canonical converter.
Require the caller to pass run ID and timestamp so tests do not depend on the
clock. Capture environment/repository provenance separately from semantic
resume identity.

Semantic compatibility includes policy/support semantics, budgets, seeds,
calculator label/fingerprint, complete caller/effective SSW configs, initial
state, schema versions, repository source fingerprint, and
Python/NumPy/SciPy/ASE/package versions. Any difference in those executable
kernel facts fails closed.

Record `PosteriorExplorationConfig.mode`, but exclude only that field from
semantic equality because resume necessarily changes `new` to `resume`.
Require the canonical `run_directory` and every other field to match.
Define repository source fingerprint from the commit plus tracked source diff;
do not include untracked run artifacts, so writing a run directory inside the
checkout cannot change its own kernel identity.

Represent allowed hardware/non-kernel drift with:

```python
@dataclass(frozen=True)
class RunSession:
    session_id: int
    created_at_utc: str
    environment: dict[str, object]
    benchmark_eligible: bool
    benchmark_ineligibility_reasons: tuple[str, ...]
```

Creation atomically writes `sessions/00000000.json`; every accepted resume
writes the next contiguous session record before dispatch. Tests decode the
session file and prove drift eligibility is durable and later propagated into
`PosteriorExplorationResult` and benchmark run records.

- [ ] **Step 5: Implement atomic JSON write helper**

Use same-directory unique temp files:

```python
def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    # open temp with O_EXCL/O_NOFOLLOW
    # write canonical JSON + newline
    # flush and fsync file
    # os.replace(temp, path)
    # fsync held parent directory
```

Reject final symlinks and require existing parent directories. Clean up only
the temp file created by this call after a failed pre-rename write.

At run creation, create `batches/` and `sessions/`, fsync both directories and
the run root, then publish the manifest and first session. Add monkeypatched
fsync-order tests so directory-entry durability is not implicit.

- [ ] **Step 6: Run codec/manifest tests**

Run:

```bash
pytest -q tests/unit/test_exploration_run_store.py -k "state or manifest or atomic"
```

- [ ] **Step 7: Commit**

```bash
git add pamssw/exploration/run_store.py \
  tests/unit/test_exploration_run_store.py
git commit -m "Add strict exploration run manifest"
```

---

### Task 7: Atomic complete-batch journal

**Files:**
- Modify: `pamssw/exploration/run_store.py`
- Modify: `pamssw/exploration/__init__.py`
- Modify: `tests/unit/test_exploration_run_store.py`

- [ ] **Step 1: Write failing batch round-trip tests**

Encode/decode a `CommittedExplorationBatch` with:

- full policy vector;
- multiple actions in slot order;
- one completed result with landing state;
- one failed result;
- credited outcomes;
- exact purpose counts and cost flags;
- budget before/reserved/spent/after;
- controller versions before/after.

Assert decoded states and all scalar facts match.

- [ ] **Step 2: Define the stored batch envelope**

Create:

```python
@dataclass(frozen=True)
class StoredExplorationBatch:
    run_id: str
    batch: CommittedExplorationBatch
    budget_before: int
    reserved: int
    spent: int
    budget_after: int
    policy_version_before: int
    archive_version_before: int
    policy_version_after: int
    archive_version_after: int
    batch_id_after: int
```

Validate equations, contiguous versions, per-slot result/outcome equality for
scalar total, complete purpose vector and exact-cost flag, and exact costs. A
batch is benchmark-eligible only when unattributed calls are zero and every
terminal outcome is posterior-observed; persist eligibility reasons in the
envelope. A same-total/different-purpose pair must fail.

Define the pre-dispatch facts separately:

```python
@dataclass(frozen=True)
class PendingBatchMetadata:
    budget_before: int
    reserved: int
    policy_version_before: int
    archive_version_before: int
    batch_id_before: int
```

`StoredExplorationBatch.from_pending` computes:

```python
spent = sum(result.force_evaluations for result in batch.results)
budget_after = metadata.budget_before + spent
policy_version_after = metadata.policy_version_before + 1
archive_version_after = metadata.archive_version_before + 1
batch_id_after = metadata.batch_id_before + 1
```

It rejects a batch whose identity does not match the pending metadata or whose
spend exceeds the reservation. It also requires all actions to share one
positive `force_budget` and:

```python
metadata.reserved == len(batch.actions) * batch.actions[0].force_budget
```

- [ ] **Step 3: Implement strict action/result/outcome codecs**

Every decoder must require an exact field set and reconstruct the public
dataclasses, allowing their `__post_init__` validation to run. Serialize
`AttemptResult.landing_state`; do not serialize arbitrary Python exceptions or
calculator objects.

- [ ] **Step 4: Write failing append/idempotency/corruption tests**

Cover:

- first append creates `batches/00000000.json`;
- byte-equivalent retry succeeds;
- different content for the same batch ID fails;
- gaps and duplicate action IDs fail;
- truncated and unknown-field canonical files fail;
- an orphan temp pre-rename file is ignored and not loaded;
- a temp file coexisting with its canonical batch fails closed;
- symlink final path is rejected.

- [ ] **Step 5: Implement `ExplorationRunStore`**

Public surface:

```python
class ExplorationRunStore:
    @classmethod
    def create(cls, config, manifest) -> "ExplorationRunStore":
        root = _create_new_run_directory(config.run_directory)
        _atomic_write_json(root / "manifest.json", encode_manifest(manifest))
        return cls(root=root, manifest=manifest)

    @classmethod
    def open_for_resume(cls, config, initial_state) -> "ExplorationRunStore":
        root = _require_existing_run_directory(config.run_directory)
        manifest = decode_manifest(_read_strict_json(root / "manifest.json"))
        _validate_resume_identity(manifest, config, initial_state)
        return cls(root=root, manifest=manifest)

    def prepare_batch_commit(self, metadata: PendingBatchMetadata) -> None:
        if self._pending_metadata is not None:
            raise RuntimeError("a batch commit is already pending")
        self._pending_metadata = metadata

    def append_batch(self, batch: CommittedExplorationBatch) -> None:
        if self._pending_metadata is None:
            raise RuntimeError("batch commit metadata was not prepared")
        stored = StoredExplorationBatch.from_pending(
            self.manifest.run_id,
            batch,
            self._pending_metadata,
        )
        self._append_stored_batch(stored)
        self._pending_metadata = None

    def reconcile_pending_commit(
        self,
        batch: CommittedExplorationBatch,
    ) -> StoredExplorationBatch:
        """Retry/acknowledge the exact pending batch without redispatch."""
        self.append_batch(batch)
        return self.load_batches()[-1]

    def load_batches(self) -> tuple[StoredExplorationBatch, ...]:
        return tuple(
            decode_stored_batch(_read_strict_json(path))
            for path in _contiguous_batch_paths(self.root / "batches")
        )
```

The store receives pending budget/version metadata from a runner-owned context
set before `controller.run_batch`; clear it only after an idempotent append.
Use the shown `prepare_batch_commit` method. Do not derive budget facts from
completion timing.

`prepare_batch_commit` accepts exactly one `PendingBatchMetadata` object; add an
interface test that rejects keyword expansion. Temp discovery scans canonical
and unique temp names together. Orphan temps are ignored; canonical/temp
coexistence is ambiguous and rejected.

Add atomic `write_abort_unknown_cost(...)` and `read_abort()` methods. The
marker records run/session/batch identity, last committed versions and spend,
and the exception class/message without claiming a cost. It is write-once and
idempotent only for byte-equivalent content; `open_for_resume` rejects its
presence.

- [ ] **Step 6: Run run-store tests**

Run:

```bash
pytest -q tests/unit/test_exploration_run_store.py
```

- [ ] **Step 7: Commit**

```bash
git add pamssw/exploration/run_store.py pamssw/exploration/__init__.py \
  tests/unit/test_exploration_run_store.py
git commit -m "Journal complete exploration batches atomically"
```

---

### Task 8: Replay-validated controller recovery

**Files:**
- Modify: `pamssw/exploration/controller.py`
- Modify: `pamssw/exploration/run_store.py`
- Modify: `tests/integration/test_exploration_controller.py`
- Modify: `tests/unit/test_exploration_run_store.py`

- [ ] **Step 1: Write a failing controller replay-equivalence test**

Run two deterministic fake-worker batches, load them from the store, construct
a fresh controller, and assert:

```python
assert _archive_fingerprint(recovered.archive) == _archive_fingerprint(uninterrupted.archive)
assert _posterior_counts(recovered.posterior) == _posterior_counts(uninterrupted.posterior)
assert recovered.policy_version == uninterrupted.policy_version
assert recovered.archive_version == uninterrupted.archive_version
assert recovered.batch_id == uninterrupted.batch_id
```

Include completed new landings, duplicates, within-batch collisions, and failed
attempts. Keep the fingerprint/count helpers private to the tests; do not expand
the public controller API for assertions.

- [ ] **Step 2: Implement replay from complete batch facts**

Add:

```python
@classmethod
def from_committed_batches(
    cls,
    initial_archive: MinimaArchive,
    policy_name: str,
    master_seed: int,
    action_force_budget: int,
    batch_log: BatchLog,
    batches: tuple[CommittedExplorationBatch, ...],
    *,
    require_exact_cost: bool,
) -> "ExplorationController":
    controller = cls(
        initial_archive,
        policy_name,
        master_seed,
        batch_log,
        require_exact_cost=require_exact_cost,
    )
    for expected_batch_id, batch in enumerate(batches):
        if batch.batch_id != expected_batch_id:
            raise ValueError("committed batches must be contiguous")
        controller._replay_committed_batch(batch)
    return controller
```

Validate `action_force_budget` as a positive non-boolean integer and require
every stored action's `force_budget` to equal this manifest/config truth before
replay. The batch envelope's self-consistent reservation is not sufficient:
reject a batch whose actions and reservation were all changed together to a
different cap.

For every batch:

1. rebuild the expected policy snapshot from current archive/posterior;
2. compare the complete probability vector and versions;
3. regenerate the planned actions from master seed, batch ID, width, and fixed
   action cap, then compare exact actions;
4. re-run `_credit_result` against cloned dispatch/shadow state using stored
   terminal results;
5. recompute the posterior-observation predicate, update Beta counts only when
   true, and compare the complete computed outcomes (including
   `posterior_observed`) to stored outcomes;
6. install shadow archive/posterior and increment versions.

Do not trust serialized posterior counts or archive prototypes directly.

- [ ] **Step 3: Add tamper and mismatch regressions**

Reject:

- changed landing coordinate/energy;
- changed discovered/inserted/collision flag;
- changed selection probability or seed;
- non-contiguous batch/policy/archive versions;
- changed action cap;
- batch-internally-consistent action cap/reservation that differs from manifest
  `action_force_budget`;
- missing batch;
- unknown-cost stored result in strict mode.

Include a positive-cost `WORKER_ERROR` with `posterior_observed=False`, followed
by a later observed batch. Uninterrupted and recovered posterior counts,
snapshots, propensities, and next planned actions must be identical; compact
event-log reconstruction must produce the same result.

- [ ] **Step 4: Restore budget by validated stored envelopes**

After controller replay, reconstruct `CampaignBudget` from bootstrap counts and
stored batch costs. Recompute every total and compare to each envelope rather
than assigning serialized `spent`.

`load_batches()` returns validated `StoredExplorationBatch` envelopes. Pass
`tuple(stored.batch for stored in stored_batches)` to
`from_committed_batches` only after validating every envelope in order. Derive
terminal state after reconstruction: a last zero-spend envelope means
`ZERO_COST_STALL`; otherwise `remaining < action_force_budget` means
`BUDGET_TAIL`.

- [ ] **Step 5: Run recovery suites**

Run:

```bash
pytest -q \
  tests/integration/test_exploration_controller.py \
  tests/unit/test_exploration_run_store.py
```

- [ ] **Step 6: Commit**

```bash
git add pamssw/exploration/controller.py pamssw/exploration/run_store.py \
  tests/integration/test_exploration_controller.py \
  tests/unit/test_exploration_run_store.py
git commit -m "Recover exploration state by replay"
```

---

### Task 9: Opt-in posterior SSW campaign runner

**Files:**
- Create: `pamssw/exploration/runner.py`
- Create: `tests/integration/test_posterior_exploration_runner.py`
- Modify: `pamssw/runner.py`
- Modify: `pamssw/__init__.py`
- Modify: `pamssw/exploration/__init__.py`

- [ ] **Step 1: Write a failing new-run analytic integration test**

Use a recording `DoubleWell2D` calculator factory:

```python
result = run_posterior_ssw(
    initial_state,
    calculator_factory,
    SSWConfig(max_steps_per_walk=1, oracle_candidates=2),
    PosteriorExplorationConfig(
        policy_name="uniform",
        batch_size=3,
        max_workers=3,
        action_force_budget=40,
        total_force_budget=170,
        master_seed=9,
        calculator_label="double-well-v1",
        calculator_fingerprint="double-well-2d:v1:a=1.0:b=1.0",
        run_directory=tmp_path / "run",
        mode="new",
    ),
)
```

Assert:

- bootstrap is counted;
- every action uses a distinct calculator;
- all actions use cap 40;
- total raw calls equal result total and purpose sum;
- total is at most 170;
- no unattributed calls;
- final batch width follows `remaining // 40`;
- unused budget is reported;
- run store contains the same number of committed batches.

- [ ] **Step 2: Implement bootstrap preparation**

Create one fresh calculator and `SurfaceWalker` with a budget equal to the
campaign cap. Call:

```python
walker.relax_true_minimum(
    deepcopy(initial_state),
    trajectory_name=None,
    quench_purpose=EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH,
)
```

Build the one-entry central `MinimaArchive` from the relaxed state/energy. Use
`POST_RELAX_VALIDATION` for its post-check. Create the manifest only after
bootstrap succeeds.

Construct the bootstrap walker from:

```python
bootstrap_config = replace(
    ssw_config,
    max_trials=1,
    max_force_evals=exploration_config.total_force_budget,
    accepted_structures_log=None,
    accepted_structures_dir=None,
    write_proposal_minima=False,
    write_relaxation_trajectories=False,
    direction_diagnostics_enabled=False,
    direction_archive_enabled=False,
    direction_archive_path=None,
)
```

Reject a caller configuration that requests shared filesystem output instead
of silently disabling it. The `replace` values above document the effective
side-effect-free bootstrap contract; only fields already required to be off are
accepted.

Persist three distinct config facts in the manifest: the caller config above,
the exact `bootstrap_config`, and an `effective_action_config` template showing
`max_trials=1`, shared outputs disabled, and the fixed
`max_force_evals=action_force_budget`; action-specific `rng_seed` remains in
each `StarterAction`. Resume compares all three facts.

Define:

```python
class BootstrapExplorationError(RuntimeError):
    def __init__(self, message: str, evaluation_counts: EvaluationCounts):
        super().__init__(message)
        self.evaluation_counts = evaluation_counts
```

Invalid geometry and budget exhaustion raise this error with the exact current
bootstrap snapshot. No manifest/run directory is published. The budget
exhaustion test asserts
`exc.value.evaluation_counts.total == raw_calculator.calls == campaign_cap`.

- [ ] **Step 3: Implement the synchronous fixed-cap loop**

Pseudocode:

```python
if budget.stop_reason is not None:
    return _assemble_result(stop_reason=budget.stop_reason)

while True:
    width = budget.next_batch_size(config.batch_size)
    if width == 0:
        assert budget.stop_reason is CampaignStopReason.BUDGET_TAIL
        stop_reason = budget.stop_reason
        break
    store.prepare_batch_commit(PendingBatchMetadata(
        budget_before=budget.spent,
        reserved=width * config.action_force_budget,
        policy_version_before=controller.policy_version,
        archive_version_before=controller.archive_version,
        batch_id_before=controller.batch_id,
    ))
    try:
        outcomes = controller.run_batch(
            executor,
            worker,
            batch_size=width,
            force_budget=config.action_force_budget,
        )
    except UnknownActionCostError as exc:
        store.write_abort_unknown_cost(
            batch_id=controller.batch_id,
            budget_spent=budget.spent,
            error=exc,
        )
        raise
    except Exception as exc:
        if not controller.has_pending_commit:
            store.write_abort_unknown_cost(
                batch_id=controller.batch_id,
                budget_spent=budget.spent,
                error=exc,
            )
            raise
        committed = controller.reconcile_pending_commit()
        outcomes = committed.outcomes
    except BaseException as exc:
        store.write_abort_unknown_cost(
            batch_id=controller.batch_id,
            budget_spent=budget.spent,
            error=exc,
        )
        raise
    stored = store.load_batches()[-1]
    budget.commit_batch(
        action_counts=tuple(
            outcome.evaluation_counts for outcome in stored.batch.outcomes
        ),
    )
    batch_spent = stored.spent
    if batch_spent == 0:
        stop_reason = CampaignStopReason.ZERO_COST_STALL
        break
```

Use one `ThreadPoolExecutor(max_workers=config.max_workers)` for the campaign.
Construct `ExplorationController(..., require_exact_cost=True)`. Never sample
before width/reservation is fixed.

Assemble `benchmark_eligible` and reasons from every session record and stored
batch envelope. A posterior-unobserved terminal fact, unattributed call, or
allowed environment drift remains visible in the final result even when the
campaign otherwise stops normally.

The controller exposes `reconcile_pending_commit()` for runner use. If
If `append_batch` raises, the runner first calls this reconciliation method: it
either retries the exact pending batch or acknowledges the already-renamed
canonical batch, installs controller shadow state, and returns that same
`CommittedExplorationBatch`. Only then does the runner load the envelope and
advance `CampaignBudget` once. It never calls `prepare_batch_commit` again
while controller/store pending state exists.

- [ ] **Step 4: Write and implement resume equivalence**

Test an uninterrupted two-plus-batch run against:

1. a run stopped cleanly after a chosen committed batch using an internal
   test-only batch limit hook;
2. process objects discarded;
3. `mode="resume"` with the same semantic config;
4. continuation to the original total budget.

Assert archive states/IDs, posterior, action IDs/seeds/propensities, versions,
purpose costs, total spend, and stop reason equal uninterrupted execution.

The test-only stop hook must not be public configuration. Put it in an internal
runner helper or inject a batch observer in tests.

Add an injected dispatch-complete/pre-append interruption test. It must prove:

1. worker raw calls occurred but no canonical batch, committed ledger cost,
   archive credit, or posterior update exists;
2. process objects are discarded;
3. explicit resume starts from the prior committed boundary and regenerates
   identical action IDs, seeds, and propensities;
4. the uninterrupted and resumed committed state/ledger match, while the raw
   uncommitted hardware calls are reported separately as outside the ledger.

Add pre-rename write failure and post-rename acknowledgement failure tests.
Within one live process, both reconcile the pending complete batch without any
worker re-execution and advance the ledger exactly once.

- [ ] **Step 5: Add failure-path integrations**

Cover:

- invalid bootstrap and bootstrap budget exhaustion create no resumable run;
- unknown-cost escaped worker failure aborts before batch commit;
- the unknown-cost path writes `abort.json` and resume fails closed;
- known zero-cost batch commits once then stops;
- resume from zero-cost stall is a no-op and preserves
  `stop_reason == ZERO_COST_STALL` rather than rewriting it as `BUDGET_TAIL`;
- post-write acknowledgement failure retries the exact batch without
  re-executing workers;
- `KeyboardInterrupt`/`SystemExit` propagate, write no ordinary action outcome,
  commit no batch, and conservatively leave an unknown-cost abort marker;
- shared-output SSW configuration is refused at the public runner boundary;
- incompatible resume initial state/config/calculator label/fingerprint,
  executable-kernel provenance, or effective transformed config fails closed;
- natural resume after budget-tail returns immediately without new actions.

- [ ] **Step 6: Implement public wrappers**

In `pamssw/runner.py`:

```python
def run_posterior_ssw(initial_state, calculator_factory, ssw_config, exploration_config):
    return run_posterior_exploration(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=False,
    )

def run_posterior_ls_ssw(initial_state, calculator_factory, ssw_config, exploration_config):
    return run_posterior_exploration(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=True,
    )
```

The internal boolean is not user-facing. Validate SSW/LSSSW config pairing as
strictly as `SSWAttemptWorker`.

Add a real analytic `run_posterior_ls_ssw` integration using `LSSSWConfig`.
Assert the same bootstrap-inclusive cap, raw-call/purpose parity, strict exact
cost, and fixed action cap as SSW. Assert that passing `SSWConfig` to the LS
wrapper and `LSSSWConfig` to the non-LS wrapper fails before bootstrap.

- [ ] **Step 7: Export only the intended public API**

Export `PosteriorExplorationConfig`, `PosteriorExplorationResult`,
`run_posterior_ssw`, and `run_posterior_ls_ssw` from package root. Keep storage
internals under `pamssw.exploration`.

- [ ] **Step 8: Run runner and legacy regression suites**

Run:

```bash
pytest -q \
  tests/integration/test_posterior_exploration_runner.py \
  tests/integration/test_exploration_controller.py \
  tests/integration/test_ssw.py \
  tests/integration/test_ls_ssw.py
```

- [ ] **Step 9: Commit**

```bash
git add pamssw/exploration/runner.py pamssw/runner.py pamssw/__init__.py \
  pamssw/exploration/__init__.py \
  tests/integration/test_posterior_exploration_runner.py
git commit -m "Add recoverable budgeted posterior runner"
```

---

### Task 10: Exact-force analytic policy benchmark and documentation

**Files:**
- Create: `benchmarks/posterior_policy_compare.py`
- Create: `tests/integration/test_posterior_policy_benchmark.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing benchmark-contract smoke test**

Run a tiny paired two-seed, three-policy campaign and assert the raw payload
contains:

```text
schema_version
policies
paired_seeds
paired_master_seeds
kernel_config
action_force_budget
total_force_budget
batch_size
max_workers
runs
environment
repository
```

For every run require:

- complete run-directory path;
- policy support flag;
- best-energy/unique-minima points indexed by cumulative physical evaluations;
- purpose counts summing to total;
- bootstrap-inclusive first curve point;
- primary points only at committed-batch boundaries;
- completed and failed attempt counts at every point;
- unused budget;
- exact propensities;
- durable benchmark eligibility and reasons from manifest/session/batches;
- no `"winner"`, `"improved policy"`, or promotion claim.

- [ ] **Step 2: Implement raw campaign extraction**

Create frozen JSON-safe run records and load committed batches from each run
store. Build primary points only for campaign states that existed. Start with:

```python
{
    "force_evaluations": manifest.bootstrap_counts.total,
    "best_energy": manifest.bootstrap_energy,
    "unique_minima": 1,
    "completed_attempts": 0,
    "failed_attempts": 0,
}
```

Then add one point after each complete committed batch:

```python
{
    "force_evaluations": cumulative_calls,
    "best_energy": best_energy,
    "unique_minima": archive_size,
    "completed_attempts": completed_attempts,
    "failed_attempts": failed_attempts,
}
```

Do not create slot-prefix points because slots shared one frozen dispatch
snapshot and those prefixes were never campaign states. If a diagnostic
slot-prefix table is ever emitted, label it `post_hoc_slot_prefix` and exclude
it from policy curves/estimands. Do not use local relaxation count as x-axis.

- [ ] **Step 3: Implement paired harness CLI**

Required CLI arguments:

```text
--seeds
--policies
--action-force-budget
--total-force-budget
--batch-size
--max-workers
--output
--run-root
```

Use explicit analytic calculator/kernel factory functions. Reject overwrite of
existing run directories/output. Label `minimal_ucb` as
`support_complete=false`.

For paired replicate seed `s`, derive exactly one deterministic
`master_seed_by_seed[s]` and pass that same value to all policy arms. Persist
the mapping and assert it in the smoke test.

The first harness is deliberately fixed to:

```python
def make_calculator():
    return AnalyticCalculator(CoupledPairWell())


def initial_state(seed: int) -> State:
    rng = np.random.default_rng(seed)
    positions = np.array([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    positions += rng.normal(scale=0.01, size=positions.shape)
    return State(numbers=np.array([1, 1]), positions=positions)
```

This is a contract benchmark for the posterior outer loop, not a claim that a
two-basin analytic potential predicts cluster or materials performance.

Do not include policy-specific SSW configs or automatic parameter tuning.

- [ ] **Step 4: Add neutral aggregation**

Define the primary estimand as the within-seed policy contrast under identical
`(G, q, B, W, master_seed)` and initial structure. Normal `budget_tail` runs
contribute their final fixed-cap campaign state with unused budget reported.
Stratify `zero_cost_stall`, aborted, drifted, or otherwise ineligible runs
separately and never pretend they are observations at exactly `G`.

Report per-policy paired-seed distributions for final energy, unique minima,
completed/failed attempts, duplicate fraction, purpose fractions, and unused
budget. If confidence intervals are included, use a paired/block bootstrap of
seed-level contrasts, name the estimator/resampling unit, and state seed count.
Do not calculate independent per-policy intervals and do not select a winner.

- [ ] **Step 5: Update README claim boundaries and example**

Document:

- operational meaning of strategy-unbiased;
- fixed per-action fidelity;
- bootstrap-inclusive campaign budget;
- sub-action tail budget remains unused;
- exact purpose ledger;
- committed-boundary recovery and unknown mid-batch crash cost;
- analytic ThreadPool-only validation;
- no TS, MACE/GPU/process, async, detailed-balance, or performance claim.

Show one explicit API example with all `PosteriorExplorationConfig` fields.

- [ ] **Step 6: Run benchmark smoke and focused suite**

Run:

```bash
pytest -q tests/integration/test_posterior_policy_benchmark.py
POSTERIOR_SMOKE_DIR=$(mktemp -d)
python benchmarks/posterior_policy_compare.py \
  --seeds 0 1 \
  --policies uniform posterior_proportional minimal_ucb \
  --action-force-budget 40 \
  --total-force-budget 130 \
  --batch-size 2 \
  --max-workers 2 \
  --run-root "$POSTERIOR_SMOKE_DIR/runs" \
  --output "$POSTERIOR_SMOKE_DIR/results.json"
```

Inspect the JSON and verify all totals directly from committed batch files.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/posterior_policy_compare.py \
  tests/integration/test_posterior_policy_benchmark.py README.md
git commit -m "Add exact-force posterior policy benchmark"
```

---

### Task 11: Completion audit and full verification

**Files:**
- Modify only if audit finds a documented gap.

- [ ] **Step 1: Run the complete suite**

```bash
pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run high-risk focused suites repeatedly**

```bash
for i in 1 2 3 4 5; do
  pytest -q \
    tests/integration/test_posterior_exploration_runner.py \
    tests/integration/test_exploration_controller.py
done
```

Expected: all five runs pass without timing-dependent order changes.

- [ ] **Step 3: Audit raw calculator parity**

Run the analytic worker and campaign recording-calculator tests and verify:

```text
raw calls
= EvalCounter total
= sum purpose counts
= AttemptResult total
= CreditedOutcome total
= run-store batch total
= campaign reported action total
```

Bootstrap raw calls must equal bootstrap ledger counts separately.
Run this parity matrix for both SSW and LS-SSW completed,
duplicate/invalid, fragmented, worker-error, and budget-exhausted terminal
routes, and compare aggregate output/call fixtures to parent commit `6b2a289`.

- [ ] **Step 4: Audit resume equivalence**

Compare uninterrupted and resumed run-store trees with the verifier from Task
9. Require identical canonical committed batch payloads after the split point
and identical final archive/posterior/budget fingerprints.

Repeat for pre-rename failure, post-rename acknowledgement failure,
dispatch-complete/pre-append process discard, zero-cost terminal resume, and
unknown-cost abort refusal. Inspect session eligibility and orphan-temp versus
canonical/temp behavior.

- [ ] **Step 5: Audit scope**

Verify no Phase-3 diff in:

```text
pamssw/exploration/policies.py
pamssw/exploration/posterior.py
pamssw/acquisition.py
```

Inspect walker diff to confirm it contains purpose scopes/signatures only and no
direction, bias, step, optimizer, selector, or RNG formula changes.

- [ ] **Step 6: Run hygiene checks**

```bash
git diff --check 6b2a289..HEAD
git status --short
```

- [ ] **Step 7: Dispatch whole-feature spec and code-quality reviews**

Review every Phase-3 acceptance criterion, exact budget/recovery claim, test
coverage, and README statement. Fix and re-review all blocking findings.

- [ ] **Step 8: Push and create the stacked PR**

Push `feature/recoverable-budgeted-posterior-runner` and create a PR with base
`feature/ssw-attempt-adapter`. Preserve the worktree after PR creation.
