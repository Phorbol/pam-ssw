# SSW Attempt Adapter and Exact Evaluation Accounting Design

## 1. Purpose

Phase 1 established an auditable, synchronous exploration controller with
exact starter-selection propensities and dispatch-snapshot credit. It used
generic workers deliberately.

Phase 2 connects that controller to the existing fixed-cell SSW escape kernel
without changing the kernel, the starter policies, or the default `run_ssw`
entry point.

The question answered by this phase is:

> Can one selected starter be executed as one isolated, reproducible SSW
> attempt whose complete calculator cost and terminal landing are reported
> exactly under a per-action force-evaluation budget?

This phase validates execution and accounting. It does not test whether
posterior-driven starter selection improves scientific search performance.

## 2. Approved Boundary

The phase uses:

- the existing serial bias-relax SSW kernel;
- the existing direction generator and scoring behavior;
- the existing per-atom-RMS execution-step control;
- the existing true-PES quench;
- analytic calculators;
- `ThreadPoolExecutor`;
- one fresh calculator and walker per action.

The starter is quenched again inside every action. That redundant work is
included in the action cost. Skipping the quench would require extracting a
new pure single-attempt kernel and is deferred until the adapter has proved
accounting parity.

The following are out of scope:

- MACE or other shared GPU calculator execution;
- `ProcessPoolExecutor`;
- calculator serialization or model lifecycle management;
- aggregate/global budget allocation across actions;
- asynchronous racing;
- Thompson sampling;
- direction-policy learning;
- reaction-network objectives;
- performance claims against the legacy SSW search;
- a top-level `run_parallel_ssw` entry point;
- cross-process resume.

## 3. Alternatives Considered

### 3.1 One-Trial Adapter Around `SurfaceWalker.run` — Selected

Each action creates a fresh walker with:

```text
max_trials = 1
rng_seed = action.random_seed
max_force_evals = action.force_budget
```

This preserves the currently tested physical kernel and limits this phase to
execution isolation, outcome extraction, and accounting.

The cost is a repeated starter quench and the construction of a one-entry
internal archive. Both costs are explicit and counted.

### 3.2 Extract a Pure Single-Attempt Walker Kernel — Deferred

This could skip the repeated quench and remove the one-entry internal archive.
It would be more efficient, but it would require separating a large section of
`SurfaceWalker.run`, changing exception and telemetry boundaries, and proving
physical parity. That is too much uncertainty for the accounting validation
phase.

### 3.3 Use a Full Multi-Trial SSW Run as One Action — Rejected

A multi-trial run changes starters internally and produces several transitions
under one logged starter action. The posterior could no longer attribute the
outcome to the selected starter. It is incompatible with the Phase-1 action
contract.

## 4. Existing Accounting Defect

`SurfaceWalker` currently wraps the supplied calculator in `EvalCounter`, but
constructs `SoftModeOracle` with the original calculator:

```python
self.calculator = EvalCounter(calculator, ...)
self.oracle = SoftModeOracle(calculator, ...)
```

Most directional curvature evaluations use a `ProposalPotential` that already
contains the counter, but direct oracle probes can call the raw calculator.
This violates the single accounting boundary and makes the risk dependent on
which optional diagnostic path is active.

The corrected invariant is:

> Every calculator evaluation reachable from a walker is made through the
> same action-local `EvalCounter`.

`SoftModeOracle` therefore receives `self.calculator`, never the original
calculator.

## 5. Evaluation Counting Semantics

`EvalCounter` remains the only budget authority for the current kernel.

For both `evaluate` and `evaluate_flat`:

1. reject the call if no budget remains;
2. increment force and energy evaluation counts;
3. invoke the underlying calculator;
4. return the result or propagate the calculator exception.

Incrementing before delegation records a calculator call that was actually
started even when the calculator raises. A call rejected by the budget is not
counted because the calculator was never invoked.

The current calculator protocol always returns energy and gradient together,
so:

```text
energy_evaluations == force_evaluations
```

Phase 2 records the exact total only. It does not add purpose weights or split
the total into HVP, proposal, and quench categories. Such labels would require
threading purpose context through the walker and are unnecessary for a
fixed-kernel starter-policy comparison.

## 6. Adapter Interface

Add an experimental adapter inside `pamssw.exploration`:

```python
class SSWAttemptWorker:
    def __init__(
        self,
        calculator_factory: Callable[[], CalculatorLike],
        config: SSWConfig,
        *,
        softening_enabled: bool = False,
    ) -> None:
        ...

    def __call__(
        self,
        action: StarterAction,
        starter_state: State,
    ) -> AttemptResult:
        ...
```

The adapter is callable so it can be passed directly to
`ExplorationController.run_batch`.

The adapter is exported from `pamssw.exploration`, not from the package root.
The Phase-1 root API remains unchanged.

## 7. Configuration Rules

The action owns the execution identity and budget. For each call, the adapter
creates:

```python
attempt_config = replace(
    base_config,
    max_trials=1,
    rng_seed=action.random_seed,
    max_force_evals=action.force_budget,
)
```

The base values of `max_trials`, `rng_seed`, and `max_force_evals` are ignored
by design.

Phase 2 requires:

```text
proposal_pool_size == 1
proposal_duplicate_rescue_optimizer is None
```

This ensures one starter action contains one SSW walk rather than an internal
winner-take-all proposal competition or an unlogged rescue proposal.

The adapter rejects configurations that enable shared filesystem output:

- `accepted_structures_log`;
- `accepted_structures_dir`;
- proposal-minimum writing;
- relaxation-trajectory writing;
- direction-diagnostic writing;
- direction-archive file writing.

It does not silently rewrite those fields. The caller must provide a
side-effect-free base configuration for threaded execution.

Passive in-memory diagnostics may remain enabled if they do not change the
escape kernel or write shared state.

## 8. Isolation and Parallel Semantics

For every action:

1. call `calculator_factory()` exactly once;
2. construct one new `SurfaceWalker`;
3. let the walker create one new `EvalCounter`;
4. seed its RNG from `action.random_seed`;
5. execute only the supplied starter;
6. return an independent `AttemptResult`.

No calculator, walker, RNG, counter, archive, or mutable output object is
shared between actions.

The first validated executor is `ThreadPoolExecutor`. Calculator instances
returned by the factory must be independent. A factory that returns the same
object is rejected when this can be detected in the integration harness; the
general runtime contract remains the factory's responsibility.

Worker completion order remains irrelevant because the existing controller
commits in slot order and credits against the dispatch archive snapshot.

## 9. Landing Extraction

`SearchResult.best_state` is not the action landing. A valid escape may land in
a basin with energy above the starter, in which case the starter can remain
the global best.

For a one-trial run:

1. if `walk_history` contains its one transition, resolve
   `discovered_entry_id` in `result.archive`;
2. return that entry's state and energy as `COMPLETED`;
3. ignore `best_state` for action credit.

Landing extraction has precedence over the final `budget_exhausted` flag. The
counter may become exactly exhausted on the last successful evaluation after a
valid landing was already produced.

If no landing exists:

- return `BUDGET_EXHAUSTED` when the budget stopped the attempt;
- return `FRAGMENTED` only when the single proposal has an explicit fragment
  rejection;
- otherwise return `INVALID` with `no_landing_minimum`.

The adapter does not infer chemical failure modes from energies or descriptor
thresholds.

## 10. Exception Mapping

Expected terminal failures are returned with the counter value owned by the
walker:

- `BudgetExceeded` before a landing while the counter is exhausted:
  `BUDGET_EXHAUSTED`;
- invalid starter geometry, or `BudgetExceeded` raised by the current geometry
  guard while counter capacity remains: `INVALID`;
- underlying calculator `Exception`: `WORKER_ERROR`;
- explicit single-proposal fragmentation: `FRAGMENTED`.

This classification uses counter state and explicit walker telemetry. It never
parses exception-message text.

The adapter catches ordinary `Exception`, not `BaseException`.

The generic controller's zero-cost exception fallback remains a final safety
net for adapter defects. Correct adapter failure paths must return their own
`AttemptResult` so nonzero spent evaluations are preserved.

## 11. Validation Strategy

### 11.1 Counter Unit Tests

Use an instrumented analytic calculator to prove:

- the underlying call count equals `EvalCounter.force_evaluations`;
- `evaluate` and `evaluate_flat` share one budget;
- the call that raises inside the calculator is counted;
- a call rejected before delegation is not counted;
- the counter never exceeds its budget.

### 11.2 Oracle Accounting Test

Enable a direct oracle probe path and prove the wrapped counter equals the
instrumented calculator's total calls. This is the regression for the raw
calculator seam.

### 11.3 Adapter Mapping Tests

Use deterministic `SearchResult` fixtures to prove:

- a higher-energy discovered basin is returned instead of `best_state`;
- a valid landing wins over an exactly exhausted final counter;
- no landing plus budget exhaustion maps to `BUDGET_EXHAUSTED`;
- explicit single-proposal fragmentation maps to `FRAGMENTED`;
- other no-landing results map to `INVALID`;
- calculator failures preserve nonzero spent cost.

### 11.4 Threaded Integration

Use analytic calculators and `ThreadPoolExecutor` to prove:

- the factory creates one distinct calculator per action;
- repeated identical actions reproduce landing, status, and evaluation count;
- completion order does not change committed outcomes;
- every outcome count equals its underlying calculator calls;
- no action exceeds its force budget;
- the central archive and posterior update once per action;
- the default `run_ssw` tests remain green.

## 12. Acceptance Criteria

Phase 2 is complete when:

- every walker calculator call passes through the action-local counter;
- underlying analytic calculator calls equal reported action counts;
- no reported count exceeds the action budget;
- a valid higher-energy landing is credited as the landing;
- threaded action results are independent of completion order;
- each action owns a distinct calculator, walker, RNG, and counter;
- expected worker failures preserve spent evaluation counts;
- Phase-1 controller and event-log replay tests remain green;
- the no-budget default `run_ssw` physical path remains available and its
  existing suite passes;
- budgeted `run_ssw` includes oracle calls that previously bypassed the limit;
- no MACE, process-parallel, performance, or thermodynamic claim is made.

## 13. Deferred Work

Only after Phase 2 passes should the project consider:

1. extracting a pure single-attempt kernel that skips starter re-quench;
2. per-purpose evaluation telemetry;
3. aggregate budget allocation;
4. process/GPU calculator factories;
5. analytic multi-seed policy benchmarks;
6. MACE runtime validation;
7. asynchronous racing.

Each item requires a separate design and ablation boundary.
