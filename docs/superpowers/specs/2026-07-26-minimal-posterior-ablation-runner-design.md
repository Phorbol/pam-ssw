# Minimal Posterior Ablation Runner Design

## Status and supersession

This design replaces Tasks 6–11 of
`docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md`
and the corresponding durable-recovery requirements in
`docs/superpowers/specs/2026-07-25-recoverable-budgeted-posterior-runner-design.md`.

Phase-3 Tasks 1–5 remain authoritative:

- exact evaluation-purpose accounting;
- exact terminal action costs;
- unchanged physical SSW path with explicit purpose attribution;
- complete in-process batch commit facts;
- fixed-fidelity campaign budgeting.

The replacement is intentionally smaller. Its purpose is to make subsequent
algorithm-component ablations easy to run and interpret, not to build a
general workflow or recovery platform.

## Goal

Provide one opt-in analytic `ThreadPoolExecutor` runner that composes the
existing posterior exploration components into a complete fixed-budget SSW or
LS-SSW campaign, plus one paired-seed three-policy ablation harness.

The completed branch must let later work replace and compare starter policies,
direction mechanisms, or uphill kernels without changing campaign accounting,
parallel dispatch semantics, or result measurement.

## Non-goals

This phase does not add:

- run manifests, sessions, repository fingerprints, or environment capture;
- atomic run stores, batch-file journals, process-crash recovery, or resume;
- asynchronous scheduling or multi-fidelity racing;
- Thompson sampling, contextual bandits, learned policies, or reward weights;
- new direction sources, uphill propagators, or SSW numerical parameters;
- transition-network, kinetic, or reaction-network semantics;
- performance or superiority claims.

The existing compact `ExplorationEventLog` remains the only durable action
record. It is not a recovery source because it does not contain landing
geometries.

## Public API

Add two thin entry points in `pamssw/exploration/runner.py`:

```python
def run_posterior_ssw(
    initial_state: State,
    calculator_factory: Callable[[], Calculator],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    ...


def run_posterior_ls_ssw(
    initial_state: State,
    calculator_factory: Callable[[], Calculator],
    ssw_config: LSSSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    ...
```

Both functions delegate to one private campaign implementation. The only
difference is whether local softening is enabled and which SSW configuration
type is accepted.

`PosteriorExplorationConfig` is reduced to fields used by this runner:

```text
policy_name
batch_size
max_workers
action_force_budget
total_force_budget
master_seed
run_directory
```

Remove `mode`, `calculator_label`, and `calculator_fingerprint`. They existed
for the cancelled resume/manifest design and would otherwise be unused core
configuration. The standalone benchmark output records its analytic
calculator identity directly.

## Bootstrap semantics

The runner accepts a raw `State` and performs exactly one bootstrap true-PES
relaxation before constructing the exploration archive.

Bootstrap uses:

- `GeometryValidator.is_valid_state` to reject an invalid raw state before
  calculator creation;
- a fresh calculator from `calculator_factory`;
- one `EvalCounter` capped by `total_force_budget`;
- `Relaxer` with `ssw_config.quench_optimizer`,
  `ssw_config.quench_fmax`, and `ssw_config.quench_maxiter`;
- `EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH` for the relaxation;
- `EvaluationPurpose.POST_RELAX_VALIDATION` for the final finite
  energy/gradient and geometry check.

All started calculator calls count toward the campaign budget. A budget
exception, calculator failure, invalid geometry, or non-finite result fails
before a successful campaign result is created.

After successful bootstrap, the runner creates a `MinimaArchive` using the
SSW configuration's energy/RMSD tolerances and inserts the relaxed minimum as
entry zero.

## Campaign execution

The runner creates:

- `CampaignBudget(total_force_budget, action_force_budget)`;
- `ExplorationEventLog(run_directory / "events.jsonl")`;
- `ExplorationController(..., require_exact_cost=True)`;
- one reusable `SSWAttemptWorker` whose calculator factory creates a fresh
  calculator for every action;
- one `ThreadPoolExecutor(max_workers=max_workers)`.

The run directory must not already exist. It is created only after bootstrap
succeeds. There is no resume mode.

After recording bootstrap counts, each iteration computes:

```text
width = min(batch_size, remaining // action_force_budget)
```

The controller plans and dispatches exactly `width` actions, each with the
same fixed action force cap. When the complete batch has been durably appended
to the compact event log, the campaign ledger commits the outcomes' exact
purpose-count snapshots.

The loop ends only with:

- `budget_tail` when less than one full action cap remains; or
- `zero_cost_stall` when a complete committed batch used zero physical calls.

No residual-fidelity action is dispatched.

## Result semantics

Return `PosteriorExplorationResult` with:

- the final archive and fixed-prior posterior;
- completed batch and attempt counts;
- completed versus failed terminal attempts;
- number of posterior-observed attempts;
- exact bootstrap, action, total, and per-purpose evaluation counts;
- total and unused force budget;
- terminal reason;
- benchmark eligibility and explicit reasons;
- run directory.

Strict mode rejects an action whose physical cost is unknown. A committed
outcome that is intentionally not posterior-observed, any unattributed
calculator call, or a zero-cost stall makes the campaign benchmark-ineligible.

The result is an optimization/exploration summary. It is not a thermodynamic
sample, kinetic model, or unbiased estimator of a stationary PES distribution.

## Failure and durability boundary

- A bootstrap failure produces no successful run directory.
- A malformed or unknown-cost batch fails closed before controller state is
  installed.
- An event-log append failure leaves the batch pending in the live controller
  for in-process retry, as already implemented.
- Process termination may lose the in-memory archive and landing geometries.
  Cross-process recovery is explicitly unsupported.
- `KeyboardInterrupt`, `SystemExit`, and other `BaseException` values remain
  uncaught.

## Minimal ablation harness

Add `benchmarks/posterior_policy_compare.py`.

For the same analytic potential, raw initial state, paired master seeds, fixed
action cap, total force budget, batch size, and worker count, run:

```text
uniform
posterior_proportional
minimal_ucb
```

The harness writes one deterministic JSON document containing:

- analytic calculator label and fixed potential parameters;
- complete execution-budget settings;
- seed and policy for every run;
- best energy and number of unique minima;
- completed/failed/posterior-observed attempts;
- exact bootstrap/action/total/purpose counts;
- unused budget, stop reason, and benchmark eligibility.

It reports raw paired observations only. It does not add significance tests,
policy rankings, plots, or performance claims.

## Verification

Required tests:

1. bootstrap calls are included exactly once and carry only bootstrap or
   post-validation purposes;
2. invalid/bootstrap-budget-exhausted runs fail before creating a successful
   run directory;
3. every dispatched analytic action has exact cost and zero unattributed
   calls;
4. campaign total never exceeds the fixed budget;
5. the final batch width follows the reservation equation and no residual
   action is dispatched;
6. `budget_tail` and `zero_cost_stall` produce correct summaries;
7. `max_workers < batch_size` executes through a real
   `ThreadPoolExecutor`;
8. identical seeds/configurations are reproducible;
9. all three policies run through the same campaign path;
10. the harness records paired raw facts without declaring a winner;
11. legacy `run_ssw` and `run_ls_ssw` behavior and defaults remain unchanged;
12. the full unit/integration suite passes.

## Completion boundary

This reduced Phase 3 is complete when:

- the two public runner functions work on the analytic backend;
- the exact fixed-budget invariants above pass;
- the three-policy harness runs and emits reviewed JSON;
- documentation states the statistical and runtime claim ceilings;
- the complete branch passes independent specification and code-quality
  review.

No recovery, resume, asynchronous scheduling, new policy mathematics, or
physical-kernel modification is required for completion.
