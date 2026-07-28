# Posterior Terminal-Outcome Validation Design

## Status

This specification is an evidence-closing milestone for the existing
posterior exploration stack. It does not introduce a new exploration
algorithm.

The following existing design decisions remain authoritative:

- one externally dispatched action contains exactly one SSW proposal
  (`proposal_pool_size=1`);
- the raw initial `State` is bootstrap-quenched before starter selection;
- all calculator calls are attributed to an evaluation purpose and charged to
  the fixed campaign budget;
- terminal action outcomes are appended to the schema-v2 exploration event
  log and may update the starter-productivity posterior;
- parallel execution uses `ThreadPoolExecutor`.

## Goal

Produce a small, reproducible analytic validation showing that the current
single-action posterior exploration path has a closed action, terminal
outcome, evaluation-cost, and posterior-replay record.

This milestone answers only:

> For the existing one-action protocol, does every dispatched action obtain
> exactly one committed terminal record with internally consistent cost and
> posterior-observation semantics?

It does not compare starter policies or claim that posterior-driven selection
outperforms uniform selection.

## Scientific and algorithmic boundary

The validation must not change:

- `SurfaceWalker`;
- starter selection probabilities or posterior update equations;
- the fixed-prior UCB-like policy;
- direction generation, ranking, or oracle parameters;
- proposal relaxation or true-quench optimizers;
- SSW bias equations;
- archive matching;
- force-budget allocation.

In particular, this phase does not add Thompson sampling, contextual features,
new rewards, adaptive weights, multi-fidelity racing, reaction-network
semantics, or batch proposal competition.

Keeping `proposal_pool_size=1` is a causal-identification requirement: the
externally selected starter action must receive the outcome of one physical
proposal, without hidden winner selection inside the worker.

## Existing implementation under validation

The validation exercises the existing path:

```text
raw State
  -> bootstrap true-PES quench
  -> MinimaArchive entry 0
  -> posterior policy snapshot
  -> one StarterAction per slot
  -> ThreadPoolExecutor
  -> SSWAttemptWorker
  -> one AttemptResult per action
  -> ExplorationController credit assignment
  -> ExplorationEventLog batch commit
  -> posterior reconstruction
```

`AttemptStatus` already defines:

- `completed`;
- `invalid`;
- `fragmented`;
- `budget_exhausted`;
- `worker_error`.

The existing posterior-observation rule is retained:

- a positive-cost exact `completed`, `invalid`, `fragmented`, or
  `budget_exhausted` terminal result is an observation;
- a zero-cost result is not an observation;
- `worker_error` is not an observation.

This rule treats a physically attempted but unsuccessful escape as evidence
about starter productivity while excluding failures that never produced a
physical PES evaluation.

## Deliverables

Add one finite validation package:

```text
runs/20260728-posterior-terminal-outcome-validation/
  run_validation.py
  evidence.json
  conclusion.md
```

`run_validation.py` must regenerate the evidence from repository code. It may
write raw temporary event logs beneath a user-supplied or automatically
created temporary directory, but the repository retains only the compact
reviewed evidence and conclusion.

At most one narrowly scoped automated test may be added if the validator
contains non-trivial invariant arithmetic. Existing terminal-status,
ThreadPool, exact-accounting, event-schema, and replay tests must not be
duplicated.

## Validation layer A: real analytic ThreadPool campaign

Run the public `run_posterior_ssw` entry point with:

- the existing analytic `DoubleWell2D` calculator;
- a raw, reproducible multi-atom `State`;
- `proposal_pool_size=1`;
- `batch_size > max_workers > 1`, so queueing and concurrent execution both
  occur;
- a fixed master seed;
- a fixed per-action force budget;
- a fixed total force budget;
- the existing uniform starter policy.

Use the same physical SSW runner path as later policy ablations. Do not mock
the walker, calculator, executor, budget manager, controller, or event log in
this layer.

The validator must independently parse the generated `events.jsonl` and
verify:

1. each committed batch contains exactly one policy snapshot, its planned
   attempt records, and one batch commit;
2. every action identifier is unique;
3. the action identifiers in each commit exactly match the preceding attempt
   records in slot order;
4. each attempt's purpose-count sum equals its recorded
   `force_evaluations`;
5. every attempt has an exact cost and no unattributed calculator calls;
6. every policy snapshot has complete support, normalized finite
   probabilities, and each attempt stores the selected starter's matching
   probability;
7. bootstrap evaluations plus action evaluations equal total evaluations;
8. total evaluations plus unused budget equal the configured campaign
   budget;
9. summed event-log action counts equal `action_evaluations`; subtracting
   those action-purpose totals from the campaign-purpose totals leaves only
   non-negative bootstrap counts whose sum equals `bootstrap_evaluations`;
10. posterior reconstruction from the committed log exactly matches the live
    posterior counts for every archived starter;
11. the number of posterior observations equals the number reconstructed
    from committed attempts.

The evidence must record the effective configuration, seed, terminal-status
counts, action costs, purpose totals, batch widths, posterior counts, stop
reason, and every invariant result.

This layer proves the real runner path and its accounting closure. It does not
guarantee that every possible failure type naturally occurs on
`DoubleWell2D`.

## Validation layer B: deterministic terminal-semantics matrix

Exercise the real `ExplorationController`, `ThreadPoolExecutor`, and
`ExplorationEventLog` with a minimal deterministic worker that returns a
fixed matrix of terminal results. This layer does not claim to validate SSW
physics; it validates controller credit and durable replay semantics for
terminal classes that cannot be reliably induced by one analytic PES.

The matrix contains:

| Terminal case | Recorded physical cost | Expected posterior observation |
|---|---:|---:|
| completed | positive | yes |
| invalid after PES work | positive | yes |
| fragmented after PES work | positive | yes |
| budget exhausted | positive | yes |
| invalid before PES work | zero | no |
| worker error | zero | no |

All six deterministic worker results declare their recorded costs exact. The
separate existing executor-failure tests cover an unmeasured
`cost_is_exact=False` transport failure; strict production campaigns reject
such an outcome rather than treating it as usable posterior evidence.

For every row, verify:

- exactly one committed attempt exists for the dispatched action;
- status, failure reason, exactness flag, force count, and purpose counts are
  preserved;
- recorded `posterior_observed` matches the existing rule;
- replayed starter-productivity counts exactly match the live controller;
- zero-cost/non-physical failures do not alter the posterior.

Existing `SSWAttemptWorker` tests remain the evidence that real walker results
and exceptions map to these terminal classes. This matrix must not duplicate
that adapter logic or manufacture new production abstractions.

## Evidence schema

`evidence.json` is a deterministic JSON object with:

```text
schema_version
git_commit
validation_scope
real_campaign
terminal_matrix
invariants
claim_ceiling
```

The validator fails closed and does not write a passing conclusion if:

- an event is missing, duplicated, uncommitted, or out of order;
- the event schema or expected provenance fields drift;
- an action has unknown or inconsistent cost;
- any evaluation count fails to close;
- policy support or saved selection probabilities are inconsistent;
- replay differs from live posterior state;
- any required invariant is false.

`conclusion.md` must state the exact commit and distinguish:

- verified execution behavior;
- verified accounting and replay behavior;
- untested physical and statistical questions.

## Claim ceiling

Passing this validation supports only the following claim:

> On the tested analytic fixed-budget ThreadPool campaign, the existing
> one-action posterior exploration stack persisted one terminal outcome per
> dispatched action, closed evaluation accounting, and reproduced the live
> starter-productivity posterior from committed events.

It does not establish:

- unbiased sampling of a thermodynamic or stationary PES distribution;
- superiority of UCB-like, posterior-proportional, or Thompson policies;
- improved minimum discovery;
- GPU/MLIP calculator correctness or performance;
- C60 or PdO production behavior;
- optimal direction or optimizer parameters.

Here “unbiased exploration” means that no candidate is silently discarded
between external action dispatch and terminal credit assignment. It does not
mean that starter selection is uniform or that the resulting trajectory is a
statistically unbiased equilibrium sample.

## Next milestone

Only after this validation passes should the same runner be used for a
paired-seed, fixed-force-budget comparison of:

1. uniform starter selection;
2. the current fixed-prior UCB-like policy;
3. optionally posterior-proportional selection as a diagnostic baseline.

That ablation must freeze the direction oracle, uphill policy, relaxation
settings, initial states, action budget, total budget, concurrency, and random
seeds. Thompson sampling remains out of scope until the simpler policies show
that action-conditioned posterior information contributes measurable value.

## Completion criteria

This milestone is complete when:

- the finite validation script runs from a clean checkout;
- both validation layers pass;
- tracked evidence is reproducible and names the tested commit;
- the full test suite remains green;
- no production exploration algorithm or numerical parameter changed;
- independent specification and code-quality reviews find no unsupported
  performance claim or duplicated mechanism.
