# Recoverable Budgeted Posterior Exploration Runner Design

**Date:** 2026-07-25
**Status:** Approved
**Repository baseline:** `6b2a289` on `feature/ssw-attempt-adapter`

## 1. Purpose

Phase 1 established auditable starter-selection policies and deterministic
synchronous batch credit. Phase 2 connected one dispatched action to the
existing fixed-cell SSW kernel with an isolated calculator, walker, RNG, and
exact total calculator-call accounting.

The branch is not yet a complete experimental platform:

- there is no opt-in runner that executes more than one posterior batch;
- a scalar per-action cap is copied to every action, but there is no campaign
  force-evaluation cap;
- total action cost cannot be decomposed by physical algorithm phase;
- the JSONL event log cannot reconstruct archive geometries or controller
  versions after process exit;
- the existing LJ benchmark uses an approximate local-relaxation budget rather
  than a common force-evaluation budget.

Phase 3 answers:

> Can the fixed SSW kernel be run through a recoverable, synchronous posterior
> exploration campaign whose action fidelity, total calculator cost, purpose
> breakdown, policy probabilities, archive updates, and committed state can be
> reconstructed exactly?

This phase makes the outer experiment surface usable for later direction,
uphill-policy, and starter-policy ablations. It does not add or tune any new
search algorithm.

## 2. Scientific Boundary

Phase 3 keeps fixed:

- the serial bias-relax SSW escape kernel;
- per-atom-RMS execution-step control;
- the existing random, momentum, and bond direction pool;
- true-PES landing quench;
- current structure identity and archive matching;
- the Phase-1 Beta(1,1) Bernoulli productivity posterior;
- `uniform`, `posterior_proportional`, and `minimal_ucb`;
- dispatch-snapshot discovery credit;
- with-replacement batch sampling;
- deterministic slot-order commit;
- one fresh calculator and walker per action;
- action-internal starter re-quench.

The starter re-quench remains redundant but fully counted. Extracting a pure
prepared-starter kernel is deferred because the current one-entry local archive
affects novelty scoring, duplicate semantics, and RNG consumption. Removing it
would change the physical trajectory and requires an independent parity design.

Phase 3 uses analytic calculators and `ThreadPoolExecutor` only.

The following remain out of scope:

- Thompson sampling or posterior sampling;
- posterior temperature, epsilon, forgetting, quotas, or learned priors;
- cost-weighted, energy-weighted, novelty-weighted, or damage-weighted reward;
- using evaluation telemetry to select starters or allocate action budgets;
- direction-policy learning or new direction sources;
- trust-region, step-length, bias, or local-softening parameter changes;
- reaction-network or transition-state objectives;
- asynchronous racing, cancellation, or completion-time feedback;
- shared atomic global counters across workers;
- `ProcessPoolExecutor`;
- MACE, GPU affinity, model sharing, or device lifecycle validation;
- canonical, kinetic, or detailed-balance sampling claims;
- claims that one starter policy improves scientific search performance.

## 3. Definitions

### 3.1 Physical evaluation

One physical evaluation is one started call to the underlying calculator's
`evaluate` or `evaluate_flat` method. Energy and gradient returned together are
one physical evaluation, not two costs.

A call rejected by `EvalCounter` before delegation is not started and is not
counted. A delegated call that raises is started and is counted.

### 3.2 Action fidelity

Action fidelity is the positive integer calculator-call cap assigned to every
normal dispatched SSW attempt. It is fixed for the entire campaign.

Phase 3 never gives a starter a larger or smaller cap based on posterior mean,
energy, novelty, previous cost, failure history, execution time, batch slot, or
worker availability.

### 3.3 Campaign budget

The campaign budget is a positive integer upper bound on:

```text
bootstrap physical evaluations
+ physical evaluations in committed action results
```

It is not local-relaxation count, wall time, optimizer iterations, HVP count, or
energy-plus-force count.

### 3.4 Committed recovery

Recovery reconstructs the state after the last atomically committed batch.
Work performed by an interrupted process before that batch was atomically
committed is not reconstructable and is not silently reported as known cost.

Phase 3 therefore guarantees exact committed cost and deterministic committed
state. It does not claim an exact lifetime hardware-call cap across arbitrary
mid-batch process crashes.

An exception that reaches the runner while the cost of a started action is
unknown is different from an abrupt process death: the runner can observe the
former. It atomically writes a terminal `aborted_unknown_cost` marker and
refuses ordinary resume. The operator must start a new campaign to make a new
budget claim. An abrupt death before any such marker or batch commit remains
outside the reconstructable ledger and may cause deterministic re-execution on
an explicitly requested resume.

## 4. Evaluation-Purpose Ledger

### 4.1 Closed purpose taxonomy

Add a closed enum:

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
```

The categories describe mutually exclusive calculator-call phases. They do not
encode a direction kind, policy, starter, reward, outcome, or cost weight.

`UNATTRIBUTED` exists for backward-compatible generic workers and migration.
A run is benchmark-eligible only when its unattributed count is zero.

### 4.2 Single counting authority

`EvalCounter` remains the only object that increments physical-call counts. It
gains a scoped purpose context and an immutable snapshot:

```python
with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
    ...

counts = counter.snapshot()
```

The counter increments the aggregate total and exactly one active purpose
before delegating to the calculator. Nested scopes use the innermost scope.
Leaving a scope restores the previous scope even after an exception.

The invariant is:

```text
sum(counts[p] for every purpose p) == force_evaluations
```

No second calculator wrapper, estimated HVP multiplier, optimizer-iteration
conversion, or post-hoc inference is allowed.

### 4.3 Walker attribution

The existing path is instrumented without changing its numerical operations:

- central initial relaxation: `BOOTSTRAP_TRUE_QUENCH`;
- action-local initial relaxation: `STARTER_TRUE_QUENCH`;
- candidate generation, directional curvature, HVPs, and optional probes:
  `DIRECTION_ORACLE`;
- true curvature and before/after true-PES checks during uphill propagation:
  `ESCAPE_TRUE_PES_CHECK`;
- calculator calls made by biased relaxation:
  `BIASED_PROPOSAL_RELAX`;
- candidate true-PES quench: `LANDING_TRUE_QUENCH`;
- post-quench finite evaluation: `POST_RELAX_VALIDATION`.

Purpose scopes must cover both `evaluate` and `evaluate_flat` calls. A rejected
budget call increments neither the total nor a purpose.

### 4.4 Result contracts

Add an immutable `EvaluationCounts` value to `AttemptResult` and
`CreditedOutcome`.

For backward-compatible generic workers, omitted purpose counts are converted
to an `UNATTRIBUTED` count equal to the scalar total. `SSWAttemptWorker` must
always supply an exact snapshot.

The records also distinguish exact from unknown cost:

```python
cost_is_exact: bool
```

- every result returned by `SSWAttemptWorker` has exact cost, including a
  zero-cost pre-calculator validation/factory failure;
- a future exception escaping the worker boundary has unknown cost;
- the generic controller may continue to record unknown-cost failures in its
  non-strict mode;
- the Phase-3 runner uses strict-cost mode and aborts before committing a batch
  containing unknown cost.

The event-log schema is bumped explicitly. Older rows are rejected by the new
strict schema rather than interpreted as purpose-complete data.

`CreditedOutcome` also persists:

```python
posterior_observed: bool
```

The productivity posterior observes only terminal facts that are evidence
about the fixed physical attempt: completed landings, budget exhaustion,
fragmentation after physical work, and physical no-landing results. It does not
observe factory/constructor failures, escaped adapter exceptions, calculator
infrastructure errors, or zero-cost invalid starters. These unobserved outcomes
remain committed audit facts, but make a policy benchmark run ineligible. This
keeps the posterior estimand equal to fixed-kernel basin discovery conditional
on an executable action rather than silently multiplying it by infrastructure
availability.

The controller also promotes its private pending payload into one explicit
commit contract:

```python
@dataclass(frozen=True)
class CommittedExplorationBatch:
    snapshot: PolicySnapshot
    actions: tuple[StarterAction, ...]
    results: tuple[AttemptResult, ...]
    outcomes: tuple[CreditedOutcome, ...]
```

Actions, results, and outcomes are aligned in slot order. `BatchLog` accepts
this complete object. The compact event log serializes the policy and scalar
outcome facts, while the recoverable run store additionally serializes the
landing states held by `AttemptResult`.

For each slot, `CommittedExplorationBatch` requires exact equality between the
result and outcome `force_evaluations`, complete `evaluation_counts`, and
`cost_is_exact`, not merely equality of their totals.

`CreditedOutcome` remains the scalar archive-credit record; state geometry is
not duplicated into it. This keeps credit comparison independent of NumPy array
equality and gives recovery one canonical source for the actual worker landing.

## 5. Campaign Budget Semantics

### 5.1 Bootstrap

The runner true-quenches the input state once to build the central one-entry
archive. Bootstrap uses a fresh calculator and the same exact counter.

Bootstrap calls:

- are included in the campaign budget;
- are recorded in the immutable run manifest after a successful bootstrap;
- are not posterior attempts;
- do not have a starter propensity;
- fail the run without creating a resumable campaign if a valid initial
  minimum cannot be established.

### 5.2 Pre-dispatch reservation

Let:

- `G` be the campaign cap;
- `q` be the fixed action cap;
- `B` be the configured maximum batch width;
- `s` be committed spend including bootstrap;
- `r = G - s`.

Before policy sampling for the next batch:

```text
k = min(B, floor(r / q))
```

If `k == 0`, the campaign stops and reports the unused budget `r`.

Otherwise the controller plans exactly `k` actions, each with cap `q`. The
reservation `k*q` therefore fits before any action is sampled.

The final batch may be narrower than `B`, but its width depends only on the
remaining campaign budget and fixed action cap. It does not depend on which
starter is sampled.

### 5.3 No residual-fidelity action

Phase 3 does not dispatch one final action with cap `r < q`. Mixing action
fidelities would make the existing Bernoulli posterior combine success
probabilities under different computational opportunities.

Unused budget below `q` is reported, not hidden or reassigned.

### 5.4 Actual spend

After a committed batch:

```text
s_next = s + sum(outcome.force_evaluations)
```

Because every action is bounded by `q` and the batch was reserved before
dispatch:

```text
s_next <= G
```

Unused reservation from actions that terminate early becomes available to a
later batch.

A fully committed batch with zero total physical calls terminates the campaign
with `zero_cost_stall`. This prevents an infinite campaign of invalid starters
or calculator-factory failures without inventing a fake cost.

`budget_tail` is reconstructed when the remaining budget is less than the
fixed action cap. `zero_cost_stall` is reconstructed from a final committed
zero-spend batch. Both are durable terminal facts; resume from either terminal
state is a no-op.

## 6. Opt-In Runner

### 6.1 Configuration

Add a separate frozen configuration:

```python
@dataclass(frozen=True)
class PosteriorExplorationConfig:
    policy_name: str
    batch_size: int
    max_workers: int
    action_force_budget: int
    total_force_budget: int
    master_seed: int
    calculator_label: str
    calculator_fingerprint: str
    run_directory: Path
    mode: Literal["new", "resume"] = "new"
```

This configuration is separate from `SSWConfig`. It has no algorithm weights
or hidden fallback values.

Validation requires:

- policy is one of the three Phase-1 policies;
- sizes, budgets, and seed are valid integers;
- calculator label and semantic fingerprint are nonempty provenance strings;
- `max_workers <= batch_size`;
- `total_force_budget >= 1`;
- new mode requires an absent or empty run directory;
- resume mode requires a valid existing manifest and contiguous committed
  batches.

The runner does not silently delete, overwrite, repair, or start over in a
non-empty incompatible directory.

### 6.2 Public API

Add explicit opt-in functions:

```python
run_posterior_ssw(
    initial_state,
    calculator_factory,
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult

run_posterior_ls_ssw(
    initial_state,
    calculator_factory,
    ssw_config: LSSSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult
```

The two functions avoid a user-facing softening boolean. Existing
`run_ssw`/`run_ls_ssw` signatures and defaults remain unchanged.

The runner owns:

- bootstrap preparation;
- central archive;
- `ExplorationController`;
- `SSWAttemptWorker`;
- `ThreadPoolExecutor`;
- campaign budget loop;
- run-store creation or recovery;
- final result assembly.

It does not accept a generic worker. Strict exact-cost behavior is tied to the
known SSW adapter.

### 6.3 Result

Return a frozen summary containing at least:

```python
PosteriorExplorationResult(
    archive,
    posterior,
    policy_name,
    completed_batches,
    completed_attempts,
    bootstrap_evaluations,
    action_evaluations,
    total_evaluations,
    purpose_counts,
    failed_attempts,
    posterior_observed_attempts,
    benchmark_eligible,
    benchmark_ineligibility_reasons,
    total_force_budget,
    unused_force_budget,
    stop_reason,
    run_directory,
)
```

The result is a campaign summary, not a `SearchResult`. It contains no
transition-state, kinetic, or reaction-network interpretation.

## 7. Recoverable Run Store

### 7.1 Canonical source of truth

The Phase-3 runner uses a new `ExplorationRunStore`. The existing compact
`ExplorationEventLog` remains available for Phase-1 generic tests, but is not
the source of truth for full recovery because it omits landing geometries.

Run layout:

```text
run_directory/
  manifest.json
  sessions/
    00000000.json
    ...
  batches/
    00000000.json
    00000001.json
    ...
  abort.json                 # present only after an observed unknown-cost abort
```

The manifest and each batch are strict, versioned JSON objects. A batch file
contains:

- run identifier and schema version;
- policy snapshot and full probability vector;
- actions and exact propensities;
- credited terminal outcomes;
- completed landing states;
- evaluation-purpose counts;
- exact-cost flags;
- budget before, reserved, actually spent, and after;
- controller versions before and after;
- action order and batch identifier.

The manifest additionally stores the caller configuration, the effective
side-effect-free bootstrap configuration, and the effective per-action
configuration transformation. Session records durably capture allowed
environment/hardware drift and the resulting benchmark-eligibility decision.

### 7.2 State serialization

State serialization preserves:

- atomic numbers;
- positions;
- cell;
- PBC;
- fixed mask;
- JSON-compatible metadata.

Arrays are serialized as finite JSON arrays. Non-finite values, shape
mismatches, unknown fields, duplicate keys, non-JSON metadata, and inconsistent
atom counts fail closed.

The manifest preserves the initial archive configuration and complete bootstrap
minimum, including energy and descriptor-relevant state.

### 7.3 Atomic commit

Each file is written to a unique temporary file in its target directory,
flushed, fsynced, atomically renamed, and followed by a parent-directory fsync.
The run store creates and fsyncs the `batches/` and `sessions/` directory
entries before publishing the manifest.

`append_batch` is idempotent:

- an existing byte-equivalent canonical batch is accepted as a retry;
- an existing batch ID with different content is rejected;
- gaps, duplicate action IDs, non-contiguous versions, or mismatched run IDs are
  rejected.

The batch file is committed before the controller installs the shadow archive,
posterior, and campaign spend in memory. A pending commit has an explicit
reconciliation operation. If a write raises after the canonical rename, that
operation validates and accepts the canonical byte-equivalent batch, installs
the controller shadow state, and advances the campaign ledger exactly once
without redispatching workers. If no canonical batch exists, it retries the
same complete pending batch, also without redispatch.

### 7.4 Recovery

Recovery:

1. validates the manifest and configuration identity;
2. starts from the serialized bootstrap archive and empty posterior;
3. reads contiguous batch files in order;
4. replays landing insertions in slot order;
5. recomputes the observation rule and replays a posterior update only when
   `posterior_observed` is true;
6. validates every stored observation flag, discovery, insertion, collision,
   landing ID, version, and cost against the reconstructed state;
7. restores `policy_version`, `archive_version`, `batch_id`, and campaign spend.

Before replay, every action cap and stored reservation is checked against the
manifest's fixed `action_force_budget`; batch-internal consistency alone is not
a trusted source of `q`.

Recovery first rejects `abort.json`. It then derives `budget_tail` or
`zero_cost_stall` from the committed ledger so a terminal campaign cannot
dispatch again. A temporary file without a canonical peer is ignored; a
temporary file that coexists with its canonical peer is treated as ambiguous
and fails closed.

Recovery never trusts a serialized posterior or mutable archive object without
replay validation.

A leftover temporary file is ignored only when the corresponding canonical
batch does not exist. An incomplete or corrupt canonical file fails closed.

### 7.5 Crash claim boundary

The store guarantees no duplicate committed credit and deterministic recovery
from complete batch boundaries.

If the process exits after dispatch but before atomic batch commit, those
physical calls are not durably known. Explicit resume may re-execute the
deterministic batch from the preceding committed snapshot. The final report
must state that committed cost is exact and interrupted-uncommitted hardware
cost is outside the ledger. A caught unknown-cost failure instead writes
`abort.json` and is not resumable.

No broader lifetime-budget claim is made.

## 8. Analytic Multi-Seed Policy Harness

Add a dedicated benchmark harness for the new runner. It does not reuse the
legacy LJ comparison's approximate local-relaxation budget.

For every paired seed and policy:

- use the same initial structure;
- use the same fixed SSW kernel configuration;
- use the same action cap `q`;
- use the same campaign cap `G`;
- use the same maximum batch width and worker count;
- use the same `master_seed` for every policy arm of a paired replicate;
- record the complete run manifest and run-store path.

The first harness compares:

- `uniform`;
- `posterior_proportional`;
- `minimal_ucb`, explicitly labeled incomplete-support.

Primary curves contain only states that actually existed: the bootstrap point
followed by committed-batch boundaries. The bootstrap point begins at its
physical evaluation count. Slot-order prefixes may be emitted only as clearly
labelled deterministic post-hoc visualizations, never as executed campaign
states.

Outputs include raw per-run records and primary derived curves for:

- best energy versus cumulative physical evaluations;
- unique minima versus cumulative physical evaluations;
- completed/failed attempts versus cumulative physical evaluations;
- duplicate fraction;
- per-purpose evaluation fractions;
- unused campaign budget;
- exact selection probabilities and support completeness.

The paired estimand is the within-seed contrast between policies under the same
`(G, q, B, W, master_seed)` and initial structure. Normal `budget_tail` runs
are compared at their final fixed-cap campaign state with unused budget
reported. `zero_cost_stall`, aborted, drifted, or otherwise benchmark-ineligible
runs are stratified and reported separately, not treated as observations at
`G`. Confidence intervals, when present, use paired/block bootstrap over
seed-level contrasts rather than independent per-policy intervals. The harness
persists the seed-to-master-seed mapping.

A tiny analytic smoke proves the harness contract. Larger LJ
experiments are evidence generation after Phase 3, not a condition for merging
the infrastructure.

The harness does not tune policies, remove unfavorable seeds, or use its output
to change defaults.

## 9. Error Semantics

- Invalid bootstrap geometry: fail before creating a committed run.
- Bootstrap budget exhaustion: fail with exact bootstrap counts.
- Known zero-cost factory failure: commit an exact zero-cost failed action; a
  zero-cost batch then stops the campaign without updating the posterior and is
  benchmark-ineligible.
- Calculator failure after a started call: count the call and return
  `WORKER_ERROR`.
- Action budget exhaustion: return `BUDGET_EXHAUSTED` with exact action cap.
- Future/adapter exception with unknown cost in strict runner mode: abort
  without committing the batch or updating archive/posterior, atomically write
  `abort.json`, and refuse resume.
- Non-finite landing or incompatible result: fail closed before commit.
- Run-store write failure: retain the exact pending commit in-process for retry;
  do not install archive/posterior state.
- Manifest/config/schema mismatch on resume: fail closed.
- `KeyboardInterrupt`, `SystemExit`, and other `BaseException` values are not
  converted to ordinary worker outcomes. Once dispatch may have begun, the
  runner conservatively writes the unknown-cost abort marker and re-raises the
  original value.

## 10. Configuration and Provenance Identity

The immutable manifest records:

- complete `PosteriorExplorationConfig`;
- complete `SSWConfig` or `LSSSWConfig`;
- initial state and bootstrapped minimum;
- policy name and support semantics;
- calculator-factory label supplied by the caller;
- calculator semantic fingerprint supplied by the caller;
- effective bootstrap and per-action transformed configurations;
- Python, NumPy, SciPy, ASE, and package versions when available;
- repository commit and dirty flag when discoverable;
- telemetry and run-store schema versions.

Resume compatibility is enforced on semantic configuration, structure,
calculator label/fingerprint, schema, repository source fingerprint, and the
Python/NumPy/SciPy/ASE/package versions that execute the kernel. These
executable-kernel differences fail closed because one posterior must not mix
physical kernels. Allowed non-kernel environment/hardware differences are
written to a new session record and make the resumed run
non-benchmark-eligible unless identical; they are not silently discarded.
The `mode` field is recorded but excluded from semantic equality because a
valid continuation necessarily changes `new` to `resume`; the canonical
`run_directory` must remain identical.

Secrets, model binary contents, and arbitrary calculator objects are never
serialized.

## 11. Required Tests

### 11.1 Evaluation ledger

- started success and started failure increment one total and one purpose;
- budget-rejected calls increment neither;
- nested purpose scopes restore correctly after success and exception;
- sum of purpose counts equals aggregate total;
- one real SSW action has zero unattributed calls;
- success, invalid, fragmented, budget exhaustion, and worker error preserve
  exact purpose snapshots.
- SSW and LS-SSW completed, duplicate/invalid, fragmented, worker-error, and
  budget-exhausted terminal routes have purpose-sum parity with raw calculator
  calls.

### 11.2 Contracts and logging

- generic omitted counts become unattributed;
- SSW results are exact-cost;
- controller-generated escaped future exceptions are unknown-cost;
- strict-cost mode refuses unknown-cost batch commit;
- infrastructure/adapter failures are posterior-unobserved and do not change
  posterior counts;
- each result/outcome slot has identical scalar, purpose, and exact-cost facts;
- event-log schema round-trips purpose counts and exact-cost flag;
- old/incomplete schema is rejected.

### 11.3 Campaign budget

- bootstrap plus committed action spend never exceeds `G`;
- reservation width is `min(B, floor(remaining/q))`;
- final narrow batch uses normal action cap;
- no residual low-fidelity action is dispatched;
- completion order does not change actions, propensities, costs, posterior, or
  budget state;
- early action termination releases unused reservation;
- zero-cost batch terminates without an infinite loop;
- zero-cost terminal resume dispatches no new action;
- no zero-budget action is created.

### 11.4 Run store and recovery

- manifest and state serialization round-trip exactly;
- one and multiple batches reconstruct archive entries, coordinates,
  constraints, metadata, posterior, versions, and spend;
- interrupted then resumed analytic run equals uninterrupted run at committed
  boundaries;
- append retry is idempotent;
- post-write acknowledgement failure does not re-execute a committed batch;
- dispatch-complete/pre-append interruption leaves no committed cost or credit,
  and explicit resume regenerates the same action IDs, seeds, and propensities;
- caught unknown-cost failure writes an abort marker and ordinary resume fails;
- truncated, tampered, non-contiguous, duplicate, incompatible, symlinked, or
  non-finite artifacts fail closed;
- an orphan temporary artifact is ignored, while temporary/canonical
  coexistence fails closed;
- session drift eligibility decisions are durable.

### 11.5 Runner

- posterior SSW and LS-SSW opt-in APIs run with analytic calculator factories;
- each action receives a distinct calculator;
- serial `run_ssw` and `run_ls_ssw` behavior remains available;
- runner refuses shared worker outputs and unknown-cost failures;
- bootstrap budget exhaustion exposes exact counts without creating a
  resumable run;
- `BaseException` propagates rather than becoming an action outcome;
- new versus resume directory rules are enforced;
- natural budget stop and zero-cost stall produce exact summaries.

### 11.6 Benchmark harness

- paired policy runs use identical seed, master seed, kernel, and budget inputs;
- raw result manifest includes provenance and exact costs;
- primary curves begin at bootstrap and advance only at committed-batch
  boundaries, with completed and failed attempt counts;
- paired/block intervals operate on seed-level policy contrasts;
- minimal UCB is labeled incomplete-support;
- no performance claim is emitted by the smoke test.

## 12. Acceptance Criteria

Phase 3 is complete only when:

- a public opt-in posterior SSW and LS-SSW runner exists;
- the legacy serial runner path remains unchanged;
- every analytic SSW worker call has exact purpose attribution;
- purpose totals equal raw calculator calls and aggregate action counts;
- bootstrap and committed actions obey one campaign force cap;
- action fidelity is fixed and independent of starter/policy outcome;
- no residual low-fidelity tail action is mixed into the posterior;
- all committed actions have exact known cost;
- the complete archive, posterior, controller versions, and budget state recover
  from atomic committed batches;
- uninterrupted and committed-boundary resumed analytic runs are equivalent;
- caught unknown-cost actions cannot resume or contaminate posterior credit;
- infrastructure failures are audit records but not posterior observations;
- executable-kernel identity fails closed on resume and allowed drift is
  durably marked benchmark-ineligible;
- benchmark curves represent bootstrap and real committed-batch states only;
- paired policy inference uses within-seed contrasts with identical master seed;
- a multi-seed analytic policy harness uses exact force-evaluation budgets;
- full existing tests and new focused suites pass;
- documentation states the exact unbiasedness, recovery, budget, and runtime
  claim boundaries;
- no new algorithm component or heuristic parameter is introduced.

## 13. Deferred Work After Phase 3

After Phase 3, the branch is ready for isolated development and ablation of:

- alternative starter posterior models;
- direction portfolios;
- alternative uphill propagators;
- action-conditioned statistics;
- later multi-fidelity or asynchronous designs.

Each requires its own design, fixed-kernel comparator, and promotion evidence.

Before expensive scientific claims, separately add:

- MACE/GPU/process runtime validation;
- periodic structure-identity upgrades;
- larger paired-seed benchmark campaigns;
- hard-crash reservation accounting if exact lifetime hardware-call caps are
  required.

Pure prepared-starter kernel extraction remains a separate parity project and
is not bundled into Phase 3.
