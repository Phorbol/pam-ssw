# GPU-native two-operator PES population design

## Status and decision boundary

This document specifies one bounded research line for fixed-composition,
fixed-cell PES exploration:

\[
\mathcal A = \{\text{direct displacement + true quench},
               \text{serial SSW + true quench}\}.
\]

Both operators act on shared starter minima and publish certified landing
minima to one epoch-frozen archive.  The first scientific gate uses a fixed
1:1 allocation.  GPU-native active-set batching is admitted only after the
two operators show a non-dominated cost/support trade-off under scalar
execution.

This is not a design for reaction networks, canonical sampling, continuous
Gaussian optimization, Bayesian allocation, MD, GA crossover, element
mutation, CCQN/PRFO, or a general asynchronous scheduler.

## Motivation

The current serial SSW action is expensive but causally active.  A selected
direction is followed by an explicit displacement, cumulative Gaussian bias,
modified-PES relaxation, and a final true-PES quench.  Existing counterfactual
evidence shows that the explicit displacement alone sometimes returns to the
starter when biased relaxation reaches another basin.  Proposal relaxation
therefore cannot be deleted as numerical overhead.

At the same time, proposal relaxation consumes most action force evaluations,
and the later biased path plus quench amplifies small direction or numerical
differences into different minima.  A terminal landing is consequently a
distributional label for the full action, not a clean label for one local
direction.

The new hypothesis is not that a direct move is a better SSW.  It is that the
two operators occupy different points on a cost/support frontier:

- a direct move provides cheap, broad basin-hopping support;
- SSW pays for orthogonal structural accommodation and curved, history-dependent
  propagation;
- a shared population can retain both without forcing either operator to solve
  every escape context.

## Scientific hypotheses

### H1: direct moves are a viable low-fidelity action family

For a shared starter and initial intent, an explicit displacement followed by
strict true-PES quench should produce a nonzero rate of certified non-starter
minima with lower median total action cost than the full SSW action.

### H2: SSW supplies non-dominated basin support

Full SSW should retain repeatable cases where the paired direct action returns
to the starter, is invalid, or reaches a worse basin.  Otherwise its additional
proposal-relaxation cost is not justified for this portfolio.

### H3: population sharing is sufficient exchange semantics

Workers need not exchange in-flight coordinates.  Publishing certified minima
to a shared archive at macro-action boundaries should provide the useful
cross-worker migration while preserving each action's Gaussian, optimizer,
and random state.

### H4: GPU batching is an execution property

Packing independent energy/force requests into one MLIP forward pass should
reduce wall time without changing logical force-evaluation counts or the
scientific allocation policy.  Batch size is an execution parameter selected
from device memory and throughput, not a search hyperparameter.

## Common action input

Every paired block freezes an immutable input:

```python
@dataclass(frozen=True)
class PopulationActionInput:
    starter_state: State
    starter_id: int
    initial_direction: np.ndarray
    direction_kind: str
    direction_diagnostics: Mapping[str, float]
    execution_sigma: float
    random_seed: int
    force_budget: int
```

The direction oracle is executed once before branching.  Both operators share
the exact direction, execution scale, starter coordinates, calculator/model,
constraints, geometry validator, and true-quench policy.  Direction-oracle
cost is charged once to the paired block and reported separately; it is not
duplicated in either arm.

Local softening may be used only inside the shared direction-oracle prefix.
It is disabled from both propagation arms so the experiment does not mix the
operator question with proposal-side LS.

## Operator definitions

### Direct operator

The direct endpoint is the current SSW frame-zero displacement:

\[
x_{\mathrm{direct}} = x_0 + \sigma u.
\]

It receives geometry validation and then the same strict true-PES quench as the
SSW arm.  It has no Gaussian term, modified-PES relaxation, short-rollout
screen, energy threshold, or fallback into SSW inside the same action.

The absence of a sequential fallback is deliberate.  Quenching a failed
direct action and then executing SSW would pay two actions and obscure whether
the cheap family reduced total force cost.

### SSW operator

The SSW arm is the cumulative-Gaussian H8 action resolved from
`feature/direction-continuation-ablation@115e56e` and the per-system profiles
used by `runs/20260802-fixed-h4-h8-equal-budget/`.  The execution protocol must
persist the complete resolved configuration and its hash.  The arm starts from
the shared starter and initial direction and uses the same strict true-PES
quench as the direct arm.  Subsequent direction reselection, Gaussian history,
step scales, and termination semantics remain unchanged.  This arm is a
reference, not a claim that the current adaptive feedback is optimal.

## Outcome contract

Every completed or censored action records the full vector outcome:

```python
@dataclass(frozen=True)
class PopulationActionOutcome:
    operator_family: str
    starter_id: int
    landing_id: int | None
    certified: bool
    returned_starter: bool
    is_new_minimum: bool
    improved_global_best: bool
    landing_delta_e: float | None
    direction_force_evaluations: int
    propagation_force_evaluations: int
    quench_force_evaluations: int
    validation_force_evaluations: int
    wall_seconds: float
    termination_reason: str
```

No mixed scalar reward is persisted.  Force evaluations are counted per
structure evaluation even when several structures share one GPU forward pass.
Kernel calls and wall time are separate execution telemetry.

## Shared-pool epoch semantics

The scientific reference scheduler is synchronous and deterministic:

1. freeze the current minima archive and policy state;
2. select the paired starter/action inputs;
3. allocate exactly one direct and one SSW arm per paired input;
4. execute every arm against the frozen snapshot;
5. sort completed records by preassigned slot id;
6. validate, deduplicate, and atomically merge certified minima;
7. expose the new archive only to the next epoch.

An in-flight SSW state is never imported by another worker.  Only a certified
minimum can migrate through the shared pool.  This avoids inventing exchange
rules for incompatible Gaussian histories.

The fixed 1:1 operator quota remains in force throughout the first online
campaign.  Completion order cannot change allocation, starter selection, or
credit.

## Stage A: zero-new-FE feasibility replay

Before implementation, reconstruct the two-family cost/support frontier from
the existing G-UP0 direct-versus-relaxed corpus and newer action histories.
The replay must distinguish source-generation cost, terminal quench cost, and
right-censored records.  It must not project unobserved sequential fallbacks.

Stage A passes only if neither family is already empirically dominated in both
certified basin support and total action cost.  If direct moves never supply a
useful certified landing, or SSW never supplies exclusive support, close the
portfolio without a fresh campaign.

## Stage B: fresh paired scientific gate

Run the unchanged two-arm comparison on C60, fixed-bottom PdO, and CuO:

- two predeclared starter contexts per system;
- three predeclared random seeds per starter;
- 18 paired inputs and 36 terminal actions;
- one shared initial direction per pair;
- maximum campaign budget: 60,000 force evaluations;
- all failures and budget-censored actions retained;
- `unattributed == 0` required.

The gate reports, by system and pooled only after system-level reporting:

- certified non-starter minima per 1,000 FE;
- global improvements per 1,000 FE;
- landing-energy distribution and paired regret;
- direct-only and SSW-only certified landings;
- propagation, quench, validation, and total FE;
- geometry-invalid, fragmented, unconverged, and budget-censored outcomes;
- wall time as telemetry, not the primary scientific result.

The portfolio survives only if all three preregistered rules pass:

1. **Direct viability:** in at least two of the six `(system, starter)`
   contexts, at least 2/3 direct seeds produce a certified non-starter landing
   and the context-median direct action FE is lower than the paired SSW median.
2. **SSW exclusive support:** in at least one context, at least 2/3 seeds have
   a certified SSW non-starter landing while the paired direct arm returns to
   the starter, is invalid, or is budget-censored.
3. **Numerical acceptability:** each operator has at least 5/6 certified
   terminal outcomes within every system.  Budget-censored actions count as
   non-certified for this rule.

A sign reversal without these repeated family effects closes the hybrid rather
than opening weights, thresholds, or operator-specific rescue rules.

This is a survivor gate, not a statistical-significance or production
superiority claim.

## Stage C: GPU-native active-set execution gate

Stage C is conditional on Stage B.  It changes execution only:

1. keep one independent optimizer/action state per slot;
2. pack all active structures requiring the same calculator purpose;
3. execute one batched MLIP energy/force request;
4. unpack results and advance each state independently;
5. remove completed slots and compact the next active batch.

Direction HVP geometries, direct-arm quench states, SSW proposal-relax states,
and true-quench states may each use this service, but different scientific
purposes are not mixed in one logical ledger entry.

The batch gate compares scalar and batched execution on frozen actions.  It
requires:

- identical action inputs and termination semantics;
- closed per-action and global ledgers;
- unchanged convergence certificates and basin classifications;
- no new invalid geometry;
- at least `1.3x` median wall-time speedup in each system.

Float32 batched and scalar paths need not be bitwise identical.  Any basin
classification change is reported as path sensitivity and fails algorithmic
equivalence for that cohort; it is not hidden behind a coordinate hash
tolerance.

## Explicitly deferred work

The following are not allowed into Stages A--C:

- sequential direct-then-SSW rescue;
- multiple Gaussian weights, widths, targets, or horizons;
- continuous or batch Bayesian optimization;
- UCB/Thompson allocation;
- MACE-feature starter models, FPS, DPP, or quality-diversity niches;
- MD, annealing, minima hopping, replica exchange, GA crossover, or element
  mutation;
- CCQN, constrained quadratic, PRFO, or saddle refinement;
- asynchronous stale-posterior updates;
- changes to the production default.

These are separate future operator or allocation hypotheses.  They are not
fallbacks for a failed two-family gate.

## Later posterior boundary

Only a surviving Stage B portfolio and an equivalent Stage C executor can
admit one allocation experiment.  That experiment compares fixed 1:1 quotas
against one hierarchical posterior over the two operator families.  It keeps
a positive minimum quota for both families and predicts the vector outcome:
validity, non-starter landing, global improvement, and total cost.

It does not construct one arm per minimum, regress terminal coordinates, or
optimize continuous Gaussian parameters.  Complete propensity and censored
cost logging are prerequisites.

## Interpretation of unbiased exploration

Here, "unbiased" means search support and statistical auditability:

- every starter admitted by the frozen starter policy retains nonzero support;
- both base operator families retain positive allocation probability;
- propensities are frozen before execution;
- failures and censored actions remain in the dataset.

It does not mean canonical sampling or detailed balance.  The shared archive,
energy objective, and adaptive selection of minima are global-optimization
mechanisms.

## Expected decision tree

```text
existing counterfactual replay
        |
        +-- one family dominated --> close hybrid
        |
        `-- complementary support --> fresh paired 1:1 gate
                                      |
                                      +-- no repeatable complementarity
                                      |       --> retain current reference
                                      |
                                      `-- survives --> batch-equivalence gate
                                                       |
                                                       +-- no wall gain/equivalence
                                                       |       --> scalar portfolio
                                                       |
                                                       `-- passes --> online fixed 1:1
                                                                    campaign
```

The controlled single-coordinate propagator remains a separate future
hypothesis if the direct family is too weak and current SSW is too expensive.
It is not bundled into this portfolio design.
