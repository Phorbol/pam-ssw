# Direction-Mode Continuation Ablation Design

## Decision

Test whether the current SSW plateau is caused by solving for a new local
softest direction at every biased macro step without explicitly following the
same moving event branch.

The experiment has three direction arms:

1. `fixed_intent_ritz`: the current detached two-column block-Krylov control;
2. `transported_direction`: use the control direction at step zero, then
   directly transport the preceding selected direction to the next biased
   geometry;
3. `continuation_lanczos`: use the control direction at step zero, then seed a
   one-column Krylov solve from the preceding selected direction.

No arm adds an overlap weight, cone, angular threshold, CBD bias strength,
posterior, selector, classifier, or adaptive HVP depth.

## Scientific question

The original SSW mechanism updates the walking direction at each biased
configuration.  Its biased-CBD rotation is intended to retain the physical
information of the initial direction while moving toward a locally softer
mode; it is not defined as repeated convergence to the globally softest
eigenvector.

The current block-Krylov path initializes `anchor_direction` and
`krylov_intents` once before the biased walk.  At every later step it rebuilds
a Krylov space from the same initial intent block.  The `previous_direction`
argument is not used by this path.  Consequently, the selected direction can
switch between local mode branches as the structure and cumulative bias PES
change.

This experiment asks two separable questions:

1. Is preserving the preceding direction itself beneficial?
2. If so, does recomputing local curvature around that preceding direction
   improve over direct transport?

## Evidence boundary

Completed C60 experiments already show:

- the raw random-plus-bond anchor is too stiff and produces only higher-energy
  new basins in the tested cohort;
- anchor-seeded lowest-Ritz refinement is softer and closer to the anchor but
  does not improve terminal outcomes;
- an energy-bounded linear combination retains substantially more anchor
  overlap at comparable curvature but does not outperform detached Ritz;
- deeper or cleaner eigensolves do not by themselves improve PES exploration;
- the current adaptive uphill updater has context-dependent width failures,
  but globally removing width feedback merely moves those failures between
  prefixes.

Older experiments also show that an explicitly weighted CBD-like objective is
substantially more expensive and did not beat the adaptive baseline on the
tested C60, PdO, and CuO cases.  A separate reference-Dimer smoke remained
under-converged.  Repeating either parameterized mechanism is therefore not
the next experiment.

The present study is a paired mechanistic ablation.  It is not a statistical
proof, a production-default promotion, or evidence for a learned direction
policy.

## Arm definitions

### 1. `fixed_intent_ritz`

This is the unchanged control:

- generate one common random-plus-bond anchor at walk initialization;
- generate the detached two-column intent block once;
- at every biased step, use the same initial block;
- run block Krylov at depth six;
- consume 12 central-difference HVPs per completed selection;
- execute the lowest total-curvature Ritz direction.

### 2. `transported_direction`

At step zero, execute exactly the direction selected by
`fixed_intent_ritz`.  This common first step is required so the experiment
isolates continuation rather than direction initialization.

At each later step:

1. take the preceding selected direction;
2. project out the current state's rigid-body and fixed-atom components;
3. normalize and sign-align it with the preceding direction;
4. evaluate one central-difference HVP to obtain total and true curvature;
5. execute the transported direction through the unchanged propagator.

The arm pays its actual cost: one HVP, or two force evaluations, per selection
after step zero.  It must not perform padding evaluations to match the control.

If projection makes the direction numerically zero, the case terminates with
an explicit `continuation_projection_degenerate` reason.  It does not fall
back silently to another direction source.

### 3. `continuation_lanczos`

At step zero, execute exactly the common control direction.

At each later step:

1. project, normalize, and sign-align the preceding selected direction;
2. use it as a one-column `IntentBlock`;
3. run the existing Krylov solver at depth twelve;
4. consume 12 central-difference HVPs per completed selection;
5. orient the lowest Ritz direction toward the preceding direction before
   execution.

This arm has the same HVP ceiling as the control at every completed selection.
It changes only the initial subspace vector and therefore tests local
mode-branch continuation without a new eigensolver.

## Paired-randomness contract

For a fixed starter and random seed:

- all arms receive the same random-plus-bond anchor;
- all arms receive the same detached step-zero intent block;
- all arms execute the same step-zero selected direction before CUDA numerical
  divergence;
- arm-specific logic begins only at step one;
- no arm consumes extra random numbers after the common context is created.

The runner must prove the common step-zero direction from recorded vector
hashes and pairwise cosine, rather than inferring equality from the seed.

## Unbiasedness boundary

This design preserves the project's global-search meaning of unbiased
exploration:

- no target product, reaction coordinate, transition label, or known pathway
  enters the action;
- each macro trial retains the same random-plus-bond initialization
  distribution;
- no terminal outcome or archive reward changes a direction during the
  experiment;
- every arm is selected externally by the paired experiment, not online by a
  posterior.

Direction correlation inside one biased walk is intentional state propagation,
not a target-specific bias.  This is not a claim of canonical sampling or
detailed balance.

## Frozen downstream algorithm

The following remain identical to the validated C60 K4 profile and the current
uphill control:

- locked starter structures and model hash;
- local-softening construction;
- cumulative Gaussian bias form;
- per-atom-RMS step semantics;
- current sigma and normalized-curvature feedback;
- safe-L-BFGS-total proposal relaxation;
- proposal `fmax`, relaxation cap, and walk displacement guard;
- maximum eight biased steps;
- true-PES escape checks;
- strict terminal true quench and preregistered fallback;
- structure matching and meaningful-outcome definition;
- no starter selector, UCB, Thompson sampling, posterior, reaction-network
  objective, or archive learning.

## Minimal implementation seam

Add two explicit research-only direction modes:

- `transported_direction`;
- `continuation_krylov`.

Do not add a general strategy hierarchy or refactor the walker.

The minimal changes are:

1. retain the preceding **selected** direction separately from the relaxed
   displacement direction;
2. construct the arm-specific direction input inside the existing biased-step
   loop;
3. reuse `_choose_exact_anchor_direction`-style one-HVP accounting for direct
   transport;
4. reuse `solve_krylov_block` for continuation Lanczos;
5. add continuation diagnostics to the existing direction record;
6. leave every production default unchanged.

The existing `previous_direction` variable currently represents the
normalized relaxed coordinate displacement.  It must not be silently reused
as the selected-mode state.  The two physical quantities need distinct local
variables.

## Diagnostics

Record for every direction selection:

- selected-direction vector hash;
- signed and absolute cosine with the preceding selected direction;
- signed and absolute cosine with the relaxed displacement direction;
- signed and absolute cosine with the original anchor;
- total and true curvature;
- participation ratio;
- Krylov residual and initial-span overlap where applicable;
- requested and consumed HVP count;
- direction-oracle force-evaluation delta;
- executed step scale;
- proposal outcome class and termination reason.

These are observer-only algebraic diagnostics.  They must add zero PES
evaluations.

## Stage C0: tests and analytic mechanism check

Before GPU execution, tests must prove:

1. all arms execute an identical step-zero direction;
2. direct transport consumes exactly one central HVP after step zero;
3. continuation Lanczos consumes exactly 12 HVPs after step zero;
4. control and continuation Lanczos have equal per-selection HVP ceilings;
5. vector orientation is sign-consistent;
6. rigid-body and fixed-atom components remain projected out;
7. a degenerate projection terminates explicitly;
8. production defaults and existing direction modes are unchanged;
9. purpose ledgers close with zero unattributed evaluations.

A small analytic Hessian test must demonstrate mode transport and a controlled
mode crossing.  It validates semantics only and cannot promote an arm.

## Stage C1: paired C60 terminal cohort

Use the same locked C60 accepted minima:

- `intermediate_accepted`;
- `plateau_accepted`.

Use seeds 42, 43, and 44 with all three arms, for 18 terminal trajectories.

Each trajectory must produce:

- a fresh strict terminal-quench certificate;
- independent starter, escape, and landing hashes;
- purpose-resolved force accounting;
- generation and quench wall sections;
- complete direction-continuation diagnostics.

Meaningful remains a certified new basin at least 0.001 eV below the starter.
Continuous landing-energy change is reported even when this threshold is not
met.

The expected first-stage scale, based on the preceding 18-case cohort, is
approximately 7,500 force evaluations and three GPU minutes.  These are
planning estimates, not budgets to be filled.  The hard stop is 10,000
accounted force evaluations for Stage C1.

## Stage C2: repeatability gate

No arm advances merely because its median direction cosine is cleaner.

An experimental arm survives Stage C1 only if:

1. it increases mode continuity as designed;
2. it has no force-certificate or geometry-validity regression;
3. it either finds a meaningful lower basin in a paired condition where the
   control does not, or reproduces the control's meaningful conditions at
   lower actual force cost.

Repeat only the control and surviving arm on the same two starters and three
seeds.  A terminal advantage is called stable only when the same paired
condition has the same meaningful/non-meaningful classification in both
executions.  Float32 CUDA energy and structure hashes are not required to be
bitwise identical.

If no arm survives or the repeat reverses the result, stop this direction
route.  Do not add a continuity weight, angular threshold, adaptive depth, or
fallback mixture to rescue it.

## Stage P: PdO transfer gate

PdO is entered only after a stable C60 survivor exists.

Run the current control and the single survivor on two locked PdO accepted
states and paired seeds 42, 43, and 44.  Freeze the PdO model, fixed mask,
geometry validator, propagator, relaxer, quench, and force budget.

PdO is a transfer test, not a second tuning dataset.  No C60 decision rule or
parameter may be changed after seeing PdO.

## Cost and reporting

For every case report:

- direction-oracle evaluations;
- biased-proposal-relax evaluations;
- true-PES escape-check evaluations;
- terminal-quench evaluations;
- post-relax validation evaluations;
- total evaluations and unused budget;
- generation and quench wall time;
- unattributed evaluations.

Compare a metric vector rather than a scalar score:

- stable meaningful terminal events;
- continuous landing-energy changes;
- mode continuity;
- new-basin rate;
- force certificates;
- invalidity and damage;
- purpose-resolved cost.

No weighted aggregate reward is introduced.

## Interpretation

- If `transported_direction` wins, repeated soft-mode refinement is redundant
  or destructive in the tested regime.
- If `continuation_lanczos` wins over both controls, moving mode-branch
  tracking is a positive mechanism.
- If continuity improves but terminal outcomes do not, continuation is only a
  diagnostic success and remains default-off.
- If neither arm improves the intermediate starter, the action set lacks a
  productive physical event family.  The next work must change direction
  content, not selectors or eigensolver parameters.

Posterior direction selection is permitted only after at least two direction
actions have stable, action-labelled meaningful outcomes under matched
contexts.  This experiment does not train or update such a selector.

## Non-goals

- no full CBD or reference-Dimer implementation;
- no OPES, metadynamics, direct quadratic uphiller, or bias-form change;
- no posterior, UCB, Thompson sampling, classifier, or MACE-feature model;
- no asynchronous scheduler or batch relaxer change;
- no archive or reaction-network redesign;
- no production-default change.
