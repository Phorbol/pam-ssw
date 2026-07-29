# Anchor-Consistent Direction Ablation Design

## Decision

Run one paired, fixed-starter C60 ablation that changes only the relationship
between the random-plus-bond physical anchor and the executed direction.

The three arms are:

1. `detached_ritz`: the current one-block, two-column, depth-six block-Krylov
   solve, with 12 HVPs per direction selection;
2. `exact_anchor`: execute the exact random-plus-bond anchor and spend one
   central HVP (two force evaluations) only to obtain the total and true
   curvatures required by the unchanged uphiller;
3. `anchor_lanczos`: start a one-column, depth-twelve Lanczos space from that
   exact anchor and select its lowest Ritz vector, with 12 HVPs per selection.

No arm adds an overlap threshold, anchor penalty, curvature target, adaptive
depth, choice-aligned-softening threshold, selector, UCB/TS update, or archive
feedback.

## Why this is the next experiment

The completed checkpoint-shooting audit found no categorical terminal
overshoot: every failed trajectory lacked a productive checkpoint throughout
its accepted uphill path.  It also found that lower curvature was not
sufficient for a productive transition.

The subsequent code audit identified a more basic disconnect:

- block-Krylov intents are sampled before and independently of the
  random-plus-bond anchor;
- local softening is built from the anchor;
- lowest-Ritz selection ignores anchor overlap;
- choice-aligned softening was disabled;
- the observed median absolute anchor cosine was only 0.050 for the balanced
  arm and 0.067 for the deep arm.

The current algorithm therefore often softens one coordinate and walks along
another nearly orthogonal coordinate.  Before optimizing a selector or adding
a constrained eigensolver, the experiment must determine whether using the
same physical anchor in direction construction improves terminal PES
outcomes.

## Alternatives considered

### Enable choice-aligned softening

This would rebuild softening around the selected Ritz direction when an
overlap threshold is crossed.  It is rejected for this phase because it adds a
threshold and changes the proposal PES after direction selection.  It tests a
different mechanism from retaining the original random-plus-bond intent.

### Penalize distance from the anchor

An objective such as

\[
u^\mathsf{T} H u + \lambda\left(1-(u^\mathsf{T}a)^2\right)
\]

or a constraint \(|u^\mathsf{T}a|\ge\eta\) would formalize a CBD-like local
mode search.  It is rejected for this phase because \(\lambda\) or \(\eta\)
would be a new, presently unidentifiable algorithm parameter.  It becomes
justified only if the parameter-free anchor-seeded space improves outcomes
but still loses excessive anchor continuity.

### Exact-anchor baseline plus anchor-seeded Lanczos

This is selected.  It exposes two separable questions without new tunable
weights:

1. Is the original physical anchor itself more effective and cheaper than the
   detached softest Ritz direction?
2. Does curvature refinement inside a Krylov space generated from that exact
   anchor improve upon the raw anchor?

## Paired-randomness contract

The exact random-plus-bond anchor must be generated before any arm-specific
Krylov intent consumes random numbers.  For a fixed starter, seed, and macro
step, all three arms must therefore receive the same anchor.

The existing detached control remains detached: after the common anchor is
created, it samples its independent random/pair intent block and solves the
current depth-six block problem.  Moving common-anchor generation ahead of
intent generation changes the random-number ordering relative to older
block-Krylov runs, so this experiment is paired internally rather than treated
as bitwise continuation of the previous cohort.

The anchor remains fixed over one proposal walk, matching current SSW walk
semantics.  Arm-specific Krylov spaces are also constructed once at the seed
state and reused during that walk.  Geometry-adaptive regeneration is
explicitly out of scope so it cannot be confounded with anchor consistency.

## Minimal code seam

### Configuration

Extend `direction_selection_mode` with two explicit research modes:

- `exact_anchor`;
- `anchor_krylov`.

Production defaults remain unchanged.

`exact_anchor` uses the common anchor directly.  It evaluates one
directional HVP so that `DirectionChoice.curvature` and
`DirectionChoice.true_curvature` are defined without changing step-scale or
bias-weight mathematics.

`anchor_krylov` constructs:

```python
IntentBlock(basis=anchor_direction[:, None])
```

and calls the existing `solve_krylov_block` with depth twelve.  No new Krylov
solver or regularizer is introduced.

### Walker ordering

At proposal-walk bootstrap:

1. generate the common exact anchor;
2. build local softening from that anchor;
3. build only the arm-specific intent:
   - detached random/pair block for `block_krylov`;
   - one-column exact-anchor block for `anchor_krylov`;
   - no block for `exact_anchor`;
4. enter the existing macro-step loop.

All downstream Gaussian bias, explicit displacement, safe-LBFGS proposal
relaxation, true-PES checks, terminal strict quench, archive matching, and
acceptance logic remain unchanged.

## Experiment cohort

Use the same two locked C60 accepted states:

- `intermediate_accepted`;
- `plateau_accepted`.

Use paired seeds 42, 43, and 44 and the three arms, for 18 trajectories.

The checkpoint-shooting result already showed no categorical overshoot, so
this ablation performs only the normal terminal strict quench.  It must not
repeat 68 diagnostic checkpoint quenches.

Frozen conditions include:

- MACE model and model hash;
- starter structures and hashes;
- Gaussian bias and local-softening parameters;
- maximum macro steps;
- safe-LBFGS-total proposal optimizer;
- strict ASE-LBFGS true quench and preregistered FIRE fallback;
- `fmax`, maximum iterations, deduplication, and meaningful energy threshold;
- no selector, posterior, archive, or cross-trajectory learning.

## Accounting

Every trajectory must record separate purpose ledgers.

Expected direction-oracle cost per accepted macro step:

- `detached_ritz`: 12 HVP = 24 force evaluations;
- `exact_anchor`: 1 HVP = 2 force evaluations;
- `anchor_lanczos`: 12 HVP = 24 force evaluations.

No unattributed evaluations are permitted.  Bootstrap is skipped because the
locked accepted state is the starter and therefore costs zero evaluations in
this experiment.

Report:

- direction-oracle FE;
- biased-proposal-relax FE;
- true-PES escape-check FE;
- terminal-quench FE;
- total candidate FE;
- measured generation and quench wall sections.

## Diagnostics

For every macro step, retain:

- selected and true curvature;
- absolute anchor cosine;
- participation ratio;
- Ritz residual and initial-span overlap where applicable;
- HVP count and force-evaluation delta;
- selected direction kind;
- proposal termination reason.

For every terminal outcome, retain:

- certified convergence;
- final maximum force;
- landing energy change;
- new-basin flag;
- meaningful-outcome flag;
- fallback usage;
- source and output hashes.

The runner must prove that the common anchor call precedes arm-specific random
consumption.  Unit tests must prove exact anchor reuse by identity, not infer
it from the seed alone.

## Interpretation

This is a descriptive three-seed mechanistic ablation, not statistical proof
or a production-default promotion.

Possible conclusions are deliberately narrow:

- if `exact_anchor` improves outcomes, detached soft-mode refinement is
  damaging the initial physical intent;
- if `anchor_lanczos` improves over both controls, anchor-consistent curvature
  refinement is a positive mechanism;
- if `anchor_lanczos` is softer but not more productive, further optimization
  of the lowest Rayleigh quotient is not the next priority;
- if none improve the intermediate starter, direction initialization itself
  needs new physical content before posterior selection can help.

No selector receives credit from these 18 trajectories during execution.
Posterior arm selection remains a later phase and may use the completed
action-labelled outcomes only after this mechanism is resolved.

## Tests and acceptance

Unit tests must fail before implementation and then prove:

1. `exact_anchor` returns the exact projected anchor and spends exactly one
   HVP;
2. `anchor_krylov` uses a one-column block equal to the exact anchor and
   consumes exactly the configured 12 HVPs;
3. detached, exact-anchor, and anchor-Lanczos paths receive the same common
   anchor for a paired seed;
4. production defaults are unchanged;
5. the two explicit modes validate while unknown modes still fail clearly;
6. purpose ledgers close with zero unattributed evaluations.

The GPU runner is accepted only when all 18 cases complete, source hashes
match, strict certificates and cost ledgers close, and a conclusion separates
verified code facts, measured outcomes, physical interpretation, and
unproven claims.
