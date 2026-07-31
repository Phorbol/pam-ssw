# Paired Continuation/Restart Selector Gate Design

## Purpose

The present `archive_ucb` selector assigns a fixed weighted score to every
minimum in a growing archive.  The exploration bonus makes newly inserted or
rarely tried nodes individually attractive.  As the archive grows, this can
turn the selector into a near one-visit-per-node scheduler even when the
global-minimum objective needs repeated continuation from a productive low
energy funnel.

The gate asks one physical question:

> Is the useful outer-loop structure only a low-energy continuation channel
> plus a full-support global restart channel, rather than a weighted bandit
> over every archive node?

Direction generation, cumulative Gaussian propagation, proposal relaxation,
true quench, matcher and force accounting remain frozen.

## Existing evidence used before new force evaluations

The seed-42 20,000-FE campaigns already contain three systems and three
selectors.  A zero-FE S-CR0 audit will compute, from the recorded starter IDs:

- number of actions and final archive size;
- unique selected starters;
- Shannon effective support `exp(H)`;
- maximum starter frequency;
- fraction of actions drawn from a previously used starter.

These are descriptive diagnostics, not reward functions.  They test the
specific growing-arm concern and cannot promote a new selector.

## Alternatives

### A. Snapshot-paired best continuation plus uniform restart — selected

At the start of each two-action cycle, freeze the current archive and select:

1. the minimum-energy entry, with entry ID as deterministic tie-break;
2. one uniformly sampled entry from the same frozen archive.

The best action executes first and the cached uniform starter executes second.
Archive insertion never invalidates the cached entry because entries are not
deleted.  The scheme therefore has exact two-lane snapshot semantics even
when executed serially.  It can later be dispatched as two parallel workers
without changing the policy distribution.

This policy has no score weights, energy temperature, PCA dimension, top-k,
cell count or posterior parameter.  Every archive node has probability
`1 / (2N)` per slot averaged over a complete pair; the best node receives the
additional continuation mass.  This is global-search full support, not a
claim of detailed balance.

### B. Metropolis plus epsilon restart — rejected

This preserves the classic chain but introduces a restart probability whose
effect cannot be separated from the existing `0.26 eV` temperature in a small
gate.  It also does not test whether an energy-weighted acceptance mechanism
is needed at all.

### C. Thompson sampling over nodes or selector families — deferred

Node-level Thompson sampling retains the growing-arm problem.  Family-level
posterior allocation is meaningful only after continuation and restart have
separable, repeatable outcome distributions.  It is a possible downstream
allocator, not the present hypothesis.

## Implementation boundary

Add one opt-in `seed_selection_mode="paired_best_uniform"`.  The default
remains `archive_ucb`.  `SurfaceWalker.run` keeps at most one cached uniform
archive entry and clears it when a new pair begins.  Existing separate
selection RNG and physical-action RNG streams are preserved.

No archive deletion, selector score, direction, propagator, optimizer,
matcher or budget behavior changes.  The final incomplete pair, if the force
budget ends after the best slot, is retained and reported rather than hidden.

## S-CR1 fixed-budget gate

Run one shared-bootstrap campaign for every `(system, seed)` and reuse the
exact minimum and bootstrap cost across four arms:

- `uniform_archive`;
- `archive_ucb`;
- `metropolis_chain`;
- `paired_best_uniform`.

Frozen matrix:

- systems: C60, fixed-bottom PdO, packaged CuO;
- first gate seed: 45, unused by the seed-42 selector evidence;
- total budget: 20,000 force evaluations per arm including bootstrap;
- one worker, identical physical-action RNG seed per arm;
- CuO uses the packaged fine-tuned model and the already promoted
  oracle-only LS scope; C60/PdO use their existing frozen production kernels.

Retain separately best-energy-vs-FE AUC, final energy drop, completed actions,
unique minima, duplicates, purpose costs, invalid/failure counts and wall
time.  No weighted aggregate reward is created.

S-CR2 paired-seed expansion is admitted only if `paired_best_uniform` has the
largest gain AUC among the two exploitation comparators (`archive_ucb`,
`metropolis_chain`) in at least two of three S-CR1 systems.  Uniform is a
mechanism control and cannot lower this admission threshold.  Strict ranking
is used; no post-hoc tie tolerance is introduced.

If admitted, run seeds 46 and 47 with shared bootstrap for
`archive_ucb`, `metropolis_chain` and `paired_best_uniform`.  Promote the new
policy only if, over all nine system-seed blocks, its paired gain-AUC
difference is positive against both comparators in at least six blocks and
has positive median against each comparator.  Otherwise close it as an
experimental negative.

## Posterior admission boundary

No posterior is implemented in S-CR0/S-CR1.  A family-level posterior gate is
allowed only if the repeated experiments show that the continuation and
restart lanes have distinct, non-degenerate outcome/cost distributions and
that a pre-action context available without new force evaluations predicts
which lane is productive on held-out blocks.

Even then, posterior arms are the two physical families, never individual
archive nodes.  Propensities must remain positive and be recorded from the
immutable pre-batch snapshot.

## Claim ceiling

The gate can decide whether a fixed two-lane selector deserves paired-seed
testing under the frozen C60/PdO/CuO kernels.  It cannot establish asymptotic
large-archive performance, canonical sampling, production superiority,
posterior learnability or multi-GPU speedup.
