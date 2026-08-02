# Step-1 Direction Counterfactual Gate Design

## Purpose

Determine whether the production continuation-direction bottleneck comes from
candidate source quality or from the static selector after a walk already has
one relaxed biased step.

The existing shared-K4 gate branches at micro-step 0. At that point
`previous_direction=None`, so it labels bond and random candidates but cannot
test the momentum candidate that dominates later production choices. This gate
adds that missing causal comparison without adding a new direction family,
scorer, posterior, or production parameter.

## Alternatives rejected

1. **Long-run momentum on/off.** This changes every later candidate pool and
   immediately creates different basin histories. It measures a whole search
   policy, not the causal value of one momentum candidate.
2. **Regression on selected production directions.** Unselected candidates do
   not have terminal outcomes, so source coefficients inherit winner-selection
   bias.
3. **Shared micro-step-1 branch-out.** Replay the same first biased step, build
   one common K4 pool containing momentum/bond/random, force each candidate,
   then continue the unchanged H8 action. This is the selected design because
   it isolates exactly one choice while retaining the real nonlinear
   propagator and true quench.

## Frozen experiment

- Systems: C60, fixed-bottom PdO, and CuO.
- Seeds: 52, 53, 54.
- Starter: one shared true-PES bootstrap minimum per system/seed.
- Propagator: production fixed H8.
- Step 0: one shared native K4 pool; execute its existing static-score winner.
- Step 1: one shared native K4 pool generated with the exact previous selected
  direction, accumulated Gaussian bias, relaxed state, archive, anchor, and
  trust-controller state produced by step 0.
- Step-1 candidate composition: production generator output without quotas or
  substitutions. A usable pool must contain exactly one momentum candidate and
  at least one non-momentum candidate; invalid bond proposals naturally fall
  back to random as in production.
- Arms: force every step-1 candidate once, then let steps 2--7 use the unchanged
  production static selector.
- Repeats: execute every frozen candidate trajectory twice. Candidate vectors,
  static ranks, step-0 prefix, and post-pool RNG state must be identical across
  repeats.
- All local softening, Gaussian bias construction, Safe-LBFGS proposal relax,
  true-PES quench, geometry validation, and matching remain unchanged.
- No starter UCB/TS, direction-type UCB, Ritz/Lanczos/Dimer, learned score,
  adaptive stopping, H4, new descriptor, or reaction-network label.

The fixed matrix contains nine shared pools, at most 36 candidates, and at most
72 terminal trajectories. A pool that terminates before step 1 is recorded as
right-censored and is not replaced by another seed.

## Replay and causal checks

The reference execution records the step-0 choice, the positions and true
energy entering step 1, the accumulated-bias parameters, the previous selected
direction, the step-1 candidate vectors and scores, and the RNG state after the
pool is generated.

Every forced arm replays from the same bootstrap state and must satisfy before
branching:

- identical step-0 selected-direction hash;
- identical step-1 input coordinates within `1e-10 Å` maximum absolute error;
- identical bias count and bias-parameter hashes;
- identical step-1 candidate-direction hashes and static ranks;
- zero direction-selection FE in the forced branch because the shared pool was
  already evaluated;
- the same post-pool RNG state before steps 2--7.

Failure of any prefix invariant invalidates the whole pool rather than being
repaired with a tolerance change.

## Outcome vector

Each candidate keeps separate, non-scalarized labels:

- terminal landing energy relative to the shared starter;
- best-energy improvement relative to the shared starter;
- certified true-quench status and final force norm;
- geometry validity and fragmentation status;
- same-basin versus different-basin landing relative to the starter;
- walk termination reason;
- full purpose ledger: shared-pool HVP, direction oracle, biased proposal
  relaxation, true-PES checks, landing quench, and wall time.

`new basin` is used only as a basin-escape label, not as a reaction-network
edge or a substitute for low-energy quality.

## Readouts

For every usable pool and each repeat report:

1. current static-winner terminal regret;
2. best-candidate identity and whether it is repeat-stable;
3. momentum terminal regret against the best non-momentum candidate;
4. pool-level bond and random family medians, avoiding a false advantage from
   the two available bond slots;
5. Spearman ordering between static score and terminal landing energy;
6. candidate cost and certificate failures.

The earlier step-0 dataset and this step-1 dataset remain separate. Step 0 asks
whether random/bond initialization contains a useful action; step 1 asks
whether continuing the previous displacement or changing direction is useful
after the PES has already responded once.

## Budget and stop rules

- Maximum new force evaluations: 60,000.
- Maximum single-GPU wall time: 35 minutes.
- `unattributed` must be zero and every forced-arm ledger must close.
- At least two usable pools per system and all their terminal quenches must have
  a strict convergence certificate; otherwise classify the dataset as
  numerically non-identifiable and stop.
- A source is called **cross-system dominant** only if it has lower pool-level
  terminal regret than every alternative in both repeats for at least two of
  three pools in every system. This is deliberately strict and creates no
  weighted mixture.
- The static selector is a **repeat-stable continuation bottleneck** only if at
  least two of three pools in every system have a repeat-stable better
  non-winner and the per-system median score/terminal Spearman is non-positive
  in both repeats.
- If candidate winner identity is repeat-stable in fewer than two pools for any
  system, no posterior/classifier gate opens.
- Any system-level sign reversal closes a universal family-prior claim.

No result changes the production direction policy directly. A positive source
result permits one source-only prospective ablation. A positive selection
result permits a separately preregistered family-level contextual model gate.
Mixed or negative results stop this branch without weight tuning.

## Claim ceiling

This is a three-system, three-context causal direction-choice gate under one
fixed H8 propagator and the current MLIP models. It can identify whether the
existing continuation pool contains useful alternatives and whether momentum
has repeatable conditional value. It cannot establish statistical significance,
generalize to variable-cell search, or prove that UCB, Thompson sampling, or a
learned classifier will improve long production searches.
