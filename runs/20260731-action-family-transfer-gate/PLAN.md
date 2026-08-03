# Action-family transfer gate

## Question

Before adding Thompson sampling, a contextual model, or a generative policy,
determine whether a repeated, low-dimensional direction mechanism has
transferable terminal value across C60 and PdO.

## Frozen experiment

- Systems: C60 and PdO.
- Starters: locked accepted minima at production trials 100 and 180.
- Seeds: 42, 43, and 44.
- Uphill horizon: 8 microsteps.
- Proposal relaxation, local softening, Gaussian-bias propagation, true
  quench, geometry validation, and force accounting are frozen.
- Arms:
  - `D0_exact_anchor`: execute the random-plus-bond initial intent directly;
  - `D1_anchor_krylov_d2`: one finite Krylov expansion from the same intent;
  - `K4_discrete`: current four-candidate discrete direction selection.

The D1 arm uses the existing `anchor_krylov` implementation at depth two. It
adds no Dimer angular step, cone, bias-strength, or convergence parameter.

## Primary outcome and promotion gate

The terminal outcome is `landing_energy - starter_energy`; lower is better.
Force evaluations are a separately reported cost, not folded into a tuned
scalar reward.

For each held-out system:

1. select the arm with the lowest mean terminal outcome on the other system;
2. require its mean terminal outcome on the held-out system to be lower than
   the equal-arm mean on that system;
3. require the selected arm not to be Pareto-dominated by another arm in both
   mean terminal outcome and mean force evaluations.

Both leave-one-system-out directions must pass. Otherwise no online posterior,
TS, UCB, classifier, or action-family weighting is promoted.

## Integrity requirements

- All 36 cases complete with strict terminal-quench certificates.
- No fragmented landing structures.
- Purpose-resolved force ledgers close with `unattributed == 0`.
- Same `(system, state, seed)` group has the same anchor hash for all arms.
- No production default changes.

