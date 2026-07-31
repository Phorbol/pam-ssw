# Direction feature learnability gate

This is a zero-new-FE test of whether direction information already computed by
the K4 oracle is sufficient to support a statistical posterior.

The immutable input is the pair of completed shared-candidate counterfactual
campaigns: 12 starter/seed groups, four exact candidates per group, and two
certified terminal repeats per candidate.

## Feature blocks

- softness: within-pool standardized negative true-PES curvature;
- intent: within-pool standardized absolute anchor overlap and random/bond
  direction type;
- combined: softness plus intent;
- static baseline: the current within-pool static score.

`score_sigma` is excluded. In the observed positive-curvature regime its
adaptive definition makes `0.5 * sigma_score**2 * curvature` equal to the fixed
0.8 eV target, so it does not provide independent curvature information.

The learned models use one fixed unit-penalty, intercept-free ridge solution.
There is no regularization sweep, feature selection, interaction, nonlinear
model, MACE descriptor, classifier, TS, or UCB.

## Validation

- leave-group-out: tests another random K4 pool while sharing state contexts;
- leave-context-out: holds out one `(system, state)` context;
- leave-system-out: trains on C60 and tests PdO, then reverses.

Only leave-system-out controls promotion. The same combined model must obtain
at least 4/6 top-1 choices and zero median regret in each held-out system. Its
global mean regret must be lower than both single feature blocks and no worse
than the static baseline.

Passing would only permit design of a later posterior experiment. Failure
closes this low-dimensional posterior route and leaves production unchanged.
