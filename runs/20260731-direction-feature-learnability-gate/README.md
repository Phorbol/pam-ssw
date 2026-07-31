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

## Result

The gate used 48 repeat-averaged candidates from 12 shared K4 groups and added
zero force evaluations. The posterior stage is **not allowed**.

Under leave-system-out validation:

- `static_score`: 2/12 top-1, mean regret 2.615 eV;
- softness: 6/12 top-1, mean regret 1.891 eV;
- intent: 3/12 top-1, mean regret 2.126 eV;
- combined: 3/12 top-1, mean regret 2.126 eV.

The combined model reached only 2/6 on held-out C60 and 1/6 on held-out PdO,
with median regrets of 1.800 and 0.626 eV. Leave-context-out was still worse:
the combined model selected 0/12 terminal winners.

A post-hoc coefficient audit found no large cross-system sign reversal.
Instead, all paid local features had weak within-system association with
terminal quality. For C60/PdO respectively, the absolute correlations were
approximately:

- softness: 0.031 / 0.130;
- anchor overlap: 0.000 / 0.028;
- direction family: 0.118 / 0.139.

Thus the failure is not repaired by changing UCB to TS or by placing a more
formal posterior over the same feature vector. The local K4/HVP descriptors do
not currently contain enough transferable information about the nonlinear
H8-plus-quench outcome. A richer posterior would require a new, independently
justified state/action representation and substantially more action-labelled
data; it is not a justified next production component.
