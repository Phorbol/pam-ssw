# K4 action-breadth order-statistic gate

## Question

The current static K4 scorer often misses the best terminal candidate, while
zero-FE single-candidate replacements do not transfer across C60 and PdO. Does
the already-generated K4 support have enough terminal value that executing two
complete candidates as an unbiased parallel action batch is worth its added
force budget?

This is not a new direction scorer and does not claim GPU wall-time speedup.
It measures the quality--FE frontier of action breadth using fully observed
terminal outcomes.

## Frozen evidence

- Source:
  `runs/20260731-direction-candidate-counterfactual-gate/repeat_evidence.json`
- SHA256:
  `cb19e5b4b9a8839ad917f287177251d179e6b0e662c4be6fa56a8d12bd347d5d`
- Two independent campaigns.
- Per campaign: 12 shared K4 pools, 48 complete candidate outcomes.
- Systems: C60 and fixed-bottom PdO.
- New force evaluations: zero.
- Each shared K4 pool cost is exactly 8 direction-oracle FE. The two campaign
  ledgers must satisfy:

  `campaign FE = sum(candidate FE) + 12 * 8`.

## Exact counterfactuals

For every pool and campaign:

1. `static-B1`: execute the existing `static_rank == 1` candidate.
2. `uniform-B1`: average exactly over all four one-candidate subsets.
3. `uniform-B2`: average exactly over all six two-candidate subsets.
4. `uniform-B3`: average exactly over all four three-candidate subsets.
5. `uniform-B4`: execute all four candidates.

For a subset, terminal quality is the lowest certified, geometry-valid landing
energy delta in that subset. Regret is relative to the best valid candidate in
the same fully observed pool. Cost is:

`8 shared-pool FE + sum(selected candidate FE)`.

Invalid candidates remain in the sampling support and contribute to the exact
probability that a subset contains at least one valid landing. No candidate
score, curvature, family, descriptor, or outcome is read before sampling.

## Frozen decision

Aggregate per `(campaign, system)` over the six paired pools using medians.
For `uniform-B2` relative to `static-B1`, define:

- fractional regret reduction;
- fractional FE increase;
- benefit--cost elasticity =
  `fractional regret reduction / fractional FE increase`.

Open one live two-action batch gate only if all four campaign-system strata:

1. have positive static median regret;
2. have no lower valid-subset probability than static;
3. reduce median regret;
4. have benefit--cost elasticity at least 1.

This scale-free rule requires the fractional quality recovery to be at least
as large as the fractional FE increase. Failure in any stratum closes the live
B2 claim. B3/B4 are frontier diagnostics only and cannot be promoted by this
gate.

No production default, posterior, ForceService, selector, direction generator,
or matcher is modified.
