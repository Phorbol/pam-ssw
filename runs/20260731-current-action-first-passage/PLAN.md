# Current D0/K4 escape first-passage gate

## Mechanistic question

For the current production Gaussian-bias walk, does a D0 or K4 action:

1. never leave the starter basin;
2. enter a different certified basin at an early macro horizon and return by
   horizon 8; or
3. become unidentifiable because of quench, geometry, fragmentation, matcher,
   or budget failure?

This gate changes no direction source, Gaussian parameter, LS setting,
proposal optimizer, true-quench protocol, or production default.

## Frozen matrix

- systems: C60, PdO;
- locked starters: `intermediate_accepted`, `plateau_accepted`;
- seeds: 42, 43, 44;
- actions: `D0_exact_anchor`, `K4_discrete`;
- independently quenched horizons: 1, 2, 4, 8;
- 24 generation paths and at most 96 checkpoint quenches;
- total cap: 15,000 force evaluations;
- unattributed force evaluations: zero.

The run reuses the production config resolved by
`20260730-starter-cell-online-gate` and the action definitions validated by
`20260731-action-family-transfer-gate`. It records trajectories externally;
no callback or new production parameter is added to `pamssw/walker.py`.

## Labels

- `RETURN_STARTER`
- `ESCAPED_CERTIFIED`
- `AMBIGUOUS_MATCH`
- `INVALID_GEOMETRY`
- `FRAGMENTED`
- `QUENCH_UNCONVERGED`
- `BUDGET_EXHAUSTED`

A certified landing counts as escaped only when the existing archive matcher
and the configured invariant descriptor threshold both say it differs from
the starter. Disagreement is retained as ambiguity, not coerced into success
or failure.

## Decision rules

- Open the discrete horizon gate only when at least two of three seeds in one
  exact `(system, starter, action)` context escape early and return at H8.
- Call an action-support gap only when both D0 and K4 return at all four
  horizons for all three seeds in one `(system, starter)` context.
- Route a system to numerical/matcher work when at least two trajectories
  contain `AMBIGUOUS_MATCH` or `QUENCH_UNCONVERGED`.
- Otherwise report context dependence and add no adaptive rule.
