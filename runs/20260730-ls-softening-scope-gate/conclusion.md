# LS-SSW softening-scope mechanism gate

## Decision

The current moving-reference exponential local-softening mechanism does not
advance to an equal-budget full-search gate.

It is numerically active, but the fixed-starter evidence does not support
either of its intended roles:

- it did not change the first selected direction in any of the 18 paired
  system/state/seed blocks;
- applying it to the direction oracle did not produce a more delocalized first
  direction;
- applying it to the biased proposal landscape did not provide a robust
  landing-energy gain;
- combining both roles was not consistently better than either role alone.

The production default remains unchanged. This is a descriptive fixed-starter
mechanism conclusion, not a universal rejection of the LS-SSW idea or of the
published algorithm.

## Frozen design

Execution commit:
`38d9296bdef1a3a4399641e58382dcef2ee2c5d4`.

The experiment uses:

- C60 and fixed-bottom PdO;
- bootstrap, middle, and late minima from the completed seed-42 uniform
  production-kernel run;
- paired direction RNG seeds 42, 43, and 44;
- four local-softening scopes:
  - `none`: no penalty in either component;
  - `oracle`: penalty only in direction HVP/scoring;
  - `proposal`: penalty only in biased proposal relaxation;
  - `both`: the pre-existing production behavior;
- the same direction portfolio, Gaussian-bias updater, proposal optimizer,
  true quench, structure matcher, and system-specific production configuration.

This gives 72 complete escape-plus-true-quench cases. The analytic local
penalty does not consume extra true-PES force evaluations. Every MACE call is
still attributed through the common purpose ledger.

## Outcome summary

Landing delta is landing energy minus starter energy; lower is better.

| System | Scope | Cases | New basins | Quench certificates | Median landing delta (eV) | Median FE |
|---|---|---:|---:|---:|---:|---:|
| C60 | none | 9 | 9 | 9 | -2.973572 | 535 |
| C60 | oracle | 9 | 9 | 9 | -1.610138 | 483 |
| C60 | proposal | 9 | 9 | 9 | -1.978760 | 531 |
| C60 | both | 9 | 9 | 9 | -0.988953 | 452 |
| PdO | none | 9 | 9 | 9 | -0.437073 | 317 |
| PdO | oracle | 9 | 9 | 9 | -0.759155 | 351 |
| PdO | proposal | 9 | 9 | 9 | -1.422791 | 425 |
| PdO | both | 9 | 9 | 9 | +0.006042 | 349 |

The apparent PdO median improvement of each single role is not robust under
the paired factorial decomposition. For every fixed
system/state/seed block, define

```text
oracle main effect =
  0.5 * [(oracle - none) + (both - proposal)]

proposal main effect =
  0.5 * [(proposal - none) + (both - oracle)]

interaction =
  both - oracle - proposal + none
```

Positive landing-energy effects are worse.

| System | Effect | Mean (eV) | Median (eV) | Better blocks |
|---|---|---:|---:|---:|
| C60 | oracle main | +0.769462 | +0.106369 | 3/9 |
| C60 | proposal main | +0.570724 | +0.106430 | 3/9 |
| C60 | interaction | -0.350318 | -0.166046 | 7/9 |
| PdO | oracle main | +0.236426 | +0.381042 | 4/9 |
| PdO | proposal main | +0.063995 | +0.388275 | 4/9 |
| PdO | interaction | +0.436747 | +0.341248 | 1/9 |

The negative C60 interaction does not rescue the combined method: both main
effects are adverse, and the `both` arm has the worst median landing delta.
For PdO the interaction is adverse in 8/9 blocks.

## Direction mechanism

The strongest result is structural rather than statistical:

- `none` and `proposal` selected the same first direction in all 18 paired
  blocks;
- `oracle` and `both` selected the same first direction in all 18 blocks;
- `none` and `oracle` also selected the same first direction in all 18 blocks.

Thus the current exponential penalty changed no first-step direction identity.
The first-direction participation ratio was likewise unchanged:

- C60 median: `0.03333` in all scopes;
- PdO median: `0.60251` in all scopes.

For the same selected direction, local softening changed inner curvature
relative to true curvature by:

| System | Mean shift | Median shift | Range |
|---|---:|---:|---:|
| C60 | +0.05265 | 0.00000 | 0.00000 to +0.19279 |
| PdO | +0.33433 | +0.25318 | +0.15485 to +0.59465 |

Positive means that the selected direction became stiffer. The current
exponential pair term is therefore not acting as a negative radial Hessian
update on these selected directions. At its moving reference distance it has
positive radial curvature and a non-zero repulsive force; its negative
curvature is transverse to the pair.

## Active-region mismatch

Production softening selects active atoms from the walk's initial anchor, not
from the final direction selected by the portfolio. That mismatch is large in
this cohort:

- C60 median absolute anchor cosine is about `0.05`;
- C60 median active-set overlap is zero; 54--58 of 63--67 recorded steps,
  depending on scope, have zero overlap;
- PdO median absolute anchor cosine is about `0.06--0.12`;
- PdO active-set overlap is only `0.0--0.2`.

The first penalty build is not numerically negligible:

| System | Median pair terms | Penalty energy | Per-atom energy | Gradient norm |
|---|---:|---:|---:|---:|
| C60 | 9 | 1.35 eV | 0.0225 eV/atom | 1.52 eV/A |
| PdO | 37 | 5.55 eV | 0.0483 eV/atom | 3.36 eV/A |

The issue is therefore not that the penalty is too small to execute. It is
that a substantial moving repulsive force is usually applied to atoms that do
not carry the selected displacement.

## Cost

C60 softening sometimes shortened the realized optimizer trajectory:

- oracle main FE effect: median `-16.5`;
- proposal main FE effect: median `-21.5`;
- combined median FE: 452 versus 535 for no softening.

This is not a useful speedup because the median landing result becomes about
1.98 eV worse in the combined arm.

PdO does not show the same cost benefit:

- oracle main FE effect: median `+8.0`;
- proposal main FE effect: median `+65.5`;
- combined median FE: 349 versus 317 for no softening.

Optimizer-path shortening by itself is not evidence of better PES
exploration.

## Interpretation and next boundary

The current implementation should be described as an
LS-SSW-inspired moving-reference local repulsive drive. It is not a faithful
implementation of the published fixed-reference, penalty-pre-relaxed,
self-adapted LS-SSW sequence.

This gate rejects the current mechanism, not the physical idea of local mode
softening. If LS research is resumed, the next question must be narrower:

1. rebuild the already-existing active-neighbor penalty on the actually
   selected direction and test that single alignment change;
2. only if alignment is positive, compare moving reference against one
   macro-step-fixed reference;
3. only if the oracle role becomes positive, consider a force-free low-rank
   Gaussian Hessian update;
4. do not add adaptive strength, generalized eigenproblems, new selectors, or
   a continuous parameter sweep before one of these mechanisms passes.

No 20k-FE or 200-step production experiment is justified for the current
softening form.

