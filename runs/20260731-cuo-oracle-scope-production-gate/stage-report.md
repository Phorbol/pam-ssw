# CuO equal-budget production gate: proposal LS versus oracle-only LS

## Frozen comparison

This gate follows the fixed-task root-cause result in
`20260731-cuo-safe-lbfgs-line-search-root-cause`.  Both arms use:

- the exact same true-PES-quenched CuO seed-42 minimum;
- 104 shared-bootstrap force evaluations, charged to each arm;
- the packaged CuO finetuned MACE model in float32 on CUDA;
- classic Metropolis starter selection;
- the same direction candidates, Gaussian bias policy, Safe-LBFGS, true
  quench, geometry checks, random seed, and 20,000 total-FE budget.

The only algorithmic config difference is:

```text
current: local_softening_scope = both
candidate: local_softening_scope = oracle
```

Artifact output paths differ by arm and are not algorithm parameters.  The
runner checks this invariant after execution.  Execution commit:
`84cacc78fe72ad34bddb4c2ff1f5413ba2d6f985`.

## Result

| Metric | current `both` | `oracle` | Effect |
|---|---:|---:|---:|
| Total FE | 20,000 | 20,000 | identical |
| Unattributed FE | 0 | 0 | identical |
| Best energy, eV | -201.672668 | **-201.840729** | oracle lower by 0.168060 |
| Drop from shared bootstrap, eV | 2.995270 | **3.163330** | +0.168060 |
| Completed macro trials | 18 | **32** | +77.8% |
| Archive minima | 19 | **33** | +73.7% |
| Proposal relaxations | 76 | **166** | +118.4% |
| Proposal FE | 17,153 | **15,173** | -1,980 (-11.5%) |
| Proposal FE share | 85.765% | **75.865%** | -9.90 points |
| Proposal FE per relaxation | 225.70 | **91.40** | -59.5% |
| Proposal line-search failures | 74/76 | **93/166** | 97.37% -> 56.02% |
| Converged proposal relaxations | 0/76 | **73/166** | 0% -> 43.98% |
| Mean accepted iterations | 100.0 | **35.44** | -64.6% |
| Median accepted iterations | 73.5 | **30.5** | -58.5% |
| P90 accepted iterations | 230 | **80** | -65.2% |
| Search wall time, s | 611.97 | 608.99 | same fixed-FE time |

The equal wall time is expected: both arms consume the same number of CuO MACE
evaluations.  Oracle-only does not make an individual MACE call cheaper.  It
redirects 1,980 evaluations away from the proposal optimizer and completes 14
additional direction-to-quench search cycles in the same physical time.

The oracle arm still has a 56% proposal line-search-failure rate.  Therefore
removing proposal LS solves the dominant CuO coupling but does not make the
remaining true-PES-plus-Gaussian objective numerically ideal.  Further Armijo
tuning is not opened here: the candidate already spends fewer total rejected
line trials (8,853 versus 9,373) while processing more than twice as many
proposal tasks.

## Cross-system reconciliation

The earlier 72-case fixed-starter scope factorial gives the following median
terminal landing changes:

- C60: oracle -1.610 eV versus current both -0.989 eV;
- PdO: oracle -0.759 eV versus current both +0.006 eV.

The present CuO full-search result points in the same direction.  Thus deleting
proposal LS is supported across the three available systems.  This does not
prove that oracle LS itself is useful: against fully unsoftened directions,
C60 favored `none` while PdO favored `oracle`.  The remaining scientific
question is now cleanly isolated as `oracle` versus `none`, i.e. whether the LS
Hessian modification improves direction selection after its force field has
been removed from coordinate propagation.

## Decision and user-facing recommendation

The gate passes.  For new long CuO/PdO/C60 research runs on this branch, use:

```yaml
local_softening_scope: oracle
proposal_optimizer: safe-lbfgs-total
```

Keep all existing LS strength, active-pair, cutoff, Gaussian-bias, and optimizer
parameters unchanged until an independent gate addresses them.

The existing named C60 validated profile is not silently mutated.  It is tied
to a completed 200-step run with `scope=both`; changing it would falsify its
provenance.  The oracle-only setting is therefore an explicit promoted
cross-system candidate on the research branch, not yet a package-wide
`LSSSWConfig` default.

## Next bounded question

Run only `oracle` versus `none` with proposal LS absent.  That gate evaluates
the LS contribution to direction generation itself.  Do not add new pair
selection, posterior selection, TS/UCB, strength schedules, or quadratic
propagators until this remaining LS question is answered.

## Evidence boundary

The full-search result is one CuO seed.  Its large numerical-efficiency effect
is independently explained by the fixed-task mechanism and agrees with the
C60/PdO fixed-starter medians.  The 0.168 eV best-energy advantage is supportive
but not a system-general quality estimate.  No thermodynamic or reaction-network
claim is made.
