# Starter-energy residual audit

## Question

After the completed direction, uphill-propagation, and relaxation gates, is the
remaining global-minimum plateau better explained by a failure to leave basins,
or by allocating expensive SSW actions to physically unproductive starter
minima?

## Frozen evidence

This audit reads the six completed 20,000-force-evaluation uniform-starter
campaigns from the C60/PdO starter-cell gate: two systems, seeds 42--44.  A
single force evaluation means one energy-and-force calculation at one atomic
geometry.  The audit adds no force evaluation.

Uniform starter selection is essential here.  At every action, each archived
minimum had equal probability of being selected.  The analysis therefore
measures an association between a starter's physical state and its observed
outcome without inheriting the present archive-UCB-like selector's preference.
It does not manufacture counterfactual outcomes for starters that were not
run.

## Result

Across 337 completed SSW actions and 114,773 exactly attributed action force
evaluations:

| system | actions | new archive minima | global-best improvements |
|---|---:|---:|---:|
| C60 | 124 | 117 | 17 |
| PdO | 213 | 180 | 19 |

The current direction and uphill machinery therefore crosses into distinct
basins frequently.  The residual problem is not a general inability to escape
the starting basin.

Global-best improvements were strongly concentrated near the low-energy floor
of the archive:

| system | current-best starter | every other starter | top energy quartile | lower 75% |
|---|---:|---:|---:|---:|
| C60 | 9/13 | 8/111 | 11/39 | 6/85 |
| PdO | 9/16 | 10/197 | 12/67 | 7/146 |

This is not only an early-search artifact.  After attempt index 10:

| system | improvements from top energy quartile | improvements from lower 75% |
|---|---:|---:|
| C60 | 6/29 | 0/65 |
| PdO | 4/52 | 3/131 |

Conditioning on the exact action indices at which improvements occurred and on
the archive size at those times, a no-energy-rank-effect null predicts a mean
selected rank fraction of 0.5.  The observed nontrivial-improvement means are
0.322 for C60 and 0.245 for PdO.  The exact Poisson-binomial upper-tail
probabilities for observing at least as many current-best starters at those
same improvement times are 0.031 for C60 and 0.021 for PdO.

## Physical interpretation

The Gaussian-bias walk usually supplies enough collective displacement to
leave a basin, and the final true-PES quench usually lands in a distinct
minimum.  However, most such minima lie above the best structure already
known.  A low-energy starter is more likely to lie on the floor of a productive
funnel: an escape from it can reveal another still-lower packing or
reconstruction.  Spending the same expensive action on a high-energy,
weakly-connected archive member more often creates diversity without lowering
the landscape floor.

This does **not** prove that always choosing the lowest-energy structure is
optimal.  Such a rule can repeatedly probe one funnel and destroy full support.
It also does not prove that the existing fixed-weight archive-UCB-like score is
correct.  The evidence isolates only one robust physical variable: starter
energy rank matters in the same direction for both systems.

## Decision

The next live force-evaluation gate is admitted at the starter-selection layer.
It should compare:

1. uniform selection over every archived minimum, as the full-support
   exploration reference;
2. the existing archive-UCB-like selector, frozen exactly as implemented;
3. the classic SSW Metropolis chain, which continues from a newly reached
   lower-energy minimum and accepts an uphill landing with its Boltzmann
   probability.

All direction construction, local softening, Gaussian-bias propagation,
proposal relaxation, true-PES quench, random seeds, and force budgets must be
frozen.  This is a baseline comparison, not permission to add Thompson
sampling, MACE-feature encoding, hard top-k pruning, or a new acquisition
formula.

The classic Metropolis chain is sequential and therefore is not yet a proposed
parallel production selector.  Its role is mechanistic: test whether preserving
local funnel continuity and an energy-dependent acceptance rule explains the
observed low-energy-starter advantage.  Only if it wins across C60 and PdO
should a parallel, full-support approximation be designed.

## Reproduce

From the repository root:

```bash
python runs/20260731-starter-energy-residual-audit/analyze.py
```

The script validates uniform policy support, exact force accounting, attempt
counts, and source-file hashes before writing `evidence.json`.
