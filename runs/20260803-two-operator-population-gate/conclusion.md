# Two-operator population gate: Stage A/B conclusion

## Outcome

Stage A admitted a fresh paired experiment from a zero-new-force-evaluation
replay. Stage B then completed the exact frozen cohort: three systems, two
starter contexts, three seeds, 18 paired inputs, and 36 terminal actions.
Every action produced a force-converged, geometry-valid certificate. The
global purpose ledger closes at 9,147 force evaluations with zero unattributed
work and no budget censoring.

The preregistered code decision is:

```text
ADMIT_STAGE_C_DESIGN
direct_viability_contexts: 2
ssw_exclusive_support_contexts: 4
numerical_acceptability: true
```

That formal decision is preserved in `evidence.json`. It is not, however, a
valid physical admission of the direct operator. A zero-new-FE audit of the
saved coordinates found that all five direct actions labelled as non-starter
landings were separated from the starter only by the archive energy predicate.
Their archive RMSD was 0.0089--0.0209 A, far below the configured 0.4 A
tolerance, while their energy changes of 1.19--3.11 meV just exceeded the
0.001 eV deduplication tolerance. The direct viability rule therefore passed
on five energy-only splits, not on five demonstrated barrier crossings.

The scientific disposition is consequently **do not advance to Stage C yet**.
This is a measurement-validity failure, not a post-hoc change of the gate
thresholds and not evidence that a two-operator population is useless.

## Physical experiment

Each pair used the same frozen starter, initial direction, displacement scale,
potential, constraints, and final true-PES quench.

The direct arm applied only

\[
x_{\mathrm{direct}} = x_0 + \sigma u
\]

before the true-PES quench. The SSW arm started from the same displacement but
then accumulated Gaussian bias and repeatedly relaxed on the biased surface
before removing the bias and quenching on the true surface.

The resulting physical picture is simple. A large instantaneous displacement
does not by itself create an escape: the true-PES gradient usually carries the
structure back into the original attraction basin. The serial biased path
suppresses this return channel long enough for the remaining degrees of
freedom to accommodate the displacement and cross a basin boundary. In this
cohort the direct landing remained within 0.021 A RMSD of its starter whenever
the archive called it new. SSW produced 17 formal non-starter landings, 16 of
which also crossed the configured geometric RMSD boundary.

This supports the causal value of biased proposal relaxation. It does not show
that the current adaptive Gaussian schedule, H8 horizon, or optimizer is
optimal.

## System-level evidence

All costs below are fully loaded action costs: the shared initial-direction
cost is charged to either arm when comparing that arm as a standalone action.

| system | formal direct non-starter | SSW non-starter | direct median FE | SSW median FE | direct median delta-E (eV) | SSW median delta-E (eV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C60 | 0/6 | 5/6 | 35.5 | 348.5 | +0.000015 | +0.302811 |
| PdO | 2/6 | 6/6 | 39.0 | 360.5 | -0.000244 | +1.101990 |
| CuO | 3/6 | 6/6 | 39.0 | 635.5 | -0.000610 | -0.452408 |

The two PdO and three CuO direct counts in this table are exactly the five
energy-only false basin splits described above. C60 already showed the cleaner
behavior: all six direct perturbations quenched back to the starter, whereas
five SSW paths reached other structures.

SSW is an exploration operator, not an energy-monotone minimizer. It found
substantially lower landings in some C60, PdO, and CuO pairs, but also higher
landings, especially from the later PdO and C60 starters. The observed SSW
landing changes span -5.192 to +4.085 eV. This wide distribution is the cost of
crossing basin boundaries and is precisely why terminal energy alone is not a
clean label for direction quality.

## Cost anatomy

The exact global ledger is:

| purpose | force evaluations | share of total |
| --- | ---: | ---: |
| shared/staged direction oracle | 1,248 | 13.6% |
| biased proposal relaxation | 6,281 | 68.7% |
| true-PES escape checks | 115 | 1.3% |
| terminal true-PES quench | 1,449 | 15.8% |
| starter validation | 18 | 0.2% |
| post-relax validation | 36 | 0.4% |
| unattributed | 0 | 0.0% |

The direct arms used 442 exclusive FE, including 424 terminal-quench FE. The
SSW arms used 8,447 exclusive FE: 6,281 for biased relaxation, 1,008 for
direction work after removing the shared prefix, 115 for escape checks, 1,025
for terminal quench, and 18 for validation. Fully loaded totals were 682 FE for
direct and 8,687 FE for SSW.

The cheaper direct propagation was not cancelled by a more expensive final
quench: its quench was also cheaper. The issue is physical support, not hidden
cost. Direct perturbation was about 9--16 times cheaper in median FE by system,
but its apparent five escapes were local refinement artefacts. Conversely,
biased proposal relaxation is the dominant SSW cost and the dominant source of
genuine basin displacement. This identifies proposal propagation as the
correct scientific optimization target after basin labels are repaired.

The summed per-pair execution telemetry was 209.7 seconds on an NVIDIA GeForce
RTX 3060. Pooled fully loaded medians were 0.78 seconds for direct and 8.58
seconds for SSW. Wall time is environment-specific telemetry; force evaluations
remain the comparison budget.

## Claim ceiling

This is a deterministic, paired, three-seed mechanism gate under one MACE
model and six frozen starter contexts. It establishes that:

- the accounting adapter closes exactly under real GPU execution;
- the current SSW path repeatedly produces large structural displacements
  that direct displacement plus quench does not reproduce;
- biased proposal relaxation, not terminal quench, consumes most SSW force
  evaluations;
- the current energy-and-RMSD archive predicate can turn millielectronvolt
  relaxation drift into a false new-basin label.

It does not establish production superiority, optimal operator allocation,
long-run minima discovery, transfer across potentials, or a valid Bayesian
posterior. It also does not justify changing the production default.

```text
production_default_changed: false
```

## One next action

Run one measurement-validity repair of the same 18-pair cohort: true-quench
each frozen starter once with the identical terminal optimizer before the two
arms branch, charge that bootstrap once to the pair, and require a geometric
basin change between equally quenched endpoints rather than allowing an energy
difference alone to create a new basin. Freeze this definition before rerun
and keep systems, starters, seeds, directions, SSW configuration, and all
budgets unchanged. Stage C batching remains blocked until direct viability is
re-evaluated under that corrected observable.
