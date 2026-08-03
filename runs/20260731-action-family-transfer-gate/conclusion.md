# Action-family transfer gate conclusion

## Decision

Do not add an online posterior, Thompson sampler, UCB coefficient, classifier,
or action-family weight over `D0_exact_anchor`, `D1_anchor_krylov_d2`, and
`K4_discrete`.

The preregistered leave-one-system-out gate fails. C60 selects K4 by mean
terminal energy, while PdO selects D0. Training on PdO therefore selects D0
for C60, where it is 1.919427 eV worse than equal-arm sampling. Training on
C60 selects K4 for PdO and improves over the equal-arm mean by only 0.078410
eV. Both transfer directions were required to pass.

## Execution

- 36/36 fixed-starter cases completed.
- 36/36 terminal quenches have strict force certificates.
- 0 fragmented landings.
- 0 unattributed force evaluations.
- 12/12 `(system, state, seed)` groups used exactly one shared initial-anchor
  hash across all three arms.
- Total cost: 12,019 force evaluations and 245.6 measured wall seconds.

| System | Arm | Mean landing minus starter (eV) | Median (eV) | Mean FE | Lower landings |
|---|---|---:|---:|---:|---:|
| C60 | D0 exact anchor | +3.921748 | +1.619049 | 372.3 | 2/6 |
| C60 | D1 finite softening | +1.228475 | +0.088135 | 408.5 | 1/6 |
| C60 | K4 discrete | +0.856740 | +3.617722 | 399.7 | 2/6 |
| PdO | D0 exact anchor | +0.454854 | -0.075439 | 324.3 | 3/6 |
| PdO | D1 finite softening | +1.035970 | +0.841309 | 289.7 | 2/6 |
| PdO | K4 discrete | +0.627797 | +0.209167 | 208.7 | 2/6 |

## Mechanism

C60 is maximally context-dependent in this cohort: D0, D1, and K4 each win
two of the six paired contexts. PdO splits between D0 and K4 at three wins
each; D1 never wins.

D1 does soften the raw intent without introducing a Dimer angle or cone
parameter. In two high-energy C60 intermediate cases it reduces D0's harmful
landing energy, but it does not become the best transferable escape rule.
Its extra curvature work also fails to reduce total action cost: on C60 its
mean total cost is higher than both D0 and K4 because proposal relaxation and
terminal quench dominate the saved direction evaluations.

K4's low mean C60 result is tail-driven: its median landing is +3.617722 eV,
but one plateau case reaches -9.030243 eV. This is another reason not to fit a
small-sample posterior to mean reward and call it a stable policy.

The negative result rejects a context-free posterior over these three action
families. It does not prove that all contextual models are impossible. A
contextual posterior should only be reconsidered after a held-out-system
feature gate predicts which of D0 and K4 wins without using terminal force
evaluations. D1 is closed at this fixed one-expansion setting; no depth sweep
is justified by these data.

## Next priority

Do not tune TS versus UCB. The next bounded statistical question is whether
the D0-versus-K4 context switch is predictable from pre-action information
already available at zero additional force cost. Candidate inputs are limited
to starter energy rank, cached pooled MACE representation, anchor
localization, and random/bond composition. If that held-out-system gate
fails, retain stochastic D0/K4 allocation across independent walkers and use
parallelism for coverage rather than learned action selection.
