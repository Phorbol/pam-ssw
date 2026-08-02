# Shared step-1 direction counterfactual conclusion

## Decision

- Direction-source prior gate: **closed**.
- Static-score posterior/UCB/TS gate: **closed**.
- Delete all-candidate continuation HVPs: **not supported**.
- Production direction policy: **unchanged**.
- Raw execution: commit `4fe9ecc3ec7af7aa0a71a7ca6a430009be36f0c4`,
  RTX 3060, MACE 0.3.14, Torch 2.8.0/CUDA 12.8, float32.

The formal gate used 27,310 force evaluations and 635.69 s wall time. Seven of
nine shared pools reached micro-step 1, producing 56 full H8 remainder plus
true-quench trajectories. Every landing has a strict force certificate; there
were no invalid landing geometries, fragmented structures, or unattributed
force evaluations. PdO seeds 52 and 53 terminated during the shared step-0
prefix and were retained as right-censored pools rather than replaced.

## Why an exact continuation snapshot was necessary

The first replay implementation reran the shared step-0 proposal relaxation in
every arm. A CUDA diagnostic showed identical Gaussian-bias parameters but up
to 0.00524 Å coordinate drift between reruns. Normalizing the resulting small
relaxation displacement changed the momentum direction by an L2 norm of
0.00474, despite a cosine of 0.9999888. A hash tolerance would have hidden a
real difference in the branching state.

The final gate therefore executes step 0 once, freezes coordinates, accumulated
bias, prior selected and realized directions, trust scales, true-PES endpoint,
anchor, RNG state, and completed trace, then resumes every arm from that exact
snapshot. All eight arms in every usable pool have identical step-1 coordinate,
bias, and candidate-direction hashes. The pause/resume path is analytically
tested against an uninterrupted walk and preserves both coordinates and total
force-evaluation cost.

## Physical result by system

| System | Usable pools | Repeat-stable best pools | Stable static misses | Median static-score/terminal Spearman, repeats 0/1 | Terminal repeat drift, median / max (eV) |
|---|---:|---:|---:|---:|---:|
| C60 | 3 | 1 | 1 | -0.60 / -0.20 | 0.0529 / 6.1487 |
| PdO | 1 | 0 | 0 | +0.80 / +0.40 | 0.0011 / 0.0025 |
| CuO | 3 | 2 | 2 | 0.00 / +0.60 | 0.0013 / 1.8218 |

C60 contains clear alternatives to the current winner, but the identity of the
best terminal candidate is repeat-stable in only one of three pools. PdO's only
usable pool strongly favors the current momentum choice, while two earlier
prefixes never reach the branch point. CuO has two stable static misses, but its
winning family is not common to C60 and PdO. There is therefore neither a
cross-system dominant direction source nor a repeat-stable universal static
selector failure.

Immediate micro-step-1 true-PES responses are much more reproducible than the
final H8 landing. Median repeat drift of the immediate energy change is 0.0110
eV for C60, 0.0168 eV for PdO, and 0.00024 eV for CuO, whereas terminal maxima
reach 6.15 eV for C60 and 1.82 eV for CuO. The later biased relaxations and
direction reselections amplify small numerical changes into different basins.
Consequently, a terminal landing is a distributional delayed-credit label for
one step-1 direction, not a clean deterministic classifier target.

## The static scorer's actual mathematical content

For every usable pool, adaptive score sigma is unclipped and the candidate
curvatures are positive. Hence

\[
\sigma_i=s\sqrt{2E_*/\kappa_i},\qquad
\frac12\sigma_i^2\kappa_i=s^2E_*.
\]

The nominal quadratic energy term is therefore identical for all four
candidates. The largest observed within-pool spread is only
`4.44e-16 eV`. Curvature does not rank these candidates; it sets their proposed
step scales. Ranking is left to continuity, anchor, novelty, and damage terms.
Because the previous realized displacement has zero continuity penalty, the
momentum candidate is static rank 1 in all seven usable pools.

This is not evidence that HVPs are wholly redundant. The selected direction
still needs curvature to determine displacement and bias strength, and
curvature-dependent novelty probes can affect the residual score. It does show
that describing the current adaptive static score as a soft-curvature ranker is
misleading: its explicit quadratic curvature contribution is algebraically
normalized away.

## Continuation-HVP value of information

The complete pool permits a zero-new-FE diagnostic projection. `uniform` and
`family rotation` pay only one selected central HVP (2 FE); current static K4
pays all four (8 FE).

| System | Current static median regret / FE | Uniform median regret / FE | Family rotation median regret / FE |
|---|---:|---:|---:|
| C60 | 5.4275 eV / 491.0 | 3.9469 eV / 475.6 | 4.0781 eV / 468.8 |
| PdO | 0.0003 eV / 273.5 | 1.3413 eV / 211.6 | 1.5398 eV / 221.7 |
| CuO | 0.4128 eV / 727.5 | 0.5914 eV / 490.8 | 0.4960 eV / 533.1 |

Removing full-pool HVP ranking helps C60's median regret, seriously harms the
only usable PdO pool, and is mixed for CuO. This reproduces the earlier
initial-direction sign reversal at a continuation state. A universal
selected-only-HVP production change remains unsupported.

## What this closes and what remains

The result closes three tempting but unjustified moves:

1. assigning a fixed positive prior to momentum, bond, or random directions;
2. fitting UCB-like weights, Thompson sampling, or a classifier to these seven
   contexts and noisy terminal winners;
3. deleting all unselected HVPs because the quadratic score term is constant.

It also identifies the next scientifically meaningful boundary. Direction
learning needs two distinct labels: a local, causal response label describing
whether the chosen direction produces a valid and calibrated immediate PES
move, and a macro-action value distribution over later biased propagation and
quench. Collapsing them into one terminal scalar is the present credit-assignment
bottleneck. Before any posterior model, the next gate must define a stable
macro action or a distributional repeated-rollout target; it must not tune the
current static weights on this dataset.

The exact continuation snapshot is retained because it enables genuine batch
branching from one paid physical prefix. It is infrastructure for causal
parallel experiments, not a new production search heuristic.
