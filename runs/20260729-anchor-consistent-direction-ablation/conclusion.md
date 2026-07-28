# C60 anchor-consistent direction ablation

## Question and claim boundary

This experiment isolates one direction-oracle question:

> Is the late C60 plateau primarily caused by the current Ritz direction being
> detached from the random-plus-bond anchor that initializes the SSW walk?

The starter, Gaussian bias, local softening, safe-L-BFGS proposal relaxation,
strict true-PES quench, archive behavior, and random seeds were frozen.  No
starter selector, UCB-like rule, Thompson sampler, posterior update, checkpoint
shooting, or reaction-network objective was active.

The evidence is a descriptive paired three-seed audit, not a statistically
powered promotion decision.  No production default is changed.

## Preregistered cohort

- System: C60 with the locked MACE model
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`.
- Starters: the exact accepted minima at trials 100 and 180 of the locked
  200-trial trajectory.
- Seeds: 42, 43, and 44.
- Arms:
  - `detached_ritz`: the existing two-vector detached block-Krylov solve,
    depth 6, 12 HVPs per selection.
  - `exact_anchor`: the normalized random-plus-bond anchor itself, one HVP per
    selection for the curvature-conditioned uphill step.
  - `anchor_lanczos`: a one-vector Krylov solve seeded by that exact anchor,
    depth 12, 12 HVPs per selection.
- Meaningful outcome: a strictly certified new basin whose landing energy is
  at least 0.001 eV below its starter.
- Bootstrap was skipped because both starters are already locked accepted
  minima.  Bootstrap cost is therefore exactly zero.

## Execution integrity

- Execution commit:
  `a0a7948706abd863d167f3435a675b050bb30f42`.
- Completed cases: 18/18.
- Strict terminal-quench certificates: 18/18.
- Quench fallbacks: 0/18.
- Independently revalidated structure hashes: 54/54.
- Recomputed evidence is byte-for-byte equivalent as structured data.
- Unattributed force evaluations: 0.
- The scientific outputs were completely written.  The outer shell was
  interrupted after the final JSON had been emitted because process cleanup
  stalled; it therefore returned 130.  All conclusions below use the
  independently reloaded and revalidated files, not the wrapper exit code.

## Terminal outcomes

| Starter | Seed | Direction arm | Landing minus starter (eV) | New basin | Force evals | Direction | Proposal relax | True quench |
|---|---:|---|---:|:---:|---:|---:|---:|---:|
| intermediate | 42 | detached Ritz | +0.000122 | no | 348 | 144 | 162 | 33 |
| intermediate | 42 | exact anchor | +15.503235 | yes | 401 | 10 | 289 | 93 |
| intermediate | 42 | anchor Lanczos | +0.000092 | no | 459 | 192 | 239 | 16 |
| intermediate | 43 | detached Ritz | +5.673767 | yes | 461 | 168 | 242 | 40 |
| intermediate | 43 | exact anchor | +0.000092 | no | 173 | 6 | 126 | 35 |
| intermediate | 43 | anchor Lanczos | +0.000092 | no | 450 | 192 | 232 | 14 |
| intermediate | 44 | detached Ritz | +0.000061 | no | 427 | 192 | 210 | 13 |
| intermediate | 44 | exact anchor | +11.702667 | yes | 412 | 10 | 244 | 149 |
| intermediate | 44 | anchor Lanczos | +0.000061 | no | 433 | 192 | 211 | 18 |
| plateau | 42 | detached Ritz | -0.676758 | yes | 393 | 120 | 226 | 38 |
| plateau | 42 | exact anchor | +0.366943 | yes | 380 | 8 | 298 | 66 |
| plateau | 42 | anchor Lanczos | +0.000122 | no | 367 | 96 | 238 | 25 |
| plateau | 43 | detached Ritz | -7.078796 | yes | 469 | 192 | 234 | 31 |
| plateau | 43 | exact anchor | +0.176056 | yes | 533 | 16 | 392 | 113 |
| plateau | 43 | anchor Lanczos | -7.079010 | yes | 447 | 192 | 231 | 12 |
| plateau | 44 | detached Ritz | +0.358612 | yes | 448 | 120 | 274 | 46 |
| plateau | 44 | exact anchor | +3.062134 | yes | 311 | 8 | 214 | 82 |
| plateau | 44 | anchor Lanczos | -0.000092 | no | 487 | 168 | 271 | 38 |

Aggregate outcomes:

| Direction arm | New basins | Meaningful lower basins | Median landing delta (eV) | Median force evals | Total force evals |
|---|---:|---:|---:|---:|---:|
| detached Ritz | 4/6 | 2/6 | +0.000092 | 437.5 | 2,546 |
| exact anchor | 5/6 | 0/6 | +1.714539 | 390.5 | 2,210 |
| anchor Lanczos | 1/6 | 1/6 | +0.000076 | 448.5 | 2,643 |

The only useful anchor-Lanczos event, plateau seed 43, reaches essentially the
same lower basin as detached Ritz (-7.079010 versus -7.078796 eV).  It uses 447
instead of 469 total force evaluations, but this single paired difference is
not a general speed result.

## What the direction diagnostics show

| Direction arm | Selections | Median absolute anchor cosine | Median true curvature | Median participation ratio | Median Ritz residual |
|---|---:|---:|---:|---:|---:|
| detached Ritz | 39 | 0.0677 | 4.0456 | 29.70 | 4.4105 |
| exact anchor | 29 | 1.0000 | 38.4610 | n/a | n/a |
| anchor Lanczos | 43 | 0.2269 | 2.1729 | 23.28 | 2.0504 |

The exact random-plus-bond anchor is not a soft escape direction.  Its median
true-PES curvature is about 9.5 times the detached-Ritz value and 17.7 times
the anchor-Lanczos value.  It often escapes cheaply, but all five new basins
are higher in energy, including +15.50 and +11.70 eV failures.

Seeding Lanczos from the anchor does exactly what its mathematics predicts:
it increases anchor continuity, lowers the Rayleigh quotient, and reduces the
Ritz residual.  That cleaner eigensolve does not improve the terminal search
outcome: it produces fewer meaningful lower basins than the detached control
and fails all three intermediate-start cases.

## Cost accounting

| Purpose | Force evals | Share |
|---|---:|---:|
| biased proposal relaxation | 4,333 | 58.56% |
| direction oracle | 2,026 | 27.38% |
| landing true quench | 862 | 11.65% |
| escape true-PES checks | 160 | 2.16% |
| post-relax validation | 18 | 0.24% |
| bootstrap, starter quench, unattributed | 0 | 0.00% |
| total | 7,399 | 100.00% |

Measured sequential section time was 138.97 s for direction plus proposal
generation and 14.43 s for terminal quenching, 153.40 s in total.  This is
hardware- and warm-up-dependent wall time; the first detached case includes
MACE/CUDA warm-up and should not be interpreted as an arm-level timing
advantage.

The exact-anchor arm saves 878--974 direction force evaluations relative to
the refined arms, but spends more on harmful proposal relaxation and true
quenching.  A cheap direction is not a cheap successful transition.

## Decision

1. Do not promote `exact_anchor` or `anchor_lanczos`.
2. Keep the current detached-Ritz mode as the experimental control, not
   because it is solved theoretically, but because it has the best observed
   terminal outcome in this cohort.
3. Do not spend the next iteration on starter UCB/TS.  All three arms fail to
   produce a meaningful lower basin from the intermediate starter, so the
   present bottleneck is still direction/event quality rather than starter
   allocation.
4. Do not simply deepen Lanczos.  The anchor-seeded solve is already softer
   and has a smaller residual without improving the outcome.
5. The next clean analysis should reuse the already paid Krylov subspace and
   expose its Ritz spectrum and anchor overlaps.  The unresolved physical
   tradeoff is not “which eigensolver is more advanced,” but whether a
   direction can retain localized bond/event intent while avoiding the very
   high curvature of the raw anchor.  Only after observing that
   curvature-overlap frontier should one preregister a new selection rule or a
   CBD/dimer-style constrained rotation.
