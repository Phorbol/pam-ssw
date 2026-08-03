# 20k production-kernel starter-cell gate: seeds 42--44

## Question and frozen comparison

Does replacing uniform sampling over all retained minima with a full-support,
MACE-FPS cell-first distribution improve the frozen production LS-SSW kernel
at the same force-evaluation budget?

Only the outer starter distribution changes. Direction generation, local
softening, serial Gaussian-bias uphill propagation, proposal relaxation, true
quench, structure matching, and archive insertion remain frozen.

Execution commits:

- seed 42: `15f6d44caaed298c2474714f991e71807ca7e57d`;
- seeds 43 and 44: `e9a9c2fb6a11ef6272a90e98726ed9fed1e6f800`.

The commits are scientifically comparable: the intervening commit only records
the seed-42 output and conclusion and does not change algorithm code.

Each campaign uses a 20,000 force-evaluation total budget, a 1,000
force-evaluation per-action cap, one worker, full archive support, no archive
deletion, and 16 MACE-FPS cells. `Gain AUC` is the best-energy improvement
integrated over the complete 20,000-FE axis and divided by 20,000.

## Fixed-budget results

| System | Seed | Policy | Actions | Archive | Duplicate rate | Best energy (eV) | Energy drop (eV) | Gain AUC (eV) | Total FE | Wall (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60 | 42 | uniform | 43 | 42 | 0.0455 | -488.786011 | 14.116394 | 13.024437 | 19,062 | 351.20 |
| C60 | 42 | cell-first | 45 | 44 | 0.0435 | -488.369476 | 13.699707 | 12.362872 | 19,683 | 349.27 |
| C60 | 43 | uniform | 38 | 36 | 0.0769 | -497.070496 | 22.400757 | 16.259338 | 19,093 | 365.18 |
| C60 | 43 | cell-first | 41 | 39 | 0.0714 | -494.172699 | 19.502991 | 16.070332 | 19,047 | 355.66 |
| C60 | 44 | uniform | 43 | 42 | 0.0455 | -491.476349 | 16.806732 | 12.624658 | 19,307 | 361.26 |
| C60 | 44 | cell-first | 44 | 42 | 0.0667 | -494.354675 | 19.684967 | 11.964803 | 19,231 | 361.50 |
| PdO | 42 | uniform | 72 | 65 | 0.1096 | -573.070923 | 4.674072 | 3.601366 | 19,232 | 461.28 |
| PdO | 42 | cell-first | 70 | 62 | 0.1268 | -574.357056 | 5.959961 | 4.117418 | 19,112 | 458.35 |
| PdO | 43 | uniform | 68 | 60 | 0.1304 | -572.708008 | 4.311035 | 3.559322 | 19,286 | 466.93 |
| PdO | 43 | cell-first | 71 | 62 | 0.1389 | -573.006104 | 4.609131 | 4.159227 | 19,105 | 483.82 |
| PdO | 44 | uniform | 73 | 58 | 0.2162 | -575.226685 | 6.829712 | 6.144845 | 19,054 | 464.25 |
| PdO | 44 | cell-first | 67 | 60 | 0.1176 | -572.446167 | 4.049255 | 3.503759 | 19,024 | 466.85 |

Paired effects are cell-first minus uniform; positive values favor cell-first.

| System | Metric | Seed 42 | Seed 43 | Seed 44 | Mean | Median | Positive |
|---|---|---:|---:|---:|---:|---:|---:|
| C60 | final improvement (eV) | -0.416687 | -2.897766 | +2.878235 | -0.145406 | -0.416687 | 1/3 |
| C60 | gain AUC (eV) | -0.661565 | -0.189006 | -0.659855 | -0.503475 | -0.659855 | 0/3 |
| PdO | final improvement (eV) | +1.285889 | +0.298096 | -2.780457 | -0.398824 | +0.298096 | 2/3 |
| PdO | gain AUC (eV) | +0.516052 | +0.599906 | -2.641086 | -0.508376 | +0.516052 | 2/3 |

## Accounting and implementation audit

An independent replay of every event log verified:

- benchmark eligibility is true and there are no failed attempts;
- recorded purpose counts sum exactly to each action and campaign total;
- used plus unused evaluations closes exactly to 20,000;
- `unattributed == 0`;
- policy probabilities are positive, normalized, and match the immutable
  pre-action snapshot;
- cell partitions are exhaustive and non-overlapping, with full archive
  support and no deletion.

MACE descriptor construction is not a meaningful runtime bottleneck here:

- C60 cell-first seeds 42--44: 0.782, 0.696, and 0.740 s;
- PdO cell-first seeds 42--44: 1.305, 1.398, and 1.250 s.

The final archives contain only 36--65 minima. This experiment therefore does
not test the asymptotic thousands-of-arms regime, and it provides no evidence
that present performance is limited by large-pool selector dilution.

## Decision

Reject MACE-FPS hard cell-first sampling as a general production starter
selector.

It passes the implementation, accounting, full-support, and online-integration
gates, but not the algorithmic promotion gate:

- C60 gain AUC is worse in all three paired seeds;
- PdO is high-variance, with a large seed-44 reversal that makes both mean
  effects negative;
- duplicate rate does not improve consistently;
- the extra representation has negligible cost but no robust search gain.

No UCB/Thompson-sampling layer, PCA threshold, cell weighting, top-k deletion,
or reward tuning should be added to rescue this negative result. Uniform
full-support starter sampling is retained as the clean outer-loop baseline for
the next inner-kernel mechanism gate.

