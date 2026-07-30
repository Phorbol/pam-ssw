# Starter representation and arm-proliferation audit

## Scope and evidence boundary

This audit does not modify or run the production walker.  It replays the saved
new-minimum structures from four seed-42, 200-trial campaigns:

- C60 fixed-intent Ritz and transported-direction;
- PdO fixed-intent Ritz and transported-direction.

The corpus contains 306 saved C60 minima and 388 saved PdO minima.  The
campaigns saved every new minimum and its starter ID, but did not serialize
per-node `node_trials` or the starter identity of duplicate/failed trials.
Consequently:

- archive growth and new-minimum label saturation are exact;
- structure and representation metrics are exact for the saved minima;
- recorded productive-transition metrics are censored diagnostics;
- the audit cannot rank online starter policies.

The complete machine-readable result is `results/evidence.json`; the four
representation matrices are stored in the two compressed NPZ files beside it.

## S0: arm proliferation is already severe

| system/arm | trials | final archive arms | new arms/trial | trials/final arm | best-case Beta(1,1) posterior std |
|---|---:|---:|---:|---:|---:|
| C60 fixed Ritz | 200 | 123 | 0.610 | 1.626 | 0.232 |
| C60 transported | 200 | 185 | 0.920 | 1.081 | 0.248 |
| PdO fixed Ritz | 200 | 195 | 0.970 | 1.026 | 0.249 |
| PdO transported | 200 | 195 | 0.970 | 1.026 | 0.249 |

The Beta(1,1) prior standard deviation is 0.289.  Even the unattainable
best-case assumption that all trials are distributed uniformly over the final
arms only reduces it to 0.232--0.249.  Actual UCB/TS allocation is less even.
This confirms that a node-as-arm posterior has too little repeated evidence to
concentrate on these campaigns.

For reference, if all final arms were still prior-only, the expected maximum
of independent Beta(1,1) Thompson draws would be 0.992--0.995.  This is not a
claim about the actual unobserved-arm count; it quantifies why the maximum of a
large prior-only action set is dominated by prior sampling extremes.

## S0: the legacy selector also has a real scaling problem

The existing `BanditSelector` was timed without changing its implementation,
using its complete score path and up to 1000 archive prototypes:

| archive entries | prototypes | one selection, s |
|---:|---:|---:|
| 100 | 100 | 0.0510 |
| 300 | 300 | 0.3895 |
| 1000 | 1000 | 4.2369 |

The measured log-log slope is 1.92.  The source of the near-quadratic behavior
is not UCB itself: normalized archive energy is rescanned for each candidate,
and descriptor density is evaluated twice per candidate against the prototype
set.

The production campaigns used about 0.019--0.026 wall seconds per force
evaluation, including all surrounding overhead.  At 1000 entries, one legacy
selection therefore costs roughly the wall time of 165--223 such evaluations.
This timing is one synthetic CPU measurement, not a cross-hardware benchmark,
but it is sufficient to reject the assumption that full legacy scoring remains
negligible at thousands of nodes.

## S1: representation comparison

Four arms were evaluated without feeding any of them back into search:

1. current 20-dimensional dynamic-range RDF/statistics descriptor;
2. fixed 6 Å, 16-bin, species-pair RDF;
3. species-mean pooled invariant MACE descriptors from all message-passing
   layers;
4. centered, non-whitened PCA95 of the same MACE descriptors.

### C60

| representation | dim | nearest-neighbor energy MAE, eV | pair distance vs energy-gap Spearman | recorded-target NN MAE |
|---|---:|---:|---:|---:|
| current RDF | 20 | 2.440 | -0.003 | 0.441 |
| fixed RDF | 16 | 2.743 | 0.401 | 0.396 |
| raw MACE | 256 | 1.141 | 0.484 | 0.323 |
| MACE PCA95 | 5 | 1.125 | 0.485 | 0.293 |

### PdO

| representation | dim | nearest-neighbor energy MAE, eV | pair distance vs energy-gap Spearman | recorded-target NN MAE |
|---|---:|---:|---:|---:|
| current RDF | 20 | 0.930 | 0.167 | 0.0864 |
| fixed RDF | 48 | 0.818 | 0.783 | 0.0833 |
| raw MACE | 512 | 0.541 | 0.630 | 0.0683 |
| MACE PCA95 | 5 | 0.575 | 0.639 | 0.0700 |

The fixed RDF control is important.  It shows that repairing the distance grid
and species channels restores physically meaningful global distance-energy
ordering, especially for PdO, but does not match MACE's local nearest-neighbor
energy continuity.  Thus the MACE result is not explained solely by the known
dynamic-RDF bin defect.

Raw MACE is the surviving scientific representation.  It improves nearest
energy locality on both systems and improves the censored recorded-target
locality.  The latter remains descriptive because duplicate and failed
outcomes are missing.

## PCA95 is compression, not an algorithmic improvement

PCA retained five components for both systems:

- C60: 96.25% of fitted variance;
- PdO: 95.18% of fitted variance.

Raw-MACE and PCA pairwise distances have Spearman correlations of 0.9990 for
C60 and 0.9983 for PdO.  However, exact nearest-neighbor identity is retained
for only 56.5% of C60 and 47.2% of PdO structures.  The raw nearest neighbor
remains in the PCA top five for 96.4% and 86.3%, respectively.

PCA95 therefore preserves global geometry and FPS-like coverage well, but it
does not preserve every local neighbor.  It should remain a compression arm
against raw MACE, not replace raw MACE by default and not be credited as a
search improvement.

## Runtime invariance and descriptor cost

A separate four-forward GPU check used a random rigid rotation, translation,
and atom permutation for C60, and a lattice-vector translation plus atom
permutation for PdO.  Relative pooled-MACE descriptor residuals were:

- C60: `2.89e-7`;
- PdO: `5.85e-7`.

Descriptor replay cost was:

- C60: 306 forwards, 7.87 s total, 0.0257 s/structure;
- PdO: 388 forwards, 8.35 s total, 0.0215 s/structure.

This cost is modest but not zero.  A future online experiment must count and
cache the descriptor forward instead of calling it invisibly.

## Offline top-k versus FPS evidence

This is a censored replay, not an online selector result.  Productive means
that a saved new-minimum transition from the starter improved the campaign's
global best.

At capacity 128:

| system | pool | productive starters retained | recorded gain retained |
|---|---|---:|---:|
| C60, N=306 | raw-MACE FPS | 40.0% | 49.5% |
| C60, N=306 | energy top-k | 20.0% | 16.9% |
| PdO, N=388 | raw-MACE FPS | 38.5% | 67.6% |
| PdO, N=388 | energy top-k | 26.9% | 28.0% |

At capacity 256, FPS retained 91.8% and 92.9% of recorded gain for C60 and
PdO.  Energy top-k retained 53.1% and 67.0%.  PdO top-k retained more
productive nodes at capacity 256, but those nodes carried less recorded global
gain.

The result rejects energy top-k as an evidence-backed default.  It provides a
reason to continue investigating MACE-FPS geometry, but hard FPS still removes
most productive nodes at aggressive capacities and therefore does not yet
qualify as the production pool.

## Decision and next gate

S0 and S1 change the next-stage priority:

1. Do not compare node-UCB with node-TS yet.  Arm proliferation prevents either
   posterior from concentrating.
2. Keep raw pooled invariant MACE as the representation reference; retain
   PCA95 only as a compression control.
3. Do not adopt hard energy top-k.
4. Do not adopt hard FPS from this replay.
5. The next clean online mechanism should test **FPS state abstraction**, not
   permanent node deletion:
   - retain the full archive;
   - assign every node to a MACE-FPS cell;
   - accumulate outcome statistics at cell level;
   - choose a cell, then choose uniformly within that cell;
   - log the exact two-stage probability and every terminal outcome.

For a 200-trial gate, a non-tuned initial cell count can be derived from
posterior resolution rather than search performance.  A balanced Beta(1,1)
posterior needs three observations to halve its prior variance, giving
`K=floor(200/3)=66` macro-arms.  This is an explicit statistical resolution
choice, not a C60/PdO energy-tuned parameter.

The next controlled matrix should keep the current direction/uphill/optimizer
kernel frozen and compare:

- full-archive uniform;
- FPS-cell uniform followed by uniform-within-cell;
- current legacy UCB-like;
- scale-calibrated classic Metropolis as the separate chain baseline.

Only if the cell abstraction survives paired multi-seed testing should the
same cell-level posterior be used to compare posterior-proportional, UCB, and
Thompson sampling.
