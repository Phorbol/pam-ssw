# LJ13 / LJ38 Benchmark Slice

This file summarizes the first literature-backed Lennard-Jones benchmark slice for the current `pamssw` implementation.

## Why LJ13 and LJ38

- `LJ13` is the canonical small magic-number icosahedral cluster in the Cambridge Cluster Database.
- `LJ38` is the first widely cited hard case with a competing non-icosahedral global minimum.

Reference target energies used in the benchmark:

- `LJ13`: `-44.326801`
- `LJ38`: `-173.928427`

These values were taken from the Cambridge Cluster Database tables used throughout the LJ global-optimization literature.

## Protocol

- Two independent random seeds per size
- Fixed budget: `60` local relaxations per run
- Same Lennard-Jones model for all methods
- Methods:
  - `ssw`: current repo implementation
  - `bh`: lightweight basin-hopping baseline
  - `ga`: lightweight cut-and-splice style genetic baseline

## Results

| Method | Size | Success Rate | Best Energy | Mean Energy Gap |
|---|---:|---:|---:|---:|
| SSW | 13 | 0.0 | -39.647686 | 5.165654 |
| BH | 13 | 0.0 | -43.899405 | 0.427396 |
| GA | 13 | 0.0 | -43.899405 | 0.427396 |
| SSW | 38 | 0.0 | -165.145167 | 12.432864 |
| BH | 38 | 0.0 | -167.055431 | 7.357303 |
| GA | 38 | 0.0 | -169.489802 | 5.513193 |

## Interpretation

- The new archive deduplication is working correctly for rigidly moved LJ minima; `LJ4` now collapses repeated rotated/translated copies into one basin.
- On this first benchmark slice, the current `SSW` implementation is clearly behind both baselines on `LJ13` and `LJ38`.
- `LJ38` remains the most informative failure case because the literature explicitly identifies it as a multiple-funnel hard benchmark.

## Caveat

These `BH` and `GA` baselines are lightweight in-repo implementations for controlled comparison under a fixed local-relaxation budget. They are not tuned reproductions of the original literature codes.

The machine-readable results are stored in:

- `benchmark_results_lj13_lj38_budget60.json`
