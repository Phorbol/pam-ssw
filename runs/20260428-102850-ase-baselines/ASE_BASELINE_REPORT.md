# ASE BH/GA baseline replacement report

Date: 2026-04-28

## What changed

- `benchmarks/lj_cluster_compare.py` now supports `--baseline-source internal|ase`.
- `pyproject.toml` now exposes optional benchmark dependency group `benchmarks = ["ase-ga>=1.0.3"]`.
- `--baseline-source ase` uses:
  - ASE `ase.optimize.basin.BasinHopping` for BH.
  - the split-out ASE-GA package `ase-ga==1.0.3` for GA data/population/start/cut-and-splice components.
  - ASE `LennardJones` for all algorithms.
- ASE-GA candidate local relaxation uses the repo L-BFGS-B `relax_minimum` through `ASECalculator(LennardJones())`. This keeps local quench treatment comparable with SSW; ASE-GA itself provides the GA operators and database workflow, not a one-call optimizer.
- ASE-BH remains the ASE optimizer path: `BasinHopping(..., optimizer=FIRE, fmax=1e-3, dr=0.35, temperature=0.8)`.
- Minima export now includes `ase_bh_*_minima.xyz` and `ase_ga_*_minima.xyz`.

## Source/API facts checked

- Local ASE version: `3.28.0`.
- `ase.optimize.basin.BasinHopping` exists in ASE and cites Wales-Doye basin hopping in its class docstring.
- In ASE 3.28, `ase.ga` is only a compatibility placeholder; it raises an ImportError saying the GA code moved to the separate `ase-ga` project.
- Installed package for this run: `ase-ga==1.0.3`.

Relevant primary links:

- ASE optimizer documentation: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
- ASE-GA project documentation: https://dtu-energy.github.io/ase-ga/
- ASE-GA package/source home indicated by ASE ImportError: https://github.com/dtu-energy/ase-ga
- Wales & Doye BH paper DOI: https://doi.org/10.1021/jp970984n
- Deaven et al. GA LJ paper DOI: https://doi.org/10.1016/0009-2614(96)00406-X
- Wolf & Landman GA paper DOI: https://doi.org/10.1021/jp9814597
- Cambridge Cluster Database LJ page: https://doye.chem.ox.ac.uk/jon/structures/LJ/

## Budget-60 metric result

Budget: 60 local minima/quench events per algorithm run.
Seeds: 0, 1.
SSW config: `max_steps_per_walk=8`, `proposal_relax_steps=80`.

| System | Algorithm | Mean gap to CCD | Best energy |
|---|---:|---:|---:|
| LJ13 | SSW | 0.4273960379 | -43.8994049623 |
| LJ13 | ASE-GA | 0.4273960392 | -43.8994049627 |
| LJ13 | ASE-BH | 1.8548068527 | -43.8994049459 |
| LJ38 | SSW | 4.8866072927 | -169.4898023967 |
| LJ38 | ASE-GA | 6.7954662126 | -167.3514701662 |
| LJ38 | ASE-BH | 11.2176086819 | -166.0234938562 |

Primary JSON artifacts:

- `lj13_ase_baselines_metric.json`
- `lj38_ase_baselines_metric.json`
- `lj13_38_ase_baselines.json` was produced before the first full-trace command was killed; the split LJ13/LJ38 files above are the authoritative budget-60 metrics.

## Curve artifact

Full budget-60 trace for ASE baselines was too expensive because trace mode reruns all algorithms. I therefore generated a budget-20 trace/curve:

- `lj13_38_ase_budget20_traces.json`
- `lj13_38_ase_budget20_curves.png`

This plot is suitable for early-generation trend comparison, not for claiming budget-60 convergence.

## Minima files

Budget-60 minima exports:

- `minima_xyz_lj13/ssw_LJ13_seed0_minima.xyz`
- `minima_xyz_lj13/ssw_LJ13_seed1_minima.xyz`
- `minima_xyz_lj13/ase_bh_LJ13_seed0_minima.xyz`
- `minima_xyz_lj13/ase_bh_LJ13_seed1_minima.xyz`
- `minima_xyz_lj13/ase_ga_LJ13_seed0_minima.xyz`
- `minima_xyz_lj13/ase_ga_LJ13_seed1_minima.xyz`
- `minima_xyz_lj38/ssw_LJ38_seed0_minima.xyz`
- `minima_xyz_lj38/ssw_LJ38_seed1_minima.xyz`
- `minima_xyz_lj38/ase_bh_LJ38_seed0_minima.xyz`
- `minima_xyz_lj38/ase_bh_LJ38_seed1_minima.xyz`
- `minima_xyz_lj38/ase_ga_LJ38_seed0_minima.xyz`
- `minima_xyz_lj38/ase_ga_LJ38_seed1_minima.xyz`

Each frame stores `algorithm`, `size`, `seed`, `index`, and `energy`.

## Interpretation

The current SSW implementation beats the ASE-family baseline on LJ38 under this short budget, and essentially ties ASE-GA on LJ13. This does not mean the method is literature-competitive with Wales-Doye or later GA implementations. The literature runs used much broader searches, tuned operators, and difficult-size campaign budgets, while this benchmark uses two seeds and budget 60.

The comparison is now stricter than the previous self-written BH/GA baseline because BH is ASE's actual `BasinHopping`, and GA uses the official ASE-GA data/operator stack. It is still not a full reproduction of the literature algorithms.

## SSW maturity assessment

The current SSW is not yet a complete production SSW implementation.

Implemented and useful:

- true-PES quench/archive loop;
- local Gaussian bias path walking;
- rigid translation/rotation projection for free non-periodic clusters;
- descriptor/frontier/acquisition diagnostics;
- duplicate accounting and minima export;
- LJ cluster benchmark harness.

Still incomplete:

- original SSW mixed-direction machinery is only partial;
- proposal relaxation remains frequently unconverged for LJ38;
- stop-reason diagnostics and bias-history analysis are still not complete enough;
- periodic/slab archive geometry is not yet MIC-safe;
- fixed-cell periodic calculator support exists, but search-layer correctness for slab/bulk is not production-grade.

## Generality status

The algorithm is not hard-coded to LJ energies, and the calculator interface can run any ASE calculator. However, current benchmark fixtures and rigid-mode projection are cluster-focused.

- Free clusters: currently the best-supported case.
- Fixed-cell bulk: calculator path can evaluate it, but archive/fingerprint/dedup and SSW directions need periodic MIC validation before claims.
- Slab: ASE/MACE calculator path can evaluate slabs, but current search geometry still has known non-periodic assumptions; slab SSW is experimental, not a production claim.
- Variable-cell bulk/slab: out of scope for the current implementation.

## Verification

- `pytest -q tests/unit`: 54 passed.
- `pytest -q`: 65 passed.
- Internal default baseline smoke: `python3 benchmarks/lj_cluster_compare.py --sizes 13 --seeds 0 --budget 4 ...` completed.
