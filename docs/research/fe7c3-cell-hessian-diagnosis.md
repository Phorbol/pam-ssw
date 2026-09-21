# Fe7C3 cell-direction diagnostic: soft bias is present

2026-09-11. Job1269206 completed on one V100 (4v100n01), elapsed81s,
exit0. Four saved cycle inputs,148 E/F/stress requests total; no search rerun.
Fe registered cumulative cost becomes61088 EFS (60940+148). CPU Cu4/EMT
prechecks are separate costs. Frozen Python source is stored beside the result.

| Seed/cycle (outer1) | lowest3 squared weight | Rayleigh (eV/A²) | residual (eV/A²) | min/max eigenvalue |
| --- | ---: | ---: | ---: | --- |
| seed7-outer1-cycle0 | 0.995107 | 16.22888 | 5.76955 | 6.30234/97.13089 |
| seed7-outer1-cycle4 | 0.968332 | 14.45586 | 7.86293 | 6.95671/174.47254 |
| seed101-outer1-cycle0 | 0.979015 | 11.00868 | 6.13358 | 6.30234/97.13089 |
| seed101-outer1-cycle4 | 0.752638 | 88.43837 | 115.04668 | 16.51470/881.41136 |

Both finite-difference steps (.005/.0025A) give consistent spectra and weights.
At the smaller step, relative Hessian antisymmetry is1.4e-6 to2.85e-6.
Fresh center energies match the old ledger within1.14e-13eV. Rotation components
of the saved unit directions are below2.5e-14; squared eigenmode weights sum to1.

The six-call plane dimer produces predominantly soft directions at these four
points. The final seed101 point has a large residual, but still75.3% of squared
weight in the soft half; a residual alone does not identify an unusable direction.
This weakens the hypothesis that poor cell-mode accuracy is the dominant cause
of the observed high-energy compressed preparation. It does not prove that
better CBD directions cannot help, or that all cell cycles are adequately solved.

Positive projected eigenvalues describe the local fixed-fractional cell chart;
they are not a full atom/cell Hessian, stationary-point certificate, new phase
identification, or global-search success. The two cycle0 configurations are the
same starting geometry with different saved directions, not independent materials.

Decision: retain current direction solver for the next controlled comparison.
Test only displacement normalization, with unchanged Safe-total/history and
convergence thresholds. Preserve the paper rule as default and keep any alternative
experimental until real end-to-end results support it.

Evidence: `research/ga_ssw/fe7c3-block-cell-hessian/result.json`, `input.json`,
`run.py`, frozen `source/`, `job.sbatch`, allocation and Slurm outputs.
