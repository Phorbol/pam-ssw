Task: replace the lightweight in-repo LJ BH/GA baselines with ASE-family baselines for a stricter comparison.

Scope:
- Keep the current SSW implementation unchanged for this benchmark.
- Add an ASE baseline mode to `benchmarks/lj_cluster_compare.py`.
- Use `ase.optimize.basin.BasinHopping` for BH.
- Use the split-out ASE GA package `ase-ga` for GA operators/population when available.
- Use ASE `LennardJones` for all algorithms.
- Produce JSON metrics, energy-gap curves, exported minima, and a literature-comparison report.

Approved by user:
- The user requested the ASE BH/GA conversion and explicitly allowed web/source-code lookup and larger LJ testing.

Acceptance:
- The code must clearly distinguish internal and ASE baselines.
- If `ase-ga` is unavailable, the benchmark must fail/report that instead of silently falling back to self-written GA.
- The final report must say whether the comparison is directly literature-grade or only ASE-API-aligned.
