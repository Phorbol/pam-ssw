# ASE BasinHopping baseline comparability audit

This is a read-only audit of the completed ASE baseline artifacts and the
frozen two-stage Ritz run.  It does not add a search mechanism or claim a
scientific ranking.

## Inputs and controls

The corrected baseline is
`research/ga_ssw/evidence/ase-basin-hopping-baseline-20260912-v2`.  Its
`plan.json` uses three inputs: `cu13` and `cu31_fixed` with EMT, and
`bicyclobutane` with GFN2-xTB; two seeds (11, 29); 150 K; `fmax=0.01`
eV/A; 400 BasinHopping outer steps; `dr=0.5` A; and request cap 6000 /
wall cap 90 s.  Each local call uses PAM Safe-total with 400 optimizer steps
and the default Safe-total history (10 unless explicitly changed).  The
inputs are copied from the two-stage evidence directory, so the structures
are comparable where that directory has the same named case.

The completed v2 execution has four 6000-request boundary arms: Cu13 seeds
11/29 and fixed Cu31 seeds 11/29. Their local-quench counts are 90/87 and
72/69, with the same number of fresh checks and no local-quench failures.
The two bicyclobutane arms stop at GFN2-xTB's `SCF not converged in 250
cycles`: search requests are 3146/3390 and fresh checks 32/38. There was no
retry. The v1 directory is a preparation failure and is excluded: its frozen
source did not export `quench`; v2 corrected the import to
`pamssw.standalone.surface`.

## Difference from the two-stage Ritz evidence

The corresponding runner is
`research/ga_ssw/run_two_stage_ritz_comparison.py`, with frozen source and
results under `research/ga_ssw/evidence/two-stage-ritz-comparison-20260912`.
Its cases include the same Cu13, fixed Cu31 vacancy, and bicyclobutane, plus
an LJ38 supplement.  It uses 100 SSW outer steps, `width=0.1` A,
`rotation_bias=100` eV/A^2, `max_gaussians=25`, 150 K, `fmax=0.01`,
`bias_fmax=0.1`, 400-step Safe-total quenches, `rotation_hvp=100`, and a
6000-request / 120 s arm cap.  The LJ arm overrides width, Gaussian count,
and temperature.  The research runner temporarily replaces the dimer
direction call with the two-stage Ritz helper; this is visible at its
`research_direction` wrapper and is not the production `config.rotation_bias`
trajectory.  Thus the two-stage result is not a fixed-100 dimer baseline.

The random streams also differ: BH calls `np.random.seed(seed)`, so ASE uses
global `RandomState`; SSW calls `np.random.default_rng(seed)`. BH's official
`dr=.5` displacement is an ASE example parameter and is unrelated to SSW's
Gaussian width. In ASE's `BasinHopping.run`, `ro` is the current unquenched
proposal position, while `Eo` is the current endpoint energy from the local
optimizer, not a pre-quench energy. The local endpoint callback through
ASE's `get_value` is a real post-quench cost. Both methods can be compared on
best endpoint energy and paid API cost; differing proposals and RNGs mean
their trajectories are not step-for-step or single-factor equivalent.

## Cost and failure boundaries

`run_ase_basin_hopping_baseline.py` counts every search evaluation in
`evaluations.jsonl` and records a denied boundary request separately. The
four Cu/EMT artifacts have sequential paid requests 1--6000 and
`ledger_count=6001`; the extra row is the denied cap attempt. The GFN2 arms
have sequential paid rows through 3146/3390 and stop on the 250-cycle SCF
exception. A local Safe-total failure raises immediately from the optimizer
adapter, so ASE stops that arm; the paid evaluation ledger remains available.
This differs from SSW, which records a failed rotation/quench as an outer-step
record and continues when its lifecycle permits. BH fresh checks use a new
calculator for each converged local landing and are additional E/F requests;
they must be included in total cost.

The baseline's `best_energy` is the best local-quench energy. A fresh check is
the evidence for force qualification and energy reproducibility; a green
controller status or best energy alone is not basin identity or chemical
success. All completed BH fresh checks have zero energy error and force norms
below 0.01 eV/A, subject to the two GFN2 arms ending at their SCF failures.

The comparison should remain per case and seed at a common paid search cap,
with fresh-check requests reported separately: best fresh-qualified energy,
qualified landing count, failure/boundary status, and physical diagnostics.
The bicyclobutane SCF failures remain failures at their actual costs, with no
retry or silent comparison to a 6000-request arm.
