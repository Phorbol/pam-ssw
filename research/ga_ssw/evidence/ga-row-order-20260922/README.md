# GA descriptor row-order correction — 2026-09-22

Classification: implementation / failure diagnosis, plus bounded real-calculator
integration. This is not independent global-search performance validation.

## Approved contract and implementation

User approved `descriptor_row_order="full_fingerprint"` as an opt-in correction;
`legacy_counts` remains the default. Candidate and frozen-reference copies use
the same complete n/d row key. Radial formulas, cutoffs, weights, tolerance,
periodic controllers and checkpoint version remain unchanged. Only full mode
adds the scientific-contract key; cross-mode resume is rejected before PES.
Original reference serialization is retained in the input contract, so this is
not automatic migration or equivalence matching of different reference files.

Core change: a9dd853 (reviewed integration of isolated commit5477ac6, basef7a257b).
The only subsequent production-source change before these jobs is API docstring
text. Tests, the production-helper audit and runners are included alongside this
record, so the final commit recovers actual executed behavior and configuration.

## Verification and complete execution history

All jobs: CPU-MISC, rush-cpu, sjtu-caoxiaoming, one task, five-minute wall cap;
Python `/home/gengjianrui/.conda/envs/mace_env/bin/python`, PYTHONPATH=.,
PYTHONNOUSERSITE=1, OMP/OPENBLAS threads1. No GPU jobs.

- CPU1448104: unmodified production, new API tests fail as expected (3), old
  controller/checkpoint tests pass (40). Raw error retained.
- CPU1448126: corrected production, 44 pass; six legal water15 permutations use
  the actual production helper. All old projection drifts exceed0.0001; all full
  mode projection drifts are0. No PES used by the permutation audit.
- CPU1448187: 44 pass again after strengthening the check that every actual
  landing projection uses sorted candidates AND references. Same six permutation
  outcomes. Command is in `validate.sbatch` (three targeted test files plus audit).
  Cu13/Al13 EMT checks exercise pause/save/load/resume, reference immutability,
  legacy contract shape, and both cross-mode zero-E/F rejections. These short
  tests do not require successful escape and are not search validation.
- CPU1448172: four short Cu13/Al13 mode arms,114 total search requests. Both modes
  hit the same expected existing guards: two parents do not meet >2 competition
  requirement; the tiny diagnostic rotation budget produces no qualified escape.
  See `lifecycle.json`. This failed protocol is retained, not counted as successful
  genetic-operation coverage and did not motivate a production parameter change.
- CPU1448176: reused historical three-minimum Cu13 full protocol. The runner
  failed serializing an Atoms object inside stage metadata after the first arm.
  Its complete numerical results were not saved. Exact paid cost is unavailable;
  the search was bounded by4000 requests. This run is excluded, and its error and
  source retained under `full-lifecycle/`. It is not counted as zero-cost failure.
- CPU1448190: one focused runner correction (scalar stage summary plus immediate
  pickle of actual result), same full protocol, new output directory
  `full-lifecycle-v2/`. Both modes complete with no failures.

## Complete GA lifecycle result (CPU1448190)

Reuse `../atomic-cu13-ga/plan.json` and its three saved distinct Cu13 minima,
ASE EMT, original seed3 and all archived GA/SSW parameters. Explicit budget4000
E/F per mode. Protocol and source are in `full-lifecycle-v2/`; only row ordering
varies between arms. Unlike the earlier short fixture, this covers initial
quench, quick walks, real TYPE0 proposal, offspring quenches, generation walk,
region selection and fine walk.

| Mode | Search E/F | Fresh E/F | Qualified observations | Offspring quenches | Status |
|---|---:|---:|---:|---:|---|
| legacy_counts |2396|26|26/26|13|completed|
| full_fingerprint |2396|26|26/26|13|completed|

Both arms have identical stage costs and best energy9.361357881570044eV on this
EMT fixture; measured times approximately2.91s each. Fresh independent calculator
calls confirm all52 observations have correct Cu13 composition and max force
<=0.01eV/Angstrom. This is numerical/composition qualification, not exhaustive
chemical or Hessian stability validation. No claim of speedup or better minima
coverage follows from this single seed. All observed geometries and results are
saved; result pickles remain local beside their JSON/EXTXYZ exports.

## Decision

Keep optional correction: it fixes the demonstrated label-dependent projection
on six legal water permutations and works through real GA offspring ingestion
and checkpoint boundaries. Preserve old default. Do not change descriptor
formulas, add parameters, promote search-efficiency claims, or repeat this input
for tuning. LS/pool state ownership remains a separate pending user decision.
