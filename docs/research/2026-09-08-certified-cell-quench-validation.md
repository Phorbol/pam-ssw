# Certified minima and experimental cell-quench validation

2026-09-08. Implementation branch: `fix/certified-cell-quench`, based on
`fb024698e6b6102eae4f6b10d5e6080d7637832d`. Existing production and benchmark
worktrees are unchanged. No remote refs were deleted or pushed.

## Result and scope

Initial uncertified quenches fail with an explicit exception and evaluation ledger.
Later uncertified landings remain diagnostic evidence and do not enter the archive
or earn productivity credit. Geometry-first matching retains the first representative;
energy differences are diagnostics rather than evidence of a new basin. Compatible
periodic images use their own cells before minimum-image comparison.

Opt-in `volume_only`, `shape`, and `slab_xy` use ASE cell relaxation in the true
quench. Proposal propagation remains atomic at its starter cell. Cell-mode archive
objectives are E+pV, with independent active-force and allowed-stress certificates.
Fixed atoms have fixed fractional coordinates. Posterior campaigns reject cell mode.

## Verification

Full-suite comparison before the final extxyz regression: candidate **1455 passed,
127 failed, 7 skipped**; isolated base **1408 passed, 127 failed, 7 skipped**.
The exact 127 failing test IDs are identical; all depend on unavailable historical
artifacts (old worktree paths or missing run outputs). This is not an all-green suite.
See `runs/20260908-certified-cell-quench-validation/test-comparison.json`.
After the extxyz fix, the focused archive/certificate/cell suite passed **63 tests**,
including public structure write/readback. `git diff --check` passed.
Independent reviews checked certificate/seed-credit logic, periodic matching,
nonzero-strain gradients, pressure, constrained atoms, output metadata, and API docs.

## Bounded CPU experiment

Four-atom periodic FCC Lennard-Jones, cutoff 2.7, initial lattice constant 1.8,
two LS-SSW trials, seed 8. FIRE primary / LBFGS fallback; force threshold 0.005
eV/Angstrom and stress threshold 0.001 eV/Angstrom^3. The independent final
calculator call is outside the search ledger and explicitly recorded once per case.
All search evaluations have a purpose; no unattributed evaluations occur.

| Case | E (eV) | E+pV (eV) | V (Angstrom^3) | max force | stress residual | Search FE |
|---|---:|---:|---:|---:|---:|---:|
| fixed-pressure0 | -19.53618132 | -19.53618132 | 5.832000 | 1.96e-15 | 5.24 | 184 |
| volume_only-pressure0 | -30.79818017 | -30.79818017 | 3.716154 | 4.54e-14 | 0.000306 | 199 |
| shape-pressure0 | -30.79818017 | -30.79818017 | 3.716154 | 1.62e-13 | 0.000306 | 232 |
| volume_only-pressure2 | -30.79817625 | -30.75179568 | 3.715493 | 1.82e-13 | 0.000309 | 198 |

Fixed-cell stress is reported, not certified. All variable-cell cases satisfy the
allowed stress criterion. The smoke also checks negative LJ binding energy, since
small derivatives alone can certify a cutoff plateau. These four short trajectories
do not measure global-search success or phase-transition discovery.

An earlier LBFGS-primary smoke expanded both zero-pressure cell cases to
V=59.234439 Angstrom^3 and E=0 (zero forces/stress), instead of the cohesive crystal.
The fixed-cell case had E=-19.53618132; the 2 GPa volume case reached
E=-30.79817629, V=3.71549684. This actual optimizer failure mode motivated the
negative-binding check and FIRE-primary rerun; no universal optimizer ranking is
claimed. Force/stress certificates are necessary but not physical stability proofs.

Reproduce from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m runs.20260908-certified-cell-quench-validation.run_cpu_smoke
```

`smoke.json` includes complete configurations, source hashes, dependency versions,
ledgers, endpoints, and scalar checks. The recorded base commit precedes implementation;
source hashes identify the actual Python files tested. ASE 3.29.0 was tested; the
declared 3.23 dependency floor was not separately runtime-tested.

## Remaining algorithm work

1. Compare basin novelty and low-energy coverage per total FE on matched seeds and
   budgets, including failed quenches; do not rank by accepted count alone.
2. Add species-aware, permutation-aware periodic basin identity and a reference
   re-quench policy for energy discrepancies. Current RMSD/cell matching is approximate.
3. Benchmark atomic, strain, and mixed proposal kernels before claiming generalized
   cell SSW; introduce a consistent atom/strain metric and bias gradient first.
4. Control cell steps and diagnose expansion/collapse; test FIRE versus LBFGS on
   cohesive crystals and nonzero-pressure systems. Do not silently accept flat
   dissociation plateaus as useful material phases.
5. Validate stress and quench behavior with actual MLIP/DFT backends. Slab stress
   depends on vacuum thickness; use a consistent vacuum convention or area-normalized
   criterion before comparing slab systems. No GPU/HPC production runs were launched.

Branch cleanup candidates and preserved unique refs are recorded separately in
`2026-09-08-branch-cleanup-manifest.md`.
