# Plan

## Fix Validation

- Add a unit test proving one sampled reference-dimer initial mode is reused across all internal climb steps of one walk.
- Run targeted reference-dimer tests.
- Run full unit suite.

## CUDA Validation

- System: `c60`
- Seeds: `0,1,2`
- Trials per seed: `40`
- Steps per walk: `8`
- Variants:
  - `paw_ucb_reference_dimer_bias_relax`
  - `paw_ucb_reference_dimer_direct_qp_adaptive50`

## Acceptance Signals

- `reference_dimer_steps` should remain populated.
- `reference_dimer_mean_abs_dot_initial` should no longer be trivially dominated by per-step fresh N0 semantics.
- Best energy, minima count, duplicate rate, and wall/force cost will be compared against `runs/20260601-reference-dimer-component-ablation/results.csv`.
