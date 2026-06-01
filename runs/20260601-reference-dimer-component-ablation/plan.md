# Plan

This ledger implements Task 6 from `docs/superpowers/plans/2026-06-01-reference-dimer-component-ablation.md`.

## Phase 1

- System: `c60`
- Seeds: `0,1,2`
- Trials per seed: `40`
- Variants: `reference_original`, `paw_current_bias_relax`, `paw_metropolis_pool_bias_relax`, `paw_ucb_reference_dimer_bias_relax`, `paw_ucb_pool_no_momentum_bias_relax`, `paw_ucb_reference_dimer_direct_qp_adaptive50`

## Phase 2

- Systems: `cuo,pdo`
- Seeds: `0,1,2`
- Trials per seed: `40`
- Variants: PAM-SSW component ablations only; external `reference_original` is currently reserved for C60.

Benchmark execution requires explicit approval after implementation checks pass. The runner is planning-only by default; it will not launch calculations unless `--execute` is passed.
