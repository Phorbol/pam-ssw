# Ritz CUDA 40t Ablation Plan

## Task

Evaluate whether the existing Ritz direction components improve the current best Direct-QP SSW production settings before implementing new penalized/block Lanczos code.

## Matrix

- Systems: C60, CuO, PdO
- Seed: 42
- Trials: 40
- Steps per walk: 8
- Device: CUDA
- Variants:
  - `direct_qp_system_baseline`
  - `direct_qp_system_baseline_regularized_ritz`
  - `direct_qp_system_baseline_rayleigh_ritz`

## Baseline Mapping

- C60/PdO baseline: `direct_qp_rank1_gated_q25_micro_adaptive50`
- CuO baseline: `direct_qp_curvature_gamma_kappa240`

## Command

```bash
python runs/20260602-ritz-cuda-40t-ablation/run_matrix.py --device cuda --trials 40 --seeds 42
```

## Acceptance

- `results.csv` contains 9 data rows.
- `results.json` and `summary.md` agree with `results.csv`.
- Each case has `output/<case>/ssw_summary.json`.
- Summary includes best energy, minima count, duplicate rate, force evaluations, Ritz/Ritz-reg selection counts, and residual risk.
