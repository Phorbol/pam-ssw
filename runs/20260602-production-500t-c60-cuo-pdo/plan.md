# C60/CuO/PdO Production 500t

Date: 2026-06-02

## Matrix

- Systems: `c60`, `cuo`, `pdo`
- Trials: 500 per case
- Seeds: `42`
- Device: CUDA
- Variants:
  - `bias_relax_current`
  - `direct_qp_rank1_gated_q25_micro_adaptive50`
  - `direct_qp_curvature_gamma_kappa240`

## Rationale

- `bias_relax_current` preserves the current production baseline.
- `direct_qp_rank1_gated_q25_micro_adaptive50` is the strongest Direct-QP C60 path observed so far.
- `direct_qp_curvature_gamma_kappa240` is the strongest Direct-QP slab path observed so far.

Reference dimer is intentionally excluded from this production matrix because the diagnostic runs showed that it remains locked and below `scored_pool`.
