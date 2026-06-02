# Direct-QP SSW C60/CuO/PdO Benchmark

- Status: completed
- Completed cases: 9
- Variants: bias_relax_current, direct_qp_rank1_gated_q25_micro_adaptive50, direct_qp_curvature_gamma_kappa240
- Dtype: float32, cuEq off

## Aggregate

- c60 / bias_relax_current: n=1, mean_best=-507.760742 eV, best_single=-507.760742 eV, mean_force=245444.0, mean_dup=0.160, direct_steps=0.0, direct_reject=0.0, wall=4929.5s
- c60 / direct_qp_curvature_gamma_kappa240: n=1, mean_best=-488.355347 eV, best_single=-488.355347 eV, mean_force=219746.0, mean_dup=0.200, direct_steps=2972.0, direct_reject=209.0, wall=4364.8s
- c60 / direct_qp_rank1_gated_q25_micro_adaptive50: n=1, mean_best=-507.782288 eV, best_single=-507.782288 eV, mean_force=213279.0, mean_dup=0.427, direct_steps=3752.0, direct_reject=41.0, wall=3912.7s
- cuo / bias_relax_current: n=1, mean_best=-202.2612 eV, best_single=-202.2612 eV, mean_force=489168.0, mean_dup=0.002, direct_steps=0.0, direct_reject=0.0, wall=15686.3s
- cuo / direct_qp_curvature_gamma_kappa240: n=1, mean_best=-202.590317 eV, best_single=-202.590317 eV, mean_force=122836.0, mean_dup=0.054, direct_steps=3984.0, direct_reject=0.0, wall=4784.5s
- cuo / direct_qp_rank1_gated_q25_micro_adaptive50: n=1, mean_best=-202.138763 eV, best_single=-202.138763 eV, mean_force=154939.0, mean_dup=0.349, direct_steps=3739.0, direct_reject=0.0, wall=5238.1s
- pdo / bias_relax_current: n=1, mean_best=-575.308105 eV, best_single=-575.308105 eV, mean_force=174158.0, mean_dup=0.104, direct_steps=0.0, direct_reject=0.0, wall=4897.6s
- pdo / direct_qp_curvature_gamma_kappa240: n=1, mean_best=-575.669617 eV, best_single=-575.669617 eV, mean_force=80151.0, mean_dup=0.255, direct_steps=2526.0, direct_reject=361.0, wall=2421.6s
- pdo / direct_qp_rank1_gated_q25_micro_adaptive50: n=1, mean_best=-576.852661 eV, best_single=-576.852661 eV, mean_force=146393.0, mean_dup=0.184, direct_steps=3686.0, direct_reject=31.0, wall=4122.2s

## Readout

- C60: `direct_qp_rank1_gated_q25_micro_adaptive50` slightly beat bias-relax best by 0.022 eV and used 32165 fewer force evaluations, but coverage dropped from 420 to 287 minima and duplicate rate rose from 0.160 to 0.427. `direct_qp_curvature_gamma_kappa240` is not viable for C60: it finished 19.405 eV shallower than bias-relax.
- CuO: `direct_qp_curvature_gamma_kappa240` is the best CuO result in this matrix: 0.329 eV deeper than bias-relax, 366332 fewer force evaluations, and 474 minima. `adaptive50` was cheaper than bias-relax but 0.122 eV shallower with much higher duplicate rate.
- PdO: `direct_qp_rank1_gated_q25_micro_adaptive50` is the best PdO result: 1.545 eV deeper than bias-relax and 27765 fewer force evaluations, with 409 vs 449 minima. `direct_qp_curvature_gamma_kappa240` is cheapest and still beats bias-relax best by 0.362 eV, but is 1.183 eV shallower than adaptive50 and has the highest PdO duplicate rate.

## Current Decision

- Keep `adaptive50` as the stronger Direct-QP production candidate for C60/PdO-like cases where basin settling matters.
- Keep `curvature_gamma_kappa240` as a slab-specific candidate only after system-level validation: it is excellent on CuO, acceptable-but-not-best on PdO, and fails on C60.
- Do not generalize fixed-kappa from CuO to all systems.
- The next approved task is the soft-mode eigen/Lanczos experiment after this production CUDA job has completed.
