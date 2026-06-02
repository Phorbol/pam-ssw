# Direct-QP SSW C60/CuO/PdO Benchmark

- Status: completed
- Completed cases: 9
- Variants: direct_qp_system_baseline, direct_qp_system_baseline_regularized_ritz, direct_qp_system_baseline_rayleigh_ritz
- Dtype: float32, cuEq off

## Aggregate

- c60 / direct_qp_system_baseline: n=1, mean_best=-497.477234 eV, best_single=-497.477234 eV, mean_force=14362.0, mean_dup=0.244, direct_steps=271.0, direct_reject=10.0, ritz=0.0, ritz_reg=0.0, wall=279.1s
- c60 / direct_qp_system_baseline_rayleigh_ritz: n=1, mean_best=-503.037689 eV, best_single=-503.037689 eV, mean_force=15870.0, mean_dup=0.220, direct_steps=309.0, direct_reject=7.0, ritz=104.0, ritz_reg=0.0, wall=313.1s
- c60 / direct_qp_system_baseline_regularized_ritz: n=1, mean_best=-501.953064 eV, best_single=-501.953064 eV, mean_force=16329.0, mean_dup=0.171, direct_steps=297.0, direct_reject=5.0, ritz=0.0, ritz_reg=113.0, wall=318.5s
- cuo / direct_qp_system_baseline: n=1, mean_best=-202.163208 eV, best_single=-202.163208 eV, mean_force=9997.0, mean_dup=0.024, direct_steps=319.0, direct_reject=0.0, ritz=0.0, ritz_reg=0.0, wall=358.4s
- cuo / direct_qp_system_baseline_rayleigh_ritz: n=1, mean_best=-201.390869 eV, best_single=-201.390869 eV, mean_force=9719.0, mean_dup=0.073, direct_steps=319.0, direct_reject=0.0, ritz=41.0, ritz_reg=0.0, wall=355.4s
- cuo / direct_qp_system_baseline_regularized_ritz: n=1, mean_best=-202.027588 eV, best_single=-202.027588 eV, mean_force=9706.0, mean_dup=0.049, direct_steps=320.0, direct_reject=0.0, ritz=0.0, ritz_reg=30.0, wall=375.9s
- pdo / direct_qp_system_baseline: n=1, mean_best=-574.851013 eV, best_single=-574.851013 eV, mean_force=12074.0, mean_dup=0.122, direct_steps=301.0, direct_reject=2.0, ritz=0.0, ritz_reg=0.0, wall=327.2s
- pdo / direct_qp_system_baseline_rayleigh_ritz: n=1, mean_best=-573.510742 eV, best_single=-573.510742 eV, mean_force=12064.0, mean_dup=0.073, direct_steps=298.0, direct_reject=1.0, ritz=58.0, ritz_reg=0.0, wall=329.6s
- pdo / direct_qp_system_baseline_regularized_ritz: n=1, mean_best=-573.100098 eV, best_single=-573.100098 eV, mean_force=12373.0, mean_dup=0.122, direct_steps=313.0, direct_reject=1.0, ritz=0.0, ritz_reg=72.0, wall=337.9s

## Readout

- C60: both Ritz modes improved the 40t best energy. `regularized_ritz` improved best by 4.476 eV and reduced duplicate rate from 0.244 to 0.171. `rayleigh_ritz` was deepest, improving best by 5.560 eV, but duplicate rate was higher than `regularized_ritz`.
- CuO: both Ritz modes hurt the fixed-kappa slab baseline. `regularized_ritz` was 0.136 eV shallower than baseline, while `rayleigh_ritz` was 0.772 eV shallower and had the highest duplicate rate.
- PdO: both Ritz modes hurt best energy in this 40t seed42 run. `rayleigh_ritz` reduced duplicate rate from 0.122 to 0.073, but it was still 1.340 eV shallower than baseline. `regularized_ritz` was 1.751 eV shallower.

## Decision

- Ritz is a promising C60 direction intervention and should be tested at 200t or 500t before any production default change.
- Ritz should not be promoted globally. CuO and PdO regressions show that pure/local soft-mode improvement can disrupt slab search behavior.
- The next implementation target should be a gated/system-aware Ritz policy, not an unconditional `direction_synthesis_mode="regularized_ritz"` default.
- Residual risk: this is a single-seed 40t ablation. Treat effect direction as a screening signal, not final production evidence.
