# Soft-Mode Eigen Experiment Summary

- Rows: 274
- CUDA available: True

## Aggregate

- c60 / constrained_ritz: n=12, rayleigh_min=-4.63879, rayleigh_mean=0.139555, hvp_mean=24.0, overlap_lanczos32=0.349, overlap_scored=0.088, micro_remaining_0p2=94.766
- c60 / lanczos: n=3, rayleigh_min=-5.67005, rayleigh_mean=-4.68399, hvp_mean=18.7, overlap_lanczos32=0.936, overlap_scored=0.003, micro_remaining_0p2=113.202
- c60 / penalized_lanczos: n=18, rayleigh_min=-4.66161, rayleigh_mean=0.133856, hvp_mean=12.0, overlap_lanczos32=0.474, overlap_scored=0.128, micro_remaining_0p2=111.540
- c60 / pool_candidate: n=12, rayleigh_min=24.1535, rayleigh_mean=35.0393, hvp_mean=1.0, overlap_lanczos32=0.052, overlap_scored=0.119, micro_remaining_0p2=478.139
- c60 / reference_dimer: n=48, rayleigh_min=4.37148, rayleigh_mean=15.0985, hvp_mean=31.0, overlap_lanczos32=0.086, overlap_scored=0.052, micro_remaining_0p2=103.577
- c60 / scored_pool_selected: n=1, rayleigh_min=24.1538, rayleigh_mean=24.1538, hvp_mean=13.0, overlap_lanczos32=0.001, overlap_scored=1.000, micro_remaining_0p2=913.957
- cuo / constrained_ritz: n=12, rayleigh_min=-486.701, rayleigh_mean=-86.3235, hvp_mean=24.0, overlap_lanczos32=0.084, overlap_scored=0.212, micro_remaining_0p2=1.632
- cuo / lanczos: n=3, rayleigh_min=-486.826, rayleigh_mean=-480.901, hvp_mean=18.7, overlap_lanczos32=0.988, overlap_scored=0.198, micro_remaining_0p2=1.435
- cuo / penalized_lanczos: n=18, rayleigh_min=-486.845, rayleigh_mean=-486.696, hvp_mean=12.0, overlap_lanczos32=1.000, overlap_scored=0.204, micro_remaining_0p2=1.888
- cuo / pool_candidate: n=8, rayleigh_min=-19.4214, rayleigh_mean=72.4095, hvp_mean=1.0, overlap_lanczos32=0.093, overlap_scored=0.211, micro_remaining_0p2=135.632
- cuo / reference_dimer: n=48, rayleigh_min=-236.39, rayleigh_mean=-114.641, hvp_mean=31.0, overlap_lanczos32=0.244, overlap_scored=0.105, micro_remaining_0p2=27.865
- cuo / scored_pool_selected: n=1, rayleigh_min=-19.4245, rayleigh_mean=-19.4245, hvp_mean=9.0, overlap_lanczos32=0.203, overlap_scored=1.000, micro_remaining_0p2=2.511
- pdo / constrained_ritz: n=12, rayleigh_min=0.289732, rayleigh_mean=1.13657, hvp_mean=24.0, overlap_lanczos32=0.273, overlap_scored=0.098, micro_remaining_0p2=1.000
- pdo / lanczos: n=3, rayleigh_min=0.175093, rayleigh_mean=0.489412, hvp_mean=18.7, overlap_lanczos32=0.547, overlap_scored=0.065, micro_remaining_0p2=1.000
- pdo / penalized_lanczos: n=18, rayleigh_min=0.318779, rayleigh_mean=2.63437, hvp_mean=12.0, overlap_lanczos32=0.212, overlap_scored=0.168, micro_remaining_0p2=1.000
- pdo / pool_candidate: n=8, rayleigh_min=7.63603, rayleigh_mean=8.95968, hvp_mean=1.0, overlap_lanczos32=0.058, overlap_scored=0.187, micro_remaining_0p2=1.000
- pdo / reference_dimer: n=48, rayleigh_min=1.90416, rayleigh_mean=5.59154, hvp_mean=31.0, overlap_lanczos32=0.059, overlap_scored=0.043, micro_remaining_0p2=1.000
- pdo / scored_pool_selected: n=1, rayleigh_min=7.63771, rayleigh_mean=7.63771, hvp_mean=9.0, overlap_lanczos32=0.043, overlap_scored=1.000, micro_remaining_0p2=1.000

## Dimer Bias Readout

- C60: reference dimer with `a=500` is locked to the initial mode (`dot_initial=1.000`) and is not soft (`rayleigh` samples 23.7 to 39.7). Reducing the bias helps: `a=5` reaches `rayleigh_min=4.371`, but still does not reach Lanczos (`rayleigh_min=-5.670`) and never converges within 30 rotations.
- CuO: dimer can become much softer than the scored pool when `a` is reduced (`a=5` reaches `rayleigh_min=-236.390`), but Lanczos/penalized Lanczos find a far softer mode around `-486.8`. `a=500` remains too restrictive and gives mixed/weak results.
- PdO: `a=500` is again locked (`dot_initial=1.000`) and close to the scored pool scale. Lowering to `a=1` improves the best dimer rayleigh to `1.904`, but Lanczos still finds softer local modes (`0.175`).
- Across all systems, dimer convergence fraction is 0.0 at the tested 30-rotation budget. That means these are fixed-budget refined directions, not converged constrained minimum modes.

## Lanczos Readout

- Plain Lanczos is the strongest eigenmode solver in the local Rayleigh-quotient sense. It is dramatically softer than scored_pool on C60 and CuO and modestly softer on PdO.
- Penalized Lanczos often recovers most of the soft-mode benefit at lower HVP cost (`hvp_mean=12`) and can preserve some proposal prior, especially on CuO.
- The softest Lanczos directions are nearly orthogonal to scored_pool on C60/PdO (`overlap_scored` around 0.003 to 0.065). This is the key warning: pure eigen softness is not the same object as the current global-optimization direction.

## Global-Optimization Interpretation

- This experiment supports the user's concern that `a=500` is too large for reference dimer in MACE-era SSW. It locks the direction on C60/PdO and wastes many force calls.
- It also supports not replacing scored_pool with pure lowest-mode Lanczos directly. The lowest modes are mathematically softer, but they do not align with the directions that production SSW currently exploits.
- The production benchmark and this diagnostic point to a hybrid direction generator: keep scored_pool semantics, then add one or two penalized/constrained Lanczos candidates as extra candidates in the pool, with true physical curvature and explicit novelty/escape scoring.
- The next implementation candidate should be `scored_pool + penalized_lanczos candidate`, not `reference_dimer` as a replacement and not plain Lanczos as the only direction.
