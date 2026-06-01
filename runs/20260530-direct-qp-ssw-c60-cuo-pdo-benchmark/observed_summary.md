# Direct-QP SSW Observed Results

- Source: scan of output/*/ssw_summary.json
- Device: CPU in this environment; CUDA unavailable / NVML blocked
- Long matrix was stopped after completed C60 and CuO 40-trial direct_qp_default cases because CPU-only runtime was multi-hour and evidence was already negative.

## Aggregate

- c60 / direct_qp_default / trials1: n=1, mean_best=-474.669067 eV, best_single=-474.669067 eV, mean_force=175.0, mean_dup=0.500, direct_steps=1.0, direct_reject=0.0, wall=12.4s
- c60 / direct_qp_default / trials40: n=1, mean_best=-475.251404 eV, best_single=-475.251404 eV, mean_force=11564.0, mean_dup=0.927, direct_steps=307.0, direct_reject=4.0, wall=761.0s
- c60 / direct_qp_strong / trials1: n=1, mean_best=-474.669067 eV, best_single=-474.669067 eV, mean_force=147.0, mean_dup=0.500, direct_steps=1.0, direct_reject=0.0, wall=10.2s
- cuo / direct_qp_default / trials1: n=1, mean_best=-198.673187 eV, best_single=-198.673187 eV, mean_force=136.0, mean_dup=0.000, direct_steps=1.0, direct_reject=0.0, wall=19.7s
- cuo / direct_qp_default / trials40: n=1, mean_best=-201.044922 eV, best_single=-201.044922 eV, mean_force=8142.0, mean_dup=0.707, direct_steps=320.0, direct_reject=0.0, wall=1201.0s
- cuo / direct_qp_strong / trials1: n=1, mean_best=-198.674713 eV, best_single=-198.674713 eV, mean_force=140.0, mean_dup=0.000, direct_steps=1.0, direct_reject=0.0, wall=20.5s
- pdo / direct_qp_default / trials1: n=1, mean_best=-568.372925 eV, best_single=-568.372925 eV, mean_force=75.0, mean_dup=0.000, direct_steps=1.0, direct_reject=0.0, wall=9.6s
- pdo / direct_qp_strong / trials1: n=1, mean_best=-568.372131 eV, best_single=-568.372131 eV, mean_force=76.0, mean_dup=0.000, direct_steps=1.0, direct_reject=0.0, wall=9.6s
