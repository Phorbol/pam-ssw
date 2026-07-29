# Proposal-fmax fixed-budget GPU ablation

## Verified execution facts

- 16/16 published campaigns passed exact manifest/index, output-path, event/action/summary FE-ledger, uniform snapshot, and action-scoped optimizer-diagnostic validation.
- Both arms use `safe-lbfgs-total`; true quench is `ase-lbfgs` with one `ase-fire` fallback, `fmax=0.01 eV/A`, and `maxiter=400`.
- 199 actions completed; failed actions are 0; diagnostics report 6/6 converged fallback uses.

| system | arm | n | attempts mean/median | total FE mean/median | unused FE mean/median | wall s mean/median | archive mean/median | duplicate mean/median | best drop eV mean/median | fallback mean/median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c60 | proposal-fmax-0.05 | 4 | 9.500000/9.500000 | 5236.500000/5253.500000 | 763.500000/746.500000 | 96.808707/92.636819 | 10.500000/10.500000 | 0.000000/0.000000 | 13.223206/15.533966 | 0.000000/0.000000 |
| c60 | proposal-fmax-0.10 | 4 | 10.000000/10.000000 | 5151.000000/5142.500000 | 849.000000/857.500000 | 90.086547/90.514527 | 10.250000/10.500000 | 0.066667/0.050000 | 12.458229/11.647110 | 0.000000/0.000000 |
| pdo | proposal-fmax-0.05 | 4 | 15.750000/15.500000 | 5217.750000/5266.500000 | 782.250000/733.500000 | 119.859296/119.262355 | 14.500000/15.000000 | 0.133591/0.167183 | 3.109985/3.110535 | 0.500000/0.000000 |
| pdo | proposal-fmax-0.10 | 4 | 14.500000/15.000000 | 5357.500000/5347.000000 | 642.500000/653.000000 | 123.162239/121.402247 | 14.250000/14.500000 | 0.081976/0.067873 | 3.470398/3.531494 | 1.000000/1.000000 |

## Purpose-ledger FE mean/median

| system | arm | bootstrap true quench | starter true quench | direction oracle | escape true PES | biased proposal relax | landing true quench | post-relax validation | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c60 | proposal-fmax-0.05 | 48.000000/48.000000 | 9.500000/9.500000 | 1476.000000/1488.000000 | 65.000000/67.500000 | 2964.250000/2953.500000 | 653.750000/633.000000 | 20.000000/20.000000 | 0.000000/0.000000 |
| c60 | proposal-fmax-0.10 | 48.000000/48.000000 | 10.000000/10.000000 | 1696.000000/1656.000000 | 74.250000/73.500000 | 2504.750000/2458.000000 | 797.000000/874.500000 | 21.000000/21.000000 | 0.000000/0.000000 |
| pdo | proposal-fmax-0.05 | 83.000000/83.000000 | 31.500000/31.000000 | 459.000000/463.000000 | 39.250000/39.500000 | 2483.500000/2585.000000 | 2089.000000/2034.000000 | 32.500000/32.000000 | 0.000000/0.000000 |
| pdo | proposal-fmax-0.10 | 83.000000/83.000000 | 29.000000/30.000000 | 533.500000/553.000000 | 42.500000/44.000000 | 1927.750000/1950.500000 | 2711.750000/2811.500000 | 30.000000/31.000000 | 0.000000/0.000000 |

## Paired loose-minus-strict (0.10 - 0.05)

| system | n | d total FE mean/median | signs -/0/+ | d best drop eV mean/median | signs -/0/+ | d wall s mean/median | signs -/0/+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| c60 | 4 | -85.500000/-132.000000 | 3/0/1 | -0.764977/-2.054199 | 3/0/1 | -6.722160/-3.415042 | 3/0/1 |
| pdo | 4 | 139.750000/118.500000 | 2/0/2 | 0.360413/0.299866 | 2/0/2 | 3.302943/3.986383 | 1/0/3 |

## Claim boundary

- These are small-n (four paired seeds per system) descriptive summaries. GPU nondeterminism is not controlled away and can be amplified by PES exploration.
- No inferential p-value is calculated; this analysis establishes neither an arm ranking nor a default decision.
- FE totals come from the closed purpose ledger. Optimizer diagnostics are action-scoped final-backend telemetry and are not an additive cost ledger when fallback occurs.
