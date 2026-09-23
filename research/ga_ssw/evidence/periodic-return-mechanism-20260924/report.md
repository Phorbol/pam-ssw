# Periodic return mechanism: zero-PES diagnostic

This is a selected diagnostic sample (first three completed `gaussian_limit` records per arm), not a whole-population return/basin estimate. Matching is approximate geometry; ordered MIC displacement is not permutation invariant. Energies are reported by role and not used to rank sub-MLIP differences.

| Case | Method | Requests | Complete / censored records | Sampled records | Stop checks consistent / inconsistent / uncheckable |
|---|---|---:|---:|---:|---:|
| aloh3 | ritz | 12000 | 13 / 1 | 3 | 13 / 0 / 0 |
| aloh3 | recovered | 12000 | 22 / 1 | 3 | 22 / 0 / 0 |
| brookite48 | ritz | 12000 | 16 / 1 | 3 | 16 / 0 / 0 |
| brookite48 | recovered | 12000 | 29 / 1 | 3 | 29 / 0 / 0 |

## aloh3 / ritz

| Record | Requests | Current E | Biased total E | Physical E at biased endpoint | True landing E | Status | Current~work tight/broad | Current~landing tight/broad | Work~landing tight/broad |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 868 | -177.56717351223392 | -175.19726064576912 | -175.27740312302774 | -177.56743385376285 | gaussian_limit | False/False | True/True | False/False |
| 2 | 965 | -177.56743385376285 | -176.2935828586351 | -176.36127336870283 | -177.3300634451495 | gaussian_limit | False/False | False/False | False/False |
| 3 | 1069 | -177.56743385376285 | -175.46388277309967 | -175.50456815552053 | -177.56735622614008 | gaussian_limit | False/False | True/True | False/False |

## aloh3 / recovered

| Record | Requests | Current E | Biased total E | Physical E at biased endpoint | True landing E | Status | Current~work tight/broad | Current~landing tight/broad | Work~landing tight/broad |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 512 | -177.566952554733 | -176.47243308449117 | -176.53831871634264 | -177.56695612909118 | gaussian_limit | False/False | True/True | False/False |
| 2 | 539 | -177.56695612909118 | -175.28662618818115 | -175.39172364434273 | -177.32899770721883 | gaussian_limit | False/False | False/False | False/False |
| 3 | 637 | -177.56695612909118 | -175.93498206379815 | -175.98483001592098 | -177.22432694383724 | gaussian_limit | False/False | False/False | False/False |

## brookite48 / ritz

| Record | Requests | Current E | Biased total E | Physical E at biased endpoint | True landing E | Status | Current~work tight/broad | Current~landing tight/broad | Work~landing tight/broad |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 0 | 715 | -427.59036474751946 | -421.95585118151837 | -422.1271342874134 | -427.59023942851314 | gaussian_limit | False/False | True/True | False/False |
| 1 | 695 | -427.59023942851314 | -422.4502968322169 | -422.6416949897583 | -427.5912905410551 | gaussian_limit | False/False | True/True | False/False |
| 2 | 727 | -427.5912905410551 | -422.4731546495148 | -422.6631625233005 | -427.59091016229337 | gaussian_limit | False/False | True/True | False/False |

## brookite48 / recovered

| Record | Requests | Current E | Biased total E | Physical E at biased endpoint | True landing E | Status | Current~work tight/broad | Current~landing tight/broad | Work~landing tight/broad |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 0 | 408 | -427.59036474751946 | -418.3709775594562 | -418.5167843247728 | -427.590820305591 | gaussian_limit | False/False | True/True | False/False |
| 1 | 372 | -427.590820305591 | -418.8116646789335 | -419.19616437029 | -425.6262458908399 | gaussian_limit | False/False | False/False | False/False |
| 2 | 360 | -427.590820305591 | -418.3497383347633 | -418.6311446668515 | -427.5911092754902 | gaussian_limit | False/False | True/True | False/False |

Stop-sign details, full per-record energy changes, geometry displacements, missing comparisons, and all request costs are preserved in `analysis.json`.
