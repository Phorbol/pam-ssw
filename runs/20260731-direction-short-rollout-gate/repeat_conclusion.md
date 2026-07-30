# Two-arm short-uphill-rollout result

## Decision

- Allow implementation of online state-reusing racing: **False**.
- Probe quality: 96/96 geometry-valid, 96/96 exact first-direction matches.
- Probe cost: 8400 force evaluations, 189.2 s serial wall.

Primary rule: choose the larger true-PES energy rise after H2.

| System | Larger-rise accuracy | Lower-rise diagnostic | Stable | Median regret (eV) | Mean vs static (eV) | Median vs static (eV) | Mean FE overhead | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c60 | 0.000 | 1.000 | 0.667 | 1.268608 | 1.037430 | 0.718102 | 105.0 | False |
| pdo | 0.333 | 0.667 | 0.833 | 0.411148 | 0.064840 | 0.000000 | 150.2 | False |

H1 is diagnostic only. No alternative metric or horizon may be
substituted after seeing this result. A passing result permits
only the next fixed-total-FE end-to-end test; it does not change
the production selector by itself.
