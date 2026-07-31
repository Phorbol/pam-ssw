# Initial K4 HVP value-of-information result

## Decision

- Live selected-only-HVP gate allowed: **False**.
- Initial all-candidate HVP deletion supported: **False**.
- New force evaluations: **0**.
- Scope: initial K4 candidate ranking only; continuation HVP policy is unchanged in every forced trajectory.

## Paired shared-pool readout

| System | Rule | Median regret (eV) | Median projected FE | Δ regret vs K4 (eV) | Δ FE vs K4 | Pass |
|---|---|---:|---:|---:|---:|---|
| c60 | static_k4 | 2.895081 | 419.0 | 0.000000 | 0.0 | baseline |
| c60 | uniform_no_hvp | 3.212437 | 386.2 | 0.317356 | -32.8 | False |
| c60 | family_rotation_no_hvp | 2.913704 | 396.4 | 0.018623 | -22.6 | False |
| pdo | static_k4 | 1.281876 | 237.5 | 0.000000 | 0.0 | baseline |
| pdo | uniform_no_hvp | 1.199280 | 187.7 | -0.082596 | -49.8 | True |
| pdo | family_rotation_no_hvp | 1.208580 | 213.4 | -0.073296 | -24.1 | True |

## Independent live D0/K4 anchor

| System | Pairs | Median D0−K4 landing ΔE (eV) | Median D0−K4 FE |
|---|---:|---:|---:|
| c60 | 6 | 2.026962 | -32.5 |
| pdo | 6 | -0.527863 | 64.5 |

The D0 comparison is kept separate because its exact-anchor action is
not one of the four shared K4 candidates. The replay changes only the
initial candidate-selection rule; all continuation direction searches,
biased relaxations, and true quenches are inherited unchanged.
