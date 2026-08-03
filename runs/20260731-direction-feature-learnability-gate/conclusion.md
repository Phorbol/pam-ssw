# Direction feature learnability result

## Decision

- Allow posterior-selector stage: **False**.
- New force evaluations: 0.
- Dataset: 48 candidates in 12 shared K4 groups.

## Held-out results

| Split | Model | Accuracy | Mean regret (eV) | Median regret (eV) |
|---|---|---:|---:|---:|
| group | static_score | 0.167 | 2.615459 | 1.952148 |
| group | softness | 0.417 | 2.501924 | 1.952148 |
| group | intent | 0.000 | 3.289463 | 1.996452 |
| group | combined | 0.167 | 3.680995 | 2.449959 |
| context | static_score | 0.167 | 2.615459 | 1.952148 |
| context | softness | 0.333 | 2.300051 | 1.952148 |
| context | intent | 0.000 | 3.321924 | 2.058990 |
| context | combined | 0.000 | 3.301505 | 1.996452 |
| system | static_score | 0.167 | 2.615459 | 1.952148 |
| system | softness | 0.500 | 1.891186 | 0.937523 |
| system | intent | 0.250 | 2.125651 | 1.059982 |
| system | combined | 0.250 | 2.125651 | 1.059982 |

## Leave-system combined model

| Held-out system | Accuracy | Mean regret (eV) | Median regret (eV) |
|---|---:|---:|---:|
| c60 | 0.333 | 3.526159 | 1.799545 |
| pdo | 0.167 | 0.725143 | 0.625732 |

The promotion decision is determined only by the fixed combined
ridge model under leave-system-out validation. Group and context
splits are diagnostics and cannot override a failed system holdout.
No hyperparameter, feature, or model family was selected after
observing the result.
