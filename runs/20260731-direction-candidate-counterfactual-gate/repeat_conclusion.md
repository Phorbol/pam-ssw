# Repeated shared-candidate direction gate

## Result

- Static selector inadequacy supported: **True**.
- Posterior selector promotion allowed: **False**.
- Stable static-winner misses: 10/12 shared pools.
- Stable best-candidate identity: 9/12 pools.
- Median absolute repeat difference in landing delta: 0.000198 eV.
- Maximum absolute repeat difference in landing delta: 5.503296 eV.
- Prospective true-curvature ablation allowed: **True**.
- Direction-family posterior promotion allowed: **False**.
- Adaptive quadratic score cost range: 2.285e-08 eV around 0.800000 eV.

## Per-run system gate

| System | Run 1 | Run 2 | Stable winner misses |
|---|---|---|---:|
| c60 | selection_bottleneck | selection_bottleneck | 5 |
| pdo | selection_bottleneck | ambiguous | 5 |

## Zero-extra-FE offline rankers

| Ranker | Top-1 hits | Median regret (eV) | Mean regret (eV) |
|---|---:|---:|---:|
| static_score | 2/12 | 1.952148 | 2.615459 |
| inner_curvature | 5/12 | 1.601433 | 2.001837 |
| true_curvature | 6/12 | 0.937523 | 1.891186 |
| random_then_static | 4/12 | 0.631737 | 2.341090 |
| loo_beta_family | 4/12 | 0.631737 | 2.341090 |

The two mechanisms must be kept separate. Better candidates are
already present in the K4 pool often enough to reject candidate
generation as the sole bottleneck. However, PdO's score-ordering
sign is not repeat-stable, so the preregistered cross-system gate
does not yet authorize an online UCB/TS or learned selector.
The leave-one-group-out beta rule selected the random family in
every held group, so its apparent aggregate gain is a fixed global
family prior rather than context-sensitive posterior learning.

The next bounded step is therefore a prospective, fixed-budget
ablation of the already-paid true-curvature ranker against the
current static score. A new force probe or online posterior is not
justified before that simpler physical baseline is tested.
