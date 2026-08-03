# Prospective true-curvature ranker result

## Decision

- Promote true-curvature ranker as production default: **False**.
- Cases: 48/48 certified and 48/48 geometry-valid.
- Total cost: 14775 force evaluations, 302.6 s.

| System | True-curvature wins | Mean ΔΔE (eV) | Median ΔΔE (eV) | Δ force eval | Pareto |
|---|---:|---:|---:|---:|---|
| c60 | 2/6 | -0.173236 | 0.718102 | -9 | False |
| pdo | 4/6 | -0.513982 | -0.707047 | -3 | True |

True curvature is a useful PdO mechanism but not a universal
replacement for intent-preserving static selection. C60's negative
mean is driven by one large win while its median and four of six
paired groups are worse. The default therefore remains
`static_score`.

This closes the single-ranker branch. The next selector experiment
must preserve the complementary intent and softness hypotheses
without introducing a tuned weighted blend: either an explicitly
costed two-arm racing policy or a contextual posterior that first
passes offline held-out calibration.
