# Prospective true-curvature ranker ablation

The repeated shared-candidate gate showed that adaptive score scaling makes
`0.5 * sigma_score**2 * curvature` exactly `0.8 eV` for every positive-
curvature K4 candidate. The current composite score therefore pays for an HVP
but cancels curvature discrimination in the regime actually observed.

This experiment changes one thing only:

- `static_score`: current composite score and adaptive score sigma;
- `true_curvature`: select the minimum true-PES curvature already returned by
  the same central-FD candidate HVP.

Both arms retain K4, H8, the same direction generator, local softening,
serial-Gaussian uphill policy, proposal optimizer, true quench, starter, seed,
and candidate force cost. No posterior, new direction source, or extra force
probe is introduced.

The matrix has C60 and PdO, locked trial-100 and trial-180 states, seeds
42–44, and both arms: 24 cases per execution. It is executed twice because
nonlinear uphill termination amplified small GPU floating-point differences
in the preceding gate.

Promotion is deliberately Pareto-based, without a weighted acquisition score:
for each system independently, repeat-averaged true curvature must have
non-positive mean and median landing-energy differences and no greater total
force cost, with at least one strict improvement. Otherwise the result remains
experimental and the production default stays unchanged.
