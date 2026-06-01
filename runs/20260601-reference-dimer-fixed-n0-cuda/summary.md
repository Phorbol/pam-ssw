# Reference Dimer Component Ablation

- Status: completed
- Completed cases: 6

## Verification

- Targeted fixed-N0 test: `1 passed`
- Reference-dimer unit tests: `14 passed`
- Full unit suite: `412 passed`

## Aggregate

| variant | cases | mean best eV | best single eV | mean minima | mean dup | mean force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-N0 reference_dimer bias_relax | 3 | -495.878 | -498.939 | 24.7 | 0.398 | 13721 |
| fixed-N0 reference_dimer Direct-QP adaptive50 | 3 | -474.669 | -474.670 | 1.7 | 0.959 | 12923 |

Previous comparison points from `runs/20260601-reference-dimer-component-ablation/results.csv`:

| variant | mean best eV | best single eV | mean minima | mean dup | mean force |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous reference_dimer bias_relax | -493.888 | -499.262 | 18.0 | 0.561 | 14134 |
| previous reference_dimer Direct-QP adaptive50 | -474.670 | -474.670 | 2.0 | 0.951 | 14474 |
| current scored_pool bias_relax baseline | -499.847 | -504.182 | 35.3 | 0.138 | 21502 |

## Diagnostics

- `reference_dimer_mean_rotations = 15.0` for every fixed-N0 case.
- `reference_dimer_converged_fraction = 0.0` for every fixed-N0 case.
- `reference_dimer_mean_abs_dot_initial = 0.99883-0.99885`.
- `reference_dimer_mean_curvature = 1961-1966 eV/A^2`.
- Direct-QP remained collapsed: `direct_qp_mean_progress = 0.1227-0.1230`, `direct_qp_mean_model_error = 11.6-18.7`, duplicate rate `0.927-0.976`.

## Verdict

The integration bug was real: the previous reference-dimer path sampled a new initial mode `N0` on every internal climb step. Reusing one `N0` per walk improves the bias-relax variant's mean best energy and coverage, but it does not make dimer competitive with scored-pool production on C60.

The remaining failure is algorithmic: the biased dimer still stays nearly parallel to its initial mode and reports curvature around `~2000 eV/A^2`. Direct-QP should not consume that biased curvature as a numerical Hessian estimate.

## Rows

- c60 / paw_ucb_reference_dimer_bias_relax / seed 0: best=-495.792175, n_minima=31, force=14701, wall=318.7s
- c60 / paw_ucb_reference_dimer_bias_relax / seed 1: best=-498.938568, n_minima=18, force=13208, wall=243.2s
- c60 / paw_ucb_reference_dimer_bias_relax / seed 2: best=-492.90271, n_minima=25, force=13253, wall=247.5s
- c60 / paw_ucb_reference_dimer_direct_qp_adaptive50 / seed 0: best=-474.669617, n_minima=3, force=12759, wall=237.1s
- c60 / paw_ucb_reference_dimer_direct_qp_adaptive50 / seed 1: best=-474.669586, n_minima=1, force=12916, wall=247.2s
- c60 / paw_ucb_reference_dimer_direct_qp_adaptive50 / seed 2: best=-474.66925, n_minima=1, force=13094, wall=224.8s
