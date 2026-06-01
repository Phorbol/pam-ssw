# Reference Dimer Hybrid Curvature C60

- Status: completed
- Unit suite: `418 passed`
- Case: C60, seed0, 40 trials, CUDA, `paw_ucb_reference_dimer_bias_relax`

## Result

| variant | best eV | minima | duplicate rate | force evals | early stops | bias steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference_dimer hybrid | -495.800 | 26 | 0.366 | 16432 | 12 | 145 |

## Curvature And Bias

| metric | value |
| --- | ---: |
| true curvature mean | -31.531 |
| biased/effective curvature mean | 1963.878 |
| direction choice curvature mean | 1963.878 |
| bias weight mean | 10.0 |
| bias weight max | 10.0 |
| mean abs dot to initial N0 | 0.998852 |
| mean rotations | 15.0 |
| converged fraction | 0.0 |

## Comparison

| run | best eV | minima | duplicate rate | force evals |
| --- | ---: | ---: | ---: | ---: |
| current C_true everywhere | -474.669 | 2 | 0.951 | 20900 |
| hybrid C_true trust + C_eff weight | -495.800 | 26 | 0.366 | 16432 |
| old fixed-N0 C_eff everywhere | -495.792 | 31 | 0.244 | 14701 |
| scored_pool current seed0 | -500.467 | 36 | 0.122 | 19144 |

## Verdict

The hybrid split fixes the `bias_weight=0` failure. Reference dimer returns from `-474.669` to about `-495.8`, so Gaussian height must not use raw `C_true` under the current weight rule.

However, hybrid does not make dimer competitive with scored_pool. The direction still does not rotate (`dot~0.999`, `converged=0`) and duplicates remain much higher than scored_pool. The useful decomposition is:

- trust/QP/model curvature: `C_true`
- Gaussian height for reference_dimer bias-relax: `C_eff` for now
- diagnostics: record both

This supports keeping reference_dimer experimental and focusing production effort on scored_pool + Direct-QP.
