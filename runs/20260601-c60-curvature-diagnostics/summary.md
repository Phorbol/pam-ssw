# C60 Direction Curvature Diagnostics

- Status: completed
- Completed cases: 2
- Unit suite: `417 passed`

## Search Results

| variant | best eV | minima | duplicate rate | force evals | walk early stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| scored_pool bias_relax | -500.467 | 36 | 0.122 | 19144 | 4 |
| reference_dimer bias_relax | -474.669 | 2 | 0.951 | 20900 | 0 |

## Curvature Comparison

Scored pool selected directions:

| kind | count | true curvature mean | true min | true max |
| --- | ---: | ---: | ---: | ---: |
| momentum | 115 | 3.586 | 0.459 | 24.947 |
| random | 39 | 39.555 | 31.980 | 46.451 |
| bond | 8 | 33.063 | 17.749 | 49.911 |

Reference dimer:

| quantity | value |
| --- | ---: |
| selected count | 278 |
| true curvature mean | -28.699 |
| true curvature min | -39.828 |
| true curvature max | -17.974 |
| biased/effective curvature mean | 1966.676 |
| mean abs dot to initial N0 | 0.998843 |
| mean rotations | 15.0 |

## Answer

Reference dimer is indeed much softer in physical curvature than scored_pool on this C60 run. Its true curvature is around `-29 eV/A^2`, while scored_pool mostly selects positive-curvature directions, especially momentum around `+3.6 eV/A^2`.

But lower curvature is not sufficient for useful SSW escape. The dimer direction remains locked to the initial random mixed mode (`dot~0.999`, no convergence), has no momentum or targeted bond semantics, and produced only 2 minima with 95% duplicates. Scored_pool wins despite not selecting the softest-curvature direction because the selected momentum/bond/random directions produce useful structural displacement and basin escape.

The early-exit default looks healthy in the scored_pool case: it stopped 4 walks early and still reached `-500.467 eV` with 36 minima and fewer force evaluations than the previous no-early-stop seed0 run.
