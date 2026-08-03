# K4 central-HVP GPU batch result

- ForceService gate allowed: **True**.
- Surviving batch sizes: **[2, 4, 8]**.
- Force evaluations: **384** (`unattributed=0`).
- Production change allowed: **False**.

| System | Batch | Equivalent | Median speedup | Force err | HVP rel err |
|---|---:|---|---:|---:|---:|
| c60 | 2 | True | 1.957 | 2.190e-05 | 5.586e-04 |
| c60 | 4 | True | 3.175 | 1.755e-05 | 5.712e-04 |
| c60 | 8 | True | 3.635 | 2.141e-05 | 5.895e-04 |
| pdo | 2 | True | 1.696 | 1.273e-05 | 2.013e-03 |
| pdo | 4 | True | 2.747 | 1.761e-05 | 2.010e-03 |
| pdo | 8 | True | 2.962 | 1.746e-05 | 2.098e-03 |

This is a fixed HVP execution microbenchmark. It does not reduce FE or establish terminal-search improvement.
