# K4 action-breadth order-statistic result

- Live B2 gate allowed: **False**.
- New force evaluations: **0**.
- Production change allowed: **False**.

| Campaign | System | Static regret / FE | B2 regret / FE | Elasticity | Pass |
|---|---|---:|---:|---:|---|
| first | c60 | 2.818085 / 423.0 | 1.101476 / 700.5 | 0.929 | False |
| first | pdo | 1.159515 / 234.5 | 0.339213 / 390.5 | 1.063 | True |
| second | c60 | 3.074295 / 415.0 | 1.182409 / 788.0 | 0.685 | False |
| second | pdo | 1.609680 / 240.5 | 0.429047 / 379.8 | 1.267 | True |

B3/B4 are diagnostics only. This gate does not claim batched GPU speedup and does not change any production default.
