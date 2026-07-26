# Analytic ablation result

Both custom modes passed the preregistered correctness gate:

- finite converged endpoints on both cases;
- exact objective-call ledger closure;
- no rejected secants or MIC branch resets;
- no fallback or non-finite termination.

Raw call counts were:

| Case | ASE FIRE | ASE FIRE2 | Total secant | Bias separated |
|---|---:|---:|---:|---:|
| anisotropic one hill | 150 | 111 | 14 | 18 |
| anisotropic two hills | 148 | 134 | 16 | 18 |

The result supports advancing the inseparable custom pair to the real GPU G1
screen. It does not support claiming that analytic bias separation is faster;
the total-secant control used fewer calls on both frozen analytic cases.
