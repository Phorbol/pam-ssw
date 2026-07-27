# Safe L-BFGS inverse-scale decomposition — evidence-limited conclusion

## Certificate outcome

The frozen 16-task C60/PdO matrix produced 29/48 certificate-satisfied rows and 19/48 incomplete rows.  Finite `maxiter` and `line_search_failed` records remain valid incomplete outcomes; they are not converted into convergence.

Fixed scale, history 0: 3/16 certificate-satisfied rows, 12 `maxiter`, 1 `line_search_failed`, and 5988 evaluator calls.
Adaptive scale, history 0: 10/16 certificate-satisfied rows, 5 `maxiter`, 1 `line_search_failed`, and 4593 evaluator calls.
Adaptive scale, history 10: 16/16 certificate-satisfied rows, 0 `maxiter`, 0 `line_search_failed`, and 1468 evaluator calls.

## Cost outcome

Across all 16 tasks, fixed scale/history 0 used 5988 calls and 122.038499 s; adaptive scale/history 0 used 4593 calls and 89.977804 s; adaptive scale/history 10 used 1468 calls and 28.747531 s.
Each arm cost includes all 16 rows.  Costs of incomplete arms are protocol costs, not cheap-success comparisons or same-endpoint speedups.

## Fixed-matrix interpretation ceiling

Adaptive scalar scaling is a positive but partial contributor on this fixed matrix: relative to fixed scale/history 0, it raises certificate coverage from 3/16 to 10/16 and reduces recorded protocol cost from 5988 to 4593 evaluator calls.

Adaptive scaling alone is not sufficient to recover the existing kernel: history 10 reaches 16/16 certificates with 1468 calls, versus 10/16 and 4593 calls for scale-only.  The remaining two-loop correction stack is therefore a strong positive candidate on these tasks, but this experiment does not separate the newest correction from older retained pairs.

This is not a generic BFGS result, a same-basin equivalence result, a full-SSW performance result, or a statistical generalization claim.  It does not justify a production-default change or an analytic bias-Hessian method.

Accepted-state and explicit-finalization records are accounting annotations.
The trace figure visualizes exact evaluations and callback-observed annotations for accounting.  A callback-observed label is not an optimizer acceptance rule.  The figure supports no endpoint-equivalence inference.

The evidence remains limited to the pinned source tasks, CUDA/model/input provenance, and reviewed safe-L-BFGS kernel recorded in `evidence.json`.
