# Safe L-BFGS history-capacity ablation — evidence-limited conclusion

## Certificate outcome

The frozen 16-task C60/PdO matrix produced 20/32 certificate-satisfied rows and 12/32 finite `maxiter` rows.  A `certificate_satisfied: false` / `maxiter` record is retained as a valid, incomplete outcome; it is not converted into convergence.

History 10: 16/16 certificate-satisfied rows and 1521 evaluator calls.
History 0: 4/16 certificate-satisfied rows, 12 finite `maxiter` rows, and 5969 evaluator calls.

## Cost outcome

Across this fixed matrix, history 10 used 1521 evaluator calls and 33.919626 s, while history 0 used 5969 calls and 112.000601 s.  These are recorded replay costs, not an optimization score.
Each arm cost includes all 16 rows; the history-0 cost includes its incomplete rows and is not a cheap-success comparison.

## Fixed-matrix interpretation ceiling

The positive retained-history contribution is limited to this fixed matrix: the history-10 arm preserves the certificate-qualified outcomes and incurs fewer recorded evaluator calls than the history-0 arm.  This is not a generic result, a same-basin equivalence result, a full-SSW performance result, or an inferential claim.

The history-0 arm recorded 4764 accepted secants, but those secants are not retained history.  Accepted-state and explicit-finalization records are reported only as accounting statistics; no trace curve or endpoint-equivalence inference is made.

The evidence remains limited to the pinned source tasks, CUDA/model/input provenance, and reviewed safe-L-BFGS kernel recorded in `evidence.json`.
