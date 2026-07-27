# Safe L-BFGS history-depth ablation (G1, one execution only)

## Frozen question and matrix

For the pinned 16 one-bias proposal-relaxation tasks, compare only adaptive
safe-total L-BFGS with history depth one and history depth ten.  The frozen
matrix is `SYSTEMS=(c60,pdo)`, seeds 42 through 49, `MAXITER=400`, and exactly
32 rows in this order: system, arm, seed.

1. `adaptive-scale-history1`: `safe-lbfgs-total`, history limit 1,
   `latest-history-pair-gamma-plus-one-two-loop-correction`.
2. `adaptive-scale-history10`: `safe-lbfgs-total`, history limit 10,
   `latest-history-pair-gamma-plus-up-to-ten-two-loop-corrections`.

Both use the total-biased-gradient secant.  Neither enables
`_safe_lbfgs_adaptive_scale_without_history`.

## Execution gate

`run_gpu_ablation.py` fails closed before calculator construction: it checks
arguments/non-overwrite, frozen source and every canonical task hash, helper
and import roots, the pinned pamssw bundle, execution commit and clean
worktree, runtime/platform schemas, model/input hashes, and CUDA provenance.
`--expected-git-commit` is mandatory; `--preflight-only` performs all gates
and creates zero calculators.  The runner does not embed a self-referential
execution commit.

Every `(system, arm)` has exactly one calculator which is serially reused for
its eight tasks; systems and arms are serial.  Rows retain task payload/hash,
requested/resolved arm policy, result/certificate/termination, complete
zero-extra-call trace, callback hashes, telemetry, wall time, and endpoint
positions/hash.  Finite `maxiter` and `line_search_failed` rows are reported
as incomplete outcomes; non-finite values, open accounting, or schema drift
are fatal.  A complete 32-row ledger and validated summary are staged beside
the target and published only by directory rename.

## Claim ceiling

Any future output establishes only fixed-task, fixed-model CUDA replay
behaviour for these C60/PdO tasks and this pinned objective.  It does not
establish endpoint equivalence, search improvement, statistical significance,
or a production-default change.
