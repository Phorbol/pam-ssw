# Safe L-BFGS history-depth ablation

## Frozen experiment

Run exactly two `safe-lbfgs-total` arms over the pinned C60 and PdO tasks:

1. `adaptive-scale-history1`, `_safe_lbfgs_history_limit=1`;
2. `adaptive-scale-history10`, `_safe_lbfgs_history_limit=10`.

Seeds are 42 through 49, `maxiter=400`, for exactly 32 rows ordered by
system, arm, then seed. One calculator is created for each `(system, arm)` and
reused only by that arm's eight serial tasks.

## Minimal execution gate

Before calculator construction, verify the requested execution commit against
`HEAD`, require a clean tracked worktree, and verify the pinned source summary,
all task payloads, the pamssw source bundle, the external fixed-replay helper,
the model, and both inputs. Record Python, NumPy, SciPy, ASE, PyTorch, MACE,
and CUDA runtime/device facts.

Rows contain raw task, arm, result, telemetry, trace, accounting, endpoint, and
wall-time facts. The runner derives no certificate or termination aggregate.
The accounting closure is trace length = force evaluations = telemetry
`evaluator_calls` = biased-proposal-relax calls, with zero unattributed calls.
`backend_evaluations` remains raw telemetry and is not part of that equality.

## Completion and claim ceiling

Refuse existing `output` or `output.partial`. Write the two system ledgers to
`output.partial`, write `summary.json` last as the completion marker, then use
an ordinary directory rename to `output`. This is a single-process convention,
not a concurrent publication API.

Future analysis may establish only fixed-task, fixed-model CUDA replay
behavior for this matrix. It does not establish endpoint equivalence, general
search improvement, statistical significance, or a production-default change.
