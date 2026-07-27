# Safe L-BFGS history-capacity ablation

## Purpose

Determine whether retained accepted secant history is a positive contributor to
the observed fixed-proposal efficiency of `safe-lbfgs-total`.

This is a single-mechanism ablation. It does not introduce a new production
optimizer, tune optimizer constants, change the biased objective, or test
analytic bias-Hessian corrections.

## Frozen comparison

Compare two arms on the same 16 frozen C60/PdO
`ProposalRelaxationTask` payloads:

- `safe-total-gradient-history10`: the current safe L-BFGS kernel;
- `safe-total-gradient-history0`: the same kernel with no retained secant
  history.

Both arms must use the same:

- total-gradient `ProposalPotential.evaluate_parts` path;
- MACE model, input structures, task payloads, device, and precision;
- initial inverse scale, maximum atomic displacement, Armijo constant,
  backtracking factor, maximum line trials, force certificate, and
  `maxiter=400`;
- PBC/MIC branch handling, curvature acceptance diagnostics, observer, and
  budget accounting.

The only permitted algorithmic difference is whether an accepted secant remains
available to the next outer iteration.

## Minimal production interface

Add a keyword-only experimental argument to `Relaxer.relax`:

```python
_safe_lbfgs_history_limit: int | None = None
```

Its contract is:

- `None` resolves to the existing `_SAFE_LBFGS_MEMORY` value of 10;
- only integer `0` or the existing capacity `10` may be supplied explicitly;
- `0` leaves the history passed to every subsequent two-loop recursion empty;
- booleans, every other integer, non-integers, and explicit use with any
  optimizer other than `safe-lbfgs-total` fail closed.

The underscore is intentional: this is an experimental injection point, not a
user-facing scientific parameter. It must not be added to `SSWConfig`, YAML,
CLI, `RelaxOptimizer`, `SurfaceWalker`, or `proposal_replay`.

The safe relaxation loop continues to compute curvature acceptance and secant
telemetry when the capacity is zero. It only declines to retain the accepted
pair. Consequently the history-zero direction is the existing empty-history
direction

\[
p_k=-\frac{1}{70}g_k
\]

followed by the unchanged atomic-step limit and Armijo line search.

## Run-local experiment

Create a new run directory rather than changing the prior observer experiment.
The runner may reuse the frozen tasks and trace recorder from
`runs/20260727-proposal-energy-traces`, but it must write a new atomic output
ledger.

Every row records at least:

- task ID and immutable task hash;
- `optimizer_kernel="safe-lbfgs-total"`;
- arm ID and resolved history limit;
- objective and safe-kernel constant descriptors;
- model/input/CUDA provenance;
- force-evaluation count, wall time, certificate, termination reason, endpoint,
  telemetry, and zero-call observer trace.

For every arm, the following equality is a hard validity gate:

\[
N_\mathrm{trace}
=N_\mathrm{EvalCounter}
=N_\mathrm{RelaxTelemetry}.
\]

The current history-10 result is rerun in the same experiment. The previous
trace ledger is descriptive provenance, not a deterministic oracle, because
float32 threshold jitter has already been observed.

## Analysis

The primary evidence is paired task-level force evaluations required to reach
the force certificate. Certificate failure and termination reason are reported
before cost comparisons; a failed arm is not presented as a cheap success.

Secondary evidence includes:

- wall time;
- total biased-energy trajectory and accepted-step monotonicity;
- line-search evaluations and callback-nonaccepted evaluations;
- accepted/rejected secants and MIC resets;
- endpoint energy and MIC-aware displacement between the two arms.

No weighted composite score or automatic promotion threshold is introduced.
Results are reported separately for C60 and PdO and at task level.

## Required tests

1. Default `None` and explicit history limit 10 are identical on a deterministic
   analytic objective.
2. Limits 0 and 10 are identical through the first outer iteration.
3. An anisotropic quadratic objective diverges only after an accepted secant can
   affect the next iteration.
4. A spy verifies that the two-loop recursion always receives an empty history
   for limit 0.
5. History zero preserves line-search accounting, curvature diagnostics, force
   termination, and MIC reset behavior.
6. Invalid limits and use with other optimizers fail before evaluator calls.
7. Both GPU arms satisfy the exact three-way call-ledger equality.
8. The manifest binds task, objective, kernel, model, input, and CUDA
   provenance.
9. Generated evidence and plots are deterministic and fail closed on incomplete
   matrices or malformed types.
10. The full test suite and stacked diff checks pass.

## Claim ceiling

This experiment can determine whether retaining prior accepted total-gradient
secants changes cost and certificate coverage for the frozen biased-proposal
matrix.

It cannot establish:

- generic superiority of BFGS or L-BFGS;
- same-basin acceleration when endpoints differ;
- the separate contributions of scalar inverse scaling and multi-secant
  corrections;
- improvement of complete SSW exploration;
- validity of bias separation or an analytic bias-Hessian method;
- statistical generalization beyond the tested tasks and float32 GPU runtime.
