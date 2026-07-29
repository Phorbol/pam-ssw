# Safe L-BFGS inverse-scale decomposition (G1, one execution only)

## Question and fixed protocol

For the frozen 16 one-bias proposal-relaxation tasks from the pinned fixed-replay
summary, how much of the history-enabled result is explained by scalar inverse
scaling rather than two-loop corrections?  The only arms are fixed scale with
history zero, latest-accepted-secant adaptive scale with history zero, and the
existing adaptive-scale history-10 kernel.  All three run each C60/PdO task
exactly once with `maxiter=400`.

## G1 execution gate

This driver is an execution artifact, not an auto-tuner.  It validates source
summary, task IDs/seeds/payload hashes, model/input hashes, and CUDA provenance
before constructing any calculator.  One complete 48-row ledger is published
by same-parent staging-directory rename only after all rows validate.  There
is no retry, resubmission, parameter adjustment, seed substitution, or
capacity expansion after observing an adverse/inconclusive result; such a
result is a valid negative G1 outcome.

No execution retry or parameter adjustment is permitted after the first
physical call.  A finite, ledger-closed `maxiter` row is a valid incomplete
scientific outcome rather than a protocol failure.

## Claim ceiling

The resulting data can establish only fixed-task, fixed-model, CUDA replay
behaviour for C60 and PdO under the listed objective constants and certificate.
It does not establish endpoint equivalence, broader SSW/search improvement,
statistical significance, or a production-default change.

## Pre-execution checks

1. Focused runner tests (including fail-closed provenance and atomic publish),
   prior trace tests, and relax history-limit tests pass.
2. The diff is whitespace-clean; the only production `pamssw` change is the
   reviewed private scale-only experiment seam, and no raw output is tracked.
3. A reviewer launches this one GPU execution only after the gate is accepted.
