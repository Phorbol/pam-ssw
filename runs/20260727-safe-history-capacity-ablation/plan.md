# Safe L-BFGS history-capacity ablation (G1, one execution only)

## Question and fixed protocol

For the frozen 16 one-bias proposal-relaxation tasks from the pinned fixed-replay
summary, does the safe total-gradient L-BFGS history capacity change the
certificate-qualified replay outcome?  The only arms are
`safe-total-gradient-history10` (`safe-lbfgs-total`, history limit 10) and
`safe-total-gradient-history0` (`safe-lbfgs-total`, history limit 0).  Both
run each C60/PdO task exactly once with `maxiter=400`.

## G1 execution gate

This driver is an execution artifact, not an auto-tuner.  It validates source
summary, task IDs/seeds/payload hashes, model/input hashes, and CUDA provenance
before constructing any calculator.  One complete 32-row ledger is published
by same-parent staging-directory rename only after all rows validate.  There
is no retry, resubmission, parameter adjustment, seed substitution, or
capacity expansion after observing an adverse/inconclusive result; such a
result is a valid negative G1 outcome.

The first physical attempt reached the fixed replay matrix but stopped at a
runner protocol bug: a finite, ledger-closed `maxiter` result was treated as
an invalid ledger instead of a reported scientific outcome.  It produced no
published artifact and is not a result.  Re-running the complete fixed matrix
after this protocol repair is required to obtain the first valid attempt; it
does not change tasks, arms, parameters, or constitute retuning.

## Claim ceiling

The resulting data can establish only fixed-task, fixed-model, CUDA replay
behaviour for C60 and PdO under the listed objective constants and certificate.
It does not establish endpoint equivalence, broader SSW/search improvement,
statistical significance, or a production-default change.

## Pre-execution checks

1. Focused runner tests (including fail-closed provenance and atomic publish),
   prior trace tests, and relax history-limit tests pass.
2. The diff is whitespace-clean and contains no production `pamssw` change or
   raw output.
3. A reviewer launches this one GPU execution only after the gate is accepted.
