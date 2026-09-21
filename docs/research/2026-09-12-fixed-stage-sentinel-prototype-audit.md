# Audit of the opt-in fixed-stage sentinel prototype

> **Superseded for recovered lifecycle facts.** See
> [`fixed-stage-control-audit.md`](2026-09-12-fixed-stage-control-audit.md) and
> [`fixed-lbfgs-rc-lifecycle.md`](2026-09-12-fixed-lbfgs-rc-lifecycle.md) for
> the authoritative dispatch, status-bit, and `x_eval`/`x_next` interpretation.

Date: 2026-09-12. Design review only; no production change or PES run.

## Corrected snapshot and field-scope interpretation

The pre-dispatch snapshot is the already evaluated input `x_k` consumed by that
RC dispatch.  LBFGS can return an unmeasured `x_next` requesting a later energy /
gradient evaluation; a Python evaluator callback sees a trial only after that
trial has been evaluated.  This audit therefore makes no geometric
non-equivalence claim from “before” and “after” labels.

The material question is which fields the stage predicate reads.  At
`climb_convg:0x5cd187`, the force scan reads the structure force descriptor at
`object+0x1d0` (`fa`).  In the complete `bfgsdriver` path, the LBFGS call is at
`0x5b41cd`; the post-call path has no direct store to `object+0x1d0` and uses
optimizer work/gradient buffers.  Static evidence therefore supports that the
predicate sees the incoming modified-objective `fa` array, rather than a
separately reconstructed bare-force array.  This is a bounded static result;
indirect callees outside this path were not claimed impossible.

The scalar energy must be named `base_energy`, not `current_bare_energy`.  The
climb caller saves the incoming structure energy to `tene0` at
`0x5caf55–0x5caf71` (`object+0x1ac8`), and `native-gaussian-caller.md:62` states
that this incoming/base value may already include LS or another upstream term.
The research predicate now records this scope explicitly; callers claiming
all-mobile/no-LS must establish that condition separately.

Any sentinel design must still preserve the evaluated snapshot and true landing
certificate according to its explicitly chosen endpoint contract. It must not
call an optimizer trial “accepted”, and a finite stage stop is not convergence.

## Minimal safe prototype

Keep it opt-in and local to the existing `surface.quench` boundary:

1. Before starting a modified quench, copy the input `Atoms` plus its evaluated
   true E/F and cell/constraint contract.
2. Wrap the existing modified-surface evaluator. After each complete E/F result,
   compute a pure stage predicate using the returned force array.
3. Use the original max-component force definition for this experimental
   predicate. Count the evaluation before raising a private sentinel.
4. Have the quench adapter catch only that sentinel and return a result with
   `stage_stop_reason='native_stage_predicate'`, preserving the exact evaluated
   geometry/E/F and request count. It must not mark the result `converged`.
5. Choose explicitly between two documented endpoint modes: independent mode
   returns the evaluated point; release mode restores the pre-dispatch snapshot
   and records the triggering trial separately. Do not silently mix them.
6. Run the existing true landing quench/evaluation after either mode. MC and LS
   response handling stay unchanged.

The sentinel must be checked in the evaluator path before the optimizer consumes
that evaluation's return, but the optimizer's own accepted/trial status remains
its private state. This avoids claiming that a trial was accepted.

## Predicate scope

The recovered native scalar gates include first/later counter limits and force,
energy-height and `limitGM` terms. The current evidence does not justify a new
Python default for every native threshold. If `limitGM` or its exact runtime
source is unavailable, the prototype may expose a clearly labeled partial
predicate with only the source-backed force and stage-counter terms. It must
report `predicate_scope='partial_native_scalar'`; silently dropping the gate
while claiming native parity is invalid. No new threshold or fallback should
be invented.

The existing Safe-total optimizer's `maxiter` remains independent. A finite
sentinel or optimizer budget is a stage stop, not convergence, and does not
replace the existing final force certificate.

## Fair bounded comparison plan

For a later two-system diagnostic, use one fixed-cell metal and one molecule
with identical backend, seed, initial true quench, width/Gaussian settings,
outer-step count and total request cap. A practical existing pair is Cu13 EMT
and trans-butadiene GFN2. Compare current default against the opt-in sentinel
with the same initial structures and seeds. Report, per arm:

- initial, triggering modified E/F, accepted optimizer geometry when available,
  and release geometry if release mode is selected;
- modified evaluation requests, optimizer iterations, trigger count and failed
  requests separately;
- final true landing E/F certificate and MC result;
- geometry equality between trigger and release snapshot, rather than assuming
  exchangeability.

The comparison is a lifecycle/endpoint diagnostic, not evidence of native
search performance. The prototype should first use a deterministic mock
optimizer that evaluates a displaced trial and verifies that independent mode
returns the trial while release mode restores the pre-dispatch copy. No real
experiment is implied by this design document.
