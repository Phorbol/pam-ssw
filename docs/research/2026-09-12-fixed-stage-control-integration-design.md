# Minimal fixed-cell stage-control integration design

> **Superseded for recovered lifecycle facts.** Use
> [`fixed-stage-control-audit.md`](2026-09-12-fixed-stage-control-audit.md),
> [`fixed-noncrystal-counter-trace.md`](2026-09-12-fixed-noncrystal-counter-trace.md),
> and [`fixed-lbfgs-rc-lifecycle.md`](2026-09-12-fixed-lbfgs-rc-lifecycle.md)
> for the now-closed counter reset/increment, single-dispatch, `x_eval` versus
> `x_next`, and LSB status semantics. The design below remains a historical
> proposal and does not define those facts.

Date: 2026-09-12. Design only; no production code or PES run.

## Native facts used

The recovered fixed-cell sequence is `addgaussian -> noncrystal_opt ->
climb_convg`. `noncrystal_opt` receives `control+0x08`, calls BFGS/LBFGS once,
and increments that field after the driver returns. The LBFGS reverse
communication result can request a later E/G; it does not establish that the
current trial is accepted. `climb_convg` reads the counter, applies separate
initial/later strict budgets, and returns stage/all-stop flags. On normal
all-stop, `climb_` restores the force, coordinate and energy snapshots saved
before the Gaussian/optimizer dispatch.

These facts define bookkeeping boundaries; they do not authorize copying the
native optimizer or its undocumented thresholds into Python.

## Small public contract

Keep the current `run_ssw` default path byte-for-byte in behavior. An explicit
stage-control mode may add only a record-level controller around the existing
quench boundary:

1. Record the true initial minimum and its E/F separately.
2. Record every biased objective evaluation, including unevaluated/trial
   geometry if the selected optimizer exposes one, and distinguish it from the
   last accepted optimizer geometry.
3. Treat `relax_steps`/`bias_stage_steps` as the selected Python optimizer's
   numerical cap. A finite cap produces `stage_budget`, never
   `converged`, unless the final evaluated certificate independently satisfies
   the force criterion.
4. After a successful biased stage, retain the returned evaluated geometry and
   E/F, then perform the existing true-PES landing evaluation and certificate.
5. On stage/all-stop or failure, retain the last evaluated geometry/E/F and the
   exact failure/cost record. Release restoration must use the pre-dispatch
   snapshot only when the selected Python mode explicitly requests native-style
   release semantics; it must not silently replace the current accepted-point
   contract.

The existing `surface.evaluate` request ledger remains the physical API cost
account. Optimizer callback count, accepted steps, trial requests, and true
landing checks must be separate telemetry fields.

## Where the boundary belongs

`surface.quench` already owns the fixed-cell optimizer call and returns
`QuenchResult(atoms, energy, max_force, converged, optimizer_steps,
evaluation_requests, optimizer_telemetry)`. That is the correct minimum
observation boundary for the existing public modes. `SurfaceCalculator` keeps
true and additive-term forces together during a modified quench, while the
outer `paper_reference` stage records true landing and MC state.

A native-style controller cannot be implemented faithfully by changing only
`QuenchResult.converged`: the optimizer's RC request state and trial geometry
are not currently exposed, and the existing ASE/Safe-total optimizers return a
last state under their own contracts. The minimum non-default change would be
an opt-in optimizer telemetry adapter that exposes callback/accepted/trial
records and a release snapshot, while preserving `surface.evaluate` and the
true landing certificate. No adapter should label a trial as accepted without
an optimizer-provided acceptance boundary.

## Parameters and scope

The already recovered native quantities that may be recorded are
`ngaus_relax_ini`, `ngaus_relax`, and the native counter semantics. Their
runtime values and full optimizer state are not a Python default specification.
No new multi-threshold defaults, native GTOL/FTOL copies, or second optimizer
controller should be introduced. The current Python `bias_stage_steps` remains
an explicit Safe-total numerical policy, not a native RC budget.

The prototype needed for a future opt-in comparison is therefore a narrow
stage ledger with: initial E/F, accepted E/F, trial E/F where available,
optimizer dispatch count, physical request count, stage-stop reason,
release-all reason, and restored snapshot identity. It should be tested first
with a deterministic mock optimizer that creates a rejected trial and a finite
budget, then with one existing EMT case. The default path must have identical
request and geometry traces, and true landing qualification must remain
unchanged.

Until such an adapter exists, retain the present quench/landing lifecycle and
report its Safe-total budget explicitly. This preserves numerical and physical
certificates without claiming native RC or release parity.
