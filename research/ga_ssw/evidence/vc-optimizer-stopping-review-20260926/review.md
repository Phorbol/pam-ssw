# VC optimizer stopping-cost review

Read-only audit, 2026-09-26. This note interprets the existing fixed-task and
short end-to-end panels; it adds no calculations or new runs.

## Finding

The lower Safe-total cost in the four end-to-end AlOH26/TiO2 pairs is consistent
with an operational advantage for this short protocol, but unequal native stopping
is a plausible contributor. It is not possible to apportion the end-to-end cost
difference between stopping, line-search/trial efficiency, and divergent outer
trajectories from the current artifacts.

The preceding fixed-task panel records the first accepted point satisfying the
shared certificate and the full native termination cost separately. In task order
(AlOH biased, AlOH unbiased cell relaxation, TiO2 biased, TiO2 unbiased cell
relaxation), first passage was Safe `26/45/102/62`, ASE `55/80/179/117`, and
SciPy `26/50/103/64`; native cost was Safe `26/45/102/62`, ASE
`63/96/221/153`, and SciPy `38/58/121/83`. The native runs therefore spent 102
ASE and 57 SciPy requests after first common-certificate passage, while Safe
stopped at that passage. Safe and SciPy first-passage totals differ by only 8
requests (235 versus 243), versus a 65-request difference in native totals (235
versus 300). This is direct evidence that overconvergence can explain much of the
fixed-task terminal-cost gap; it does not quantify the same effect in end-to-end
searches.

The full-workflow runner passes `gradient_tol/sqrt(6)` to SciPy or
`gradient_tol/sqrt(2)` to ASE for the true cell quench (`run_vc_e2e_optimizer_panel.py:200-203,220-223`). Those tolerances are also passed through the baseline bridge. The bridge separately applies the sufficient norm conversions to biased joint calls (`vc_lbfgs_baseline_runner.py:41-56`). Safe true relaxation instead stops on the accepted physical force/stress certificate (`cell_relax.py:8-23`) and independently checks that certificate (`cell_relax.py:50-56`). Baseline status is checked against the caller's common convergence certificate, but the native solver retains its own stopping rule; SciPy also retains its default relative-energy `ftol` (`lbfgs_baselines.py:211-241,251-254`), while ASE uses native force stopping (`:297-324`). The conversions are sufficient, not necessary, for the shared norm criterion (`vc-mature-baseline-norm-contract.md:7-26`).

The end-to-end wrapper records terminal adapter request/step counts, status,
certificate, and metadata, not each accepted iterate (`vc_lbfgs_baseline_runner.py:57-61`). Its analyzer can close total and phase costs, but cannot recover first common-certificate passage inside those solves. The frozen-task first-passage data is separate input/objective evidence; subtracting its overrun from end-to-end totals would be invalid because the search paths diverge.

## Scope for the held-out SiO2 panel

The completed end-to-end panel supports a descriptive operational statement for
these inputs: Safe-total used 9,562 search requests and returned 12/12
qualified landings, versus SciPy's 11,371 and 11/12 and ASE's 15,020 and 10/12
(`vc-e2e-optimizer-panel-20260926/decision.md:13-30`). The runs used matched outer
protocols, but native stopping and step geometry remain part of each numerical
implementation (`protocol.md:1-14`). This is not an isolated optimizer or
line-search effect, nor a stable or universal speedup.

A held-out SiO2 run remains useful if it asks whether these whole implementations
retain operational cost and landing qualification under the already fixed
protocol. Keep total paid requests, failures, fresh force/stress qualification,
and native terminal costs visible; do not label any difference optimizer
superiority. If attribution of excess work is required, future logs need accepted
iterate request indices and the shared certificate values collected without new
calculator evaluations. No such attribution should be inferred from this panel.
