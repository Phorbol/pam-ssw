# History400 retains convergence on all31 frozen Cu13 failure-selected stages

Evidence `research/ga_ssw/evidence/cu13-failed-quench-safe400/`; runner `research/ga_ssw/compare_cu13_safe_memory400.py`. The predeclared set is all31 original Cu13 direction-only biased-quench failures, with each original stage start and frozen Gaussian history. Change: only process-local Safe memory10→400. No public default or production file changes, no history sweep, and no optimization retries. Limits per case201 optimization E/F,200 accepted steps,.01 eV/Å; shared6262 E/F/120 seconds, one CPU.

| Kernel | Force-converged/31 | Optimization E/F | New fresh E/F |
|---|---:|---:|---:|
| Safe10 (archived)|31/31|3058|not repeated|
| Native ELF (archived)|29/31|2779|not repeated|
| Safe400|31/31|2804|31; all pass|

Total new2835 E/F. Elapsed from the original plan-file write to the repaired final summary is44 seconds, inside the original120-second wall limit; cumulative optimization/fresh execution was4.16 seconds, which excludes the intervening analysis and is not substituted for total elapsed. Per-case requests include initialization and failed trials. Safe400 has fewer calls than Safe10 in19 cases, more in5, equal in7. Across the29 cases also converged by native:20 fewer,5 more,4 equal. The remaining two native failures also converge with Safe400. These are selected local subproblems, not unbiased efficiency/success estimates for a complete walker or independent global-search validation.

The initial research runner called a nonexistent `RelaxResult.converged` attribute **after** each optimization had returned and its termination_reason had been obtained. This caused a postprocessing error before fresh checks, not failure of the numerical optimization. Original script/log/results retain these errors. No optimization was rerun. `repair-no-optimization-rerun.py` classifies the retained accepted endpoints by the fixed force criterion and independently recomputes successful endpoints; `repaired-summary.json` and `comparison.json` are the qualification/comparison artifacts. All31 fresh checks were completed44 seconds after the original plan write. The reusable runner now tests the returned termination_reason rather than the nonexistent attribute; this correction was not used to overwrite the archived original run.

Decision: no observed convergence regression on this entire31-case set. Retain memory400 as a research ablation, not a new default. Five cases cost more, and this set was selected using earlier failed quenches. The hard-C60 single-stage success plus this regression justifies a prospective independent full-walker comparison, not a claim that400 is optimal or necessary for all PES.

## Current prospective experiment, not launched

The earlier hard-C60/seeds29,43 proposal is superseded by the cheaper full-lifecycle check in `safe-history-newseed-e2e-plan.md`: C4H6/GFN2 paper-LS and Cu13/EMT ordinary SSW, seeds29,71, history10/400,2 outer steps,8 runs. This replaces the old proposal rather than creating a second campaign. Hard-C60 remains selected local-subproblem evidence.
