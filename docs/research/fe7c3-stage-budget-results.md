# Bounded biased stages complete Fe7C3 proposals, without lower-energy discovery

2026-09-11. Job1270433 completed on one V100,4v100n04,213s,exit0.
Frozen source and prospective plans: research/ga_ssw/fe7c3-block-stage-budget.
Total6609 charged E/F/stress requests, including10 fresh checks; two denied
requests belong to censored strict arms and are not charged. Registered Fe
cumulative75693 (=69084+6609). No retry, resource change or cap extension.

## Controlled question and implementation

Only SSWConfig.bias_stage_steps changes: strict None versus25 Safe-total
accepted steps per Gaussian. Both preserve finalfmax.001eV/A and
stress.0001eV/A^3, MACE-OMAT-0-small float64, p0, history10,14 Gaussians,
five cell cycles, lattice_frobenius fraction.15 and25 partial atom iterations.
The native ELF has compiled ordinary25/initial5/half10 values, but uses a
callback counter and strict greater-than comparison. Our uniform accepted-step
cap does not reproduce that counter or the initial-stage5 rule. No default is
promoted. Neither native expiry handling nor native PES/main was executed.

The Python stage-cap path returns the Safe-total accepted point, retains
converged=False, evaluates true energy, and records the stage as iteration_budget.
Only normal finite maxiter qualifies. Failed line searches, nonfinite energies,
forces or coordinates cannot complete a stage. Initial/final and LS prequench
budgets retain relax_steps. Nine targeted checks cover actual Cu/EMT stages,
checkpoint replay, true quenching, Cu7 vacancy combined VC force/stress checks,
and injected failures;18 existing strict/reference/block checks passed too.

## Complete attempt accounting

Two requested outer steps per arm: cell-only, then combined cell+atomic.
Every arm completed the first cell-only proposal. All proposed endpoints were
MC-rejected; no arm improved its best energy.

| Arm | Seed | Search + fresh EFS | Combined proposal | Completed Gaussians | Combined endpoint deltaE eV |
| --- | ---: | ---: | --- | ---: | ---: |
| Strict | 7 |1997+2| budget censored |10| unavailable |
| Strict |101|1997+2| budget censored |8| unavailable |
| Stage25 |7|1034+3| certified landing |14|17.318582|
| Stage25 |101|1571+3| certified landing |14|17.583746|

Stage25's28 stages all stopped at their25-step limit, without biased force
convergence. Their completed biased quenches cost366 and371 requests. Final
combined true quenching still passed independently: maxforce0.00069231 and
0.00085168eV/A; maxstress0.000002692 and0.000005474eV/A^3, respectively.
Fresh checks use a newly instantiated calculator. This is numerical force/stress
stationarity, not a full Hessian stability, magnetic, or DFT validation.

The strict arms' combined evaluation_failed labels originate from the exact
request cap, not a claimed intrinsic optimizer crash. Denied rows and prior
outputs remain preserved. Stage25 finished combined proposals in810/1349
search requests after its cell-only stage; both strict counterparts consumed
1773/1775 and remained incomplete. Since strict time-to-completion is censored,
no exact speedup ratio is inferable. Measured whole-arm wall times were
48.60/63.54s strict and24.88/46.14s Stage25; scheduler213s also includes process
startup and orchestration.

## Structure identity and decision

Three recorded pymatgen profiles (strict/default/loose) agree: both Stage25
combined endpoints differ from their initial structures, from their cell-only
endpoints, and from each other. Matching same-seed cell-only endpoints across
arms confirms that the unchanged preceding cell block is reproduced to the
identity criterion. This remains approximate structural identity, not proven
basin connectivity. The new combined endpoints are17.32/17.58eV uphill and
rejected at300K. Thus this pilot supports a bounded-stage completion mechanism,
not better global-minimum discovery or a universally best budget.

Decision: retain the explicit experimental cap and strict baseline. Do not
retune history, final stress/fmax, cell displacement or this cap on these seeds.
The primary open problem shifts to proposal energy and useful basin selection.
Next evidence work should close native first/ordinary stage schedule, bias
height/force and reference-state semantics against the Python walker; then
register broader harder-system tests of low-energy discovery at equal total
cost. Avoid changing several controls simultaneously or calling more Gaussian
boundaries a scientific success. These reused seeds are developmental, not an
independent final assessment.

Evidence: comparison-summary.json (exact charged/denied accounting and all
three identity matrices), stage-cost-summary.json, every per-arm result.json,
search-result.json, evaluations.jsonl, plan-used.json, frozen source manifest,
allocation-gpu.csv and Slurm logs. A read-only summarizer originally named
summarize_block_metric_comparison.py was reused unchanged for the two budget
arms; its generic metric label does not denote another algorithm change.
