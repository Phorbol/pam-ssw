# Explicit bounded biased stages

2026-09-11. Experimental implementation contract; no efficacy conclusion.

Problem: the combined BlockSSW atomic walker advances a Gaussian only after
biased force convergence. Native VC climb_convg permits a stage to finish on
its own iteration budget. Requiring full modified-surface stationarity can
spend the search budget before true-surface quenching is reached.

Evidence: native-vc-climb-convg-stage-stop-followup.md and
native-vc-optimizer-handoff.md. Native climbstep counts optimizer callbacks;
Safe-total iterations count accepted steps. These are not interchangeable.
The native saved scart/sfa snapshot is not proven to be a Wolfe-accepted point.
Our adaptation uses the known Safe-total accepted point contract.

Design: optional SSWConfig.bias_stage_steps, a positive integer, explicitly
supplied by the experiment, no numerical default. None retains strict force
convergence and the existing relax_steps limit. An explicit cap affects only
biased Gaussian quenching. A normal maxiter termination with finite coordinates,
energy and force can complete a stage; line-search/evaluation failures cannot.
The event retains the failed force certificate and reports iteration_budget.
The true energy is still evaluated before the completed boundary is recorded.
Initial/final true quenching, LS prequenching and MC selection remain unchanged.

Hypothesis: inexact intermediate minima may allow more valid proposals per
oracle budget. Counter-hypothesis: truncated stages lose directional continuation,
increase rotation failures, or increase final quench cost enough to erase gains.
Report all oracle costs and failures. Do not equate more Gaussians with better
search. Delete or retain only as an ablation if fair real-system comparisons
show no gain; do not promote based on interface tests or a selected seed.

Validation: preserve strict baseline and exact checkpoint replay; exercise
normal budget termination, nonfinite/line-search failure exclusion, and fresh
true quenching on Cu/EMT. These certify lifecycle behavior only. Fe7C3/MACE
and broader independent cases are required before a search-efficiency claim.
A cap sensitivity study must be declared prospectively and kept distinct from
final evaluation. The existing 25 cell-interleave iterations are not evidence
for a 25-step biased-stage default.


Implementation and verification: `paper_reference.py` and `atomic_climb.py`
now support the opt-in. Nine targeted budget tests pass, including real
Cu/EMT two-stage continuation and exact checkpoint replay, a full fixed-cell
walker with a fresh true-force check, a Cu7 vacancy combined VC walk with a
fresh force/stress certificate, and injected failure exclusion. The prior
strict/extraction/block/reference regression subset also passes. None of
these small-system checks establishes efficiency on difficult landscapes.

The exact ELF initialized fields were subsequently recovered at0x541b4c0:
ordinary25, initial5, half10. This is compiled data, not proof of all runtime
input overrides; the third field is not consumed by the inspected callee.
A prospective Fe7C3/MACE strict vs stage25 pilot is frozen in
`research/ga_ssw/fe7c3-block-stage-budget`, job1270433, one V100, four arms
and8000 total EFS cap. The uniform25 accepted-step cap is an explicitly
independent adaptation; it does not reproduce the native first-stage5 rule.
