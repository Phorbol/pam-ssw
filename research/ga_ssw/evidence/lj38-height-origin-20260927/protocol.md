# Saved-stage height origin: bounded diagnostic

2026-09-27. Scope: explain existing SSW bias construction, without changing the algorithm.
Inputs: all 82 recorded Gaussian stages in the existing four three-outer-step LJ38 stage probes,
not a newly selected high-height subset. Original probes were outcome-selected; results cannot
estimate population failure rates or establish causal search gains.

Competing explanations: large height is driven by (A) physical resistance at the finite forward
probe, including local repulsion; (B) accumulated Gaussian force; or (C) implementation/reporting
mismatch. Reconstruct the exact existing forward-force rule with frozen recorded centers,
modes, widths and preceding Gaussian terms. Check its match to recorded height before interpreting.
Measure center versus forward-probe pair geometry, physical energy, projected physical force,
projected historical bias force and original rotation/quench cost. No width/height tuning.

Budget: exactly two FullPairLJ evaluations per saved stage (164 maximum), no MLIP, GPU or search,
no new trajectories; short local analysis only. Preserve initial and outer-step residual costs;
do not silently equate captured inner work with complete search. A height mismatch stops interpretation.
If physical forward-probe force dominates extremes, inspect whether the premise of local quadratic
curvature is reasonable before suggesting adaptive width; this does not authorize a new policy.
If Gaussian history dominates, inspect accumulation/projection; if both are normal, close this lead.
