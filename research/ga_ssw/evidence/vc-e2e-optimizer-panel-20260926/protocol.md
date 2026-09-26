# Full joint-VC optimizer development comparison

Goal: test whether frozen-task observations survive complete proposals, quench
and MC selection. Hypotheses: (1) Safe saves full-task cost at comparable landing
quality; (2) rotation, trajectory divergence or failed quench dominates, making
frozen-task improvement irrelevant to search.

Reuse run_vc_ssw and temporary_lbfgs_baseline without public/core API changes.
Joint atomic/log-strain VC is NOT the paper block-VC schedule. Three solvers share
metric, width, bias, direction, history500 and strict release. Both biased and
unbiased phases use the selected numerical implementation. Native stopping remains
part of each implementation; no line-search-only causal claim. Use the same native
norm conversions as the frozen panel for true quenches via the existing context
parameter. Fixed-task first passage remains separate prior evidence; this panel
measures complete native costs and final physical certificates, not hidden first
passages inside each solve.

Inputs are archived initial qualified AlOH26 (uploaded example) and TiO2
phase87/48atoms (literature structure). New seeds71/83 and three outer steps probe
repeated proposals, not global success probability. Source-specific width/count
remain .2/14 and .6/10; effective parameters frozen in plan.json. No LS or
alternative direction. Not an independent assessment of these materials because
related saved configurations were used for the frozen-task development test.

Three steps/two seeds can expose gross qualification loss or a dominant shared
failure before long trajectories. Ceiling6000 search requests/240s per arm bounds
failure, not a target to consume. Twelve arms at most72000 search +48 fresh calls;
two GPUs max, two40min allocations max. Startup bounded by360s external timeout;
no automatic retry of failed searches. Stalled in-flight evaluations counted
separately. Existing CPU preflight's unbounded-cell negative evidence remains.

Implementation: research-only worker, targeted CPU EMT full-path accounting check,
root review, frozen source before GPU, independent fresh checks of every recorded
landing (including MC rejects/failed certificates), zero-PES periodic structure
comparison. Failed/censored arms stay in the denominator. No volume clipping or
loosened certificate to obtain success. If shared pre-optimizer failure dominates,
close optimizer expansion and diagnose only that blocker. Mixed results retain
default; no parameter sweep. Main report measures both qualification and costs.
