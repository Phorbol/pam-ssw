# Eight new-seed whole-search runs: fewer calls with history400, no accepted new-basin evidence

The reviewed experiment has now **executed**. The original `plan.json` remains a preparation record; actual status/versions/runner SHA are in `research/ga_ssw/evidence/safe-history-newseed-e2e-prepared/execution.json`. `executed-runner.py` records the approved additions for execution metadata and explicit budget-censoring detection. The production-source hash check passed; no public default changed. All8 declared runs completed, with no retries or censoring, **20831 total E/F in51.47 seconds** on one CPU thread. Versions: ASE3.29.0, NumPy2.5.2, tblite0.7.0.

Same original inputs and frozen system settings; seeds29/71;2 outer attempts; only process-local Safe history10 versus400 changes. Bounds48000 E/F/600 seconds shared,6000 per run including3 fresh checks. Every run records3 force-qualified minima (initial+2 landed), and all24 independent fresh checks pass. All16 attempts reach the Gaussian-count stopping condition and then perform true quench/MC: `gaussian_limit` here is not a failed local quench. Both histories have zero true/biased-quench failures in this short set, so the result demonstrates cost differences rather than an improved failure rate.

| System/seed | History | Total E/F (search+fresh) | Valid new landing records | Rejected | Fragmented landing | Accepted near-return |
|---|---:|---:|---:|---:|---:|---:|
|Cu13/29|10|1075 (1072+3)|2|2|0 indicated|0|
|Cu13/29|400|1036 (1033+3)|2|2|0 indicated|0|
|Cu13/71|10|1377 (1374+3)|2|2|0 indicated|0|
|Cu13/71|400|1234 (1231+3)|2|2|0 indicated|0|
|C4H6/29|10|4799 (4796+3)|2|1|0|1|
|C4H6/29|400|3372 (3369+3)|2|1|0|1|
|C4H6/71|10|4571 (4568+3)|2|2|1 (C4H4+H2)|0|
|C4H6/71|400|3367 (3364+3)|2|2|0|0|

“Valid landing” means fresh force-qualified true-PES quench output, not Hessian-certified stability or a distinct minimum. Fragmentation is separately classified. Across all runs14/16 proposals are rejected; both accepted proposals return near their respective initial trans-butadiene geometry. History totals:10 uses11822 E/F,400 uses9009, about23.8% fewer for this fixed8-run set. These are accounting observations on known systems/new seeds, not a general speedup estimate.

## Geometry and chemistry

**Cu13:** All8 landed candidates differ visibly from initial according to sorted pair-distance RMS .271–.563 Å and same-atom aligned RMSD1.64–2.36 Å. They have energies10.078–10.493 eV versus initial9.36136 eV and are all rejected. Minimum-spanning-tree longest edges2.428–2.473 Å remain comparable to normal Cu–Cu contacts, and gyration radii2.40–2.56 Å show no separated-fragment signature. These diagnostics do not resolve symmetry/permutation equivalence between every pair of candidates or prove distinct stable basins.

**C4H6 seed29:** Both histories first find intact same-bond-graph structures with carbon-chain dihedral18.76°/19.47°, which are rejected. The second proposals return to179.72°/180.04° and are accepted. Their same-atom aligned RMSDs to the relaxed initial are .00350/.00053 Å; these accepted events are near-return, not evidence of new accepted basins. The histories also relax the common raw starting geometry slightly differently, consistently with including initialization in the intervention.

**C4H6 seed71:** History10 first proposes an intact341.53° conformer, then a C4H4+H2 configuration; both are rejected. The latter has minimum distance .77693 Å and intercomponent minimum-spanning-tree link4.095 Å. Its small residual force does not turn a separated-fragment configuration into a useful molecular conformer. History400 instead proposes intact97.11° then342.35° structures, both rejected. The97° structure has no Hessian qualification; small force alone does not distinguish a minimum from a saddle/flat torsional configuration. It must not be advertised as a new stable conformer.

`offline-comparison.json` retains component formulas, carbon dihedrals, minimum-spanning-tree scales, geometries and energies for all24 observed minima; raw trajectories, all MC-rejected structures and per-step costs remain in each `runs/*/result.json` and evaluation log. Offline analysis used0 additional E/F.

## Interpretation and decision

Increasing history alone reduced E/F cost in each of the four paired seed/system comparisons, so the selected local-subproblem observation is not confined to that one replay. Nevertheless this short experiment shows **no accepted new-basin advantage**, and only one observed fragmentation difference. Both arms already complete all quenches; candidate distributions also change. A molecule's force check and a cluster's geometric novelty are not Hessian or global-search certificates.

Retain this as evidence for an explicit, research-level memory ablation. Do not claim400 is optimal, unseen-system generalization, or a default change justified by this experiment alone. Do not compare against “native whole SSW”: only the earlier local native kernel was tested. Previous qualification/stage/replay costs remain separate and are not mixed into this20831-E/F denominator.
