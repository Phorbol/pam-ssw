# Late fragmentation is observed; fixed early stopping is not justified

The capture adapter reproduced all12 completed outer records (status, acceptance, request cost and energies) and the four initial costs. Successful CPU1493165 used8900 search requests in45s. CPU1493151 previously paid the same8900 requests in85s but failed while building summaries (`SSWResult.best` is Atoms, unlike checkpoint.best). Its geometry/error/cost files remain intact. Combined17800 requests and130 CPU seconds stay within the original24000-request/12-minute bound. The fix changed reporting only. Repaired derived evidence is `stage-probe-repaired-runs/analysis.json`.

Within the selected first three outer steps, both global-direction arms remain connected at every saved biased endpoint. Paper seed25092501/outer0 first separates atom23 after the ninth Gaussian (index8); paper seed25092502/outer2 first separates atom18 after the tenth (index9). Both1.3sigma and1.5sigma thresholds agree. The final true quenches retain the fragments. Those stages were selected from earlier failures, so these are diagnostic examples, not fragmentation-rate estimates.

The two modes are already localized at the first stage: participation ratios0.089/0.078; maximum atom weights0.430/0.426. Near and after separation the same modes become more localized, eventually carrying0.761/0.908 of squared amplitude on the separated atom. Localization correlates with this escape but does not prove causation; the early motions are not instantly disconnected. Sampled real directional curvature is mostly positive while rank-one-biased curvature is negative, so one must not treat the biased curvature sign as a true-PES saddle crossing.

CPU1493179 then quenched eight frozen intact endpoints with unchanged Safe-total/history500, true fmax0.01, full-pair LJ.452 quench+8fresh requests,28s;8/8 force-qualified,8/8 connected at both cutoffs. No new SSW trajectories or search parameters. Results:

| Seed/outer | Gaussian count | True-quench energy/eV | Difference from outer start/eV |
|---|---:|---:|---:|
|25092501/0|1|-163.016909|+3.813400|
|25092501/0|4|-163.028956|+3.801353|
|25092501/0|7|-165.423100|+1.407209|
|25092501/0|8|-167.855351|-1.025042|
|25092502/2|1|-168.435942|+0.000038|
|25092502/2|4|-163.897030|+4.538951|
|25092502/2|7|-165.333838|+3.102142|
|25092502/2|9|-167.555374|+0.880607|

Seed25092502's first-stage quench geometrically returns to the outer start (proper RMS0.001033 Angstrom). The other seven have different fixed-cutoff graphs from their starts; one also has a clearly lower energy. All are force-qualified structures, not Hessian-certified minima. None is the target GM. The full selected landings have energies-160.632084 and-163.851330eV and are fragmented. Therefore a useful lower-energy intact landing exists along one of these paths and is lost when only the final stage is quenched. This directly supports a local lost-opportunity mechanism, not a universal stopping rule or performance improvement.

Decision: do not change H, add a stage8 stop, add confinement, or launch another longer LJ search. The best observed depth was selected after seeing these paths. Existing C60 depths2/3/6 quenches found no repair (8 returned defects,4 non-cages); earlier C4H6 depth experiments also failed to preserve full-path reaction coverage. Those are contrary evidence against general fixed early stopping. Retain intermediate-landing utilization as a hypothesis requiring a prospective cross-system comparison before any new controller/API. Close the depth scan here. Return to the already approved real-C60 input-order robustness check of the full direction bundle; it addresses a separate existing qualification gap without changing SSW or adding parameters.

Sources and artifacts: `stage-probe-plan.md`; `early-quench-plan.md`; `early-quench-runs/summary.json`; [prior C60 counterevidence](../c60-local-defect-20260925/early-quench-plan.md); `analyze_lj38_stages.py` and `quench_lj38_early.py`. Scalar/geometry analysis adds no PES calls.
