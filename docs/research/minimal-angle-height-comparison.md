# Analytic87° height: four new runs and three-policy interpretation

2026-09-10. The explicit `MinimalAngleHeightPolicy` is now callable through `run_ssw(height_policy=...)`. Default forward-force and native-derived discrete initialization logic are unchanged. Nonpositive analytic height maps to the existing `nonpositive_height` failure; zero-resultant/domain errors preserve the attempted old history, preparation point/background force and all paid requests. No silent continuation/quench or height floor was added. The new branch and pure policy, existing native height and paper driver pass **16 tests** together.

The analytic policy was then tested on **4/4 new predeclared terminal trajectories**, C4H6/GFN2 and Cu13/EMT, seeds3/17, two outer steps each. Input extxyz files and full configs were loaded directly from the earlier frozen8-run comparison; that baseline was not rerun or overwritten. The runner asserts byte identity of the core surface, direction, dimer, Gaussian, cluster-frame, native-rotation and Relaxer files against the frozen baseline, saves the paper-driver branch diff, and snapshots new code. All numerical/PES settings remain paired; the intervention is the explicit height policy.

The approved limit was6000 total E/F per run (5997 search plus3 fresh reserve),24000 total and600 shared wall seconds, one CPU thread. Actual **new cost is7909 E/F,18.29 seconds**, including32 initialization calls and12 independent fresh checks. All8 new outer attempts yielded certified landings; all12 fresh checks pass .01 eV/Å (maximum0.00905758); no run was censored. Old8-run cost remains20616 E/F, so this pair of development experiments cost **28525 search+fresh E/F**. Independent Cu qualification, described below, adds880 separately; combined cost is29405. No offline analysis adds PES calls.

## What the minimal rule changed

All123 prepared stages (76 C4H6,47 Cu13) have angles87° within1.5e-13° floating-point error. There is no weight growth/history reweighting by construction. Heights range .06266–.90780 eV for C4H6 and .24838–1.58002 eV for Cu13. This isolates the angle criterion instead of using the previous native profile's2.0/5.6 eV initial heights, which had already satisfied87° in all132 stages without a single update.

| System / policy | Seed3 total E/F | Seed17 total E/F | Sum | Certified landings / accepted |
|---|---:|---:|---:|---:|
|C4H6 forward-force|3978|4182|8160|4 /2|
|C4H6 native conservative|2743|4248|6991|4 /2|
|C4H6 minimal-angle|4276|1604|5880|4 /3|
|Cu13 forward-force|1187|1209|2396|4 /2|
|Cu13 native conservative|1444|1625|3069|4 /1|
|Cu13 minimal-angle|1217|812|2029|4 /3|

Relative to forward force, minimal-angle reduces summed calls28.0% on C4H6 and15.3% on Cu13 across these two seeds. **Seed3 is slightly more costly on both systems**; aggregate savings come from seed17, where the second attempt terminates on lower true energy before exhausting the Gaussian cap. The final quench still finds no structure below the common initial minimum. These are costs of two completed attempts, not evidence for equal-cost superior global search.

## C4H6: intact conformations versus reaction-space coverage

The zero-cost analyzer applies the same explicit C/H graph cutoffs and species-preserving graph-isomorphism comparison across all12 runs. All four new minimal-angle landings remain connected C4H6 with the initial bond graph. Seed3 landing1 changes the initial-chain dihedral from180° to18.42° and is rejected; landing2 is179.79° and accepted. Seed17 landings are180.11° and180.05°, both accepted. No new landing has energy below the shared initial-314.365038063089 eV.

The earlier forward-force arm had one C4H4+H2 landing in four attempts and native conservative had two in four; minimal-angle has zero in four. Those fragmentations remain in the combined data and are not discarded as numerical failures. For conformational optimization, maintaining an intact molecule is relevant; for unrestricted PES/reaction exploration, eliminating such structures is not automatically a benefit. No Hessian or electronic-structure refinement was added for these molecular endpoints.

## Cu13: independent all-endpoint refinement changes the identity assessment

Before further PES calls, the same old strict-fingerprint threshold failed to match even the common raw initial snapshots, so raw unmatched labels were explicitly withheld as novelty claims. The parent subsequently qualified **all18 endpoints from all6 Cu13 runs**, including initial/rejected records, under one frozen independent diagnostic plan. This is a separate expense, not search feedback:880 E/F,3.025 seconds, strict fmax1e-5, sorted-pair-distance tolerance1e-4 Å, and internal finite-difference Hessians at h=1e-4 and5e-5 Å with global rigid modes excluded.

The refined endpoints form four fingerprint groups. Both finite-difference steps give positive internal eigenvalues at each representative:

| Refined group | Energy (eV) | Lowest internal eigenvalue, approximately (eV/Å²) | Policy coverage |
|---|---:|---:|---|
|0|9.36135788|see full spectrum artifact|all three, common initial|
|1|10.46286188|.18272|forward and minimal|
|2|10.48849784|.34016|forward and native|
|3|10.17423741|.07900|native only|

Thus coverage sets are forward `{0,1,2}`, native `{0,2,3}`, minimal `{0,1}`. Every newly reached higher-energy group was MC rejected but remains in the discovered structure archive. **There is no global-minimum energy improvement, while PES coverage is complementary across height profiles.** The cheaper minimal-angle runs cover fewer refined groups here; a lower E/F count alone is not a search-quality win. Native's distinct higher-energy group provides coverage evidence despite its higher cost and no accepted energy improvement.

Qualifications apply to the **refined geometries**, not automatically to the raw search snapshots. Raw-to-refined maximum fingerprint change reaches .03405 Å; same-basin continuity has not been established. Sorted distance fingerprints are not injective, and four groups are not a complete basin enumeration or statistical success probability. Complete endpoints, before/after geometry, spectra and costs are in `research/ga_ssw/evidence/height-cu13-qualification/result.json`.

## Artifacts and decision boundary

New search: `research/ga_ssw/compare_minimal_angle_height.py`; plan, source diff/snapshots and four complete runs under `evidence/minimal-angle-height-two-system/`. Combined zero-PES analyzer: `analyze_minimal_angle_comparison.py`, with12-run fixed denominator and4-run new-experiment denominator; output `offline-analysis.json` in the new directory. It retains all36 fresh observations, raw strict-reference distances and a separately labeled posthoc qualification field, plus each observed search/fresh request index. Original8-run files remain unchanged.

The current result justifies retaining the analytic policy as an explicit conservative alternative with a clear mathematical target. It does not justify making it a universal default, combining policies through a new heuristic scheduler, or asserting a broadly better SSW algorithm. The two-seed, two-step development results expose a meaningful tradeoff between cost, intact-molecule exploration and higher-energy PES coverage; wider predeclared testing would be needed for a stronger efficiency claim.
