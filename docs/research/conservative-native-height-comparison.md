# Eight-run height comparison: initialization dominates; no87° growth activated

2026-09-10. All **8/8 predeclared runs** are terminal: two systems, seeds3/17, forward-force versus conservative native-derived height. All16 outer attempts produced stored force-certified landings; all24 independent fresh checks including initial records passed fmax .01 eV/Å (maximum0.00963724). No run was censored and no failure was omitted. Total cost, including initialization and fresh checks, is **20,616 E/F and53.00 seconds**. This is a two-outer-step development comparison with unequal realized cost, not a fixed-cost efficacy campaign.

The main algorithmic conclusion is that **none of the132 native-profile height stages grew its weight**:76 C4H6 stages and56 Cu13 stages directly satisfied the angle criterion. There were also zero actual historical weight rewrites. Native heights were exactly2.0 or5.6 eV; all resulting preparation angles were2.64–37.42°, already far below87°. Thus this experiment primarily probes the loaded initial-height profile, not the claimed advantage of adapting height to the87° target. Do not describe a lower cost in one system as validation of87° adaptation.

## Full denominator and cost

| System | Height arm | Seed3 total E/F | Seed17 total E/F | Sum, including fresh | Valid/rejected landings | Fresh passed |
|---|---|---:|---:|---:|---:|---:|
| C4H6/GFN2 | forward force |3978|4182|8160|4 /2 rejected|6/6|
| C4H6/GFN2 | native conservative |2743|4248|6991|4 /2 rejected|6/6|
| Cu13/EMT | forward force |1187|1209|2396|4 /2 rejected|6/6|
| Cu13/EMT | native conservative |1444|1625|3069|4 /3 rejected|6/6|

C4H6 initial relaxations cost15 requests each; Cu13 initialization cost1 each. Every run used3 fresh checks. Every search/request ledger reconciles with the saved serial result and sequential calls.jsonl indices. Initialization, all rejected attempts and final certificates are included. The native profile reduced total C4H6 calls by14.3% across these two seeds, while increasing Cu13 calls by28.1%; those are realized-cost differences for two completed steps, not matched-cost discovery efficiencies.

The analyzer preserves both the cumulative search request of each landing and the later actual request at its independent fresh certificate. It does not pretend those fresh checks occurred during exploration or charge them to a different denominator. Per-run records, all24 observations, status and every height remain in `offline-analysis.json`.

## C4H6: conformational and fragmentation outcomes

The graph criterion is exactly the preceding experiment's recovered H/C length table plus.1 Å: same cutoffs for every arm/seed. Graph matching preserves species and tests graph isomorphism; it does not infer bond orders. The smallest pair-distance margin to any graph cutoff among these observations is0.06047 Å, so these particular assignments are not threshold-roundoff decisions. Listed dihedrals use the four-carbon order from the initial chain; a bond-changed structure's value is only that geometric diagnostic.

| Seed / arm | Landing1 | Landing2 |
|---|---|---|
|3 / forward|intact C4H6, initial-chain dihedral19.46°, rejected|intact C4H6,179.83°, accepted|
|3 / native|intact C4H6,181.74°, accepted|intact C4H6,180.54°, accepted|
|17 / forward|intact C4H6,179.87°, accepted|C4H4 + H2, rejected|
|17 / native|C4H4 + H2, rejected|C4H4 + H2, rejected|

Three species-preserving graph classes occur overall: the initial connected graph and two distinct fragmented graphs with the same component formulas. Fragmented landings are **1/4 versus2/4**, forward versus native. They are retained as discoveries of force-small reaction-space structures; for a bound-conformer task they would not count as successful conformer coverage. No bond-order/isomer name, Hessian stability or physical reaction rate is inferred.

The common initial fresh energy is-314.365038063089 eV. Neither arm discovers a lower-energy snapshot. The forward seed3 first landing is-314.27273175 eV with a substantial carbon dihedral change; the native seed3 candidates remain close to the original dihedral. Fragmented energies lie around-311.93 to-312.18 eV and are MC rejected. These few observations provide no overall search-quality advantage for the native initial heights.

## Cu13: relative fingerprint diagnosis with strict limitations

The offline analyzer reuses the14 existing strictly quenched Cu13 reference groups under `evidence/cu13-safe-total/strict-validation`, their sorted all-pair-distance fingerprint, and their original1e-4 Å maximum-component threshold. **None of the12 new Cu observations matches at that strict threshold**, including all four copies of the initial candidate, whose distance to reference group0 is0.00010231 Å. This exposes the effect of comparing fmax .01 snapshots to strict fmax1e-5 references; it must not be turned into12 new basins.

Closest-reference distances, offered only as relative diagnostics:

| Seed / arm | Landing1 nearest group / distance Å | Landing2 nearest group / distance Å |
|---|---|---|
|3 / forward|9 /0.012489|10 /0.012152|
|3 / native|10 /0.003575|0 /0.002011|
|17 / forward|0 /0.000986|0 /0.001902|
|17 / native|10 /0.004415|2 /0.695034|

All Cu structures remain one component at the existing3.3 Å connectivity cutoff. Native seed17 landing2 is unusually distant from the available strict reference set, but no new quench/Hessian was spent and a sorted-distance fingerprint is not injective. It remains an unassigned observed candidate, not a certified new minimum or novel phase. The common initial energy9.361357969484 eV remains lowest for both arms. Accept/reject decisions and all higher-energy candidates remain included.

## Interpretation and reproducibility

The configured native weights are much larger than the realized forward-force heights (C4H6 .0603–.9332 eV; Cu13 .2665–1.6019 eV). At the displaced point one width from its center, a Gaussian supplies force `W*exp(-1/2)/width`; large initialization can therefore satisfy87° before that condition has any control over the result. This is a mechanistic explanation of the saved preparation traces, not a post hoc tuned policy recommendation.

The evidence supports retaining the conservative native profile as an explicit reproduced initial-height option. It does not support replacing the baseline or claiming87° adaptation has been tested. Isolating the angle criterion requires a separately declared minimal-height experiment or another principled control, not unreported tuning of w_level until these same seeds improve.

Artifacts: `research/ga_ssw/evidence/conservative-native-height-two-system/plan.json`, terminal `summary.json`, all8 run directories, and the frozen source snapshots. Zero-PES analyzer: `research/ga_ssw/analyze_native_height_comparison.py`, copied to `offline-analyzer.py`; full result: `offline-analysis.json`. It refuses to publish unless all8 expected terminal rows exist, checks every call count, and uses no calculator or new relaxation.
