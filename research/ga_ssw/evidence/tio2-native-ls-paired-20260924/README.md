# TiO2 paired SSW / NativeLS development experiment

Protocol: plan.json. Inputs phase87 and phase139 are fixed by prior source order, not selected by search outcomes. Same OMAT-small/omat_pbe, seed59, fixed cell, recovered CBD and Safe-total/history500 in both arms. True fmax=.03 and bias fmax=.1 eV/A, true/bias relax_steps400; native LS table and .1/50 force-or-step-limit prequench inherited existing lifecycle protocol. Explicit periodic-images counts all frozen image bonds; no assumption of native-MIC parity. No GA or variable cell.

First gate: two true initial quenches, 2000 E/F each max, one fresh endpoint check each. Compare both LS neighbor geometries without new PES. One V10010min. Failures retained, no expensive paired search if either initial fails. This gate supports numerical suitability, not phase stability/model accuracy.

Planned paired search:4 arms, each60000 total E/F requests including initial, LS, failed and censored work;2 independent fresh checks/arm. One V100120min hard limit;1750s per arm. Previous3-step TiO2 LS cost502–619 requests/step motivates testing beyond the earlier short lifecycle;~100 attempts is a rough estimate, not guaranteed or a power calculation. Full request and actual calculate costs reported; common prefixes used if wall capped. No result-dependent extension or parameter adjustment.

Hypotheses: LS changes the local escape surface to discover distinct/lower-energy structures, versus extra preparation cost or weak/irrelevant deformation yielding no equal-cost benefit. Endpoints: true-quench qualification, approximate geometry identity with two existing tolerances, best energy vs total cost; LS response and frozen neighbor counts diagnose activation. Responses approaching20meV/atom do not alone show benefit. Fixed-cell results are compared only within each starting cell; no cross-phase stability claim. One seed and two development inputs cannot establish general superiority.

Source frozen from Git commit in plan; source, input and shared ledger hashes recorded. Root reviews qualification before submission. Large raw ledgers/checkpoints remain in this study directory; reports/config/source paths retained in Git. C60/MH1 cage plus reference-energy acceptance remains a separate long-run criterion.

## Case-selection update before any paired search

User emphasized scientific case selection. GPU1468829 performs only the above initial qualification; the long four-arm search has NOT been submitted and is deferred. TiO2 phase87/139 originate in variable-cell material discovery and are not established LS corrugated-PES benchmarks. Their candidate role must be justified separately from coordinate provenance. No search protocol or input is selected by an LS result.
