# C4H6 / MH-1 lifecycle and cost pilot

Category: implementation/cost diagnosis, seed59; not independent efficacy evidence.
GPU1469509 completed; CPU1469713 offline audit completed with no accounting,
fresh-coverage or missing-input errors in any arm. CPU1469681 failed in the
analysis script with an undefined output variable; its log is retained. Only
that analyzer was corrected, and no PES calculation was repeated.

| Method | Outer attempts / force-qualified landings | Search E/F requests | Actual calculator calls | Fresh qualified | Search seconds |
|---|---:|---:|---:|---:|---:|
| SSW | 12/12 | 7651 | 6746 | 13/13 | 111.3 |
| Paper LS | 12/12 | 7512 | 6572 | 13/13 | 107.5 |
| Native-inspired LS | 12/12 | 8108 | 7166 | 13/13 | 114.9 |

Total23271 search requests +39 fresh checks. No denied or failed E/F requests.
All36 climbs reached their Gaussian limit and subsequently obtained a
force-converged true-PES landing. That climb stop label is not a failed quench.
All minima observations, including rejected landings and repeated visits, are
included once; there is no duplicate addition of record.landing.

The graph classifier found3/4/1 within-arm graph classes respectively, with
butadiene torsion changes also observed. These short, single-seed counts do not
rank the methods or reproduce the paper's reaction network. Native-inspired
LS response after12 updates is0.035925eV/atom, while paper LS reaches0.699347;
both targets are0.7. Equal targets do not mean equal effective softening.

Decision: the existing execution chain is ready for a longer frozen comparison;
no parameter/default changes. See [next protocol](../c4h6-mh1-coverage-20260924/plan.md).
Its400-attempt limit differs from the paper's400 visited minima. The pilot's
observations will not be pooled with its new seeds.

Reproduce offline readout on a CPU compute node: `python analyze.py` (requires
no existing analysis.json/report.md). The analyzer reads the raw sibling arm
directories and the previously committed torsion/graph helpers; it performs no
PES evaluations. Existing derived outputs are intentionally not overwritten.

Authoritative small outputs: [summary](summary.json), [effective settings](effective_config.json),
[analysis](analysis.json), [detailed report](report.md), and each arm's summary
and fresh-checks.json. Large raw result.json, requests.jsonl and checkpoint.pkl
remain under this exact shared-storage evidence directory, not copied intoGit.
Execution source57e5097; core tree46f0049ed02c773621ac614093386220227e3d3a.
Input/model/ledger provenance is in summary.json and the fixed plan.
