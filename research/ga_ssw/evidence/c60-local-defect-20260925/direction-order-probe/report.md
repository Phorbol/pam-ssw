# C60 fixed relabeling qualification: completed, no new Ih recovery

## Result and scope

One fixed permutation of the same qualified physical local-defect cage gives **0/4 joint Ih-and-energy hits**, compared with **2/4** under the original ordering. This difference remains at the four matched search-cost prefixes. These are two numeric seeds and two methods on one physical input, not four or eight independent structures or a reliable success-probability estimate. The original successful geometries remain valid; this result does not erase them.

| Method | Seed | Original search requests | Relabeled search requests | Original joint-hit trajectory | Relabeled joint-hit trajectory |
|---|---:|---:|---:|---|---|
| SSW | 1101 | 4106 | 4078 | no | no |
| native LS | 1101 | 4090 | 4378 | yes | no |
| SSW | 1102 | 4301 | 4742 | yes | no |
| native LS | 1102 | 3801 | 4248 | no | no |

All four new arms completed ten attempts. All 40 landing observations passed fresh force/convergence/composition/cell/PBC qualification. No landing was fragmented under the existing 1.8-A graph criterion. There were 14 intact-cage observations across arms (4/1/5/4), all without Ih recovery; their best energy remains about 1.223 eV above the qualified Ih reference. A graph change or intact cage is not the requested repair success. Force qualification is not a Hessian or chemical-accuracy certificate.

A joint hit requires the same qualified landing to match the Ih graph at all three frozen cutoffs (1.64/1.7/1.8 A) **and** the existing 0.01-eV reference window. No new hit exists to align geometrically; the independent geometry checker therefore correctly returns an empty row list. The original two first-hit geometries remain independently aligned in the parent panel. Repeated hits are not extra successful trajectories.

## Costs and verification

GPU1493205 completed with exit0 in43m37; CPU1493206 completed with exit0 in3s. New work charged17446 search requests plus44 fresh checks, within the48000+44 and60-minute limits. All four record-cost sums reconcile, with zero requests outside saved records, zero missing/failed fresh checks and no search-cap boundary. Whole-job time includes work outside the per-arm search timers; no runtime speedup claim is made. Original work16298+44 remains charged separately; relabeling is new evidence but prior-source rereads are not.

The zero-PES preflight established exact coordinate reindexing and preserved species/cell/PBC. All original numerical/model/runtime/resource settings were retained. The source plan and input permutation were fixed before execution (runner01d33fc, corea040dcffe6d029ded010da341d7cd29a99afae9a). Core includes the already approved compact-observer change; no kernel, public startup option, RNG contract or checkpoint format was changed for this panel.

Parent checks: `compare_order.py` reproduces the old two first-hit trajectories before reading new outcomes; its new report explicitly requires joint same-frame criteria. Assertions over `analysis.json` confirm four completed arms, ten qualified landings and eleven fresh checks per arm, no censoring and exact saved-cost reconciliation. Independent read-only review also reconciled all four raw request ledgers, summaries, result totals and record costs: each arm has one initial request plus its ten outer records; fresh ledgers contain eleven checks per arm, and search/fresh denials are zero. It independently reproduced the original two joint-hit trajectories and new zero-hit result at common prefixes4078/4090/4301/3801. The reviewer used the same underlying experiments, not independent new scientific evidence. Raw summaries/checks/ledgers/trajectories remain under `runs/`, not all tracked by Git.

Readout commands (zero PES):

```sh
python research/ga_ssw/evidence/c60-local-defect-20260925/analyze_escape.py --plan research/ga_ssw/evidence/c60-local-defect-20260925/direction-order-probe/plan.json --output research/ga_ssw/evidence/c60-local-defect-20260925/direction-order-probe/analysis.json
python research/ga_ssw/evidence/c60-local-defect-20260925/verify_hits.py --runs research/ga_ssw/evidence/c60-local-defect-20260925/direction-order-probe/runs --output research/ga_ssw/evidence/c60-local-defect-20260925/direction-order-probe/hit-geometry.json
python research/ga_ssw/evidence/c60-local-defect-20260925/compare_order.py
```

These entrypoints refuse to overwrite existing reports. Preserve the existing output or choose a new analysis artifact when reanalyzing. Evidence: [analysis](analysis.json), [joint/prefix comparison](comparison.json), [geometry check](hit-geometry.json), [protocol](plan.md), [original result](../direction-probe/report.md).

## Interpretation and decision

The null result is not explained by unfinished computation, failed true quenches, or fragmentation. It shows that the earlier positive behavior has not demonstrated robustness under this fixed input relabeling. It does **not** identify startup tie-breaking as the cause: assigning random components to different atom indices changes physical random directions even under the same numeric seed. Two explanations remain: systematic label-dependent startup geometry and ordinary trajectory variability/changed physical random draws; this four-arm panel cannot distinguish them.

Keep the complete direction mechanism experimental. Do not tune to recover the two hits, add permutations, extend the random-C60 production budget, or promote a new default. A short C4H6 transfer panel was considered but deferred because mere trajectory differences and small class counts would not resolve a retain/promote decision. The separate random-cloud C60 cage-plus-energy acceptance remains unmet.

The separate mathematical startup symmetry question is now ready for discussion: preserve the current API and restrict shuffling to research inputs, or expose an explicit legacy-compatible optional symmetrized startup. The latter has a distributional startup symmetry argument, not demonstrated efficiency benefit, and changes settings/RNG/checkpoint identity. See the [bounded proposal and native-source boundary](../../../../../docs/research/2026-09-25-direction-startup-order-proposal.md). No public option has been implemented. Native first-call `cart_copy` chronology remains unresolved but does not block the independent qualification conclusion.
