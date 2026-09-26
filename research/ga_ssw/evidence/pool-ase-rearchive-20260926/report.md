# C4H6 historical pool re-archive: ordered vs ASE-v1

This zero-PES replay found substantially fewer archive entries under ASE permutation matching on all four frozen C4H6 histories. It did **not** find a merge across the previously recorded connected, element-labeled graph classes among the mapping-divergence cases. A number of matcher-defined merges have energy differences above the configured 0.001 eV diagnostic tolerance, so the result identifies cases for review; it does not establish basin identity or a matcher error.

Each arm contains 400 raw attempts. Candidate input order was the initial structure followed by every numerically qualified landing (`converged=true`, true surface, finite saved `max_force <= 0.03 eV/Å`) in record order, with accepted and MC-rejected landings both included. No connectivity or graph class filtered inputs. The four complete denominators and outcomes were:

| Arm | Raw `minima` | Qualified landings (accepted / rejected) | Observations re-archived | Ordered entries | ASE entries | Mapping divergence events |
|---|---:|---:|---:|---:|---:|---:|
| SSW seed 61 | 401 | 400 (140 / 260) | 401/401 | 53 | 20 | 257 |
| SSW seed 67 | 400 | 399 (120 / 279) | 400/400 | 49 | 21 | 278 |
| NativeLS seed 61 | 401 | 400 (131 / 269) | 401/401 | 171 | 38 | 334 |
| NativeLS seed 67 | 401 | 400 (122 / 278) | 401/401 | 146 | 38 | 316 |

“Mapping divergence” means the two archive replays selected different representative observation indices or classified the candidate as new in only one archive. For `ordered_new_ase_duplicate` cases, counts were 33, 29, 134, and 108 respectively. The reverse `ordered_duplicate_ase_new` occurred once in SSW seed 67 and once in NativeLS seed 61. Remaining divergence events assigned an observation to different representatives after the entry histories had already diverged.

The two reverse events are not evidence that ASE failed to match the same geometry pair that ordered matching accepted: each archive's greedy representative set had already evolved differently. The result includes both RMSD metrics from the current candidate to each archive's current assigned representative, but no history-independent shared-representative replay was run. Attribute these rows only to a mapping difference in the two sequential archive replays.

## Representative energy, RMSD, and topology diagnostics

For every divergence event, the result records the candidate and both representative source record IDs, both assigned-reference RMSDs under both metrics, candidate-minus-representative energy differences, and previously recorded graph-class IDs where available. Across the mapped references in these events, no candidate crossed a **known** recorded graph class under either assigned representative. All candidate/reference graph classes in the divergence rows were known; the full input sets also contained 7, 9, 14, and 14 structures, respectively, outside the previously recorded topology classes. These class labels are coarse graph annotations, not basin ground truth.

Among all mapping-divergence references, the maximum ASE RMSD to its assigned representative was 0.0871, 0.0977, 0.0968, and 0.0991 Å by arm, respectively, within the fixed 0.1 Å matcher threshold. Maximum absolute candidate/representative energy differences were 4.77, 4.77, 5.42, and 6.64 meV. The archive's energy-mismatch counters (`|ΔE| > 1 meV`) were:

| Arm | Ordered mismatch hits / max | ASE mismatch hits / max |
|---|---:|---:|
| SSW seed 61 | 58 / 4.98 meV | 61 / 4.98 meV |
| SSW seed 67 | 115 / 4.42 meV | 67 / 4.77 meV |
| NativeLS seed 61 | 66 / 5.40 meV | 59 / 5.42 meV |
| NativeLS seed 67 | 66 / 3.37 meV | 101 / 6.64 meV |

For cases where ordered matching created a new entry but ASE mapped to an existing entry (33, 29, 134, and 108 cases), the maximum ASE RMSD was 0.0797, 0.0977, 0.0958, and 0.0991 Å; maximum absolute energy differences to the ASE representative were 4.47, 4.05, 5.27, and 6.64 meV. None crossed a known recorded graph class. These are approximate geometric merges within the frozen RMSD rule; energy and graph annotations do not certify basin truth.

These energy differences flag geometry matches whose saved energies differ by more than the configured tolerance. Because observations were archived from existing records and not independently re-evaluated, they do not by themselves prove a bad merge or distinguish PES/model drift from geometric matcher error.

Taken together, the full four-arm replay and its zero observed crossover between known recorded graph classes support proceeding to the separately approved bounded development comparison. The 0.1 Å results remain approximate geometry-based identity evidence, not exact basin labels.

## Cost and limits

All arms completed fully in 4.4–14.7 seconds of measured replay time. Ordered/ASE RMSD comparison calls were 5,189/1,163; 4,672/1,419; 29,577/1,933; and 23,417/2,025. Total RMSD-call time was 0.56/2.39 s; 0.50/2.87 s; 2.98/3.89 s; and 2.28/4.04 s, respectively. ASE required fewer pair comparisons because its archive remained smaller, while its individual comparisons cost more. These offline timings say nothing about search performance or online restart benefit. The recorded historical E/F costs remain in each result and were not spent again.

There was a scheduler allocation mismatch: the array requested one CPU per task, but Slurm allocated two CPUs per task. Three tasks overlapped at peak, so the observed peak allocation was six CPUs for roughly nine seconds; every task still completed within 18 seconds. No further CPU jobs are planned in this task.

The full raw observation maps, representative pairs, energies, RMSDs, graph annotations, source hashes, matcher source hashes, and costs are in the four per-arm JSON files in this directory. The exact candidate rule and replay protocol are in [protocol.md](protocol.md); the committed replay script is [rearchive.py](rearchive.py).
