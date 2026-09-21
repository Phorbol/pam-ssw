# MH1 native-derived LS result

This is a bounded development comparison on two saved C60 first-state inputs;
it is not an independent success-rate study. The native-derived LS run used
24,000 search E/F calls in total and 22 fresh checks, with the completed
equal-budget baseline as the reference.

The detailed search analysis is in
[`analysis.json`](../../research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920/analysis.json),
and the lifecycle audit is in
[`ls-diagnostics.json`](../../research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920/ls-diagnostics.json).
The native-LS arms reached the 12,000-request cap and ended with
`evaluation_failed` at the terminal request. The first starting point retained
11 landing records and the second retained 9; the corresponding baseline had
11 and 11.

All 22 pre-quench stages qualified by force, all 22 LS updates were performed,
and the response-unit checks were consistent. The lifecycle audit charged 197
pre-quench E/F calls for `c60_17093-first` and 154 for `c60_17094-first`.
Force qualification does not imply complete physical qualification: one first-
state landing was disconnected.

| Saved input | Best LS energy (eV) | Best baseline energy (eV) | LS minus baseline (eV) |
|---|---:|---:|---:|
| `c60_17093-first` | -62179.906115 | -62185.311131 | +5.405016 |
| `c60_17094-first` | -62169.771792 | -62166.280165 | -3.491627 |

Both best LS candidates were accepted and connected. Per-landing energies,
acceptance and geometry remain in the linked analysis rather than being
repeated here.

At the 1.64, 1.7 and 1.8 Å diagnostic graphs, the first start has 10/11
connected LS landings: index 11 is the disconnected landing (two components),
with energy -62179.059995752 eV and rejected status. The second start has 9/9
connected LS landings. The two LS best landings are connected, but neither is a
complete C60 cage graph under the recorded cage criterion; complete-cage
count is 0/2 for the two LS best structures. These are connectivity and graph
diagnostics, not chemical stability certificates.

The separate `.01` refinement, job 1415068, completed with 12 and 11
optimization calls plus 2 fresh calls (25 E/F total). The resulting energies
were -62179.90651145753 and -62169.772146650226 eV, both with `fmax < .01`.
This post-hoc termination check is excluded from the 24,000-call search
budget. Independent analysis 1415090 completed: both refined structures are
connected at all three graph cutoffs, neither is a cage candidate, and cost
accounting matches. At the common .01 threshold, LS remains higher than the
baseline on the first input and lower on the second; this does not establish
a general ranking.

The evidence supports keeping native-derived LS as an explicit experimental
component. It does not support making it the default, tuning it from these two
starts, or combining it with another change. The next useful check is a
cross-system validation on the existing material cases rather than another C60
scan.
