# C60 matched controls: what the completed experiments isolate

All four runs use the same non-Ih C60 input, MACE-OMAT-0-small CPU float64,
seed3 and numerical settings. They are mechanism tests on one development case,
not an independent multi-seed performance benchmark.

| Run | Full E/F cost including2 fresh checks | Result |
|---|---:|---|
| Ordinary SSW, paper zero-margin exit |179|Second-Gaussian exit, MC accepted, same basin under stricter endpoint diagnosis|
| Paper-LS, zero-margin exit |1395|12 Gaussian stages, C58+C2, +8.68518986 eV, MC rejected|
| Paper-LS plus optional native-derived terminal reconnection1.7 Angstrom |1462|Identical climbing, connected60 with2/4-coordinate defects, +4.95160934 eV, MC rejected|
| Ordinary SSW, native-derived0.1-eV exit margin |1315|12 Gaussian stages, C58+C2, +8.68518658 eV, MC rejected|

Ordinary zero-margin versus margin trajectories share the first177 logged
physical requests exactly, then differ when the margin arm continues climbing.
The zero-margin landing received an additional23-request strict two-endpoint
quench: aligned RMSD5.82e-5 Angstrom, energy difference1.96e-7 eV and identical
90-edge graphs support a return to the same local basin. The LS/reconnection
pair shares all first1381 E/F/coordinate records and all climbing records
exactly; the optional component changes only the unbiased landing input.

The shared fragmentation composition is not evidence that LS alone causes
fragmentation. Ordinary SSW reaches that composition when its climb is allowed
to continue. Nor does avoiding early exit automatically improve discovery: this
single margin counterfactual spends more and finds no lower-energy intact cage.
Do not promote either the0.1-eV margin or reconnection to a default from these
results. Native0.1-eV masked exit is recovered to Allopt handoff, but other
native lifecycle/direction details differ; this is not whole native parity.

Zero-PES atom-resolved diagnosis: the saved paper anchor is exactly reproduced
from seed3. It selects atoms10 and48 (zero-based), with mixing coefficient
1.4507417694. Those two atoms account for99.5349% of its squared norm. Both the
ordinary margin and paper-LS landings detach atoms34 and48 as C2. The LS saved
completed quenches remain connected through Gaussian8 at both1.64/1.7-Angstrom
graph cutoffs, then have C58+C2 with6.07-Angstrom separation at Gaussian9 and
16.24-Angstrom separation at Gaussian12. These are discrete geometry snapshots,
not a transition-state pathway or proof of causality for every seed.

The2013 paper (DOI10.1021/ct301010b, p1839 equations1–2) explicitly normalizes
the global Maxwell direction and uses the raw two-atom coordinate difference
for the local contribution. The observed locality therefore does not by itself
identify a formula bug. Next recover the release's actual local/global direction
normalization and Ratio_Local consumer. Keep the initial-direction test separate
from LS spectrum changes; no guessed mixing rule or retrospective seed selection.

Evidence under `research/ga_ssw/evidence/`: the four `hard-c60-mace-omat-*`
run directories; `c60-anchor-locality.json`; original LS
`whole-run/results/paper-seed3/fragmentation-stage-audit.json`; the reconnection
`root-prefix-audit.json`; and the margin source-diff/prefix reports. MACE accuracy
for these isolated configurations, Hessian stability and a C60 GM discovery
claim remain unqualified. Whole costs also include4 original precheck requests,
23 strict endpoint requests and125 terminal counterfactual requests (including
45 invalid-source requests), for4503 E/F across these explicitly listed tests.
No GPU or Slurm job was used. Wall times are not controlled speed comparisons.


## Direction-scale follow-up (zero PES)

The release's pair-local generator independently normalizes its local vector
before mixing (native call0x5d7dd7), with Ratio_Local-derived coefficient in
slot4. This is a source distinction from the2013 paper equation. An algebraic
same-C60/same-seed check preserves the global draw, selected pair andlambda:
raw local norm9.9967867 becomes a weighted contribution14.502756 against unit
global norm; unit-normalizing only this local term changes the selected-pair
squared share from99.5349% to66.5822%. This controlled algebraic change is not
a full native mode: native neighbor cooperation, constraints, pair selection
and coefficient distribution are separate. It has no measured search benefit.
Evidence: `c60-normalized-local-scale.json` and its audit script.


## Completed local-direction controls

The unit-local arm uses138EF (one Gaussian); the native-cooperative-unit-local
arm uses170EF (two Gaussians). Both preserve the first17 physical requests of
the ordinary baseline exactly. Their global/pair/lambda draws and outer MC RNG
state agree; the helper uses a cloned stream and accepts5 neighbors in6 draws.
The selected-pair squared direction shares are66.58% and22.58%, respectively.
Both landings are intact three-coordinate60-atom graphs with energy changes
about-1.1e-4eV, which are not by themselves evidence of distinct basins.

Two strict fmax.001 quenches require another21EF total and give aligned
RMSD<6e-5Angstrom with the saved strict ordinary/initial references, identical
bond graphs and energy differences below4e-7eV. Thus both trials support return
to the original basin, not improved discovery. An initial postprocessing field
used the wrong reference energy; root corrected it from the saved strict
reference energies, retaining the raw result and executed script. No additional
PES was needed for that correction. See c60-local-direction-strict-validation.
Total completed cost for the explicitly listed C60 diagnostic series is4832EF.

To separate early-exit interaction from direction geometry, the existing
raw+0.1eV-margin control is joined by two predeclared unit-local/cooperative
+0.1eV-margin controls, each at2000EF/900s. These are single-input research
ablations; they do not change the production zero-margin sampler or its defaults.
No further margin/radius/mixture sweeps are authorized by this experiment plan.

## Completed direction × early-exit controls

All six runs use the same seed-3 MACE-OMAT CPU setup and initial input.  The `.1` column is the margin01 counterfactual; it changes only the frozen comparison `true_energy < current_energy` to `true_energy < current_energy - 0.1`.  `EF` includes two fresh checks.  `G` is the number of recorded climb/Gaussian stages; `MC` is the top-level accepted flag.  Graph status is the final fresh check, not a claim of chemical stability.

| Direction | margin | EF (search+fresh) | G | MC accepted | final graph | strict same-basin qualifier |
|---|---:|---:|---:|---|---|---|
| unit-local | 0 | 138 (136+2) | 1 | yes | 1 component, connected, 60×degree-3 | strict validation: same basin as ordinary strict (RMSD 5.24e-5 A) |
| unit-local | .1 | 1984 (1982+2) | 12 | no | 1 component, open; `{1:1,2:8,3:51}` | numerically converged candidate, MC rejected; no strict basin claim |
| native-cooperative | 0 | 170 (168+2) | 2 | yes | 1 component, connected, 60×degree-3 | strict validation: same basin as ordinary strict (RMSD 5.87e-5 A) |
| native-cooperative | .1 | 1744 (1742+2) | 12 | no | 1 component, open; `{1:1,2:1,3:57,4:1}` | numerically converged candidate, MC rejected; no strict basin claim |
| ordinary | 0 | 179 (177+2) | 2 | yes | 1 component, connected, 60×degree-3 | strict validation: same basin as ordinary strict |
| ordinary | .1 | 1315 (1313+2) | 12 | no | 2 components, C58+C2; `{1:2,3:58}` | fragmented; no same-basin claim |

For the explicitly listed C60 diagnostic series (including earlier LS/reconnection and strict checks), the cumulative cost is **8,560 EF = prior 4,832 + current margin01 unit-local 1,984 + current margin01 native-cooperative 1,744**.  The six displayed rows retain their own exact row costs above.  The margin01 rows all reached their Gaussian limit without accepted landing, while the zero-margin rows accepted an energy-lowering candidate.  This single seed and changed stopping rule do not establish a general benefit or harm, and the one-component open cages are not closed-cage successes.

Primary artifacts are each row's `results/ssw-seed3/summary.json`, `result.json`, `fresh-checks.json`, and `evaluations.jsonl` under `research/ga_ssw/evidence/`.

Final unit-local margin candidate: E=-503.826127458518eV, delta+9.9483435003478eV, fmax0.00981078777664eV/A. Final cooperative margin candidate: E=-505.9934999480359eV, delta+7.780971010829887eV, fmax0.00541716640848eV/A. These are independently recomputed numerical landings despite MC rejection, not absent landings or qualified closed cages. Further same-input parameter sweeps are stopped; the remaining source question is the actual globalcompress move, not a new reward or threshold.
