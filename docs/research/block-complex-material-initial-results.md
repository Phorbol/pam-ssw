# Blocked SSW-crystal: first complex-material endpoint diagnostics

2026-09-10. Offline analysis of existing seed-3 runs. **No new energy, force,
stress, relaxation or Hessian calls** were made. Reproduction script and full
geometry output: `research/ga_ssw/evidence/block-complex-geometry/diagnose.py`
and `geometry.json`. Source results are
`research/ga_ssw/evidence/block-brookite48-seed3/result.json` and
`research/ga_ssw/evidence/block-aloh26-seed3/result.json`; each directory retains
input, plan, source snapshot, original result and last evaluated structure.

Both runs completed one cell-only proposal with an independently checked
low-force/low-stress landing, and both rejected its higher energy. The next
proposal scheduled atomic SSW but hit the declared 300-second development
budget. Neither case completes the cell+atomic SSW branch; neither gives an
algorithm-efficiency or stable-phase result.

## Structural findings

| Quantity | Brookite-derived initial → landing | AlOH initial → landing |
|---|---|---|
| Preserved composition | Ti16 O32 (48 atoms) | Al8 O14 H4 (26 atoms) |
| Volume, Å³ | 525.972608 → 609.908864 (+15.96%) | 273.784997 → 309.618289 (+13.09%) |
| Density, g/cm³ | 4.034247 → 3.479050 | 2.692127 → 2.380557 |
| Stored ΔE, eV per cell | +6.864787992 | +0.303659884 |
| MC accepted | false | false |
| Fresh initial / landing fmax, eV/Å | 0.00942185 / 0.00898472 | 0.00811367 / 0.00987434 |
| Fresh initial / landing max absolute stress component, eV/Å³ | 0.000158685 / 0.0000291850 | 0.000116444 / 0.000144675 |

The physical observables above are from the existing `fresh_checks`, not newly
recomputed. Both fresh energy errors were zero in each run. The stored plan
uses MACE OMAT-small, zero external pressure, atomic fmax 0.01 eV/Å and stress
tolerance 0.001 eV/Å³. Small force/stress residuals establish those numerical
certificates only; no endpoint Hessian or finite-wavevector phonon stability
was tested here. These observations are model-PES results, not DFT validation.

### Brookite-derived 48-atom structure

At all three Ti–O distance cutoffs 2.2, 2.3 and 2.4 Å, the initial structure
has all 16 Ti sites six-coordinate. The landing coordination distributions
are:

| Cutoff, Å | Number of Ti sites by coordination |
|---|---|
| 2.2 | CN4: 5; CN5: 7; CN6: 4 |
| 2.3 | CN4: 5; CN5: 3; CN6: 7; CN7: 1 |
| 2.4 | CN4: 5; CN5: 2; CN6: 5; CN7: 4 |

The exact higher-coordination counts are threshold-sensitive, but five
four-coordinate Ti sites persist across this range. This supports substantial
coordination reorganization accompanying expansion; it does not identify a
particular polymorph or establish a stable new phase.

Periodic species-pair shortest distances (initial → landing, Å): Ti–O
1.868376 → 1.768335; O–O 2.514631 → 2.402740; Ti–Ti 2.972994 → 2.983292.
There is no near-zero atomic overlap in these endpoints. The expansion and
coordination reduction still require physical scrutiny; absence of overlap
is not a model-domain or phase-stability certificate.

### Al8O14H4 structure

All four H sites have exactly one O neighbor at cutoffs 1.1, 1.2 and 1.3 Å,
for both endpoints. Atom indices below are zero-based and retain the input
ordering. H20 changes nearest oxygen from O16 (0.981475 Å) to O11
(1.061264 Å); H4–O1, H8–O2 and H18–O15 retain their nearest O identities.
This is evidence of a change in endpoint proton coordination, not an observed
transition-state trajectory or rate.

| Al–O cutoff, Å | Initial coordination distribution | Landing distribution |
|---|---|---|
| 2.2 | CN4: 4; CN5: 3; CN6: 1 | CN4: 6; CN5: 1; CN6: 1 |
| 2.3 | CN4: 4; CN5: 3; CN6: 1 | CN4: 6; CN6: 2 |
| 2.4 | CN4: 4; CN5: 2; CN6: 2 | CN4: 6; CN6: 2 |

Shortest periodic distances (initial → landing, Å): Al–Al
2.724188 → 2.716985; Al–H 2.253803 → 2.266206; Al–O
1.689523 → 1.728318; H–H 3.033717 → 2.972618; H–O
0.981475 → 0.975601; O–O 2.428677 → 2.441553. No near-zero overlap
is present. The changed proton partner and Al coordination distinguish this
landing from simple affine volume change.

## Structural identity and method limitations

Periodic neighbor counts use ASE `neighbor_list`, including multiple periodic
images when distinct neighbors occur in adjacent cells; only zero-shift
self-interaction is excluded. A 6 Å neighbor radius contains every reported
species-pair minimum and every coordination threshold. Density uses ASE atomic
masses and 1.66053906660 g/cm³ per amu/Å³. Chosen bond-distance ranges are
explicit diagnostic sensitivity checks, not universal bond definitions.

Pymatgen `StructureMatcher` was run with `scale=False`,
`primitive_cell=True`, `attempt_supercell=True`, and symmetric matching for
three `(ltol, stol, angle_tol)` settings: `(0.1,0.15,2°)`, `(0.2,0.3,5°)`,
`(0.3,0.5,10°)`. **Neither initial/landing pair matched under any of these
settings.** `stol` is Pymatgen's normalized site tolerance, not Å; all settings
and installed package versions are in `geometry.json`. Disallowing volume
rescaling preserves the physical density change. Matcher non-equivalence plus
coordination changes provides structural evidence beyond energy-only identity,
but is still an approximate criterion, not a proof of distinct stable basins.

## Full cost and incomplete work retained

| Cost/status | Brookite48 | AlOH26 |
|---|---:|---:|
| Initial quench E/F/stress requests | 10 | 100 |
| First cell-only proposal requests | 269 | 293 |
| Second, incomplete combined proposal requests | 330 | 720 |
| Total search requests | 609 | 1113 |
| Existing fresh-check requests | 2 | 2 |
| Total recorded search + fresh requests | 611 | 1115 |
| Recorded search elapsed seconds | 300.080440 | 300.010241 |
| Additional requests in this diagnosis | 0 | 0 |

The second proposal records `atomic_evaluation_failed` with the explicit error
`RuntimeError: declared development request/wall budget exhausted` in both
cases. Its atomic portion accounts for 182 and 567 requests, respectively;
these are included in the incomplete-proposal totals, not extra costs.
The runner's top-level `completed` means return/serialization completed; it
does not mean both requested proposals or the atomic branch finished.

Thus each system has one valid, rejected cell-only landing and one
budget-censored combined attempt. Calling the second attempt an intrinsic
numerical failure would misclassify the evidence; omitting its cost would also
misrepresent it. All rejected landings should remain discoverable in the
exploration archive, while the accepted chain remains at the initial state.
The two endpoints stored under the result key `minima` are not promoted here
to Hessian-qualified minima. Further efficacy claims require a completed
combined branch, comparable baselines, repeated seeds and separate structural
and curvature qualification where relevant.
