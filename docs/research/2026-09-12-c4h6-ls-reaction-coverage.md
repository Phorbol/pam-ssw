# C4H6 SSW / LS reaction-coverage comparison (2026-09-12)

Six frozen-source CPU trajectories completed at their **120 s wall limits**.
The 40000-request and 400-outer-step limits were not reached. All 129 returned
landings passed independent cold GFN2-xTB force checks at 0.01 eV/Angstrom.
This establishes executed search and force-qualified candidate connectivity,
not 129 distinct minima, TS connectivity, PBE accuracy or an LS speedup.

Evidence: `research/ga_ssw/evidence/c4h6-ls-reaction-coverage-20260912/`.
`root-final-audit.json` checks source hashes, exact ledger sequences, request
costs and fresh certificates; `root-coverage-audit.json` contains per-landing
connectivity classes and cumulative search requests. Source/initial/backend/
settings/seeds are frozen in the same directory. Search cost is API E/F
requests, not electronic iterations. All cap denials are separate ledger rows.

## Question and fixed protocol

Guan, Shang and Liu, *Local-Softening Stochastic Surface Walking for Fast
Exploration of Corrugated Potential Energy Surfaces*, JCTC (2024),
DOI [10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081),
section 3.1 / Figure 5 uses trans-butadiene, target 0.7 eV/atom, T=150 K,
NG=25 and ds=0.1 Angstrom for reaction/isomer exploration. H2 loss and radical
channels are explicitly included; every fragmentation must not be labelled a
failed optimizer. The author full text is locally available as `literature/215.txt`
in the original GA-SSW research archive. Global energy optimization remains a
separate mainline objective; its earlier negative results are not reclassified.

The current run uses G2 trans-butadiene, GFN2-xTB accuracy .001, seeds 11/29,
staged Ritz pre5/total100, width .1, NG25, inner .1 / outer .01 eV/Angstrom,
Safe-total history10 with 400 local iterations. The same shared SSW settings
are used by all three arms; only LS is added. Explicit recovered HC tables,
paper initial fraction .03 / xi .2 / learning rate1.8, and native scale5 /
amp_c2 / eta.005 / max_change.01 / frequency10 / presteps100 are retained.
Native target700 meV/atom equals paper target.7 eV/atom in units, but the two
controllers need not reach that target at the same rate. No post-run tuning.

## Measured costs and common-prefix comparison

| Method | Seed | Search requests | Fresh requests | Returned landings |
|---|---:|---:|---:|---:|
| SSW | 11 | 23047 | 20 | 20 |
| SSW | 29 | 20409 | 22 | 22 |
| Paper LS | 11 | 20793 | 22 | 22 |
| Paper LS | 29 | 20614 | 21 | 21 |
| Native-rule LS | 11 | 23953 | 22 | 22 |
| Native-rule LS | 29 | 22361 | 22 | 22 |

Total **131177 search + 129 fresh = 131306**. Each arm has one wall denial
and no backend-failure ledger row. The driver's `evaluation_failed` terminal
label here is the explicit wall-cap exception, not SCF failure. Nonfatal
rotation failures remain in records. Fresh max energy discrepancy is
7.39e-13 eV. Other bounded CPU tests ran during parts of this campaign, so wall
throughput is not an isolated performance measurement.

Compare only completed, fresh-qualified landings in the common request prefix:
seed11 up to20793, seed29 up to20409. Counts below **include initial connectivity**.
A landing's cumulative cost is matched one-to-one to the stored true quench
record, with exact serialized geometry/energy equality; repeated structures
are not assigned the first occurrence's cost.

| Seed | Method | Qualified landings | Connectivity classes | Fragmented landings | Best energy change (eV) |
|---|---|---:|---:|---:|---:|
| 11 | SSW | 17 | 4 | 1 | -0.14333364 |
| 11 | Paper LS | 22 | 5 | 0 | -0.14332296 |
| 11 | Native-rule LS | 20 | 6 | 2 | -0.14333391 |
| 29 | SSW | 22 | 4 | 2 | -0.14333541 |
| 29 | Paper LS | 20 | 4 | 0 | -0.14332841 |
| 29 | Native-rule LS | 20 | 4 | 2 | -0.14332863 |

Graphs use HC_BOND_LENGTHS + .1 Angstrom with element-labelled isomorphism.
The full collection has nine graph classes: four match G2 butadiene,
cyclobutene, methylenecyclopropane and bicyclobutane; one has C2H4 + C2H2
components; four are unassigned C4H6 connectivity candidates at higher energy.
Graphs do not resolve bond orders, stereochemistry or barrier connectivity.
These are heuristic structural labels; no TS/radical assignment or basin
stability is inferred solely from graph and force threshold. All best structures
match the cyclobutene graph on this oracle. GFN2 puts it below trans-butadiene,
unlike the paper's stated PBE ordering: **this is not numerical paper reproduction**.

## Controller behavior and decision

Paper LS has 24/28 completed-prequench response records and ends at
0.70133527 / 0.70000047 eV/atom. Its initial response was ~0.00320, so the
adaptive stage actually reached its target during these trajectories.
Native-rule LS has 35/27 normal-update events and ends at only
0.10998151 / 0.07816610 eV/atom. Its recovered bounded per-update change
limits the ramp, despite the same nominal target. Every update is retained;
response records are not confused with a paper `ls_update` object (not exposed).

Keep both explicitly named rules. These results show different structural
coverage on some trajectories, but no reproducible low-energy advantage or
universal ordering of the methods. Do not change native update limits to
force equal final response, do not promote a new default, and do not chase
this one molecule with a parameter sweep. Continue the cross-system baseline
comparison and core interface completion. More independent seeds/oracles and
stability/path checks would be needed for a stronger scientific claim.

## Post-hoc curvature qualification, no reoptimization

All nine graph classes were represented by their lowest stored-energy
fresh-qualified landing across all six arms (deterministic arm/index tie break).
On the same GFN2 oracle, independent cold force calls at +/- .001 Angstrom
construct the full 30x30 Cartesian Hessian. Six global rigid tangent modes
are removed, including for the two-component case; relative fragment modes
are retained. This added549 requests (9x61), all ledger sequences verified.
Evidence: `c4h6-connectivity-hessian-20260912`, full Hessians/bases/force
ledgers and frozen input/source are retained. No optimization or parameter tuning.

| Graph class | Smallest internal eigenvalue (eV/Angstrom squared) |
|---|---:|
| 0 | 0.00341303 |
| 1 | 0.19349588 |
| 2 | 0.07551913 |
| 3 | 0.61324705 |
| 4 | 0.64047742 |
| 5 | 0.39718036 |
| 6 | 0.95399877 |
| 7 | 0.01419756 |
| 8 | 1.54465531 |

All selected representatives retain fmax<=.01 and have positive internal
curvature at this finite-difference resolution. Hessian skew spectral norms
range9.91e-5–3.39e-4 eV/Angstrom squared. These support local-minimum
interpretations, but finite gradients and a single displacement size do not
prove exact stationarity. Classes0 and7 have particularly soft modes.
The class0 C2H4+C2H2 graph can be a weakly bound complex: disconnected
covalent graph components do not prove separated, unbound products. No TS,
rate, radical state, PBE ordering or extra performance advantage is inferred.
