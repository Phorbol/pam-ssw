# Independent ASE SSW family: progress as of 2026-09-11

The research worktree `research/ga-ssw-behavior-parity` now contains independent
Python/ASE implementations spanning SSW, joint and sequential VC, LS, RC and GA.
They do not require LASP or Java to execute a search. This is an experimental
family implementation, not complete native behavior parity, a merged stable
release, or established general-purpose superiority. The worktree HEAD remains
92adf28 with research changes outside that commit; the main checkout remains
on main. Frozen experiment sources, rather than HEAD alone, identify each run.

## Latest continuation: optional cluster geometry and complex cells

The recovered cluster reconnection now has a calculator-free ASE implementation,
explicit optional SSW/LS integration and 54 isolated native-instruction geometry
comparisons. The default remains disabled. The full matched C60 LS run costs
1,462 E/F including two fresh evaluations, versus the original 1,395. The first
1,381 coordinates/energies/forces and all climbing records match exactly. The
terminal operation reconnects the fragments, but the final minimum remains
4.95160934 eV above the starting state, with one two-coordinate and one
four-coordinate carbon. MC rejects it. Thus implementation and a narrowly
controlled physical-model test are complete, while search benefit is unproven.
See `hard-c60-mace-omat-reconnect-single-step/root-prefix-audit.json` and report.

The ordinary matched SSW run costs179 E/F and returns to the same basin according
to an additional23 E/F stricter endpoint comparison. The native-derived0.1-eV
release-margin counterfactual completed1315 E/F and also reaches C58+C2,
MC-rejected. Fragmentation is therefore not specific to LS in this seed;
continuing the climb does not establish benefit. No default change is inferred. Reconnection's earlier terminal diagnostic spent80 E/F, with a further
45 E/F invalid-source attempt retained in the cost ledger.

Complex official TiO2 phase-87/139 coordinates have completed26 E/F/stress
prechecks. Phase-87 PQC and joint VC both reached their original1,500-request
ceilings without proposals. PQC continuation adds448 requests and yields a
force/stress-qualified but5.10869-eV-higher, MC-rejected landing; cumulative cost
1,948. The corrected joint continuation adds568 requests and stops at Gaussian index8
with an unconverged biased quench at300 optimizer iterations, without a true
landing. Another500 requests from an incorrectly selected failed-stage boundary
are explicitly excluded from the comparison and retained as development cost.
No speed or relative-search ranking follows. The next bounded check isolates
the final frozen biased objective rather than raising search budgets.
TYPE4 TiO2@Au24O4 has514 atoms, fixed atoms1..297 and217 active atoms. Its initial
constrained quench and checks cost35 E/F including preflight; it has no complete
SSW proposal yet. All of these runs use CPU MACE-OMAT; no V100/Slurm job was sent.

## Latest MACE-OMAT C60 result

User-selected MACE-OMAT-0-small, CPU float64, single thread; no V100 or Slurm job.
Checkpoint SHA256:
`0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`.
The isolated environment used MACE0.3.16, ASE3.26.0, NumPy2.0.2 and Torch2.8.0.
The initial structure was re-quenched on MACE, with unchanged SSW/LS parameters,
seed3, memory400 and limits2,000 E/F/900 seconds.

| Observation | Result |
|---|---|
| Complete numerical lifecycle | 12 biased stages, true-PES quench, fresh checks and MC completed |
| Cost | 16 initial + 1,377 outer + 2 fresh = 1,395 E/F; 556.698 seconds |
| Prechecks | 4 additional E/F; entire MACE campaign1,399 |
| Initial fresh force | 0.00939360 eV/Angstrom, connected non-Ih C60 |
| Landing fresh force | 0.00575745 eV/Angstrom; numerical stationarity passes |
| Landing structure | C58+C2; nearest interfragment distance16.26250 Angstrom |
| Relative MACE energy | +8.68518986 eV above the MACE initial minimum |
| MC | Rejected; current state remains the initial structure |

There is a converged and fresh-matched **numerical landing**, but no successful
intact-C60 discovery. The small force on separated fragments is not proof of
the target kind of minimum. No Hessian, DFT accuracy or generalization claim is
made. MACE removes the observed xTB SCF failure for this experiment; it does not
remove the fragmentation outcome. Absolute energies across backends are not
compared. OMat is a materials-training source, not specific validation for
isolated C60 and its fragments; see the linked upstream inventory in the run's
review protocol.

Sources: `research/ga_ssw/evidence/hard-c60-mace-omat-single-step/whole-run/report.md`,
`whole-run/results/paper-seed3/result.json`, `fresh-checks.json`, `review.json`
and `precheck-accounting.json` in the campaign root. The first launch failed at
an unrelated tblite import before any PES call. The executed revision inlined
the unchanged graph helper, removing that dependency. The initial precheck
script was subsequently edited; its original full executed source is not
preserved. Its measured outputs remain available, and the additional clean-
environment initial check reproduced energy and forces exactly. The complete
search has its own executed source snapshot.

## Implemented mechanics and outstanding boundaries

| Layer | Implemented | Not yet closed |
|---|---|---|
| Ordinary SSW | Random anchor, dimer/Ritz softening, conservative Gaussian climbing, true quench, MC, failure state and cost accounting | Complete native CBD/Broyden history and all native caller behavior; difficult-case efficiency |
| Local optimization | Safe-total working backend, explicit memory option across consumers; isolated native LBFGS/MCSRCH/MCSTEP comparison | No universal optimizer winner; complete native BFGSDRIVER parity absent |
| VC | Joint atomic/log-strain escape, consistent E+pV and stress pullback; separate cell/atomic block algorithm and posterior cell quench | Native stress producer details; broad multi-seed material efficiency and phase validation |
| LS and VC-LS | Frozen pair/reference state, soft-only prequench, response updates, native-derived lifecycle; periodic image bonds and stress | Consistent cross-system gain; complete C60 cluster compression/vapor handling |
| RC and RC-VC | Tree/forest topology, generalized coordinates, force/torque pullback, pose/strain coupling, unrestricted final quench | Closed-loop constraints; complex molecular-crystal physical validation |
| GA | TYPE0–4 proposal/initialization families and independent quick/archive/fine controllers with periodic/substrate paths | Exact DCCD and Java scheduling parity, broad GA efficacy, comparison against ASE-GA/USPEX |
| ASE interface | E/F calculators; stress-capable calculators for VC; fixed-substrate path | This is not a guarantee that every backend, constraint or chemical domain is supported/validated |

Implementation inventory and public entry points:
`pamssw/standalone/__init__.py`, `ssw-family-current-status.md`.
Historic gap inventories explicitly marked superseded are not current status.

## What the existing experiments support

* Native numerical instructions were actually executed in isolation for31
  frozen Cu13/EMT biased problems: Safe-total31/31 versus native LBFGS29/31.
  Native was somewhat cheaper on the common-success set. This is not a full
  LASP or global-search ranking (`native-lbfgs-frozen-emt-replay.md`).
* Memory400 helped selected frozen problems and a four-pair Cu13/C4H6 new-seed
  study, but has not established a general search advantage. Default remains10
  (`explicit-lbfgs-memory-design.md`, `safe-history-newseed-e2e-results.md`).
* Eight Al8O14H4/brookite48 MACE searches and12 domain-specific strict minimum
  qualifications completed. One joint-VC Al8O14H4 landing remains0.2005702 eV
  below its same-domain refined start; two finite-difference steps give positive
  finite-cell projected Hessians. This is model-local evidence, not a phase/GM
  or universal VC efficiency claim (`material-gate-qualification.md`).
* Ordinary/paper/native-LS C4H6 experiments do not establish a consistent LS
  winner. Butane and water-dimer RC, Cu adsorption and small RC-VC tests provide
  limited real-system integration evidence, not large molecular-system validation.
* Hard C60 has now separated several issues: finite-difference force precision
  affects dimer stopping; xTB can fail electronically after rotation; MACE can
  complete the numerical cycle yet return fragmented high-energy structures.

## Reverse-engineering status and next priorities

Recovered evidence includes Gaussian/high-angle height behavior, LS pair terms
and force-consumer lifecycle, curvature sampling, optimizer line-search calls,
parts of cell and rigid-coordinate transformations, and Java GA operations.
Some isolated instruction/geometry fixtures are verified. This is targeted
algorithm reconstruction, not a recovered complete original source tree.

The immediate missing piece is the action chain for the C60 SI settings
`globalcompress=0.0001`, `vapor_cri=1.7`, `Ratio_Local=50`, present in both SSW
and LS-SSW inputs. The inspected native caller confirms only a conditional vapor
check and local-stop path; it does not yet establish compression or downstream
MC handling (`c60-native-fragmentation-boundary.md`).

Next priorities, in order:

1. Close those existing native cluster-control semantics and distinguish
   intermediate fragmentation, true landing and native stopping. Do not invent
   a connectivity restraint solely to rescue this seed.
2. Compare ordinary SSW and LS on the same MACE model with declared common
   budgets and seeds, including a mechanism-specific ablation only after its
   equations and action are established. The present single LS run cannot
   isolate LS's contribution to fragmentation.
3. Consolidate SSW/VC difficult-case reliability and multi-seed cost/coverage
   evidence; broaden RC/GA validation after core issues, rather than prioritize
   more GA scheduler features now.
4. Close the remaining XXXII molecular-crystal force/stress qualification and
   RC-VC evidence, then prepare a documented, tested stable integration.

No production-readiness, full-family native equivalence, or exhaustion of all
possible algorithmic improvements is claimed. The latest CPU experiment is
terminal; there is no GPU job associated with this campaign.
