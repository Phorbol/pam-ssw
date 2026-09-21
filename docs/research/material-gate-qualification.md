# Eight-run material gate: completed qualification

2026-09-10. The frozen qualification completed: 5,708 E/F/stress requests,
3,256.17 seconds, within the predeclared 6,000-request / 3,600-second CPU cap.
All 16 source endpoint aliases passed fresh original-tolerance checks; all 12
domain-specific refinement/Hessian tasks completed. Search cost remains a
separate 15,467 requests across eight runs.
`research/ga_ssw/qualify_material_gate.py` refuses to import/load MACE until the
manifest contains eight runs, execution is no longer running, and all eight
processes have return codes. Failed runs remain in the denominator; absent result
files are recorded as missing. Process completion does not certify any candidate.

## Frozen procedure and resource bound

Run with the same checkpoint as every source run, CPU float64, one CPU thread.
The script checks declared model paths and available source plan checksums,
records its own model checksum/package versions, and snapshots its script and
`pamssw`. It never changes gate results or feeds refined geometries into a search.
The prepared launcher is in `research/ga_ssw/material-gate-qualification-plan.json`.

The shared budget is **6,000 E/F/stress requests and 3,600 seconds**, including
model loading and qualification. A calculator request already in progress cannot
be preempted by the wall guard; the next request is denied. There is no GPU or
Slurm launch. Failed calculator requests count. Calls denied before calculator
execution cost zero oracle requests and remain explicit pending/error records.
The BLAS and torch thread counts are set to one. No automatic restart or repeated
budget extension is provided.

1. Load **all valid recorded landings**, including every common start and every
   MC-rejected landing, across all eight source slots. Order them by endpoint
   index, arm (`fixed`, `pqc`, `block`, `joint`), system and run name. Energies do
   not select the order.
2. Fresh-check every unique exact geometry with MACE cache reset. Byte-identical
   positions/species/cell/PBC share this request, with every source alias retained.
   Failures are shared too, not silently retried. Record full E/F/stress, energy
   discrepancy, source force threshold and source stress threshold separately.
   For the fixed-cell arm only force enters the constrained-space certificate;
   residual stress stays diagnostic. If any fresh check is unfinished, no
   refinement or Hessian work starts.
3. Traverse the same deterministic endpoint order for refinement and curvature.
   Identical geometries share this work only with identical objective domain,
   pressure and strain metric; fixed and variable-cell qualifications stay separate.
   Safe-total uses the existing Cu qualification settings: force 1e-4 eV/Å,
   maximum stress component 1e-5 eV/Å³, maximum step 0.2, 100 steps and 101 local
   optimizer requests. The convergence norm uses the currently accepted state.
   A further independently recomputed E/F/stress certificate costs one request.
   Unconverged refinements remain failed or pending, never removed or relaxed by
   changing tolerances.
4. Only after a refined point passes its domain certificate, central differences
   at h=1e-4 and 5e-5 build the Hessian in that point's own coordinates. Fixed:
   Cartesian 3N minus three translations. Variable: symmetric log-strain chart,
   3N+6 minus three translations, preserving the declared strain length and E+pV.
   Store completed columns incrementally, eigenvalues and antisymmetry. A partial
   Hessian remains partial, and all its paid calls count. Remaining endpoints stay
   explicitly pending when the global budget is exhausted.

The precisions and Hessian construction follow
`qualify_vc_cu4_minima.py`; `qualify_rutile_joint_hessian.py` also motivates removing
translations and diagnosing strain-coupled modes. These are numerical validation
settings, not calibrated universal phase-classification thresholds.

## Cost and interpretation

Two Hessian step sizes cost 4d requests: AlOH26 fixed/variable d=75/81 costs
300/324; brookite48 d=141/147 costs 564/588. Each refinement can cost another
101+1 requests. All candidates may therefore exceed the shared budget. The
report preserves the eight-run denominator and all original endpoints regardless
of which Hessians finish. `completed` means the requested checks finished, not
that all candidates passed or are stable.

Before/after atoms, volume ratio, cell difference and atom-order displacement
are recorded. The existing periodic StructureMatcher tolerance sweep compares
identity without cell rescaling; disagreement is ambiguous. It does not establish
that refinement stayed in the original basin. Report refined and original
candidates separately, especially when the identity comparison changes.
A finite-cell positive Hessian is not a larger-supercell phonon certificate, a
phase identification, a DFT result, or a global minimum claim. Small gradients
alone do not settle molecular dissociation, proton bonding or model extrapolation;
retain the gate's independent geometry/coordination analysis.

## Preparation checks

`tests/research/test_qualify_material_gate.py`: six passed. Covers eight-slot
retention, rejected endpoints, fresh-first ordering, exact common-start sharing,
different fixed/variable Hessian dimensions, pending rows and partial columns on
budget exhaustion, gate refusal while running, and real periodic Cu EMT fixed
stress separation plus a variable-cell refinement/two-step Hessian. The latter
is wiring/geometry verification, not a material search performance result.
Those preparation tests did not launch MACE. The subsequently authorized frozen
qualification has now completed; the original gate artifacts were not modified.

## Terminal evidence and scientific boundary

Full per-endpoint results: `research/ga_ssw/evidence/complex-vc-gate-qualification/result.json`;
readable derived table: `report.md` in the same folder. All 12 strictly refined
structures have positive translation-projected finite-cell Hessians at both
predeclared steps. This qualification applies to refined points, not an
assertion that the original finite-force coordinates were exact stationary points.

For the accepted Al8O14H4 joint-VC landing (task 10), fresh refined fmax is
9.90850e-5 eV/Å and stress residual 3.30555e-7 eV/Å³. Both Hessian steps give
minimum eigenvalue approximately 0.206651 in the declared coordinate metric;
the two symmetric matrices differ by spectral norm 3.02556e-6. Refinement moved
any same-order atom by at most 0.010736 Å, changed volume by factor
0.999928114, and retained structure identity at all three recorded tolerances.
These are consistency observations, not a rigorous basin-preservation theorem.

Compared with the independently refined variable-cell initial structure (task 2),
its energy is lower by 0.2005702035 eV under MACE-OMAT-0-small at zero pressure.
This supports a lower-energy finite-cell local minimum with the previously
recorded proton/Al–O rearrangement. It does not establish a stable experimental
phase, complete phonon stability, a DFT result, or superior search probability
from a single seed. Fixed-cell and variable-cell domains are not pooled.
