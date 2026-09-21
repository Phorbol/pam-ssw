# Executable isolated RC forest and bounded molecular evidence

2026-09-10. The independent forest extension is now executable through a full reduced-coordinate SSW proposal, unrestricted true Cartesian quench and outer MC. It is a nonperiodic subset, not complete RC-SSW/native parity. The source and mathematical design remain in `rc-forest-design.md` and `rc-geometry-contract.md`.

## Implemented API and objective

`pamssw.standalone.rc_forest.RigidForestChart(atoms, trees, anchor=0)` accepts component dictionaries with `bodies` in global atom indices, `parents` in topologically ordered local body indices and `joints` as global oriented endpoint pairs. Additional parser provenance fields are ignored. Components are disjoint but bodies within a component share their joint endpoints. All atoms must be covered. Every root must be noncollinear; a point or linear root is explicitly unsupported because its three rotation coordinates would be redundant.

The anchor root pose is fixed once. Other roots retain three translations and three finite angle-axis rotations; every internal torsion remains. Geometry uses the existing exact exponential/Frechet map and chain products. `ForestSurface` exposes the same scalar physical energy and exact scaled Jacobian gradient, with translations in Å and explicit `rotation_length`/`torsion_length` in Å/radian. No force-transmission heuristic or separate bodywise torque subtraction is inserted.

`pamssw.standalone.rc_forest_reference.run_rc_forest_ssw(atoms, surface, trees=..., anchor=0, steps=..., config=RCForestSSWConfig(...), rng=...)` runs the complete lifecycle. Config requires root-rotation and torsion metric lengths, Gaussian width and dimer rotation bias. Shared `_run_reduced_ssw` was extracted from the existing single-chain driver: its public API and scientific policy remain unchanged. The forest uses the same fixed initial direction anchor, dimer, accumulated Gaussian objective, Safe-total relaxation, unrestricted final quench, force certificate, cost ledger and MC. Each proposal chart rebuilds from the selected true minimum. Recorded forest stage coordinates are named `scaled_coordinates`; the original single-chain field stays `scaled_torsions`.

Only force-certified candidates are called valid in the result schema. This says neither distinct minimum nor chemical success. Rejected valid candidates remain in the result; failed stages retain their E/F cost. Native lambda transmission, native angle-coordinate gauge, periodic rigid-body/cell coupling, loops, linear/point roots and topology inference remain absent. Finite rotation derivatives are exact, but an explicit domain rejection for angle-axis rank loss near full turns is not yet implemented; no global nonsingular-chart claim is made.

## Verification

`python -m pytest -q tests/standalone/test_rc_geometry.py tests/standalone/test_rc_reference.py tests/standalone/test_rc_forest.py`: **11 passed**. The forest tests include finite six-relative-DOF Jacobian differences, scaled energy-gradient differences, retained relative root motion with the anchor fixed, rigid internal distances, a mixed butane-chain/water forest with eight DOFs, exact reduction to the single-chain map, incompatible/degenerate topology rejection and full biased-path preservation of an MC-rejected landing. The stage-wiring test uses an artificial surface and is not molecular efficacy evidence. Existing ASE/NumPy deprecation warnings remain.

## Negative evidence: manually assembled water input dissociates

`research/ga_ssw/probe_rc_water_dimer_gfn2.py`, evidence `research/ga_ssw/evidence/rc-water-dimer-gfn2/`: one seed-3 proposal, 400 E/F and 30 s ceilings, GFN2-xTB accuracy .001, CPU one thread. The run used **70 search + 2 independent fresh calls = 72 E/F, 0.149 s**. It completed numerically and was MC accepted, but the *initial unrestricted quench* increased O–O from 2.90 to **17.99 Å**; final O–O was **16.30 Å**. Both full-force checks were below .01 eV/Å. This is a dissociation plateau and explicitly fails bound-dimer physical qualification. Original inputs, snapshots, every E/F record and results are preserved without tuning or overwriting.

## Source correction: actual S22 water-dimer reference

At the parent's explicit direction, a separate same-config probe replaced only the manually assembled input by `ase.data.s22.create_s22_system('Water_dimer')`. The OHH/OHH ordering is asserted. Installed ASE's module attributes the S22 geometries to Jurecka, Sponer, Cerny and Hobza, PCCP 2006, 8, 1985–1993. Exact source module metadata, ASE version and complete water-dimer database record are saved in `s22-water-source.json`; the input coordinates also appear in the frozen plan. No benchmark interaction energy was used as a GFN2 target.

`research/ga_ssw/probe_rc_s22_water_dimer_gfn2.py` and `research/ga_ssw/evidence/rc-s22-water-dimer-gfn2/` preserve the separate experiment. Same seed 3, one proposal, Lrot=Ltor=2 Å/radian, width .4 Å, rotation bias100 eV/Å², two Gaussians, HVP limit20, quench limit200 and all other settings unchanged. These are explicit development settings, not paper defaults or optimized values. Total **181 search + 2 fresh = 183 E/F, 0.374 s**, within the same 400 E/F/30 s budget. All 183 trace rows reconcile.

| Measurement | Initial unrestricted minimum candidate | End of biased proposal | Final unrestricted candidate |
|---|---:|---:|---:|
| O–O distance (Å) | 2.835014 | 2.993035 | 2.836939 |
| Shortest intermolecular O–H (Å) | 1.877098 | 2.064951 | 1.880918 |
| Cosine between molecular H–O–H bisectors | -0.612160 | -0.440076 | -0.629072 |
| Fresh maximum full force (eV/Å) | 0.00775126 | not a true-quench certificate | 0.00606377 |
| Fresh energy (eV) | -276.168540871316 | — | -276.168457692994 |

Intramolecular O–H lengths stayed .958–.969 Å; the final structure remains a compact dimer. The final energy difference is +0.0000831783 eV and MC accepted it. Fresh energies agree within 1.2e-13 eV. Relative pose changes during climbing, while each frozen body's internal distances stay fixed, and final unrestricted relaxation can change those internal distances.

This establishes a real molecular, bound-complex pipeline feasibility result under GFN2-xTB. It does **not** establish a new distinct minimum: water permutations/donor–acceptor exchange and small relaxation residuals could explain the apparent endpoint difference. No Hessian, strict structural identity, repeated-seed success or Cartesian matched-cost comparison has been performed. Both the negative manual-input result and the S22 result remain visible; the second is a source-provenance correction, not an unreported search-parameter retuning.
