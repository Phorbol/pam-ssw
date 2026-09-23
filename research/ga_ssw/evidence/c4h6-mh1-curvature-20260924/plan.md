# C4H6 / MH-1 pilot curvature check

## Decision relevance

Pilot fresh checks establish a force threshold only. A negative projected internal eigenvalue on any saved twisted butadiene frame would limit interpreting force-qualified observations as stable distinct minima and could change later coverage denominators. At a nonstationary geometry, local negative curvature alone does not prove a first-order saddle or establish that no nearby minimum exists. This three-frame check does not rank the short pilot's methods and cannot certify the unexamined pilot structures.

## Fixed frames and provenance

Each input comes from the indexed line in that arm's completed `fresh-requests.jsonl`; its `atoms` are used unchanged. The matching index in `result.json` supplies the pilot minimum's original energy and maximum-force fields. Record both source paths, request/minimum indices, and their stored values in every result.

| Arm / minimum index | Saved CCCC torsion | Saved Fmax (eV/Å) | Purpose |
|---|---:|---:|---|
| `ssw` / 0 | 180.00° | 0.02529 | planar reference |
| `paper_ls` / 1 | 25.23° | 0.01423 | twisted conformer |
| `native_ls` / 3 | 103.59° | 0.02180 | intermediate-torsion conformer |

These are three specified frames, not a geometry-deduplicated sample. Do not add frames automatically based on the results.

## Calculation

Use the completed lifecycle pilot's exact MH-1 model and `omol` head on CPU, with `float64`, one fresh calculator instance per frame, `enable_cueq=False`, and `enable_oeq=False`. For each frame, request one fresh ASE energy/force pair, then one `MACECalculator.get_hessian(atoms)` call. Do not relax, alter coordinates, run xTB, or add a finite-difference fallback. Preserve any per-frame exception and continue with the remaining specified frames.

The requested environment is `/home/gengjianrui/.conda/envs/mace_env`; its MACE source has `get_hessian` at `mace/calculators/mace.py:770`. It calls `compute_hessian=True`; the installed implementation returns one model's raw array with shape `(3N, N, 3)` (`(30,10,3)` here). Reshape that known layout to `(30,30)` or fail explicitly on a different shape. The pilot uses ordinary eV and Å conversions (`energy_units_to_eV=1`, `length_units_to_A=1`), so the unconverted Hessian is eV/Å²; record and verify the instantiated factors. Save the raw returned Hessian separately.

Report new energy, new maximum per-atom force norm, differences from the indexed pilot result, separate wall times for fresh E/F and Hessian calls, calculator configuration, model path/hash, software versions, source identities, Hessian shape and finite-value status. Also report the unmodified Cartesian spectrum, maximum absolute and Frobenius antisymmetric residuals, and the symmetrized spectrum after projection away from the six rigid translations/rotations. Save the rigid and internal bases, projected matrix, all eigenvalues, and the lowest internal eigenvector in Cartesian coordinates. Count negative signs without imposing a numerical cutoff; report values and magnitudes, and flag numerical interpretation as uncertain near zero rather than claiming certification. No mass weighting or frequency conversion is part of this curvature check.

Expected cost is one paired fresh E/F calculation plus one analytic Hessian request per frame. The MACE implementation obtains the Hessian through derivatives of the 30 force components. Measure elapsed time rather than estimating it. The CPU Slurm file uses one node/task, `CPU-MISC`, `rush-cpu`, account `sjtu-caoxiaoming`, and a ten-minute wall limit; it leaves CPU and memory binding to the system and fixes OMP/MKL/OpenBLAS/torch threads to one. It is prepared for review only; do not submit it in this task.
