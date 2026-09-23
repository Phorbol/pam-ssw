# C4H6 / MH-1 connected-class curvature qualification

## Question and decision boundary

For each method/seed arm, choose one representative from every connected element-labeled graph class observed in the fixed 200,000 charged E/F request prefix. The representative is the lowest-energy saved minimum in that class, with minimum index as the tie-break. Compute MH-1 local curvature on those saved geometries to test how many observed connected classes have a representative with strictly positive or negative projected internal spectrum at that geometry.

This qualifies local curvature only. A nonzero fresh force means the geometry is not established as stationary; positive curvature there cannot be called a minimum, and negative curvature cannot be called a first-order saddle or proof that no nearby minimum exists. Graph identity does not validate bond order, radical character, electronic state, or reaction chemistry. Do not alter or relabel the existing graph observations.

## Frozen source and selection

The only selection input is `../c4h6-mh1-coverage-20260924/analysis.json`, whose common prefix is 200,000 charged E/F requests and whose class mapping is global across six arms. Include only rows with `cumulative_requests <= 200000`, `graph.component_count == 1`, and an integer `graph_class_id`. Select one row per `(arm, seed, graph_class_id)` by `(energy_eV, minimum index)`. Keep repeat geometries across classes or arms if they occur; do not deduplicate.

The committed `manifest.json` records every selected arm, seed, minimum index, global class ID, energy, cumulative request cost, and source `result.json` path/hash, along with the analysis and plan hashes. Expected class counts are SSW 5/5, paper LS 6/8, and native LS 10/8 for seeds 61/67 respectively: 42 frames total. A mismatch stops preparation or execution instead of changing the rule.

## Calculation and budget

Use the exact model path/hash and `omol` head from the completed lifecycle pilot, `float64`, CPU, standard eV/Å units, and the existing MACE ASE calculator. The independent runner creates one shared calculator, resets its ASE cache for every manifest frame, then requests one fresh energy/force pair and one analytic Hessian. It reuses `rigid_basis` and `hessian_matrix` from the completed three-frame curvature runner by importing that file; the helper's hash is recorded in the manifest. Keep original input coordinates unchanged. Save each input and frame result as soon as possible, save the raw Hessian immediately after it is returned, and retain exceptions without fallback. No quench, finite-difference Hessian, parameter change, or automatic expansion is allowed.

For each C4H6 frame, record fresh energy and maximum per-atom force norm, energy/Fmax deltas from the source minimum, separate E/F and Hessian CPU times, Hessian shape/finiteness/antisymmetry, all 24 internal eigenvalues after removal of six external rigid modes, and the lowest internal eigenvector; preserve the raw Hessian. Fresh numerical qualification uses the existing coverage criteria only: `fmax <= 0.03 eV/Å` and `abs(energy_error_from_source) <= 1e-6 eV` (see the coverage analysis protocol). These are not curvature thresholds. Count a spectrum as a positive-, negative-, or exact-zero-curvature representative only when fresh numerical qualification passes; report unqualified fresh evaluations separately, even if their Hessian completed. Count any strictly negative internal eigenvalue as negative curvature and a fully strictly positive internal spectrum as positive curvature; an exact zero eigenvalue is reported separately as sign-ambiguous. There is no numerical cutoff or forced classification near zero. Nonzero Fmax bars a stationary-point claim regardless of spectrum.

The total action cap is exactly 42 fresh E/F pairs plus 42 analytic Hessian calls. Every manifest frame gets a result record even when one stage fails, completed and failed outcomes remain separate, and any failure gives the overall process a nonzero exit code. The final summary reports per arm the number of numerically qualified positive-spectrum representatives, negative-curvature representatives, exact-zero ambiguous representatives, fresh-numerically-unqualified evaluations, and incomplete classes separately; an unqualified spectrum does not count toward curvature classes. It does not relabel class coverage as physical minima coverage.

## Resource and stop policy

The prepared Slurm script requests `CPU-MISC`, `rush-cpu`, account `sjtu-caoxiaoming`, one node/task, and 20 minutes. It leaves CPU/memory binding to the system and fixes OMP/MKL/OpenBLAS/torch threads to one. This is a preparation artifact only: do not submit it until the main agent has reviewed the plan, manifest, and runner. Do not modify core code, coverage inputs/analysis, existing graph labels, or the prior three-frame curvature evidence.
