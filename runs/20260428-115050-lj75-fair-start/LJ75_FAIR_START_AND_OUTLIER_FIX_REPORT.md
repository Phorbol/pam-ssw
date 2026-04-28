# LJ75 Fair-Start And Outlier Diagnostic Report

## Scope

This run addresses two issues found during the larger LJ-cluster comparison:

1. The energy-decrease curves were not fair at the first point because ASE-GA was initialized through its own `StartGenerator`, while SSW/BH used the repo random cluster seed.
2. The SSW LJ75 minima export contained structures with one atom far outside the cluster.

The goal here was not to tune SSW to the LJ75 global minimum, but to make the benchmark accounting fair and remove an unphysical proposal-path artifact without introducing an LJ-specific search mechanism.

## Root Cause

The large-displacement SSW LJ75 artifact came from the proposal step, not from the final true-PES quench itself.

- SSW generates a trial by walking on a locally biased PES, then quenches on the true PES.
- The proposal L-BFGS-B step previously had no coordinate trust bound.
- On a finite LJ cluster, a locally biased move can push a single atom outside the interaction range.
- Once the atom is far from the cluster, the LJ force is very small or effectively absent under the finite cutoff used by the benchmark calculator, so the true-PES quench can leave a disconnected atom outside the cluster.
- The archive previously accepted those relaxed structures because deduplication and descriptors did not enforce finite-cluster connectivity.

This is a generic SSW stability problem: biased local PES relaxation needs a finite proposal trust region. The finite-cluster fragment guard is only a benchmark validity filter and is disabled by default for the public SSW API.

## Code Changes

Changed files:

- `pamssw/config.py`
  - Added `proposal_trust_radius`, `walk_trust_radius`, and optional `fragment_guard_factor`.
  - `fragment_guard_factor=None` by default, so the core SSW algorithm is not specialized to LJ clusters.
- `pamssw/relax.py`
  - Added optional coordinate bounds to `Relaxer.relax(...)` through L-BFGS-B bounds.
- `pamssw/walker.py`
  - Proposal relaxations now use the configured coordinate trust radius.
  - A cumulative per-atom walk displacement clip limits runaway atom motion relative to the selected seed node.
  - Optional non-periodic finite-cluster fragment guard rejects proposals whose nearest-neighbor geometry indicates a disconnected fragment.
  - Rejected proposals no longer terminate the whole search early; the selected node records a failed/duplicate-style trial and the outer loop continues.
  - Added `walk_displacement_clips` and `fragment_rejections` diagnostics.
- `benchmarks/lj_cluster_compare.py`
  - Added fair-start step-0 traces for SSW, ASE-BH, ASE-GA, and internal GA/BH traces.
  - Forced the first ASE-GA initial population member to match the shared random initial cluster.
  - Saved/restored Python and NumPy global random states around ASE-GA and seeded them inside the baseline call, because ASE-GA internals can otherwise make repeated runs drift even when local `rng` objects are passed.
  - Enabled `fragment_guard_factor=3.2` only in the LJ benchmark harness.
  - Added SSW outlier diagnostics to benchmark summaries.
- Tests:
  - `tests/unit/test_relax.py`
  - `tests/unit/test_walker_policy.py`
  - `tests/integration/test_lj_benchmark_reporting.py`

## Fair-Start Curve Correction

Before this fix, the plotted first point for ASE-GA was not the same physical start as SSW/BH. This made the curve visually misleading.

After the correction, LJ13 seed0 starts identically:

- SSW: step 0, best_energy -40.24277387226225, gap 4.084027127737755.
- ASE-BH: step 0, best_energy -40.24277387226225, gap 4.084027127737755.
- ASE-GA: step 0, best_energy -40.24277387226225, gap 4.084027127737755.

Generated curve:

- `runs/20260428-115050-lj75-fair-start/lj13_38_fair_start_budget20_curves.png`

Budget-20 fair-start metric:

- LJ13:
  - SSW mean gap: 0.42739603788184155
  - ASE-GA mean gap: 2.235593348672129
  - ASE-BH mean gap: 2.7775454355452425
- LJ38:
  - SSW mean gap: 6.336094472144836
  - ASE-GA mean gap: 9.254376027843975
  - ASE-BH mean gap: 12.26957998943297

## LJ75 Before And After

Before the trust/fragment controls, SSW found:

- Best energy: -364.98103022579147
- Gap to CCD LJ75: 32.51130077420851
- Best exported SSW frame max nearest-neighbor distance: 41.64876344497262
- Worst exported SSW frame max nearest-neighbor distance: 61.46640497712593

That confirms a severe isolated-atom artifact.

After the trust/fragment controls, LJ75 seed0 budget60 gives:

- SSW best_energy: -381.49794547522214
- SSW gap: 15.99438552477784
- ASE-GA best_energy: -377.568114808174
- ASE-GA gap: 19.924216191825963
- ASE-BH best_energy: -373.387373333605
- ASE-BH gap: 24.104957666394967

Final exported-minima geometry:

- SSW best frame:
  - energy -381.4979454752221
  - max nearest-neighbor distance 1.0958366184443724
  - median nearest-neighbor distance 1.0841292044094333
- SSW worst frame:
  - energy -277.9178483656908
  - max nearest-neighbor distance 3.291504331690055
  - median nearest-neighbor distance 1.0910137129054394

The original tens-of-sigma isolated atom is removed. The worst SSW frame is still the rough initial relaxed structure, not the final best basin.

SSW diagnostics in the final benchmark JSON:

- `walk_displacement_clips`: 4
- `fragment_rejections`: 1
- `proposal_relax_unconverged`: 410
- `true_quench_unconverged`: 22

Artifacts:

- `runs/20260428-115050-lj75-fair-start/lj75_seed0_budget60_ase_metric_final.json`
- `runs/20260428-115050-lj75-fair-start/lj75_final_geometry_check.json`
- `runs/20260428-115050-lj75-fair-start/minima_xyz_lj75_final/`

## Relaxation Diagnostics

The LJ75 depth sweep shows that proposal relaxation is still under-resolved:

- steps_per_walk=8, proposal_relax_steps=80:
  - gap 32.51130077420851 before the guard
  - proposal_relax_unconverged 472
  - true_quench_unconverged 28
- steps_per_walk=8, proposal_relax_steps=160:
  - gap 21.66192625632948
  - proposal_relax_unconverged 307
  - true_quench_unconverged 17
- steps_per_walk=14, proposal_relax_steps=160:
  - gap 19.368436862647002
  - proposal_relax_unconverged 550
  - true_quench_unconverged 27

This supports the computational-chemistry concern that LJ75 needs more careful SSW path and relaxation settings than the quick LJ13/LJ38 budget.

## Generality Check

The patch avoids using GA/BH-style reseeding or recombination as a primary mechanism.

- The proposal trust radius is a generic numerical stability control for biased local PES relaxation.
- The walk displacement clip is a generic trust-region limiter and respects fixed atoms.
- The fragment guard is disabled by default and skipped for PBC. It is only enabled in the LJ benchmark harness because finite disconnected clusters are invalid minima for this benchmark.
- No LJ motif, LJ size, magic number, decahedral/icosahedral template, or hard-coded literature structure is introduced.

For bulk/slab systems, the current SSW implementation still needs MIC-aware geometry/search descriptors before claiming production periodic correctness. Calculator-level PBC is not enough; search-level geometry must handle periodic distances and surfaces consistently.

## Conclusion

The current SSW implementation is more stable on LJ75 than before and beats the current short-budget ASE-BH/ASE-GA baselines for LJ75 seed0 budget60 in this run. However, it is not yet a literature-grade LJ global optimizer and should not be described as a complete production SSW implementation.

Next steps before formal LJ75 or larger-cluster claims:

1. Tune SSW proposal relaxation convergence and walk length on LJ55/LJ75 with fixed fair-start accounting.
2. Add trace plots and minima exports for LJ75 and at least one hard non-icosahedral case such as LJ38 under repeated seeds.
3. Keep benchmarking against ASE-BH/ASE-GA and Cambridge Cluster Database energies, but report short-budget results separately from literature-scale global-search claims.
4. Implement MIC-aware descriptor/distance support before applying the same benchmark protocol to bulk or slab systems.
