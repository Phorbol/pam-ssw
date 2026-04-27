# Summary

M1 doc-aligned core milestone is verified.

Implemented scope:

- `pamssw.coordinates`: minimal Cartesian `GeneralizedCoordinates`-style layer through `CartesianCoordinates` and `TangentVector`.
- `pamssw.metric`: documented metric names and Euclidean/mass-weighted implementations, with atom-cell-block explicitly unavailable until variable-cell coordinates exist.
- `pamssw.calculators`: `EnergyResult` contract with optional stress metadata.
- Core `walker`: smooth SSW path proposal only; archive/value controls remain outer-loop seed selection and reward statistics.
- Removed unsupported public controls: `cluster_reseed_interval`, `proposal_pool_size`, and exposed acquisition weights.
- Archive degeneracy diagnostic now uses descriptor bins and feeds into effective acquisition weights.
- M2 curvature-inverting bias cleanup: Gaussian bias weight now follows the documented curvature inversion rule, and the soft-mode oracle no longer mixes in an undocumented random direction after candidate scoring.
- M3 direction acquisition skeleton: SSW inner walk now uses a documented candidate-direction framework. Enabled candidates are previous/soft direction, random Cartesian directions, and bond directions only when `local_softening_pairs` are configured. Cell candidates remain unavailable until variable-cell coordinates exist.
- M4 direction novelty: candidate direction scoring now includes descriptor novelty gain evaluated by probing the candidate displacement against the archive. This remains score-only and does not alter the proposal PES.
- M5 trust-region feedback: Gaussian curvature inversion now supplies only the initial local bias seed. After each modified-PES proposal relax, the walker evaluates the true PES energy change, compares it with the local quadratic prediction, and updates the next-step trust radius and bias scale. The controller is system-agnostic and uses no LJ-, cluster-, reseed-, or recombination-specific operations.
- M5 diagnostics: `SearchResult.stats` now reports trust-region step count, mean local-model error, shrink/expand counts, and damage events.
- M6 task modes: proposal ranking now uses documented `SearchMode` presets and lexicographic keys instead of exposing raw proposal reward weights. Global-minimum mode ranks best-energy improvement before secondary discovery and coverage terms; reaction-network mode ranks validated edge discovery first; crystal-search mode is represented as a preset for low-enthalpy/diversity search. `SSWConfig` exposes only `search_mode`, not individual reward weights.
- M7 bounded prototypes: archive density/novelty now uses a fixed-capacity weighted prototype set. Full minima entries are still retained for graph nodes and duplicate detection, but occupancy scoring no longer requires a full KDE over every archived minimum. `SSWConfig.max_prototypes` is the single public archive-capacity knob, and prototype diagnostics are reported in `SearchResult.stats`.
- M8 adaptive step target: SSW walking now derives the target energy climb from archive energy scale and escape/damage feedback, with `target_uphill_energy` kept as the compatibility fallback and initial scale. Early damage feedback is warmed up before shrinking the multiplier so sparse noisy events do not shut down exploration.
- M9 force/energy accounting: all true-PES calculator calls are routed through an `EvalCounter`, including proposal PES calls, true quench calls, curvature probes, and diagnostic energy checks. `SSWConfig.max_force_evals` can stop new work when the budget is exhausted, and the final stats report force/energy counts and budget status.
- M10 observable frontier policy: archive nodes now receive frontier/dead status from directly observed statistics: visits, low-energy window, descriptor sparsity, success rate, and duplicate-heavy failed trials. Bandit selection uses this frontier score and penalizes dead nodes, with frontier diagnostics reported in `SearchResult.stats`.

Explicit support claim after M1:

- Supported: Cartesian fixed-cell molecules/clusters with fixed atom masks.
- Representable but not claimed as full algorithm support: slab/bulk `State` metadata.
- Not yet supported: variable-cell walking, fractional/cell generalized coordinates, atom-cell-block metric, validated periodic/cell-reduced duplicate matching.

Verification:

- `pytest -q tests/unit tests/integration`
- Output: `runs/20260427-151600-epam-generalization/logs/pytest_m10_frontier_policy.out`
- Result: `54 passed`
