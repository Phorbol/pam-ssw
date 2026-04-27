# Status

Current phase: M11 LJ13 smoke benchmark verified

- Removed GA/BH-style reseed/recombination/motif mechanisms from core `pamssw`.
- Added Cartesian coordinate/tangent layer and metric module.
- Removed unsupported public reseed/pool/acquisition-weight config knobs.
- Added `EnergyResult` calculator contract while preserving tuple unpacking compatibility.
- Replaced undocumented Gaussian bias weight constant with `sigma^2 * max(curvature + target_negative_curvature, 0)`.
- Removed undocumented random mixing from soft-mode oracle candidate selection.
- Added documented direction candidate framework with soft, random, and local-softening bond candidates; cell candidates remain explicitly unimplemented because variable-cell coordinates are not available yet.
- Added score-only descriptor novelty gain for direction candidate scoring. This uses archive descriptors only to rank candidate directions and does not add archive forces to the inner PES.
- Added trust-region feedback control for the local SSW bias. Curvature inversion now provides the initial Gaussian bias seed; true-PES energy change after the proposal relax updates the next-step trust radius and bias scale.
- Added trust-region diagnostics to `SearchResult.stats`: steps, mean model error, shrink/expand counts, and damage events.
- Added documented search modes and lexicographic proposal ranking. Global-minimum mode prioritizes best-energy improvement before near-low-energy minima, novelty, coverage, and duplicate avoidance; reaction-network mode prioritizes validated edge discovery. No raw proposal-weight knobs were added to `SSWConfig`.
- Added bounded archive prototypes for density/novelty occupancy. Full minima entries remain available for graph nodes and duplicate checks, while archive density uses fixed-capacity weighted prototypes controlled by `max_prototypes`.
- Added prototype diagnostics to `SearchResult.stats`: prototype count, maximum capacity, maximum prototype weight, and mean prototype weight.
- Added adaptive energy-scale step target control. The walking target now uses archive energy scale and escape/damage feedback, with `target_uphill_energy` retained as a compatibility fallback and initial scale.
- Added adaptive step diagnostics to `SearchResult.stats`: target, multiplier, escape rate, and damage rate.
- Added exact true-PES evaluation accounting through a calculator wrapper. Proposal PES calls, true quench calls, curvature probes, and diagnostic true-PES calls all increment force/energy evaluation counters when they invoke the underlying calculator.
- Added optional `max_force_evals` budget gating and budget diagnostics to `SearchResult.stats`.
- Added observable frontier and dead-node status for archive nodes. Frontier/dead labels now come from visits, low-energy window, descriptor sparsity, success rate, and duplicate-heavy failed trials rather than committor-style terminology.
- Bandit selection uses observable frontier score and strongly penalizes dead nodes.
- Added frontier diagnostics to `SearchResult.stats`: frontier node count, dead node count, and mean frontier score.
- Added SSW diagnostic fields to LJ benchmark summaries: force evaluations, budget exhaustion, number of minima, duplicate rate, frontier nodes, and dead nodes.
- LJ13 seed0 budget5 smoke benchmark completed with no stderr and valid JSON output. Smoke result is a runtime/accounting check only, not a performance conclusion.
- Full tests: `55 passed`.
