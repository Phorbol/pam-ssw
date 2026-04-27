# Status

Current phase: M14 geometry-risk scoring discarded after unchanged quick gate

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
- LJ13/LJ38 seeds 0/1 budget60 quick gate completed with no stderr and valid JSON output. SSW did not beat BH/GA; observed failures are high duplicate/dead-node behavior for LJ13 and LJ38 seed1, plus high-energy over-exploration for LJ38 seed0.
- Added mode-aware node acquisition policy. Global-minimum mode now weights low energy more strongly than reaction-network mode, while reaction-network mode weights frontier value more strongly.
- Baseline fallback selection now avoids dead nodes when live archive nodes exist.
- Quick gate rerun after M13 worsened SSW mean gaps, so this implementation is marked for discard and should be reverted while preserving the benchmark record.
- M13 implementation logic has been reverted; benchmark records remain in the ledger.
- Full tests after revert: `55 passed`.
- Tested a doc-backed cheap geometry `S_risk` candidate-direction scorer based on relative minimum-distance collapse and pair-distance stretch. It was score-only and did not enter the force loop, quench, or calculator.
- M14 quick gate was unchanged versus M12: SSW LJ13 mean gap `3.295909127321515`, SSW LJ38 mean gap `18.134550242531958`; BH/GA values were unchanged. Because there was no primary-metric improvement, the implementation was discarded while benchmark records were retained.
- Full tests after reverting M14: `55 passed`.
- Added M15 direction acquisition diagnostics. `SearchResult.stats` now reports direction choices, candidate evaluations, mean candidate-pool size, and selected soft/random/bond/cell counts. This is observability only and does not alter candidate generation, scoring, PES force evaluation, or relaxation.
- Full tests after M15 diagnostics: `56 passed`.
