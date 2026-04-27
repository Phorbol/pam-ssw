# Status

Current phase: M7 bounded archive prototypes verified

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
- Full tests: `45 passed`.
