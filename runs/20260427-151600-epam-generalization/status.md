# Status

Current phase: M3 direction acquisition skeleton verified

- Removed GA/BH-style reseed/recombination/motif mechanisms from core `pamssw`.
- Added Cartesian coordinate/tangent layer and metric module.
- Removed unsupported public reseed/pool/acquisition-weight config knobs.
- Added `EnergyResult` calculator contract while preserving tuple unpacking compatibility.
- Replaced undocumented Gaussian bias weight constant with `sigma^2 * max(curvature + target_negative_curvature, 0)`.
- Removed undocumented random mixing from soft-mode oracle candidate selection.
- Added documented direction candidate framework with soft, random, and local-softening bond candidates; cell candidates remain explicitly unimplemented because variable-cell coordinates are not available yet.
- Full tests: `34 passed`.
