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

Explicit support claim after M1:

- Supported: Cartesian fixed-cell molecules/clusters with fixed atom masks.
- Representable but not claimed as full algorithm support: slab/bulk `State` metadata.
- Not yet supported: variable-cell walking, fractional/cell generalized coordinates, atom-cell-block metric, validated periodic/cell-reduced duplicate matching.

Verification:

- `pytest -q tests/unit tests/integration`
- Output: `runs/20260427-151600-epam-generalization/logs/pytest_m2_curvature_bias.out`
- Result: `31 passed`
