Current phase: review P0 fixes implemented.

Implemented:
- P0-1: Trust-region prediction now uses true-PES curvature for the selected direction, while direction scoring still uses biased-PES curvature.
- P0-2: Added `bias_weight_max` to `SSWConfig` and clipped Gaussian bias weights.
- P0-3: Added `GeometryValidator` for NaN/Inf and non-periodic overlap checks at key walk/relax points.
- P0-4: Added configurable `hvp_epsilon` to `SSWConfig` and `SoftModeOracle`.
- Prior review items: public API docstrings, configurable bandit weights, fixed candidate-budget bond directions, proposal pool size, bond-pair diagnostics, and removal of CELL dead candidate.

Verification:
- `pytest -q`: 82 passed in 4.15 s.

Notes:
- The proposal pool remains SSW-only. It supports multiple independent SSW walk candidates when `proposal_pool_size > 1`; no reseed, recombination, or GA/BH-style operation was introduced.
- PBC overlap validation is intentionally not implemented yet because it requires MIC-aware geometry to avoid false positives.
