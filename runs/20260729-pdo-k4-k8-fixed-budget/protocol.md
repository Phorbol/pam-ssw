# PdO K4/K8 fixed-budget protocol

- System: `PdO.xyz`, periodic slab, bottom 35% fixed.
- Arms: `oracle_candidates=4` and `oracle_candidates=8`.
- Seeds: 42 and 43.
- Per-arm total budget: 20,000 force evaluations.
- Maximum macro steps: 200.
- Frozen shared policy: B8, 300-step safe-LBFGS proposal relaxation,
  `proposal_fmax=0.05`, SciPy L-BFGS-B true quench at `fmax=0.03`.
- Disabled adaptive direction blocks: direction-type UCB, probe, archive
  direction reuse, and plateau evolution.
- Primary comparison: completed macro steps, best-energy decrease, archive
  minima, duplicate rate, and purpose-level force allocation at fixed total
  force budget.
- Promotion rule: no cross-system K4 claim unless both seeds improve or match
  K8 in final energy decrease while completing at least as many macro steps.
- Wall time is descriptive because the four arms execute sequentially.
