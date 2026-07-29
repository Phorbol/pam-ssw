# GPU G1 frozen ablation

Purpose: compare proposal relaxation only, with direction generation, uphill
policy, starter policy, true quench, random seeds, and force budgets held fixed.

- Systems: C60 and fixed-bottom PdO slab.
- Proposal optimizers: ASE FIRE, ASE FIRE2, safe total-gradient L-BFGS, and
  analytic bias-separated L-BFGS.
- Policy: uniform. This avoids making the optimizer ablation depend on a
  UCB-like or Thompson-sampling policy.
- Seed: 42.
- Concurrency: batch size 2, ThreadPool workers 2.
- Budget: 1,000 force evaluations per action and 3,000 total per campaign,
  including bootstrap.
- Proposal trust radius: disabled for every arm.
- True quench: unchanged SciPy L-BFGS-B.
- Campaigns run serially so the eight arms do not contend with each other.
- No automatic retry after a campaign starts making physical evaluations.

This gate tests execution validity and optimizer behavior. One seed cannot
establish a statistically stable performance advantage.
