# Equal-budget multi-seed SSW survivor gate

Only `safe-lbfgs-total` advances from fixed-task replay. FIRE2 was not
consistently cheaper; bias-separated L-BFGS was already rejected.

- Systems: C60 and PdO
- Paired seeds: 42, 43, 44
- Backends: production ASE FIRE and safe total-gradient L-BFGS
- Policy: uniform
- ThreadPool workers: 2
- Action budget: 1000 force evaluations
- Campaign budget: 3000 force evaluations including bootstrap
- All direction, starter, bias, quench, archive, and geometry controls fixed
- Common input to every arm: the exact certified bootstrap minimum frozen by
  the one-bias replay. The runner still performs and charges its normal
  bootstrap confirmation quench.

Primary outputs remain separate:

- best-energy drop from the certified bootstrap minimum;
- unique minima;
- completed attempts;
- exact purpose-resolved evaluation counts;
- proposal-relaxation certificate/termination diagnostics.

No weighted score is used. Three seeds provide a survivor gate and variability
diagnostic, not a population-level significance claim.

The first attempted execution from the raw structure stopped fail-closed before
any proposal because the repeated float32 bootstrap missed its force
certificate. Repeating until success would introduce selection bias, so this
paired comparison instead freezes the already certified minimum for every arm.
