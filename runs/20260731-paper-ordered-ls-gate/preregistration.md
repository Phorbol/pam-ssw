# Paper-ordered LS mechanism gate

## Question

Does reconstructing the documented LS-SSW application order change the
direction and landing behavior that was absent from the current
active-neighbor, moving-reference implementation?

This is a mechanism gate, not a production benchmark.

## Fixed cohort

- system: C60 only, because the paper directly documents the C-C application;
- starters: bootstrap, middle, and late fixed minima from the seed-42
  20k-force-evaluation run;
- random seeds: 42, 43, and 44;
- arms:
  - `none`: no local-softening potential;
  - `current_active`: current active-neighbor, moving-reference implementation;
  - `paper_ordered`: whole C-C neighbor network, frozen at the macro-step
    starter, followed by one softened-PES pre-relaxation and reuse of the same
    penalty through climbing.

There are 27 fixed cases.

## Frozen components

Starter, RNG seed, direction portfolio, direction scoring, Gaussian bias
updater, proposal optimizer, true quench, matcher, and archive handling remain
unchanged.

## Paper-ordered constants

- exponential penalty;
- `A_CC = 0.03 * 3.61 eV = 0.1083 eV`;
- dimensionless `xi = 0.2`, giving decay length `xi * r0`;
- covalent C-C neighbor graph established once at the starter;
- no distance cutoff after graph construction;
- no `A_pq` self-adaptation in this first gate;
- softened-PES pre-relaxation uses the existing safe total-objective L-BFGS,
  `fmax = 0.05 eV/A`, and the existing true-quench iteration ceiling.

Self-adaptation is excluded so this experiment isolates the application
topology and execution order.

## Primary evidence

1. softened-PES pre-relaxation convergence, force evaluations, displacement,
   and real-PES response `P_LS`;
2. first selected direction identity, inner/true curvature, and participation
   ratio;
3. landing-energy delta, new-basin status, total force evaluations, and wall
   time.

## Interpretation boundary

- The gate may reject the present active-neighbor implementation.
- It may support a paper-ordered C60 mechanism.
- It cannot establish a system-general LS default.
- PdO is excluded until a non-system-specific rule for identifying the
  tight-bond network is justified.

No strength, neighbor cutoff, active count, optimizer, or threshold is changed
after results are observed.
