# Block-Krylov Stage-2 preregistration amendment

This amendment was written before any Stage-2 terminal search energy was
generated or inspected.  It changes only the Stage-2 arm set and the
predeclared tie-break for PdO transfer.

## Evidence motivating the amendment

The preregistered Stage-1 fixed-state audit at commit
`839c5bc72dd2c04550bbe228e7638607d0dfdc12` used equal 12-HVP budgets on
three C60 and three PdO states.  The 1x6 deep-refinement allocation produced
the lowest curvature and residual on all six states.  Excluding it from the
terminal search would therefore leave the strongest Stage-1 mechanism
untested.

## Amended C60 arms

Stage 2 uses exactly these arms, in this stable order:

1. `discrete`
2. `variational_breadth` (`block_krylov_blocks=6`,
   `block_krylov_depth=1`)
3. `balanced_refinement` (`block_krylov_blocks=2`,
   `block_krylov_depth=3`)
4. `deep_refinement` (`block_krylov_blocks=1`,
   `block_krylov_depth=6`)

Seeds remain 42, 43, and 44.  The total force-evaluation budget remains 6000.
No selector, optimizer, softening, starter, proposal, quench, or acceptance
parameter changes between paired arms.

The existing survivor rule is unchanged: an arm survives only if it improves
the final best energy in at least two of three paired seeds, every paired
archive-coverage ratio is at least 0.8, and every budget ledger closes.

If multiple block arms survive C60, the single PdO transfer arm is chosen by:

1. lowest median paired final-energy delta versus `discrete`;
2. then largest paired best-energy-AUC improvement;
3. then stable order `deep_refinement`, `balanced_refinement`,
   `variational_breadth`.

This amendment does not change the claim ceiling: three seeds are a survivor
gate, not a significance claim.
