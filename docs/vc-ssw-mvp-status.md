# VC-SSW MVP Status

This branch implements the first variable-cell SSW/LS-SSW path behind
`coordinate_mode="variable_cell"`. The fixed-cell path remains the default and
is unchanged.

## Implemented

- Generalized coordinate `q = [active fractional atom coordinates, log-cell deformation]`.
- Cell DOF modes: `volume_only`, `shape_6`, `full_9`, and `slab_xy`.
- Stress-derived generalized gradients with mandatory finite-difference stress verification by default.
- Atom-only, bond, cell-only, coupled, soft-cell, soft-coupled, and momentum direction candidates in the same q-space.
- q-space Gaussian bias and LS-SSW local-softening gradient conversion.
- Variable-cell proposal relaxation and true quench through q-space L-BFGS-B.
- Variable-cell archive descriptors that include lattice information.

## Known MVP Limits

- No variable-cell proposal pool. VC currently generates one proposal per trial.
- No duplicate-rescue optimizer path for VC trials.
- No fragment-guard rejection path in the VC walker.
- No production validation yet on the design-doc integration cases: LJ crystal, TS.cif TTT, and PdO slab.

These are follow-on phases. They should be migrated from the fixed-cell walker
only after the three integration validation cases pass.
