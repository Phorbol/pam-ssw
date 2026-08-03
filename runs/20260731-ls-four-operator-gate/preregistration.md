# G-LS0A: C60 four-operator mechanism audit

## Question

Does the paper-ordered C60 LS response arise from a fixed-geometry pair
operator, from the prestrained geometry's true PES Hessian, or from their
interaction?

This is a direction-mechanism gate.  Lower curvature alone is not a search
success and cannot promote LS to production.

## Frozen cohort

- system: non-periodic C60;
- fixed starters: bootstrap, mid, and late states from the seed-42 20,000-FE
  uniform run;
- direction RNG seeds: 42, 43, and 44;
- one native candidate pool generated at the original state for each
  `(state, seed)` block;
- paper pair graph: frozen whole C-C covalent-neighbor graph;
- initial pair strength: `0.1083 eV = 0.03 * 3.61 eV`;
- exponential decay length: `0.2 * reference pair distance`;
- no cutoff and no adaptive pair strength;
- no proposal-side LS.

There are exactly nine blocks.  Candidates are transported to the prestrained
geometry by one proper-rotation Kabsch alignment with fixed atom indices; no
permutation or structure matching is performed.

## Operators

For every frozen candidate direction:

```text
A = H_PES(x0)
B = H_PES(x0) + H_LS(x0)
C = T^T H_PES(xR) T
D = T^T [H_PES(xR) + H_LS(xR)] T
```

`xR` is obtained by the existing paper-ordered softened-PES pre-relaxation.
The A/B pair shares one true-PES central-FD HVP stencil.  The C/D pair shares a
second stencil.  LS Hessian actions are analytic and consume no MACE force
evaluations.

## Recorded evidence

- all four candidate curvatures and their ranking;
- selected candidate identity for A/B/C/D under the frozen production scorer;
- selected-direction angles after rigid-frame transport;
- radial and transverse LS curvature contributions;
- pair-space radial expressivity `||B u||^2 / ||u||^2`;
- pre-relax displacement, `P_LS`, certificate, and FE;
- exact per-purpose FE with `unattributed=0`;
- model, state, config, candidate-pool, and execution-commit hashes.

## Decision

- Open a complete action gate only if B, C, or D changes candidate ranking or
  selected identity in a repeatable direction across the nine blocks.
- A scalar curvature shift without an action change closes the corresponding
  mechanism.
- Low pair expressivity is an abstention certificate; it does not permit a
  strength increase or new top-k pair rule.
- No result from this C60 mechanism gate establishes a system-general default.

## Budget and stop conditions

- G-LS0A new true-PES ceiling: 500 FE after reuse of available prestrain
  artifacts;
- every HVP and pre-relax evaluation must have a declared purpose;
- stop on non-finite HVP, improper alignment, inconsistent candidate pool,
  incomplete nine-block cohort, any unattributed FE, or mismatched A/B and C/D
  true-HVP stencils;
- no posterior, selector, optimizer, Gaussian, pair graph, or strength changes
  are allowed after observing results.
