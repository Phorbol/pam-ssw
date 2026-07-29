# Block-Krylov direction ablation evidence

All 12 preregistered C60 cases completed at exactly 6000 force evaluations.
No block-Krylov arm passed the preregistered survivor gate, so PdO was not run.
The production default remains `discrete`.

| arm | best energy at seeds 42 / 43 / 44 (eV) | seeds better than discrete | median paired delta (eV) | minimum archive coverage | mean direction FE |
| --- | --- | ---: | ---: | ---: | ---: |
| `discrete` | -494.8329 / -485.1450 / -493.8854 | baseline | baseline | 1.000 | 1664.0 |
| `variational_breadth` (6x1) | -493.3100 / -481.1663 / -493.1502 | 0/3 | +1.5229 | 1.083 | 1372.3 |
| `balanced_refinement` (2x3) | -484.7932 / -490.9145 / -485.2259 | 1/3 | +8.6595 | 0.750 | 1528.0 |
| `deep_refinement` (1x6) | -494.1933 / -493.6766 / -483.4366 | 1/3 | +0.6396 | 0.818 | 1850.7 |

Positive paired deltas are worse than `discrete`. The three-seed rule is a
survivor gate, not a significance claim.

## Mechanistic interpretation

The fixed-state audit showed that 1x6 refinement produced the lowest
curvature and Ritz residual on all six physical states. That local variational
advantage did not produce robust terminal-search improvement:

- 6x1 breadth reduced direction cost and increased archive coverage, but all
  three final energies were worse than `discrete`.
- 1x6 refinement spent more evaluations on direction selection and retained
  fewer minima on average. A softer, better-converged local mode was therefore
  not sufficient to make the present Gaussian-bias uphill walk more
  productive.
- The evidence rejects promotion of any block-Krylov allocation. It does not
  show that block Krylov is intrinsically inferior; it shows that lowest local
  curvature is not an adequate standalone objective under the present
  direction/uphill coupling.

## Reproducibility ceiling

An unscored repeat of `discrete`, seed 43, at the same commit, input, model,
configuration, GPU, and 6000-FE budget changed the final best energy from
-485.1450 to -490.7659 eV. The archive size and completed trial count remained
11 and 10, while the purpose-level FE allocation diverged after the
trajectories separated.

This same-seed 5.6209 eV spread shows runtime-level path sensitivity large
enough to affect individual paired outcomes. It does not change the
preregistered decision (no arm survives), but it limits any ranking claim
between the failed block arms.

## Next clean experiment

Do not add another selector or a deeper eigensolver yet. The next direction
experiment should freeze starter states and the uphill policy, then apply one
complete escape plus true quench to each equal-HVP direction arm. That
measures productive escape and landing-basin quality directly, separating
direction quality from long-run starter selection and chaotic trajectory
branching.
