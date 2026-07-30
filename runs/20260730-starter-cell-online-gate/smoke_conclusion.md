# Starter-cell online mechanism smoke

## Scope

This is a committed-code, seed-42 mechanism smoke.  It compares
`uniform` with `fps_cell_uniform` on C60 and PdO under the frozen,
unsoftened posterior-harness action kernel.  Each campaign has a 6,000
force-evaluation budget.  The 12-trial resolution reference and three desired
observations per cell give `max_cells=4`.

This is not the production-kernel selector decision:

- C60 uses K12 rather than the production K4 control;
- local softening is disabled;
- PdO uses strict ASE-LBFGS to FIRE quenching at `fmax=0.01`, rather than the
  evidence-backed search protocol using SciPy L-BFGS-B at `fmax=0.03`;
- there is one seed and only 9--18 actions per arm.

## Mechanism checks

All four campaigns are benchmark-eligible:

- every terminal action is posterior-observed;
- every force-evaluation ledger closes;
- `unattributed=0`;
- no worker action failed;
- every event-log action probability equals its immutable policy snapshot;
- every cell snapshot retains every archive starter with positive support;
- every cell marginal equals
  `1 / nonempty_cells / members_in_selected_cell`.

The last C60 partition has cell sizes `[2, 2, 2, 3]`; the last PdO partition
has `[3, 2, 2, 7]`.  The mechanism therefore progressed beyond node-uniform
selection instead of remaining in the `archive_size <= max_cells` limit.

MACE descriptors were cached once per starter visible to a policy snapshot:

| system | descriptor forwards | descriptor wall time |
|---|---:|---:|
| C60 | 9 | 0.201 s |
| PdO | 14 | 0.321 s |

This is below 0.3% of the corresponding cell-campaign wall time.  Descriptor
cost remains separately reported and is not silently counted as a PES force
evaluation.

## Raw smoke outcomes

| system | policy | actions | total FE | archive entries | best-energy drop | wall time |
|---|---|---:|---:|---:|---:|---:|
| C60 | uniform | 9 | 5,425 | 10 | 15.919 eV | 126.43 s |
| C60 | FPS-cell uniform | 9 | 5,260 | 10 | 14.277 eV | 96.65 s |
| PdO | uniform | 15 | 5,125 | 15 | 3.59045 eV | 121.29 s |
| PdO | FPS-cell uniform | 18 | 5,387 | 15 | 3.59058 eV | 128.24 s |

These values must not be interpreted as a policy ranking.  The arms follow
different trajectories, the action counts differ under the same FE budget,
and the cohort is one short seed.  In particular, the lower C60 cell-arm FE is
not optimizer acceleration, and the near-identical PdO energy is not evidence
of equivalence.

## Decision

The implementation gate passes: full support, exact propensities, MACE caching,
cell activation, event replay, and FE closure all work.

The algorithm gate remains open.  The next experiment must freeze the
system-specific production kernels:

- C60: K4 LS-SSW, safe-LBFGS proposal, ASE-LBFGS to FIRE true quench at 0.01;
- PdO: K8 LS-SSW, safe-LBFGS proposal, SciPy L-BFGS-B true quench at 0.03.

The first production-kernel comparison remains only `uniform` versus
`fps_cell_uniform`, using paired seeds and equal force budgets.  Legacy
UCB-like and Metropolis are added only if state abstraction survives this
gate; TS is still out of scope.
