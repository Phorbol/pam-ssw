# PdO matcher/numerical ambiguity result

## Decision

- Classification: **residual_energy_only_local_same**.
- Offline relabel as escaped allowed: **False**.
- Strict re-quench gate required: **True**.
- Production matcher change allowed: **False**.
- New force evaluations: **0**.

## Pair decomposition

| Starter | Seed | Arm | H | ΔE (eV) | All MIC RMSD (Å) | Movable MIC RMSD (Å) | Descriptor Δ | Mechanism |
|---|---:|---|---:|---:|---:|---:|---:|---|
| intermediate_accepted | 42 | K4_discrete | 1 | 0.417908 | 0.746059 | 0.923829 | 0.023017 | descriptor_collision_geometry_split |
| intermediate_accepted | 43 | D0_exact_anchor | 1 | -0.441284 | 0.733390 | 0.908141 | 0.027426 | descriptor_collision_geometry_split |
| intermediate_accepted | 43 | K4_discrete | 1 | 0.279968 | 0.231033 | 0.286083 | 0.073602 | energy_only_archive_split |
| intermediate_accepted | 43 | K4_discrete | 2 | 2.827942 | 0.879520 | 1.089091 | 0.038112 | descriptor_collision_geometry_split |
| intermediate_accepted | 44 | D0_exact_anchor | 1 | 0.734619 | 0.779387 | 0.965098 | 0.044063 | descriptor_collision_geometry_split |

The gate uses the effective 0.001 eV energy tolerance, 0.4 Å indexed MIC RMSD tolerance, and 0.1 descriptor threshold. The movable-region subgate reuses exactly the same RMSD tolerance. No threshold was fitted to these five outcomes.
