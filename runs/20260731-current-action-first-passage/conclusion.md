# Current D0/K4 escape first-passage result

## Decision

- Discrete horizon gate contexts: **0**.
- Reproducible action-support gap contexts: **0**.
- Numerical/matcher gate systems: **pdo**.
- Production default changed: **False**.
- Force evaluations: 11454/15000.
- GPU kernel wall time: 238.9 s.

## Context readout

| System | Starter | Arm | H8 reached | Early escape→H8 return | All four return | Generation FE | Checkpoint FE |
|---|---|---|---:|---:|---:|---:|---:|
| c60 | intermediate_accepted | D0_exact_anchor | 0/3 | 0/3 | 0/3 | 704 | 512 |
| c60 | intermediate_accepted | K4_discrete | 3/3 | 0/3 | 1/3 | 1137 | 2774 |
| c60 | plateau_accepted | D0_exact_anchor | 1/3 | 0/3 | 0/3 | 790 | 642 |
| c60 | plateau_accepted | K4_discrete | 1/3 | 0/3 | 0/3 | 995 | 483 |
| pdo | intermediate_accepted | D0_exact_anchor | 0/3 | 0/3 | 0/3 | 776 | 326 |
| pdo | intermediate_accepted | K4_discrete | 0/3 | 0/3 | 0/3 | 656 | 263 |
| pdo | plateau_accepted | D0_exact_anchor | 0/3 | 0/3 | 0/3 | 620 | 289 |
| pdo | plateau_accepted | K4_discrete | 0/3 | 0/3 | 0/3 | 282 | 205 |

## Checkpoint labels

| Label | Count |
|---|---:|
| AMBIGUOUS_MATCH | 5 |
| ESCAPED_CERTIFIED | 26 |
| INVALID_GEOMETRY | 8 |
| RETURN_STARTER | 20 |

The gate classifies whether the frozen current actions leave and
return to their starter basins. It does not optimize a horizon,
alter Gaussian bias parameters, or train a selector/posterior.
