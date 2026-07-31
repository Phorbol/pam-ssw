# CuO independent-bootstrap diagnostic

The first CuO seed-42 selector attempt is excluded from selector comparison.
Its output is retained under
`seed42-cuo-independent-bootstrap-invalid-output/`.

The runner independently true-quenched the raw ARC input once for every
starter mode.  That assumption was harmless to numerical precision in the
earlier C60/PdO seed-42 smoke, but it failed on CuO:

| Mode | Bootstrap FE | Bootstrap energy (eV) | Final energy (eV) |
|---|---:|---:|---:|
| `uniform_archive` | 100 | -198.676666 | -201.871643 |
| `archive_ucb` | 225 | -201.044769 | -201.962112 |

The bootstrap minima differ by 2.368103 eV before the selector acts.  The
Metropolis case was interrupted after its first action once this confound was
confirmed.  None of these rows is evidence for or against a starter policy.

The corrected protocol performs one true-PES bootstrap quench per
system/seed, reuses the exact minimum coordinates and energy in all three
arms, and charges the same bootstrap force-evaluation count against every
arm's total campaign budget.

This diagnostic also established the actual slab constraint: the historical
`z <= quantile(z, 0.35)` rule fixes two degenerate Cu layers, or 24 of 54
atoms, rather than exactly 35% of the atoms.
