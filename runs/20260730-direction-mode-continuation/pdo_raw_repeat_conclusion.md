# Raw-PdO direction-continuation transfer

- Decision: `transported_direction_supported`
- Bootstrap: 0.630371 eV drop, 85 force evaluations, 3.819 s
- Shared initial Ritz: 72 force evaluations

| arm | meaningful | median landing ΔE (eV) | direction FE | total FE | generation s | quench s |
|---|---:|---:|---:|---:|---:|---:|
| fixed_intent_ritz | 3 | -1.343445 | 360 | 1527 | 24.635 | 9.749 |
| transported_direction | 3 | -3.250854 | 12 | 1023 | 14.172 | 8.877 |

Claim ceiling: one raw-input PdO slab, three paired seeds, one macro action per seed and arm; this tests transfer of the direction-continuation mechanism, not long-run PdO search superiority.
