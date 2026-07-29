# Raw-PdO direction-continuation transfer

- Decision: `transported_direction_supported`
- Bootstrap: 0.630371 eV drop, 85 force evaluations, 4.376 s
- Shared initial Ritz: 72 force evaluations

| arm | meaningful | median landing ΔE (eV) | direction FE | total FE | generation s | quench s |
|---|---:|---:|---:|---:|---:|---:|
| fixed_intent_ritz | 3 | -1.104309 | 336 | 1484 | 24.083 | 9.361 |
| transported_direction | 3 | -1.580627 | 14 | 1259 | 17.100 | 11.118 |

Claim ceiling: one raw-input PdO slab, three paired seeds, one macro action per seed and arm; this tests transfer of the direction-continuation mechanism, not long-run PdO search superiority.
