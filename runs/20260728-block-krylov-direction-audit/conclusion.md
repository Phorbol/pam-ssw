# Stage-1 block-Krylov direction audit

The CUDA MACE audit completed at commit
`839c5bc72dd2c04550bbe228e7638607d0dfdc12` on three preregistered C60
states and three preregistered fixed-mask PdO states.  Every non-degenerate
allocation consumed exactly 12 central-FD HVPs and 24 force evaluations; all
purpose ledgers closed with zero unattributed evaluations.

Across all six physical states, increasing Krylov depth reduced both the
selected Rayleigh quotient and the Ritz residual:

| allocation | mean curvature | mean residual | mean initial-span overlap | mean participation ratio |
|---|---:|---:|---:|---:|
| 6x1 breadth | 11.2855 | 11.3200 | 1.0000 | 2.4162 |
| 3x2 shallow | 7.2336 | 7.3470 | 0.8914 | 2.7164 |
| 2x3 balanced | 5.0381 | 6.2798 | 0.7660 | 4.9095 |
| 1x6 deep | 2.4436 | 2.7254 | 0.5003 | 20.7461 |

The ordering was the same for curvature in all six states, and 1x6 also had
the smallest residual in all six states.  Thus the current fixed-state
evidence says that insufficient refinement of each random/pair intent is a
stronger bottleneck than insufficient intent breadth.  Deeper refinement also
moves farther from the initial intent span and produces substantially more
collective modes, especially for C60.  Whether those modes improve basin
escape is not established by this audit and must be tested by the paired
fixed-total-force-budget search.

Claim ceiling: this result supports only direction algebra, fixed-state mode
quality, and exact force accounting.  It does not establish improved terminal
energy, basin discovery, or chemical pathways.
