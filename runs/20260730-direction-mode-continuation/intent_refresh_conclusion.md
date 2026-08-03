# Two-vector intent-refresh paired screen

- Decision: `reject_continuation_intent_ritz2`
- Execution commit: `b193539b87c667a1f6d7a20d592f006a0c71af26`
- Matrix: C60 2 locked states × 10 seeds × 3 arms; PdO 1 strict-bootstrap state × 10 seeds × 3 arms.
- Conclusion: adding the original random/bond intent to the transported mode does not supply useful local directional renewal. The direction stage is closed without a depth or mixing-weight sweep.

## Action-level results

The shared initial Ritz cost is excluded from each arm and reported below the table.

| system | arm | cases/cert. | median landing ΔE (eV) | new basin | action FE | direction FE | proposal FE | quench FE | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60 | transported | 20/20 | +2.581741 | 19 | 7,146 | 258 | 5,144 | 1,520 | 139.224 |
| C60 | intent-refresh Ritz(2) | 20/20 | +8.735001 | 19 | 7,300 | 488 | 5,398 | 1,196 | 141.475 |
| C60 | fresh Ritz | 20/20 | +0.000061 | 10 | 7,669 | 2,472 | 4,314 | 687 | 148.824 |
| PdO | transported | 10/10 | -1.398895 | 10 | 4,254 | 56 | 2,670 | 1,450 | 105.757 |
| PdO | intent-refresh Ritz(2) | 10/10 | -1.748871 | 10 | 4,310 | 112 | 2,580 | 1,540 | 108.194 |
| PdO | fresh Ritz | 10/10 | -2.014709 | 10 | 5,973 | 1,368 | 2,952 | 1,547 | 148.712 |

Shared costs: C60 step-zero Ritz 480 FE, giving 22,595 FE total; PdO bootstrap 85 FE plus step-zero Ritz 240 FE, giving 14,862 FE total.

## Paired landing comparison

Lower landing ΔE is better.

| system | comparison | wins / ties / losses | median paired ΔΔE (eV) |
|---|---|---:|---:|
| C60 | intent-refresh vs transported | 8 / 1 / 11 | +0.000061 |
| C60 | intent-refresh vs fresh Ritz | 5 / 0 / 15 | +4.787338 |
| PdO | intent-refresh vs transported | 5 / 0 / 5 | +0.077057 |
| PdO | intent-refresh vs fresh Ritz | 6 / 0 / 4 | -0.735626 |

## Mechanistic attribution

The experiment used the same two-HVP cost as residual–Ritz, but replaced the Hessian residual with the original SSW random/bond-mixed intent.

| system | observations | lower Ritz curvature | upper Ritz curvature | median gap | upper-mode overlap with previous |
|---|---:|---:|---:|---:|---:|
| C60 | 122 | 7.1377 | 40.6526 | 32.9165 | 0.00844 |
| PdO | 28 | 4.0714 | 8.8739 | 4.9376 | 0.02900 |

The selected direction still had median consecutive-direction cosine 0.99994 on C60 and 0.99958 on PdO. The independent intent was nearly orthogonal to the previous mode but much stiffer on the later biased PES, so the 2×2 lowest-Ritz solution correctly collapsed back onto transport.

This is not evidence for adding a mixing weight: forcing a high-curvature intent into the selected direction would override the local quadratic model rather than improve it. Together with the pure-transport, residual-trigger, residual–Ritz, and 12-HVP continuation results, this closes the low-cost continuation line. Fresh local Ritz remains the direction control; the next isolated mechanism should be the uphill propagator.

Claim ceiling: paired action-level survivor screen on two C60 states and one PdO state; no cross-cohort equality, long-horizon, significance, or universal PES claim.
