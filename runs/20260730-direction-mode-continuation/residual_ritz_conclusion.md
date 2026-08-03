# Two-HVP residual–Ritz paired screen

- Decision: `reject_residual_ritz2`
- Execution commit: `80d1216b16ae28a6575d483e9365026c9271055d`
- Matrix: C60 2 locked states × 10 seeds × 3 arms; PdO 1 strict-bootstrap state × 10 seeds × 3 arms.
- Conclusion: a two-HVP Rayleigh–Ritz correction is a real cost reduction, but it does not recover transport's lost landing quality consistently. It must not replace fresh Ritz or advance to a 200-step campaign.

## Action-level results

The shared initial Ritz cost is excluded from each arm and reported below the table.

| system | arm | cases/cert. | median landing ΔE (eV) | new basin | action FE | direction FE | proposal FE | quench FE | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C60 | transported | 20/20 | +1.304230 | 18 | 7,792 | 254 | 5,387 | 1,929 | 149.246 |
| C60 | residual–Ritz(2) | 20/20 | +4.658936 | 20 | 7,006 | 556 | 5,116 | 1,095 | 132.631 |
| C60 | fresh Ritz | 20/20 | +0.000061 | 12 | 7,785 | 2,544 | 4,423 | 616 | 148.116 |
| PdO | transported | 10/10 | -1.743591 | 10 | 4,479 | 62 | 3,056 | 1,281 | 111.302 |
| PdO | residual–Ritz(2) | 10/10 | -1.336975 | 10 | 4,229 | 144 | 2,675 | 1,324 | 103.522 |
| PdO | fresh Ritz | 10/10 | -2.256104 | 10 | 5,516 | 1,200 | 2,739 | 1,479 | 136.037 |

Shared costs: C60 step-zero Ritz 480 FE, giving 23,063 FE total; PdO bootstrap 85 FE plus step-zero Ritz 240 FE, giving 14,549 FE total.

## Paired landing comparison

Lower landing ΔE is better.

| system | comparison | wins / ties / losses | median paired ΔΔE (eV) |
|---|---|---:|---:|
| C60 | residual–Ritz(2) vs transported | 10 / 1 / 9 | -0.322861 |
| C60 | residual–Ritz(2) vs fresh Ritz | 5 / 0 / 15 | +1.683914 |
| PdO | residual–Ritz(2) vs transported | 3 / 0 / 7 | +0.172760 |
| PdO | residual–Ritz(2) vs fresh Ritz | 4 / 0 / 6 | +0.653320 |

## Mechanistic attribution

The low-rank correction did what its local mathematics promised but not what the search required:

- Median consecutive-direction cosine remained 0.9943 on C60 and 0.9883 on PdO, compared with 0.99999/1.0 for pure transport and 0.9588/0.5607 for fresh Ritz.
- It reduced median true curvature from 6.5299 to 3.2966 on C60 and from 3.0900 to 1.3348 on PdO, close to fresh Ritz on C60 but not on PdO.
- Thus the two-dimensional solve mostly softened the transported vector without providing enough directional renewal. Lower local curvature alone did not predict a better quenched basin.

This rejects the specific “one residual vector is sufficient” hypothesis. It does not reject continuation subspaces in general, but there is no evidence-based reason to tune depth 3/4/5 sequentially. The next direction experiment, if pursued, should test one structurally distinct source of new span—history or fresh intent mixed into a small paid subspace—against the same transport/fresh controls and a fixed budget.

Claim ceiling: paired action-level survivor screen on two C60 states and one PdO state; no long-horizon, statistical-significance, or universal PES claim.
