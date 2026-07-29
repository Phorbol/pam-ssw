# U2 uphill feedback factorial result

## Decision

No U2 arm advances to a full-walk experiment. Retain the current production
uphill updater and stop local tuning of its feedback parameters.

The factorial identifies `sigma` feedback as the dominant mechanism behind
both the late-prefix failure and its repair. Resetting only the normalized
Gaussian curvature does not help. However, resetting `sigma` globally trades
one set of failures for another: it repairs a bias-5 failure and one bias-8
failure, but loses the force certificate on one bias-3 task and the other
bias-8 task. It therefore fails the pre-registered survivor rule.

The correct next research priority is direction generation/evaluation, not a
new sigma floor, feedback mixture, OPES-like bias, or quadratic uphill solver.

## Evidence

- Artifact: `gpu_factorial.json`
- Artifact SHA256:
  `cbfb9d0db958c883d33b1175ab4a52b41b57f055f00740c13db770e2a991f2ad`
- Execution commit: `7ed43e18a5888e1a297eecd1bc112e8040468cc8`
- GPU: NVIDIA GeForce RTX 3060
- System: C60
- Attempted/completed frozen prefixes: 6/6
- Four-arm replays: 24
- Accounted force evaluations: 3,101
- Total wall time: 60.735 s
- Observer-only force evaluations: 0
- Unattributed force evaluations: 0

## Four-arm result

Paired differences are relative to the current controller
`feedback_on_on`.

| Arm | Meaning | Final true energy (eV) | Direction progress | Orthogonal displacement | Force evaluations | Certificate rate |
|---|---|---:|---:|---:|---:|---:|
| `feedback_on_off` | reset normalized bias curvature only | -0.045 | -0.040 | +0.532 | +1.67 | -0.333 |
| `feedback_off_on` | reset sigma only | +1.174 | +1.096 | -1.554 | -1.33 | 0.000 |
| `feedback_off_off` | reset both | +1.054 | +1.052 | -1.610 | -2.67 | 0.000 |

Higher final true energy and direction progress are desired; lower orthogonal
displacement and cost are desired. Certificate rate must not regress.

The marginal factorial effects, averaged across the other factor, are:

| Factor changed | Final true energy (eV) | Direction progress | Orthogonal displacement | Force evaluations | Certificate rate |
|---|---:|---:|---:|---:|---:|
| sigma feedback off | +1.136 | +1.093 | -1.848 | -2.83 | +0.167 |
| normalized-curvature feedback off | -0.082 | -0.042 | +0.238 | +0.17 | -0.167 |

This cleanly assigns the aggregate U0/U1 improvement to the width/explicit-step
channel, not to the Gaussian-curvature feedback channel.

## Why the apparently positive sigma result is not promoted

The average is not a stable policy result:

- Bias 3, seed 2002: current converged in 51 evaluations; resetting sigma did
  not converge in 83 evaluations.
- Bias 5, seed 2007: current did not converge and moved backward with very
  large orthogonal displacement; resetting sigma converged in 56 evaluations
  with positive progress.
- Bias 8, seed 2004: resetting sigma repaired a current non-convergence.
- Bias 8, seed 2008: current converged, while resetting sigma did not.

Thus small current sigma is sometimes the cause of collapse, but the
feedback-free base sigma is sometimes too large. A lower bound, interpolation,
or conditional switch could fit these six tasks, but would be exactly the
post-hoc heuristic parameter design this project is avoiding.

## Algorithm conclusion

The current adaptive controller is not uniformly beneficial, but its failure
cannot be removed by a single clean global intervention. Its Gaussian
curvature matching is not the bottleneck; the context-dependent step-width
decision is. Since a globally reset width fails the stop rule, the updater
stays unchanged.

The next clean hypothesis is upstream: direction quality and direction-context
information may determine when a given width is safe. That hypothesis should
be tested through direction-source/score attribution and posterior prediction,
with the current updater frozen, rather than by adding more local uphill
parameters.
