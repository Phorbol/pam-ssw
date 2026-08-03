# U3 Gaussian versus curvature-matched quadratic result

## Decision

Reject the unbounded quadratic bias. Do not run it as a full-walk arm, do not
add a Gaussian/quadratic mixing weight, and do not try to rescue it with a
cutoff or quartic stabilization.

Retain the Gaussian bias shape. This experiment does not prove that the
Gaussian is uniquely optimal; it shows that its finite, decaying tail is an
active part of the present serial SSW propagator rather than a replaceable
second-order decoration.

## Isolated mechanism

Only the newest bias in each frozen production proposal was changed:

```text
Gaussian:   V(q) = w exp[-q^2 / (2 sigma^2)]
Quadratic:  V(q) = w - 0.5 (w / sigma^2) q^2
```

At the bias center, both terms have the same value, zero gradient, and
directional curvature `-w / sigma^2`. The historical Gaussian prefix, newest
center and direction, `sigma`, `weight`, explicit starting displacement,
optimizer, `fmax`, and `maxiter` were identical. No continuous parameter was
introduced or fitted.

The difference appears away from the center. The Gaussian force reaches a
finite maximum and then decays. The quadratic force grows linearly without
bound. Because production starts the proposal relaxation at `q = sigma`, this
higher-order difference is already physically active at the first optimizer
evaluation.

## Authoritative evidence

- Artifact: `gpu_shape.json`
- Artifact SHA256:
  `1918535962e769906c2c209f1d153128d8a9ea9521ed35e2bbb6b4dfa47d6458`
- Execution commit:
  `156c88565953f4543f9de1bcc116efab96931bb1`
- GPU: NVIDIA GeForce RTX 3060
- MACE model SHA256:
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`
- Completed paired tasks: 8/8 (C60 6, PdO 2)
- Accounted force evaluations: 2,970
- Observer-only force evaluations: 0
- Unattributed force evaluations: 0
- Total wall time: 64.792 s

Cost decomposition:

| System | Bootstrap FE | Frozen-prefix capture FE | Paired replay FE | Accounted FE | Accounted wall time |
|---|---:|---:|---:|---:|---:|
| C60 | 49 | 1,636 | 722 | 2,407 | 48.302 s |
| PdO | 38 | 18 | 507 | 563 | 13.398 s |

## Paired result

Values below are quadratic minus Gaussian. Higher true energy and direction
progress alone are not success: an uphill endpoint must also remain
force-resolved, localized enough to represent a controlled proposal, and
computationally finite.

| System | Certificate rate change | Median final true-energy change | Median direction-progress change | Median orthogonal-displacement change | Median FE change |
|---|---:|---:|---:|---:|---:|
| C60 | -0.667 | +260.454 eV | +31.741 | +20.326 | +61.0 |
| PdO | -1.000 | +295.017 eV | +12.235 | +10.597 | +65.5 |

The raw termination behavior is decisive:

- C60 Gaussian converged in 4/6 tasks; the two failures reached `maxiter`.
- C60 quadratic converged in 0/6 tasks; all six reached `maxiter`.
- PdO Gaussian converged in 2/2 tasks.
- PdO quadratic converged in 0/2 tasks; both ended in line-search failure.
- Mean terminal gradient norm on C60 increased from `0.099` to `396.833`.
- Mean terminal gradient norm on PdO increased from `0.041` to `69.825`.

The hundreds of eV of additional true energy therefore do not represent
cleaner barrier crossing. They are non-stationary, spatially delocalized
runaway endpoints produced by an objective whose negative curvature never
turns off.

## Trust-radius finding

The profile carries `proposal_trust_radius=1.5`, but the active production
backend is `safe-lbfgs-total`. Current code applies that coordinate box only to
`scipy-lbfgsb`; the safe L-BFGS path ignores it and always reports
`active_bound_fraction=0`.

This is a verified API/documentation mismatch. It means the U3 result measures
the actual production algorithm: Gaussian self-localization versus an
unbounded quadratic under the same safe-L-BFGS line search. It also means that
the configured coordinate trust radius must not be cited as an active safety
mechanism for current production runs.

Adding a bound would create a different algorithm: a constrained quadratic
propagator. That remains a legitimate future arm, but it is not a rescue of
this failed shape substitution and must be tested as a separate mechanism.

## Integrated uphill conclusion

The completed U0/U1/U2/U3 sequence now supports a narrow conclusion:

1. A fixed calibrated Gaussian did not beat the current updater.
2. Removing either step-width feedback or normalized-curvature feedback did
   not give a stable cross-prefix gain.
3. Replacing the finite Gaussian tail by its center-matched, unbounded
   quadratic approximation fails strongly on both C60 and PdO.

Therefore retain the current adaptive Gaussian updater as the production
baseline, but do not claim that its feedback law is optimal. Its strongest
validated property is the localized bias shape, not the mathematical
complexity of the feedback controller.

## Next bounded question

The next redundancy test should isolate cumulative bias history:

- current cumulative Gaussian prefix versus newest-Gaussian-only;
- same frozen state, newest bias, direction, optimizer, and budget;
- no decay factor, history length, or learned selector;
- late C60 prefixes only, because PdO late prefixes were right-censored in the
  existing protocol and should not be replaced by searched-for favorable
  seeds.

This tests whether serial bias accumulation is essential to irreversible
uphill propagation or is avoidable optimizer burden. Only after that discrete
mechanism is resolved is a separately constrained quadratic/CCQN propagator
worth implementing.
