# Strict true-quench replay amendment

The fixed-starter escape run completed all 36 cases, but only 4/36 SciPy
L-BFGS-B landing quenches satisfied the frozen `fmax=0.01 eV/A` certificate.
This amendment was written before any strict-refine landing energy was
generated.

The already stored 36 `escape.xyz` configurations are therefore replayed
without regenerating a direction or rerunning the uphill walk. Each escape is
quenched on the true MACE PES using the previously validated
`ASE-LBFGS primary + certificate-triggered ASE-FIRE fallback` protocol:

- `fmax=0.01 eV/A`;
- `maxiter=400` for each stage;
- one fresh evaluator counter per escape;
- all primary and fallback force evaluations charged to
  `landing_true_quench`;
- the post-relax validation call charged separately;
- no bias, local softening, selector, retry, or direction evaluation.

The replay reports certificate coverage, new-basin/downhill-landing counts,
landing energy relative to the same fixed starter, fallback use, and true
quench force evaluations. Direction-arm interpretation is permitted only when
the strict certificate closes; this replay still cannot promote a production
default by itself.

