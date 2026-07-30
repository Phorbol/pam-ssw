# U3 bias-shape mechanism screen

## Question

Does the finite Gaussian tail contribute to serial SSW propagation beyond its
center value, zero gradient, and negative directional curvature?

## Controlled comparison

For each frozen production proposal task, retain the same state, historical
Gaussian prefix, newest center, direction, sigma, weight, explicit starting
displacement, optimizer, force threshold, iteration limit, and coordinate
trust bound. Replace only the newest term

`w exp(-q^2 / (2 sigma^2))`

with

`w - 0.5 (w / sigma^2) q^2`.

The two terms have the same value, gradient, and directional Hessian at their
center. There is no fitted parameter.

## Budget and stopping rule

- C60: seeds 2002/2003/2004/2006/2007/2008 at bias counts 3/5/8.
- PdO: seeds 2001 and 2005 at bias count 1.
- Two paired replays per captured prefix; right-censored captures are retained
  with their exact cost and are not replaced.
- No shape interpolation, cutoff, quartic stabilization, or parameter sweep.
- Advance quadratic bias to a full-walk test only if its uphill progress and
  true-energy gain improve without worse certificates, orthogonal motion,
  trust-bound activity, or force-evaluation cost on both systems.
