# Uphill feedback U2 factorial plan

## Goal

Attribute the C60 late-prefix U0/U1 result to the two physical state variables
already controlled by the current feedback law:

1. explicit step/Gaussian width `sigma`;
2. effective Gaussian curvature `weight / sigma**2`.

No new bias form, continuous parameter, selector, optimizer, or full-walk
policy is introduced.

## Frozen protocol

- System: C60 only.
- Model, bootstrap, K=4 direction generation, proposal optimizer, fmax,
  iteration cap, geometry constraints, and production bias-weight bounds:
  identical to U0/U1 schema 3.
- Attempted prefixes: seeds 2002/2003/2004/2006/2007/2008 at bias counts
  3/5/8; a prefix that terminates is retained as censored and not replaced.
- Four arms form a 2x2 factorial:

| Arm | sigma | effective curvature |
|---|---|---|
| `feedback_on_on` | captured current | captured current |
| `feedback_on_off` | captured current | curvature-matched base |
| `feedback_off_on` | feedback-free base | captured effective |
| `feedback_off_off` | feedback-free base | curvature-matched base |

For each arm, reconstruct `weight = sigma**2 * effective_curvature`, then apply
the shared production weight bounds.

## Decision rule

A single-factor arm advances to a paired full-walk test only if it:

- recovers the late-prefix certificate/direction failure seen in U0/U1;
- does not reproduce the bias-3 certificate regression;
- is Pareto-nondominated in final true energy, direction progress, orthogonal
  displacement, and force evaluations.

If neither single factor passes, do not tune a mixture. Retain the current
updater and return the research priority to direction generation/selection.

## Tasks

- [x] TDD the four physical arm constructions and fail-closed matrix analyzer.
- [x] Run the six pre-registered C60 captures and four-arm replays on CUDA.
- [x] Preserve successful/censored capture costs and total wall time.
- [x] Write the claim-bounded attribution report.
- [x] Run the U0/U1/U2 core regression set, commit, and push the branch.
