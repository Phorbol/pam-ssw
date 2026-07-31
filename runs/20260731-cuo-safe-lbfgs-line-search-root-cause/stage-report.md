# CuO Safe-LBFGS proposal line-search root cause

## Question and frozen task

The third-system CuO selector run spent 83--85% of all force evaluations in
proposal relaxation, with 58--62% of Safe-LBFGS line trials rejected and nearly
every proposal relaxation ending as `line_search_failed`.  This gate freezes the
first production proposal task from one shared true-PES-quenched CuO seed-42
minimum.  The state, selected direction, explicit Gaussian displacement,
Gaussian bias, fmax=0.05 eV/A, and 300-step cap are identical between arms.

The frozen task contains one Gaussian bias and 38 active-neighbor local-softening
pair terms.  Only the LS term, its hard cutoff, the PES numerical precision, or
the optimizer is changed.  This is a proposal-objective mechanism experiment,
not a terminal-search-quality comparison.

Execution commit: `2dfb489`.  CuO package SHA256:
`914133194053c563c23d8372a8911491b730d7f3f77f83c7d414b554ecde7944`.
The shared bootstrap used 108 force evaluations and retained 24 fixed atoms out
of 54.

## Exact depth-1 result

| Fixed-task arm | End | FE | Wall s | Rejected line trials | Final max force, eV/A |
|---|---:|---:|---:|---:|---:|
| production LS, float32 | line search failed | 262 | 8.59 | 124 | 0.486 |
| LS without 2 A cutoff, float32 | line search failed | 245 | 8.00 | 106 | 0.550 |
| production LS, float64 | 300-step cap | 781 | 164.84 | 479 | 0.508 |
| LS without cutoff, float64 | 300-step cap | 833 | 175.13 | 531 | 0.505 |
| no proposal LS, float32 | converged | **59** | **2.02** | **1** | **0.0385** |
| production LS with FIRE, float32 | 300-step cap | 166 | 5.66 | n/a | 0.341 |

Removing only proposal LS is the sole arm that reaches the frozen force
certificate.  Removing the hard cutoff does not rescue either precision.
Double precision prevents premature failure but spends 3.0--3.2 times the
production-LS FE and still does not approach the requested force threshold.
FIRE shows that the failure is not cured by avoiding Armijo backtracking.

The no-LS endpoint also has lower true-PES energy (-199.480 eV) than the two
LS Safe-LBFGS endpoints (about -198.56 eV).  This does not by itself prove a
better SSW landing, but it rules out the interpretation that the LS arms merely
stopped at an equally relaxed modified-PES state.

## What the failed line actually sees

At the failed production float32 base point, repeating the identical CuO PES
evaluation changes the reported true energy by 1.526e-5 eV.  At the smallest
line trial the gradient predicts a total decrease of 2.56e-7 eV, whereas the
measured total change is +1.51e-5 eV.  The Gaussian and analytic LS energy
changes agree with their gradients at that scale; the float32 true-PES energy
resolution is larger than the decrease Armijo is asked to certify.

This is only the termination mechanism.  It is not the primary physical
problem: in float64, where the line search can keep accepting steps, the
softened objective still fails to reduce the max force below about 0.5 eV/A in
300 steps.

The current exponential LS term has a nonzero first derivative at every pair's
reference distance.  With strength 0.15 eV and decay length 0.3 A, each term
starts with a repulsive radial derivative of magnitude 0.5 eV/A.  The 38-term
task begins with about 6.00 eV of LS energy.  Thus the current proposal scope is
not a curvature-only modification: it is a persistent many-pair force field
that competes with the true PES and Gaussian escape bias throughout relaxation.
Rebuilding it at each SSW microstep does not remove that first-order force.

## Decision

The following are rejected as production fixes:

- increasing the number of Armijo trials or reducing minimum alpha;
- running the CuO model in float64;
- merely removing the LS hard cutoff;
- replacing Safe-LBFGS by FIRE for the same softened objective.

The minimal next candidate is `local_softening_scope="oracle"`: retain the
existing LS transformation only while selecting a direction, and remove it
from proposal relaxation.  This deletes an identified coupling and introduces
no new parameter.  It does **not** claim that oracle LS is beneficial.  Existing
C60/PdO fixed-starter scope data favor `oracle` over current `both` in median
landing energy, but `oracle` versus `none` still changes sign by system.

The next gate is therefore a shared-bootstrap, same-selector, equal-20,000-FE
CuO `both` versus `oracle` search.  Admission requires the proposal
line-search-failure burden and FE/action to fall without degrading the achieved
best energy under the same budget.  Only after that gate should `oracle` versus
`none` be revisited as a direction-generation question.

## Claim ceiling

This gate identifies the CuO proposal-relaxation bottleneck and eliminates
three tempting numerical fixes.  It does not establish a new production
default, a universal LS formula, or a system-general search-quality gain.
The incomplete diagnostic outputs named `*-invalid-output` are excluded.
