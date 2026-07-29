# Energy-Bounded Anchor Direction Design

## Purpose

Test one minimal direction mechanism that lies between the two failed extremes
already measured on C60:

- the lowest Ritz vector is soft but retains little of the random-plus-bond
  event intent;
- the exact anchor retains the intent but is roughly an order of magnitude
  stiffer and did not produce a meaningful lower basin.

The new arm must use the already-paid anchor-seeded Krylov space, add no force
evaluation, add no learned selector, and introduce no new scientific weight or
overlap threshold.

## Evidence that motivates the experiment

The completed C60 frontier audit contains 45 anchor-Lanczos direction
selections.  In that cohort:

- median lowest-Ritz true curvature was approximately 1.24;
- median lowest-Ritz anchor overlap was 0.227;
- median reconstructed exact-anchor curvature was 42.44;
- no single Ritz eigenvector retained more than half of the anchor.

The anchor is nevertheless represented exactly by the complete 12-dimensional
Ritz basis.  A linear combination, rather than another eigenvector choice, is
therefore the smallest mechanism capable of recovering event intent.

## Physical constraint

For a unit direction \(n\), step length \(\sigma\), local true-PES Hessian
\(H\), and an approximately stationary starting configuration, the quadratic
energy rise is

\[
\Delta E_{\mathrm{quad}}(n)
  = \frac{1}{2}\sigma^2 n^\mathsf{T} H n.
\]

The walker already has an uphill energy scale \(E_{\mathrm{target}}\).  The new
direction is the normalized vector in the paid Krylov space with maximum
signed overlap with the random-plus-bond anchor, subject to

\[
\frac{1}{2}\sigma^2 n^\mathsf{T}H_{\mathrm{true}}n
  \le E_{\mathrm{target}}.
\]

Equivalently, its true curvature must satisfy

\[
n^\mathsf{T}H_{\mathrm{true}}n
  \le \kappa_{\max}
  = \frac{2E_{\mathrm{target}}}{\sigma^2}.
\]

This is not a claim that the quadratic model predicts the complete finite
uphill move.  It is a direction-construction constraint using the same local
curvature approximation already used by the walker.

## Why this is not another heuristic score

The selection has no expression of the form

\[
-\kappa + \lambda \langle n,a\rangle
\]

and therefore adds no arbitrary exchange rate between curvature and overlap.
It reuses:

- the existing random-plus-bond anchor \(a\);
- the existing true-Hessian products;
- the actual execution step length \(\sigma\);
- the existing uphill energy target.

The mechanism therefore gives `target_uphill_energy` one consistent physical
role: it bounds the local harmonic cost of the selected displacement.

## Scope restriction

The first experiment supports only:

- `direction_selection_mode="energy_bounded_anchor"`;
- `step_length_mode="per_atom_rms"`;
- `step_rms_scope="all_atoms"`;
- one anchor-seeded Krylov block.

For a normalized full-coordinate direction under `all_atoms`,

\[
\sigma =
\min(
  \text{target_step_rms}\times\text{sigma_scale},
  \text{max_step_rms}
)\sqrt{N_{\mathrm{atoms}}}.
\]

This is the requested execution step.  If the paid Krylov subspace contains a
direction satisfying the energy bound, the execution path applies it exactly.
If the subspace is infeasible at that requested step, direction selection
returns the lowest true-curvature subspace direction and execution shortens
the step to

\[
\sigma_{\mathrm{exec}} =
\min\left(
  \sigma_{\mathrm{requested}},
  \sqrt{\frac{2E_{\mathrm{target}}}
             {n^\mathsf{T}H_{\mathrm{true}}n}}
\right)
\]

for positive curvature.  Thus the physical contract is a joint direction-step
constraint: it never executes a locally predicted uphill displacement above
the existing energy target merely because the requested RMS step is
infeasible.  Negative-curvature directions need no cap.  The mode must reject
other step semantics instead of silently using an approximate scale.

## Projected solve

Let \(Q\) be the orthonormal Krylov basis, let

\[
A = \frac{1}{2}
  \left(Q^\mathsf{T}H_{\mathrm{true}}Q
       +Q^\mathsf{T}H_{\mathrm{true}}^\mathsf{T}Q\right),
\qquad
b = Q^\mathsf{T}a.
\]

Diagonalize \(A=V\Lambda V^\mathsf{T}\) and express the anchor as
\(c=V^\mathsf{T}b\).  If the anchor already satisfies the curvature bound, use
the exact anchor.  Otherwise solve the scalar secular equation for the
resolvent-filtered coefficients

\[
y_j(t) \propto \frac{c_j}{\lambda_j+t}
\]

such that

\[
\frac{y(t)^\mathsf{T}\Lambda y(t)}
     {y(t)^\mathsf{T}y(t)}
=\kappa_{\max}.
\]

Choose the root continuously connected to the anchor and reconstruct
\(n=QVy/\|QVy\|\).  Orient \(n\) so that \(a^\mathsf{T}n\ge0\).

If the paid subspace contains no direction satisfying the curvature bound,
return its lowest true-curvature direction, mark the selection infeasible, and
apply the analytic execution-step cap above.  This is a defined physical
fallback, not a second scoring policy or a new tunable parameter.

## Required diagnostics

Each direction record must contain:

- `energy_bounded_anchor_feasible`;
- `energy_bounded_anchor_active`;
- `energy_bounded_anchor_overlap`;
- `energy_bounded_anchor_curvature_limit`;
- `energy_bounded_anchor_quadratic_energy`;
- `energy_bounded_anchor_true_curvature`;
- `energy_bounded_anchor_exact_curvature`;
- `energy_bounded_anchor_step_scale`;
- `energy_bounded_anchor_energy_target`;
- `energy_bounded_anchor_requested_step_scale`;
- `energy_bounded_anchor_execution_step_scale`;
- `energy_bounded_anchor_execution_quadratic_energy`;
- `energy_bounded_anchor_step_capped`;
- the unchanged Krylov HVP request and consumption counts.

`energy_bounded_anchor_step_capped` ignores relative changes at or below
`1e-12`, so an active constraint that differs from the requested step only by
floating-point roundoff is not reported as a physical cap.

The complete Ritz spectrum remains diagnostic.  No unexecuted direction
receives a terminal outcome label.

## Retrospective preregistration

Before implementation, the existing spectrum was used only to determine
whether this mechanism is nondegenerate.  With
`target_uphill_energy=0.8 eV`, `target_step_rms=0.08 A`, and 60 atoms:

\[
\sigma=0.08\sqrt{60}=0.6197\ \text{A},
\qquad
\kappa_{\max}=4.1667\ \text{eV A}^{-2}.
\]

Using the recorded projected spectrum:

- 44/45 selections have a feasible direction;
- median reconstructed overlap is 0.592, versus 0.227 for lowest Ritz;
- reconstructed curvature is bounded at 4.167 when feasible;
- the sole infeasible selection has lowest curvature 4.580;
- exact-anchor curvature remains 38.74--46.79.

These are counterfactual geometric diagnostics, not outcome evidence.

## Experiment

Run a fresh paired C60 terminal cohort:

- the same two locked accepted starters;
- seeds 42, 43, and 44;
- `anchor_lanczos` as the control;
- `energy_bounded_anchor` as the new arm;
- 12 HVPs per direction selection in both arms;
- unchanged proposal relaxation, escape checks, and strict terminal quench.

The runner must execute all 12 cases from one locked execution commit and
produce fresh convergence certificates and structure hashes.

## Decision rule

The new arm is only a direction-mechanism candidate.  It is not promoted to a
default.

Report:

- meaningful lower basins per arm;
- new basins per arm;
- landing-energy deltas;
- total and purpose-resolved force evaluations;
- direction selection count and exact HVP cost;
- overlap and curvature distributions;
- invalid, damaged, fallback, and strict-quench failure counts.

Interpretation:

- if it improves terminal outcomes at equal HVP cost, expand to PdO before any
  posterior direction selector;
- if it changes trajectories but not outcomes, keep it experimental and next
  audit the physical anchor source;
- if it is worse or frequently infeasible, reject the energy-shell mechanism;
- do not add a tunable multiplier to rescue a negative result.

## Non-goals

- no UCB, Thompson sampling, posterior update, or action credit;
- no new direction source;
- no Dimer/CBD implementation;
- no optimizer or quench change;
- no asynchronous scheduler change;
- no production default change;
- no reaction-network logic.
