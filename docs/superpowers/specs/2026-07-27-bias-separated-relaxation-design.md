# Bias-Separated Proposal Relaxation Design

## Status and scope

This design continues the exact-accounting, fixed-budget posterior runner at
commit `2d922c1`. It does not change the starter posterior, direction oracle,
Gaussian-bias construction, SSW outer loop, archive semantics, or default
optimizers.

The immediate scientific question is narrower:

> Can the known analytic Gaussian-bias structure reduce the physical
> force-evaluation cost of biased proposal relaxation without degrading its
> convergence or changing the SSW search controls?

The work is staged as P0--P3. A later stage is entered only after the preceding
stage produces auditable positive evidence.

## Prior evidence

The real CUDA MACE smoke at
`runs/20260726-233346-posterior-mace-c60-pdo-smoke` attributes about 63% of all
physical calls in both C60 and PdO campaigns to biased proposal relaxation.
All relaxation calls together account for about 73.5% of C60 and 87.5% of PdO
calls. Proposal relaxation is therefore the first optimization target.

An older unmerged experiment in the `proposal-hybrid-optimizer` worktree
already tested a five-step FIRE warm-up followed by a custom L-BFGS. Across
three C60 and three CuO seeds it did not consistently beat the existing
optimizers in force cost or search outcome. That candidate coupled warm-up
length, FIRE dynamics, L-BFGS memory, scaling, damping, and step limiting, so
its result cannot identify a useful block. This design does not reuse it.

## Mathematical structure

For a plain SSW proposal,

\[
U(x) = E(x) + \sum_{j=1}^{m} B_j(x),
\]

where \(E\) is the true PES. Each implemented Gaussian hill is

\[
B_j(x)=w_j\exp\left[-\frac{q_j(x)^2}{2\sigma_j^2}\right],
\qquad
q_j(x)=d_j^\mathsf{T}\Delta_{\rm MIC}(x,c_j),
\qquad \|d_j\|=1.
\]

On one fixed minimum-image branch,

\[
\nabla B_j
=-\frac{B_jq_j}{\sigma_j^2}d_j,
\]

\[
H_{B_j}
=B_j\left(\frac{q_j^2}{\sigma_j^4}
-\frac{1}{\sigma_j^2}\right)d_jd_j^\mathsf{T}.
\]

The Hessian contribution is rank one. For positive weight it is negative in
the intended escape region \(|q_j|<\sigma_j\), zero at
\(|q_j|=\sigma_j\), and positive outside it. Directly inserting this Hessian
into a positive-definite BFGS inverse is therefore not a safe local patch; it
would require an indefinite trust-region solver. Adding it to a BFGS model
already trained on total biased gradients would also double-count the bias.

For two accepted points, the exact endpoint identity is

\[
y_U
=\nabla U(x_{k+1})-\nabla U(x_k)
=y_E+y_B.
\]

This permits one clean ablation:

- **total-secants:** update L-BFGS with \(y_U\);
- **bias-separated secants:** update the same L-BFGS implementation with
  \(y_U-y_B\), while applying the resulting positive metric to the total
  proposal gradient \(\nabla U\).

The second form is a preconditioned descent method for the biased PES when the
maintained inverse approximation is positive definite. It is not claimed to
be a Newton model of \(U\). Its value must be established empirically against
the otherwise identical total-secant solver.

## P0: make relaxation evidence correct

P0 changes reporting and redundant evaluation behavior, not optimization
mathematics.

1. Add one component evaluation record containing total, true-PES, Gaussian
   bias, and optional softening energy/gradient components. One call to the
   true calculator must populate all components; component reporting may not
   issue another PES call.
2. Add relaxation telemetry with:
   - resolved backend;
   - objective-call count;
   - reporting-cache hits;
   - explicit finalization calls;
   - unified per-atom maximum force norm;
   - convergence flag and termination reason;
   - accepted and rejected step counts where the backend exposes them;
   - accepted and rejected secant-pair counts for custom L-BFGS.
3. Cache only exact repeated points requested for reporting. Do not suppress or
   reorder evaluations requested by an optimizer.
4. Reuse the first and final backend evaluations when classifying a relaxation
   instead of unconditionally re-evaluating both points.
5. Require the runner bootstrap to satisfy the same per-atom force certificate
   it advertises. Backend success alone is not a certificate.
6. Keep post-relax physical validation separately accounted until an explicit
   result-gradient contract can replace it without weakening validation.

The default `ase-fire` proposal and `scipy-lbfgsb` true quench remain unchanged.

## P1: one orthogonal proposal-relax boundary

`ProposalPotential` gains an `evaluate_parts()` method. Its existing
`evaluate()` delegates to this method and preserves the public two-tuple
contract.

`Relaxer` retains its current constructor and accepts an optional component
evaluator. Existing backends use only total energy and total gradient. Custom
bias-separated logic may read analytic components, but it may not access the
raw calculator or evaluate outside the caller's accounting scope.

The new optimizer is proposal-only. True-PES quenching stays on the existing
backends because no bias structure exists there and mixing the two questions
would prevent attribution.

The first production comparison uses plain SSW only:

- no local softening;
- no alternative/rescue optimizer;
- no internal proposal pool;
- no change to direction, bias, starter, archive, or budget controls.

## P2: minimal optimizer ablation

The candidate family has one implementation and one scientific switch:

```text
safe-lbfgs-total
bias-separated-lbfgs
```

Both use:

1. the standard L-BFGS two-loop recursion;
2. inverse initial scaling
   \(\gamma_k=(s_k^\mathsf{T}y_k)/(y_k^\mathsf{T}y_k)\);
3. a numerical curvature test
   \(s^\mathsf{T}y>\sqrt{\epsilon_{\rm mach}}\|s\|\|y\|\);
4. a descent check on the total gradient;
5. the same maximum atomic displacement rule;
6. the same monotone Armijo line search;
7. identical memory and stopping semantics.

No optimizer constants are exposed as new `SSWConfig` parameters in this
phase. The line-search constants and finite-memory size are fixed
implementation choices shared by the two candidates, not tuned per system.
Any failed line search or non-finite component terminates with an explicit
reason; there is no silent switch to FIRE or another optimizer.

`ase-fire2` may be measured as an existing algorithmic control if the installed
ASE provides it, but it is not part of the bias-attribution comparison.

### PBC branch gate

The bias-separated secant is locally meaningful only while every Gaussian
hill remains on the same MIC image branch at both accepted endpoints.
Component evaluation therefore records the integer MIC image signature for
each hill. If any signature changes between accepted endpoints, the solver:

1. does not form or accept that secant pair;
2. clears the complete L-BFGS history;
3. records one `mic_branch_reset`;
4. continues from the current total gradient with no silent interpolation
   across the branch.

The total-secant solver applies the same reset so the only scientific
difference between the custom modes remains the gradient used to form a valid
same-branch secant.

### Why explicit bias Hessian is deferred

A locally exact model

\[
H_U^{\rm model}=H_E^{\rm QN}+\sum_j H_{B_j}
\]

is mathematically cleaner but generally indefinite in the escape region. It
requires a trust-region or truncated-CG method with explicit negative-curvature
handling and PBC branch controls. That is P3 research, not a P2 patch.

## Verification ladder

### Analytic correctness

- Gaussian energy, gradient, component sum, and HVP agree with finite
  differences on nonperiodic and stable MIC branches.
- Total and bias-separated secants satisfy their endpoint identities.
- Fixed atoms are removed from all active gradients and secants.
- Total and bias-separated modes are bitwise identical before the first
  accepted secant pair.
- Every accepted step is a descent step for total biased energy.
- Rejected secants and line-search failures are explicit.

### Frozen local objectives

Use deterministic quadratic and anisotropic true PES examples with one and
multiple Gaussian hills. Compare:

```text
ase-fire
ase-fire2
safe-lbfgs-total
bias-separated-lbfgs
```

All solvers receive the same start, static proposal objective, `fmax`, and
maximum steps. Report raw objective calls, convergence, final force, final
energy, and termination reason. Do not retune a failed candidate.

### Real CUDA MACE SSW

Stage G1 is a one-seed screen:

```text
systems:   C60, PdO
seed:      42
backends:  ase-fire, ase-fire2, safe-lbfgs-total,
           bias-separated-lbfgs
policy:    uniform
q:         1000 physical calls/action
budget:    3000 physical calls/campaign
batch:     2
workers:   2
```

All other settings, structures, model, precision, masks, and PBC values are
copied from the accepted real smoke. Campaigns execute serially outside the
already validated two-worker action pool so they do not contend for one GPU.
`proposal_trust_radius` is set to `None` for every G1/G2 arm because the
current backends do not implement that field with equivalent semantics.
Walker-level `walk_trust_radius` remains frozen. The manifest reports, for
every backend, coordinate-bound behavior, per-step atomic-displacement
behavior, PBC-axis behavior, and observed bound/reset diagnostics.

A candidate is not advanced if it has a line-search failure, non-finite state,
geometry failure, unknown/unattributed cost, or worse bootstrap/landing force
certificate semantics than the baseline.

Stage G2 expands only surviving candidates and `ase-fire` to paired seeds
`42, 43, 44`. The primary measurements are:

- biased-proposal physical calls;
- optimizer objective requests, reporting-cache hits, and actually charged
  calculator calls;
- total physical calls and wall time;
- converged/unconverged proposal counts;
- termination-reason counts;
- best-energy improvement from the certified bootstrap per total call;
- geometry and active-force certificates for retained archive frames.

Three seeds provide engineering evidence only. Results are reported as paired
raw differences and medians, without significance or superiority claims.

The two custom modes are an inseparable experimental block. They enter G2 only
if both pass G1, and G2 always executes them on the same systems and seeds. A
single surviving custom arm cannot be used to support a bias-separation claim.

## P3 entry criterion

Explicit low-rank bias-Hessian/trust-region work is permitted only if:

1. both custom modes pass all correctness and production safety checks;
2. for each real system, the median paired difference
   `bias-separated - total` in charged biased-proposal calls is negative;
3. bias-separated proposal-convergence coverage is not lower than total
   secants on either system;
4. at least two of three paired seeds per system have a negative charged-call
   difference;
5. neither mode has fewer certified retained landing frames in more than one
   of three paired seeds per system;
6. every system has at least one recorded secant-decision difference caused
   by the analytic bias contribution, rather than a reporting-cache-only
   difference.

If these conditions are not met, the bias-separated block is rejected. The
project keeps only independently useful P0 measurement/cache changes and any
standard backend whose benefit is separately demonstrated.

## Claim boundary

This work can establish local optimizer cost and convergence behavior under
the tested MACE objectives and small fixed-budget SSW campaigns. It cannot
establish canonical unbiased sampling, physical kinetics, symmetry-distinct
minima counts, global-search superiority, or MACE model accuracy.
