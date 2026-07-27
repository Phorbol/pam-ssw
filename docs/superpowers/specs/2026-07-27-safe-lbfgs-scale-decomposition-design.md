# Safe L-BFGS inverse-scale decomposition

## Purpose

Decompose the positive fixed-matrix result of the existing history-enabled
safe L-BFGS kernel without introducing a new production optimizer or tuning any
numeric constant.

The preceding history-10/history-0 experiment changed two mathematical
mechanisms together:

1. the initial inverse-Hessian scale used by the two-loop recursion;
2. the low-rank inverse-BFGS corrections represented by retained secants.

For an empty history the current kernel uses

\[
H_k^{(0)}=\frac{1}{70}I.
\]

For a nonempty history it instead uses the latest accepted pair to set

\[
H_k^{(0)}=\gamma_k I,\qquad
\gamma_k=\frac{s_k^\top y_k}{y_k^\top y_k},
\]

before applying the two-loop corrections. The previous result therefore
identified the combined history-enabled mechanism, not retained multi-secant
history alone.

## Frozen three-arm comparison

Replay the same 16 immutable one-bias `ProposalRelaxationTask` objects with
exactly three arms:

1. `fixed-scale-history0`
   - no retained two-loop history;
   - fixed inverse scale \(1/70\).
2. `adaptive-scale-history0`
   - no retained two-loop history;
   - after the first accepted positive-curvature secant, use only its scalar
     \(\gamma_k=(s_k^\top y_k)/(y_k^\top y_k)\);
   - do not apply any rank-one or rank-two secant correction.
3. `adaptive-scale-history10`
   - the unchanged production safe-total kernel;
   - adaptive scale from the newest retained pair plus the existing two-loop
     corrections with capacity 10.

All arms optimize the same total biased objective

\[
\Phi(x)=E_{\rm true}(x)+V_{\rm bias}(x)+V_{\rm soft}(x)
\]

and use the same total gradient, movable-atom projection, maximum atomic
displacement, Armijo condition, backtracking sequence, curvature gate, MIC
branch semantics, force certificate, and `maxiter=400`.

The first comparison, `fixed-scale-history0` versus
`adaptive-scale-history0`, changes only the scalar initial inverse scale after
an accepted secant. The second comparison,
`adaptive-scale-history0` versus `adaptive-scale-history10`, measures the
remaining contribution of the complete two-loop correction stack. It does not
separate the newest correction from additional retained pairs; that requires a
later history-1/history-10 experiment only if the scale-only arm does not
explain the observed gap.

## Minimal private implementation seam

Add one keyword-only experimental flag to `Relaxer.relax`:

```python
_safe_lbfgs_adaptive_scale_without_history: bool = False
```

It is deliberately private and must not enter `SSWConfig`, YAML, CLI,
`RelaxOptimizer`, `SurfaceWalker`, or the ordinary proposal-replay API.

The flag accepts literal booleans only. `False` is a no-op for every existing
optimizer path and preserves current behaviour. Only `True` activates the
experimental mode, and then it is valid only when:

- `optimizer == "safe-lbfgs-total"`;
- `_safe_lbfgs_history_limit == 0`.

`True` with a default/`None` history limit, history limit 10, or any non-safe
optimizer, and every non-boolean flag value, fails before the first evaluator
call. The default `False` path is bitwise-behaviour compatible with the current
kernel.

When the flag is true, an accepted pair which passes the existing relative
positive-curvature gate becomes the latest scale pair, but the two-loop history
passed to `_lbfgs_inverse_product` remains empty. A MIC image-branch change
clears the scale pair exactly as it clears ordinary history. Rejected secants
do not update it.

No arbitrary scalar is exposed. The scale is derived from the same accepted
secant already computed by the algorithm.

## Experimental protocol

Use the pinned source summary
`62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04`:

- C60 and fixed PdO;
- seeds 42 through 49;
- one Gaussian bias per frozen task;
- 16 tasks, 3 arms, 48 rows;
- independent calculator per arm within each system, never shared across arms;
  each arm calculator is reused only for its sequential frozen task replays;
- identical MACE model, device, precision, input, task payload, observer, and
  force certificate;
- one execution only, without retry, retuning, seed replacement, or arm
  expansion after physical calls begin.

Every ledger row explicitly records the resolved history limit, the private
flag value, and the resulting scale policy. For every row,

\[
N_{\rm trace}=N_{\rm EvalCounter}=N_{\rm RelaxTelemetry}
\]

is a hard validity gate. Finite `maxiter` outcomes remain valid incomplete
scientific outcomes and are not converted to convergence.

## Interpretation

Certificate coverage is reported before cost. Force-evaluation count is the
primary cost; wall time is descriptive. Energy/force trajectories, rejected
line-search trials, accepted/rejected secants, MIC resets, and endpoint
differences are secondary diagnostics.

The experiment can answer:

- whether adaptive scalar scaling alone improves the fixed-scale empty-history
  baseline;
- whether the existing two-loop corrections add value beyond that scale on
  this fixed matrix.

It cannot establish:

- same-endpoint or same-basin acceleration;
- the separate contribution of one secant versus older retained secants;
- generic L-BFGS superiority;
- full-SSW/global-search improvement;
- statistical generalization;
- benefit from bias-separated secants or analytic Gaussian-bias Hessians.

Bias separation and analytic bias-Hessian corrections remain out of scope
until the scalar-scale contribution is identified.
