# PAM adaptive Gaussian reuse audit for fixed-cell `run_ssw`

This audit records the implemented experimental public interface. It documents
the reuse boundary; no new PES run is included here.

`PAMCurvatureGaussian` is implemented in
`pamssw/standalone/pam_gaussian.py`. Its callable contract is
`choose(mode, anchor, center, terms, base_width, rotation_bias)` and returns
`width`, `weight`, physical `k_true`, inner `k_inner`, unclipped values and
clipping flags. Public history terms and centers retain `(N, 3)` Cartesian
shapes; the helper flattens compatible arrays only for internal dot products.
Widths are Å,
weights are eV, and curvatures are eV/Å². It normalizes both `anchor` and the
returned mode direction, restores the rank-one rotation contribution as
`k_true = mode.curvature + a (d·anchor)^2`, and adds the analytic Hessian
contribution of existing `ProjectedGaussian` terms to obtain `k_inner`.
For `height_width`, it chooses
`width=sqrt(2*target_uphill_energy/max(abs(k_true),curvature_floor))`, then
clips to `[min_width,max_width]`; weight is
`width²*max(k_inner+target_negative_curvature,0)` clipped to the configured
weight bounds. These are existing experimental parameters, not new defaults.

The public fixed-cell wiring now accepts `gaussian_policy` in
`paper_reference.run_ssw`; `run_ls_ssw`, `run_native_ls_ssw`, and `run_ga_ssw`
forward the optional policy. `atomic_climb` calls `choose` after the mode solve
and before constructing `displaced`, then adds the resulting Gaussian to the
fixed-cell objective. The block interface remains available separately through
`BlockSSWConfig.atomic_gaussian_policy`.

## Compatibility with fixed-cell `run_ssw`

The mathematical coordinate and force convention is reusable: the current
`ProjectedGaussian` is the same rank-one Cartesian Gaussian objective, and
`SurfaceCalculator` evaluates the accumulated terms. The public adapter calls
`choose` after the mode solve, then constructs
`displaced = center + width*d` with the selected width. It uses the returned
PAM weight verbatim; it does not recompute a forward-force height or request an
unneeded background force. History terms passed to the helper retain their
`(N, 3)` shape.

Curvature restoration must use the actual rotation bias and anchor. For the
fixed-bias path this is the configured positive `rotation_bias`; for the
staged path it is the recorded `actual_rotation_bias` and presweep anchor.
Reusing `config.rotation_bias` or the pre-presweep anchor in the staged path
would give a mathematically inconsistent `k_true`. LS terms can remain in
`rotation_surface` and therefore in the curvature input; the recorded legacy
`k_true` scope includes the frozen LS contribution when LS is active, while the
rotation-only Gaussian bias is excluded. This is an independent numerical
policy and does not claim exact native parity.

The policy itself does not provide a no-premature-stop rule. In
`paper_reference`, a failed biased quench still stops the walk unless the
explicit `bias_stage_steps` path marks a finite max-iteration budget. Adding
adaptive width/height therefore cannot silently change lifecycle stopping;
that question requires a separately specified arm and failure accounting.

## Minimal future comparison

For the implemented interface, compare two fixed-cell arms with
identical starts, seeds, calculator, direction solver, total request budget,
fresh final certificates and stopping rules: existing fixed Gaussian policy
versus `PAMCurvatureGaussian(mode="height_width")`. Use at least one metal
system already in the fixed-cell evidence (Cu13 or Cu31) and one molecular
system (C4H6/bicyclobutane). Report width/weight clipping, physical and
modified-quench failures, fresh certificate cost, distinct landing identity,
and every request in the common denominator. This isolates the adaptive
Gaussian policy; it does not test native LASP parity or variable-cell behavior.
