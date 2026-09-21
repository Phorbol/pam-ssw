# VC conservative height policy

`run_vc_ssw` now accepts the existing `ConservativeNativeHeightPolicy` as an
explicit `height_policy`, alongside the unchanged `MinimalAngleHeightPolicy`.
The implementation reuses the fixed-cell policy and does not add a new
criterion, weight rule, or default. `height_update_budget` is an explicit
positive integer loop ceiling and defaults to 1000 only for API parity with
the fixed-cell driver; it is validated before the first oracle request.

For each VC Gaussian stage, the background is evaluated on the physical plus
already frozen Gaussian surface. The generalized-dimer curvature contains the
analytic rotation-only term
`-rotation_bias * (direction · rotation_anchor)^2`; the VC driver adds that
rank-one term back before passing curvature to the conservative policy. The
recorded `curvature_scope` states whether the direction surface contains only
the physical enthalpy E+pV or the physical enthalpy plus frozen LS. This is a declaration of
the supplied surface, not a claim of native LS-curvature parity.

The policy returns the complete rewritten Gaussian history. VC terms and
`frozen_gaussians` are reconstructed from that returned history before the
biased quench, so level-based historical rewrites are not followed by an
append that would reintroduce old weights. Preparation input, returned
preparation, curvature scope, and update budget are retained in each stage
record. `height_policy=None` and the minimal policy retain their prior paths.

Regression coverage in `tests/standalone/test_vc_conservative_height.py`
checks an actual two-stage VC call path and pre-oracle budget validation. The
tests use ASE EMT only; no MACE/PES campaign or native VC parity claim is made.
