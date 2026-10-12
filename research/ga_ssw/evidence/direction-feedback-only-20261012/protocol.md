# Feedback-only direction source ablation

This branch isolates the displacement-feedback contribution in
`LocalDirectionState.update` for free, nonperiodic, fixed-cell research only.
The parent is immutable baseline `6c70ef4b1a568866f084a9670a01eeae496d79fc`.

`update` still invokes the original complete generator with its original seed,
coefficients and RNG. That preserves local pair/group selection, random draws,
route diagnostics and group-marker transitions. The generated direction is
discarded. The returned direction is only the existing normalized
saved-Gaussian-center displacement, scaled by the existing c9 rule
(`coefficients[9] > 1e-6`), then passed through `_native_normalize`. No extra
cluster-frame projection is applied to the seed. A zero displacement therefore
returns a zero vector with `release_all=True`, even when the diagnostic full
generator found a local direction. `initial()` is unchanged.

This is a source ablation, not a configurable production mode, checkpoint
variant, or claim about search performance. No GPU/PES search was run.

## Verification

Run from the worktree root:

```bash
PYTHONPATH=. python research/ga_ssw/evidence/direction-feedback-only-20261012/verify_feedback_only.py
PYTHONPATH=. pytest -q tests/standalone/test_native_direction_control.py
```

The dedicated parent-comparison script passes: initial directions match;
update RNG consumption, local route and group marker match the parent; the
returned nonzero direction matches the c9-scaled displacement formula; and a
zero displacement releases to true-surface relaxation.

The unchanged baseline-oriented test module reports 12 passed and two expected
semantic conflicts: `test_stage_state_uses_saved_center_not_previous_direction`
expects the full local-plus-feedback vector, while this branch intentionally
returns feedback only; `test_state_zero_displacement_still_generates_local_move`
expects local motion at zero displacement, while this branch intentionally
releases. These tests were not edited. Seven ASE/NumPy deprecation warnings
were emitted by existing fixture construction.

This numerical/interface check does not establish physical or global-search
benefit. The branch should be evaluated only under the separately frozen
matched-escape protocol before drawing a scientific conclusion.
