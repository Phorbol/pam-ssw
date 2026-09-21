# Experimental staged direction helper

`SSWConfig(pre_rotation_hvp=k, rotation_bias=None)` enables an explicitly
experimental two-stage direction solve: an unbiased dimer presweep followed
by the selected `dimer` or `ritz` main solver, with the original total HVP
budget shared between stages. The presweep direction is recorded as the
actual anchor and `max(pre.curvature, 0)` is recorded as the actual rank-one
rotation bias used by the main solve.

The combination is deliberately strict. `rotation_bias=None` requires
`pre_rotation_hvp`; specifying both fields is rejected. Existing positive
fixed `rotation_bias` configurations retain their prior single-stage path and
defaults. The helper is a numerical research variant, not complete native CBD
parity; the native Broyden/history state machine remains outside this path.
