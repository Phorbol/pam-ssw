# C60 paper-LS stage 10: bounded offline diagnosis

Source facts from `results/paper-seed3/result.json`: stages 0–8 converged;
stage 9 (the tenth rotation) failed with residual
`0.039116868954878925` at tolerance `0.02`. The run used 100 rotation
requests and ended with 1017 recorded E/F requests, without a landing. The
saved frozen LS table is in `preparations.json`; the nine completed Gaussian
records are in `result.json`.

`research/ga_ssw/replay_hard_c60_stage10_offline.py` replays ordered cache
calls `917..1016` (915 is the preceding quench and 916 its true-energy check),
using only the saved E/F cache. It reconstructs `ClusterFrame` for
`direction_only`, projects and normalizes the saved anchor, and uses only the
frozen LS term in the rotation callback, matching `paper_reference.py`.
Gaussian terms are introduced only after direction convergence and therefore
are excluded here. The replay consumed 100 force evaluations (`99` HVP
secants plus the center), matched every requested coordinate exactly in order
(`max_coordinate_error=0`, strict tolerance `1e-9`), and used `extra_PES=0`.

The residual falls rapidly from `98.3489` to below `0.03`, then settles into a
two-cycle near the budget limit. In the final ten checks it alternates between
approximately `0.0273459` and `0.0391169`; the corresponding curvatures
alternate near `-64.587403` and `-64.590775`. The final residual reproduces the
archived `0.039116868954878925` exactly. The maximum 2D projected
antisymmetry norm is `0.0813618` at the first plane, while the final ten plane
values alternate between about `0.0269145` and `0.0064256`. For the projected
matrix `A`, the exact identity is
`||A-A^T||_F = sqrt(2) * |a01-a10|`; the tail values therefore quantify the
off-diagonal asymmetry directly. They are of the same order as the alternating
residuals (`0.0391169` and `0.0273459`), but this numerical proximity does not
prove that asymmetry causes the plateau. Consecutive final directions change by
only about `0.0071` degrees, so the failure is a stable-direction residual
two-cycle rather than large direction wandering. This is direct evidence of
oscillatory/stagnating finite-secant rotation at the request budget, rather
than monotonic residual descent. It is not evidence that the failure is caused
by a numerical floor in the physical backend: the replay uses the recorded
cache and does not separate backend roundoff from the rotation map.

The direction history is stored in `offline-stage10/summary.json`; the scalar
plane tail is in `offline-stage10/plane-tail.json`; the coordinate match audit
is in `offline-stage10/evaluation-match.json`. A
falsifiable follow-up is a same-frozen-objective comparison with a specified
growing-subspace/Ritz solver under the same 100-force budget. No solver,
tolerance, or budget was changed here, no GFN2/PES call was made, and no new
algorithmic gain is claimed.
