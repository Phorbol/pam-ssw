# Fe7C3 frozen-objective native LBFGS comparison

Job1254745 completed in3m48s on one V100. Four preselected real Fe7C3-80
MACE LS+Gaussian targets were retained, with exactly the same initial q,
objective and gradient as the archived Safe-total history10/500 comparison.
No native main, BFGSDRIVER, Gaussian accumulator or full LASP walk executed.

| Frozen target | Native EFS / final norm | Safe10 EFS / final norm | Safe500 EFS / final norm |
|---|---:|---:|---:|
| ls_all, seed7 | 305 / 0.703112 | 308 / 1.13128 | 307 / 0.101062 |
| ls_all, seed101 | 306 / 0.335354 | 312 / 0.205154 | 313 / 0.0825497 |
| ls_filter, seed7 | 306 / 0.14776 | 314 / 0.210008 | 314 / 0.0726477 |
| ls_filter, seed101 | 234 / 0.019711 | 320 / 0.326872 | 243 / 0.000900593 |

The common gate is max(max atomic-gradient norm, six-cell-block L2)<=0.001.
Native qualifies0/4, Safe10 qualifies0/4, and Safe500 qualifies1/4. Three native
cases reached300 accepted steps; filtered seed101 returned a native failure
after229 accepted steps. Total native cost is1151 EFS, comprising1147 optimizer
requests and4 independent final checks, under the1320 cap. Failed-environment
job1254721 is separately archived with0 PES calls. All four final evaluations
reproduce the last accepted objective/gradient within1e-8 and all initial
comparisons pass that gate. Native accepted X agrees with the supplied E/G
coordinates throughout. Cases with no qualified biased stationary point stay
in the denominator.

The earlier Safe experiment also paid4 shared failed-point reconstruction
checks; those costs remain in its2435-EFS total and are not included in this
per-optimizer table or silently discarded. This native run reused their
reconstruction evidence and checked each first requested E/g against Safe.
Emulation wall time does not measure native hardware speedup.

This comparison changes the optimizer configuration: native history400 and
scalar step bound0.5 versus Safe histories10/500 and joint-block bound0.2.
The native GTOL900 profile is a recovered release setting, not a valid ordinary
strong-Wolfe recommendation. Results cannot isolate MCSRCH alone. No biased
gradient certificate proves a physical minimum, basin transition or global
search improvement. These four difficult local targets do not establish a
universal optimizer ranking.

Decision: retain Safe-total and its current default history. Neither original
LBFGS substitution nor longer Safe history has demonstrated a complete VC-SSW
efficiency gain. Do not expand this target set or retune optimizer settings
from these outcomes. Continue only a distinct, source-supported core question.

Artifacts: `research/ga_ssw/evidence/fe7c3-frozen-native-lbfgs/` includes the
frozen source, copied four inputs/references, dependency manifest, plan,
allocation, raw evaluations and `diagnostic/audit-summary.json`. Reproduce the
offline audit with `research/ga_ssw/audit_fe7c3_native_lbfgs.py`.

Zero-new-PES replay also resolved the filtered seed101 failure. At recorded
request233 the last accepted secant has sTy=-0.0002889063867. The next
native direction has gTp=+0.003238863592; execution visits the non-descent
branch at0x6eac29 and ends IFLAG=-1/MCSRCH INFO=0. All233 requested q values
match the recorded sequence exactly. This supports retaining Safe-total's
positive-curvature screening; it does not prove a full BFGSDRIVER run would
lack its own recovery. See `native-failure-trace.json` and the reproducible
`trace_fe7c3_native_failure.py`.
