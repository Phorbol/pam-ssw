# Hard-C60 soft-mode accuracy: interpretation before solver selection

The saved dimer fails at100 force requests with direct residual0.0391169.
The valid fresh Ritz attempt returns residual0.0133925 in9 requests under
the same stated calculator settings and frozen objective. The numerical
threshold is0.02 eV/Å²; neither result is a true-Hessian certificate.

For forward force differences, a force perturbation changes the computed
operator by delta(Hn)=(delta F0-delta F1)/h. At fixed normalized n, its
contribution to the rotational residual is (I-nn^T) delta(Hn). This follows
directly from r=Hn-(n^T Hn)n; it does not assume that the measured force
difference bounds the unknown exact force error. The rank-one rotation bias
is analytical and unchanged, so it cancels in this perturbation comparison.

At the shared center, the measured difference between projected total forces
has norm1.88637e-6 eV/Å. Dividing by h=1e-4 Å gives0.0188637 eV/Å²; its
component tangent to the Ritz direction has norm0.0187753 eV/Å². The dot
product with n is the parallel component, not the tangent component.
Replacing only the new center force by the saved old center force while
retaining the new endpoint gives a diagnostic residual0.0230065, crossing
the0.02 threshold. This mixed-record sensitivity calculation is not a new
physical evaluation, not a convergence certificate, and not an alternative
solver run. It demonstrates sensitivity to an observed same-point force
evaluation difference; its microscopic origin remains unassigned.

Before selecting a solver or relaxing tolerances, the bounded check fixes
center and both returned directions and evaluates exactly three points at
three predeclared tblite accuracy levels0.001/0.0001/0.00001, with a fresh
calculator for every point. It has9 E/F and60-second CPU caps. These are
numerical convergence probes, not fitted search parameters or an unseen
validation set. Preserve raw forces and apply the same frozen LS/frame
callback. Do not invert projected forces to claim recovered raw forces.

Sources: `ritz-results-v2/sensitivity-diagnostic.json`, the shared
`frozen-objective-v2.json`, and ordered dimer calls917..1016 under
`research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step/`.
The invalid initial Ritz attempt cost9 E/F; the valid Ritz attempt cost9 E/F;
all remain separate from the full-step1017 E/F and the9-E/F check below.

## Completed fixed-direction precision check

The nine evaluations are complete in two segments: the first collected three
valid points at accuracy0.001, then failed in post-processing (`KeyError: layers`).
The continuation collected only the remaining six points. Their original status
files are retained; summed active wall time is10.1809188290 seconds, not a
continuous nine-point run. No failed collector is relabeled successful.

| tblite accuracy | saved dimer direction residual | saved Ritz direction residual |
|---|---:|---:|
| 0.001 | 0.0082338096 | 0.0164519684 |
| 0.0001 | 0.0151336724 | 0.0125472081 |
| 0.00001 | 0.0142752410 | 0.0123216340 |

All six fixed-direction residuals satisfy0.02; their ordering changes. These
are fresh finite-difference evaluations, not reruns of either rotation solver.
Thus the original100-request dimer failure versus9-request Ritz success does
not establish a robust solver ranking. The observed force differences divided
by the finite-difference step are comparable to the stopping tolerance.

Source: `precision-sensitivity-v1/results-derived.json` and its referenced raw
records. A final paired dimer run is separately declared in
`dimer-precision-paired-v1/plan.md`: same solver and objective, fresh calculator
per arm at0.001/0.00001, reuse within each arm,100 force requests per arm. This
isolates an actual rotation's numerical behavior from fixed-direction checking.
It does not test full SSW search recovery or justify an adaptive tolerance.

## Completed paired rotation

The actual paired run completed with128 charged E/F requests and69.0426962 s
total wall time. At accuracy0.001 the unchanged dimer exhausted100 requests,
residual0.0297392733,50.7905 s. At accuracy0.00001 it converged in28 requests,
residual0.0151592221,18.0811 s. Both arms used a fresh calculator initially and
reused it within the arm; all solver/LS/frame parameters were unchanged.

Higher force accuracy is sufficient to recover this frozen rotation within its
original request budget. This supports a numerical accuracy explanation for
this failure and removes the immediate justification for replacing dimer with
Ritz. It does not identify the microscopic source of the previous force
differences, establish an optimal precision, or guarantee full search recovery.
No adaptive threshold, solver switch, or new default was introduced.

Source: `dimer-precision-paired-v1/result.json`, raw logs, and frozen runner.
The next full-step comparison must retain the original failed-run denominator
and change only calculator accuracy; its whole trajectory may change, so it is
not a continuation from this frozen stage and not independent generalization.

## Calculator meaning and scope

The tblite Python API describes `accuracy` as controlling SCC numerical
thresholds; it is not a force-error tolerance in eV/Angstrom. The official CLI
documentation specifies that smaller values tighten convergence. Consequently,
changing0.001 to0.00001 must not be described as guaranteeing100-fold smaller
force errors. Sources checked2026-09-10:
[Python API](https://tblite.readthedocs.io/en/latest/api/python.html) and
[upstream CLI manual](https://github.com/tblite/tblite/blob/main/man/tblite-run.1.adoc).

The installed0.7 ASE adapter at
`/tmp/pam-ssw-tblite-20260909/tblite/ase.py` passes the parameter to its API
calculator and invokes `singlepoint(self._res)`, retaining previous results
between compatible geometry updates. This confirms the availability of restart
history in the evaluated implementation, but does not prove that history alone
caused the observed same-point force difference. Fresh-per-point qualification
and reused-calculator search remain distinct protocols.

For another ASE calculator, its own convergence controls and measured projected
force consistency must be considered. This experiment does not justify inserting
a tblite-specific accuracy default into generic SSW or calling a calculator's
numerical-accuracy parameter an absolute force-error bound.

## What transfers to other calculators and VC coordinates

For a fixed orthogonal projector and unit direction, the operator norm of
`I-n n^T` is at most1. If endpoint force errors have known norm bounds
`epsilon0, epsilon1`, their contribution to residual error is at most
`(epsilon0+epsilon1)/h`. This is a conditional bound, not a measured error bar
for the present xTB run. Finite-separation truncation adds another error; smaller
`h` alone reduces neither term simultaneously. Repeated or tighter evaluations
provide convergence evidence but do not by themselves supply rigorous bounds.

The relevant norm is the complete projected force vector, not the maximum
per-atom norm used to terminate a local quench. Passing a0.01 eV/Angstrom
quench threshold therefore does not certify a0.02 eV/Angstrom-squared rotation
residual obtained by dividing force differences by1e-4 Angstrom.

For VC/RC generalized coordinates, replace the Cartesian force by the gradient
pulled back to the declared coordinate chart and metric before applying this
analysis. Atomic force accuracy alone does not qualify stress-derived cell
components. This is an interface requirement and a reason to retain separate
atomic and cell qualification, not a new empirical switching policy.

## Completed whole-run accuracy control

The single-factor accuracy0.00001 full seed3 paper-LS run used the previous
memory400 algorithm snapshot and unchanged2,000-E/F/900-s limits. It completed
10 biased quenches, then failed at Gaussian index10 with
`SCF not converged in 250 cycles`. Total cost1,030 requests consists of1 initial,
1,028 outer-step requests (including the failed backend request), and1 fresh
initial check; elapsed738.9192 s. No true landing exists. The earlier0.001 run
completed9 biased quenches and failed in rotation; both whole runs remain
unsuccessful. These different terminated trajectories cannot establish search
efficiency or a universal accuracy default.

Artifacts: sibling directory
`hard-c60-gfn2-paper-ls-memory400-accuracy1e5-single-step`, with frozen input,
runner diff, actual import provenance, raw logs, and `results/paper-seed3/comparison.json`.
Analysis distinguishes converged/rejected landings from absent or unconverged
landings, and matches fresh structures through `minima` indices rather than
outer-step indices. Initial fresh checks never certify a nonexistent landing.

The complete cache-only replay subsequently consumed all1,030 archived requests.
Independent comparison of all1,029 successful request geometries gave maximum
coordinate difference0 (1,028 preceding the failure, plus the final fresh initial
check). The captured stack places failed call1029 at the post-rotation displaced
background-force evaluation, rather than inside dimer. The replay injects a
RuntimeError containing the recorded CalculationFailed message; it does not
reproduce a real SCF failure or constitute a second calculator experiment.

Both original and higher-accuracy runs had already reached biased C58+C2
intermediates, with shortest interfragment distances3.7733313/6.9251679 Angstrom.
The latter had true energy8.2615566 eV above its initial minimum. These are
climbing intermediates, not true-PES stationary products. They motivate checking
the missing published cluster controls before more optimizer/SCF tuning, but
do not alone prove LS causes fragmentation or that a particular remedy helps.
See `c60-native-fragmentation-boundary.md` for verified SI inputs and the limited
native vapor-check caller evidence. No fragment rejection or compression policy
was added to the Python core.
