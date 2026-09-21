# XXXII rotation: separate finite-difference accuracy from eigensolver convergence

The total-budget RC-VC memory400 run completed four biased quenches then failed
at Gaussian index4: dimer99HVP/100API, residual0.061228 above0.02. The budget10
control instead exhausted1498searchAPI in its second biased quench. Neither
has a landing. More memory alone does not complete this RC-VC trajectory.

Thirteen fresh fixed-direction evaluations at the failed point separate a
numerical error from insufficient direction refinement: forward residual at
h1e-4 is0.061228, central0.051886. At h2.5e-5 the central result is0.051710,
still above0.02. Two-direction projected asymmetry drops from0.07055 forward
to0.000409 central at h1e-4. Smaller difference steps alone would not make the
saved direction pass; no tolerance relaxation or automatic step sweep follows.

The existing Cartesian reference solver already retains a fully
reorthogonalized Krylov basis and performs symmetric Rayleigh-Ritz extraction
with a final direct residual, whereas the generalized dimer retains a two-vector
rotation subspace. The next experiment adapts that existing solver to flat
coordinates, preserving the same shifted operator H-beta*n0*n0^T. This is a
numerical substitution, not recovered CBD parity or a new move family.

The mathematical symmetric-operator assumption is explicit. Official SciPy
[eigsh documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
identifies its real symmetric/Hermitian input and ARPACK/Lanczos basis. Our
unrestarted small reference is not ARPACK; finite-difference HVPs are only
approximately symmetric/linear, so a small projected residual is insufficient.

Predeclared comparison: first successful and fifth failed frozen RC-VC rotation
points from the same saved trajectory, same seed3 anchor, beta100, h1e-4,
tolerance0.02. Each of forward-Ritz and central-Ritz gets at most100 combined
E/F/stress API requests, including the final direct residual. Central calls
count twice per HVP. Four arms total at most400API/30s each on the qualified
converted-GAFF CPU backend. Existing dimer costs remain in the campaign. No
whole search or production integration is authorized by a favorable frozen
comparison alone; a whole-step follow-up must separately preserve total budget
and verify true landings and geometry. Delete/retain decisions depend on actual
residuals and total force costs, not solver identity or this selected failure.

## Frozen results and whole-step control

Four arms cost164API: at stage0 forwardRitz57calls residual0.08359 failed,
centralRitz24calls residual0.01905 passed; at stage4 forwardRitz57calls residual
0.21253 failed, centralRitz26calls residual0.004739 passed. Direct residuals
are from a fresh HVP of the returned direction, not projected eigenvectors
alone. This supports the combined central-difference/subspace experiment, not
Ritz with arbitrary secants or unconditional solver superiority.

A whole-step research-only substitution now uses centralRitz in the existing
RC-VC loop, with100API per rotation including the finaldirect check, identical
1500totalAPI/120s, memory400, per-quench1498 numericalbudget ceiling, seed3
and all physical parameters/geometry. The old100HVP argument is explicitly
interpreted as100API in this probe; no silent doubled force budget is allowed.
Initial and candidate fresh checks remain within1500. Frozen code/input and
all failures are retained; this does not alter the public generalized dimer.

## Whole-step1500-budget result and separate completion experiment

CentralRitz completed five biased quenches and failed in the sixth only on the
shared1498searchAPI limit; every encountered rotation passed within24–26calls.
Total1499API inclfreshinitial, no landing. This is completion of more stages,
not proven search efficiency. The previous dimer400 stopped on rotation rather
than total budget after four stages. The declared1500-budget experiment ends
with these failures, all retained.

A separate workflow-completion experiment starts again from the same input
and frozen centralRitz algorithm, budget5000API/120s including2fresh. Its
source is distinct; it is not merged into the1500 comparison or called a fair
efficiency win. Cost rationale: firstfive stages require approximately1370API;
12 times observed largest stage (~325API) plus roughly1000API for unrestricted
true cell quench fits5000. This explicit engineering ceiling is not a claim
of convergence or a fitted physical parameter. No optimizer/rotation tolerance
changes accompany it. A failure is retained as failure; final force, stress,
geometry and independently recomputed energy are required before calling the
workflow numerically complete.
