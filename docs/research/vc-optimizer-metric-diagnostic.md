# Distinguish VC search metric from local-optimizer conditioning

The phase-87 joint continuation reaches a concrete failure: the ninth frozen
Gaussian objective did not meet its generalized gradient tolerance within300
Safe-total iterations. That fact alone does not identify a wrong gradient,
insufficient memory, excessive atom/cell stiffness contrast, or an unsuitable
search direction. Recover and check that objective before changing the solver.

The implemented chart is q=(X,L*s), with six orthonormal symmetric log-strain
components s and explicit L=5 Angstrom in this experiment. At the reference
chart, the physical cell-gradient block scales like V*(sigma+pI)/L; its local
elastic Hessian scales like V*C/L^2. Thus L affects both direction selection and
numerical conditioning. The generalized modified-gradient threshold .005 is
not the final physical force (.01) or stress (.001 eV/Angstrom^3) threshold.
The PQC atomic modified-gradient threshold is .01. No equal-cost/equal-objective
optimizer ranking follows directly from the whole PQC/joint trajectories.

Primary sources checked on2026-09-11:

* SSW-crystal, DOI10.1039/c4cp01485e, publisher pp17847–17848, equations3–4;
  local `literature/benchmark-sources/vc2014/vc2014-author.pdf`, SHA256
  21bebe8ff58de8c94205c13d93d2722c55fd0ab0cc70a9db15396796b06f1bf4.
  It uses a separate CBD cell module and a lattice-vector step bounded by15%
  of the cell Frobenius norm. This does not specify our joint log-strain metric.
* ASE `FrechetCellFilter` and `UnitCellFilter` source:
  https://docs.ase-lib.org/_modules/ase/filters.html . The former scales cell
  coordinates/gradients with a factor defaulting to atom count; installed
  ASE3.26 has the same stated extensive-factor recommendation. This is a local
  minimization convention, not evidence for the optimal SSW proposal metric.
* ASE `CellAwareBFGS` source:
  https://docs.ase-lib.org/_modules/ase/optimize/cellawarebfgs.html . It initializes
  the cell Hessian from an isotropic elasticity model and takes supplied bulk
  modulus/Poisson ratio. Its defaults are not TiO2 measurements and will not be
  imported as fitted material constants. No new ASE installation was made.

A prospective optimizer-only diagnostic must preserve the frozen physical-plus-
Gaussian scalar objective. Simply changing L and recreating unit-normalized
Gaussian directions changes the search problem. Instead, an invertible linear
reparameterization y=A*q has E_y(y)=E_q(A^-1*y), g_y=A^-T*g_q. The original gradient
and step criteria are evaluated on A^T*g_y and A^-1*dy, respectively. This permits
preconditioning while preserving coordinates, biases, and acceptance energy.
It is an exact chain-rule statement; no scale A has been selected or validated.

A dimensional size argument alone does not select a universal L: fixed-density
elastic curvature scales extensively, while force balancing, Hessian balancing,
and normalized collective displacement impose different scaling objectives.
An unmeasured elastic constant or a convenient atom-count formula must not be
silently declared an optimal metric. First inspect the actual terminal gradient,
cell, secant/line-search behavior and objective consistency. Only then choose a
minimal matched diagnostic, charge the reconstruction cost, and retain the
current default unless cross-system end-to-end evidence supports a change.


## Matched frozen-objective result (2026-09-11)

The nine-term stage8 objective was reconstructed and held fixed, including the
L=5 chart, displaced starting point, gradient tolerance .005, maxstep .2 and
300 iteration limit. Only Safe L-BFGS memory differed. Memory10 exhausted300
iterations using305 EFS calls (113.10s), ending at max-block gradient .107867.
Memory400 converged in213 iterations /218 EFS calls (81.40s), ending at .00447720.
All300/213 secants were accepted; both arms rejected four line-search trials.
Thus rejected curvature updates do not explain this particular failure.
The physical-request JSONL ledgers contain exactly305 and218 records; including
one height-reconstruction check the diagnostic cost is524 EFS.

Independent directional derivatives at the start and saved failed endpoint
used13 EFS each, with atomic, strain and mixed directions at two finite-difference
steps. Maximum absolute directional errors were9.92e-8 and7.73e-8 respectively.
This supports local objective/gradient consistency at these points, not a global
correctness proof. Together these results motivate a whole phase87 step with
memory400 under the original1500 EFS budget. They do not justify changing the
production default or claim improved basin discovery. That whole-step arm is
currently running separately from the frozen-stage diagnostic.

Evidence: `research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step/`
`joint-stage8-memory-compare-v1/`. Completed campaign expenditure before the
new whole-step arm is5110 EFS, including the invalid500-call continuation.


The whole raw-input memory400 arm subsequently exhausted1500 EFS (512.73s),
after initial quench50 EFS and completed Gaussian indices0–8. The original
memory10 arm used60 initial EFS and completed indices0–6 at the same total cap.
Both have zero valid proposals. Initial relaxed energies differ by0.000278325eV
because the changed optimizer also acts during initialization; their later
biased objectives are therefore not frozen-objective solver comparisons.
The whole-run ledger records1501 attempts, including one zero-cost budget
rejection; sum(after-before)=1500 in both arms. A separately budgeted continuation
from the memory400 completed index8 is being prepared to obtain a true landing,
without changing the original capped outcomes. Completed campaign cost6610EFS.


The bounded memory400 continuation is now complete. Starting from completed
Gaussian8, it used14 rotation+1 height+180 biased-quench+1 true-energy+82
true-cell-quench+1 certificate=279 EFS, then one independently reloaded MACE
certificate. Total additional280 EFS,103.592s; cumulative whole attempt1780EFS.
The final E+pV is-423.644140497097eV, +2.613617502095eV over its own initial
minimum; fmax .0085175750eV/Angstrom, stressmax2.3955272e-5eV/Angstrom^3,
volume763.349452Angstrom^3. MC rejects. Fresh E/F/stress equal the terminal
certificate exactly in this environment. Thus the numerical chain completes,
but there is no improved-minimum discovery. This is not a native LASP parity
or universal memory400 claim. Whole capped1500 outcomes remain censored.
Completed phase87/139 campaign expenditure is6890EFS including all documented
failed/invalid continuations and diagnostics. Evidence: joint-memory400-
continuation-root/result.json, ledger.jsonl, root-audit.json, runner-executed.py.


## Second porous initial structure: phase139

Both raw-input seed3 arms exhausted1500EFS, with zero valid proposals. Both
completed Gaussian indices0–7. Memory10 reached a budget exception before
recording the next Gaussian; memory400 recorded a failed biased quench atindex8.
That extra failed record is not an extra completed stage. Initialization cost22
EFS in each arm, but resulting force certificates and initial coordinates differ.
The pair does not demonstrate a transferable complete-search improvement from
memory400. Keep the default unchanged. Additional3000EFS; completed cumulative
phase87/139 campaign cost9890EFS. No fresh landing calls because neither exists.
Evidence: ../research/ga_ssw/evidence/tio2-phase139-joint-memory-compare/.

## Phase139 independent central-Ritz falsification control,2026-09-11

From the same original48-atom input and exactly matching common initial
minimum, seed3, memory400 and1500 total EFS including22 initialization calls:

| Direction solver | EFS | CPU seconds | Completed biased stages | Valid proposals |
|---|---:|---:|---:|---:|
| Existing forward dimer |1500|561.505|8|0|
| Optional central Ritz,100 E/g cap |1500|571.683|8|0|

Both exhaust the oracle budget in biased quench at index8; this is not a
300-iteration or line-search failure demonstrated by these runs. The top
wrapper marks censoring, while the inner record says biased_quench_failed
with evaluation_failed at the unfinished stage. Several Ritz residuals
are smaller, but no lifecycle or discovery advantage is observed. The
XXXII result therefore does not justify switching the universal default.
The three parameter groups match fieldwise; common-start coordinates,
energy, forces and stress agree exactly. The ledger records cost deltas
only, so equality of every one of the22 initialization requests cannot
be reconstructed and is not claimed.

Evidence: `tio2-phase139-joint-central-ritz-memory400/` and its
`run-central-ritz-seed3/result.json`. A post-hoc metadata collector used
a different environment and was explicitly marked invalid as runtime
provenance. `environment-root-recheck.json` records a subsequent check
under the declared launch environment (NumPy2.0.2/ASE3.26.0); it is
not an original in-process capture. Frozen source and launch arguments
remain available. The zero-PES failed system-Python launch is retained.

The experiment exposed a diagnostic omission in VC stage records: they
did not retain rotation versus biased-quench calls. The public VC driver
now records four actual surface-counter deltas per stage (rotation,
height, biased quench, true check), including failed operations, plus
optimizer iterations, attempted evaluations, rejected trials and final
gradient norm. This changes no objective, solver or parameter. Cu/EMT
p=0 and0.005 eV/Å³ before/after complete trajectories match every
requested geometry/E/F/stress exactly (244 total verification EFS).
Cost tests first fail for the missing fields, then pass;341 tests pass
and1 is skipped in the reference-compatible user-site environment.
This observation change cannot retroactively recover missing phase costs
from the old TiO2 runs. Evidence: `vc-stage-cost-observer/`.

Two independent fresh MACE EFS evaluations of the saved index8 accepted
points reconstruct each frozen nine-Gaussian objective (no optimization or
continuation): dimer/Ritz biased gradient norms are0.377403/0.353175
versus0.005. Atomic parts are0.366598/0.348698; strain parts are
0.0896585/0.0560570. Neither censored target is close to its requested
stationarity, and the residual cannot be attributed solely to the cell
block. These are two distinct nearby frozen objectives, not same-point
solver residuals. Added cost2 EFS; listed TiO2 series cost is now11392
EFS=9890+1500+2. Evidence: `tio2-phase139-frozen-endpoint-diagnostic/`.
