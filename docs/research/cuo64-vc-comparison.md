# CuO64: external crystal source and joint-VC test

2026-09-11. This is a new material check of independent Python/ASE kernels,
not a LASP/PES parity test or a prediction of the real CuO phase diagram.

## Source and declared geometry

Original8atom Cu4O4 CIF: [AFLOW tenorite prototype](https://aflow.org/p/AB_mC8_15_a_e-001/),
[raw CIF](https://aflow.org/p/AB_mC8_15_a_e-001/aflow.cif), based on Åsbrink and
Norrby1970, DOI[10.1107/S0567740870001838](https://doi.org/10.1107/S0567740870001838).
CIF SHA256 `1e049233a6473831b5163049cc6ce427d153e77db059da00629e0c17c3f56f4e`.
Cu/O labels are explicit in the source; no prototype A/B element guess was used.
The actual source8cell is4.6837×3.4226×7.49694485Angstrom, beta137.572448°,
volume81.07982947Angstrom³. The AFLOW page explains its vector/origin protocol
and corrects an older99.54°/120.34° transcription error. We preserve the
source CIF cell; c'=c+a is an equivalent unimodular representation, not a
replacement by an unverified handwritten cell.

The test uses an exact2×2×2 repeat,64atoms, with no added atoms, defects or
initial random distortion. This makes additional periodic atomic modes available
while retaining the supplied infinite crystal as the initial geometry. It is
not a claim that the source publication used64atom SSW. Source and page are
frozen in `evidence/cuo-aflow-source`; the earlier benchmark inventory's absent
CuO label refers to the uploaded example corpus only.

## Completed model and initial-state qualification

Job1250571, one V100, COMPLETED0:0,90s scheduler time,85.82s script time.
Total806 E/F/stress:3 representation checks,22 strict-quench requests,1 fresh
endpoint,780 Hessian finite-difference requests. MACE-OMAT-0-small uses the
same frozen model SHA as TYPE4, float64, no cueq and one thread.

Unimodular rebasing agrees in energy to1.42e-14eV, force to3.72e-15eV/Angstrom,
stress to1.63e-16eV/Angstrom³. Exact replication agrees in energy per atom to
6.22e-15eV, force to1.12e-14eV/Angstrom and stress to4.10e-16eV/Angstrom³.
These are backend representation checks, not experimental accuracy claims.

The fixed coordinate scale is L=V_raw64^(1/3)=8.65633930Angstrom. It relates
log strain to displacement on the box length scale and is held fixed throughout
qualification/search. This is a declared independent geometric metric, not a
native LASP parameter or a proven optimal atom/strain balance. Quench uses
pressure0, fmax1e-4eV/Angstrom, stress_tol1e-5eV/Angstrom³, maximum atom/strain
coordinate step0.2Angstrom,500iterations and existing history10.

Fresh endpoint: E=-317.72045454230eV, volume707.68453819Angstrom³,
fmax1.96654e-5eV/Angstrom, stress max2.87040e-6eV/Angstrom³. **Volume increases
about9.10% from the supplied crystal in this model**; numerical convergence
does not establish experimental lattice accuracy or magnetic-state fidelity.
That preparation energy/volume change is not credited to SSW.

Complete joint Hessians use the same exact force/stress pullback,3N+6
coordinates with only three global atomic translations removed (195dof).
At steps1e-4 and5e-5Angstrom their smallest eigenvalues are0.2021454529 and
0.2021454698eV/Angstrom², both with zero negatives. The matrix step-difference
spectral norm is2.87024e-7. Eigenvalue magnitudes depend on the stated scaled
coordinate metric; these are not mass-weighted phonon frequencies. Raw matrices,
projection basis and all E/F/stress requests are retained in
`evidence/cuo64-vc-input-qualification/qualification`.

## Completed PQC/joint comparison

Job1250737 compares PQC (fixed-cell climbing followed by joint cell quench)
and joint atomic/log-strain climbing. Seeds7/101, two consecutive attempts
per arm, four arms total; each has2000 E/F/stress including three reserved fresh
checks and480s. The input is the same strictly qualified64atom endpoint.
Full source/configs/inputs are frozen in `evidence/cuo64-pqc-joint-comparison`.

Common numerical choices are width0.6Angstrom, rotation rank-one bias100eV/Angstrom²,
NG14, history10, maximum step0.2Angstrom, dimer separation1e-4Angstrom,
rotation residual0.02eV/Angstrom²,300quench iterations, T300K and pressure0.
These retain existing independent benchmark settings, not source CuO paper
parameters. Final fmax0.001 and stress_tol0.0001 apply to both arms; joint biased
gradient_tol0.001 matches the atomic force norm threshold while retaining its
separate cell-coordinate block. The kernels differ in accessible proposal
coordinates and are not claimed to have identical proposal distributions.

All eight requested attempts, failed/censored work, and independent fresh checks
are retained in the four arm result directories. The comparison has zero valid
search landings, so it provides no PQC-versus-joint effectiveness or phase
conclusion. A separately frozen pair of joint history10/500 target attempts has
no completed run/result and is not part of this comparison or any claimed gain.
No cap extension or parameter retuning follows this result.

## Frozen history diagnosis and whole-SSW history-500 control

The follow-up fixed-objective diagnosis used 997 E/F requests: history10
reached 300 optimizer steps for both seeds (312 and 318 optimizer requests),
while history500 converged in 161 and 193 steps (166 and 195 optimizer
requests). The four final fresh checks and two failed-point checks are included
in that total. The two failed-point biased-gradient norms reproduce the source
values to 2.11e-15 and 1.62e-15, establishing the frozen failure objective;
history10 is an independently recomputed fresh quench, not a bitwise baseline
trajectory replay. Details and the recomputable accounting are in
`evidence/cuo64-frozen-quench-history/diagnosis/summary.json`.

The subsequent whole-SSW history500 control used two seeds, each with a 1,997
search-request budget plus one fresh request, for 3,996 E/F total. Seed 7
reached stage 13 and seed 101 stage 12; both preceding climbs converged, then
their biased quenches stopped with `BudgetExhausted` after 32/35 relaxation
steps and biased-gradient norms 1.70803568/0.72466590. Each second recorded
attempt had zero charged requests and was not started. There were no search
landings. This is a budget-censored control record, not evidence of whole-SSW
improvement or optimizer superiority.

The cumulative CuO evidence cost is 806 + 7,992 + 997 + 3,996 = 13,791
E/F/stress requests. The still-frozen pair of joint history10/500 target
attempts has no result and is excluded from these totals.
