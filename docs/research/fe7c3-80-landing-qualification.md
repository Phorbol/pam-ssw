# Fe7C3-80 PQC landing qualification

This note audits the two higher-energy PQC candidates from the completed
Fe7C3-80 two-seed comparison.  It uses only the saved qualification artifacts
for job `1252705`; no new PES job was run.  The backend is the same MACE-OMAT
finite-cell model and fixed Fe56C24 topology used by the comparison.  The
qualification source, plans, raw evaluation ledgers, endpoint structures and
243-mode matrices are retained under
`research/ga_ssw/evidence/fe7c3-80-landing-qualification/`.

## Numerical qualification

Each endpoint was independently cell-quenched with the registered strict
criterion (`fmax <= 1e-4 eV/A`, `stress_max <= 1e-5 eV/A^3`) and then checked by
a fresh evaluation.  The full joint finite-difference Hessian used 243
non-translation modes at `h=1e-4` and `5e-5 A`; each step used 486
endpoint force/stress evaluations, or 972 for both steps at one endpoint.

| candidate | search input E (eV) | strict endpoint E (eV) | volume (A^3) | quench E/F requests | fresh fmax (eV/A) | fresh stress max (eV/A^3) | Hessian min at 1e-4 / 5e-5 | MC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PQC seed 101 | -668.3832839488 | -668.3832854992 | 736.4452240 | 18 | 8.3609328e-5 | 6.6590912e-7 | 0.09803349 / 0.09803352 | rejected |
| PQC seed 7 | -674.1725479584 | -674.1725492957 | 719.7272101 | 16 | 7.7288712e-5 | 7.5578211e-7 | 0.18693609 / 0.18693610 | rejected |

Both quench optimizers and fresh endpoint checks passed.  Neither Hessian
had a negative eigenvalue.  The two-step finite-difference diagnostics were
also small: spectral changes were `1.8654e-6` (seed 101) and `8.5935e-7`
(seed 7); skew norms at `h=1e-4` were `4.9005e-6` and `4.1964e-6`,
respectively.  These are numerical qualification diagnostics for the saved
finite-cell model, not a phonon or chemical-stability certificate.

The complete per-endpoint costs were:

| candidate | quench | Hessian | endpoint checks (source + fresh) | total charged E/F/stress requests | wall |
|---|---:|---:|---:|---:|---:|
| seed 101 | 18 | 972 | 2 | 992 | 27.19 s |
| seed 7 | 16 | 972 | 2 | 990 | 34.12 s |
| **total** | **34** | **1944** | **4** | **1982** | — |

The scheduler job `1252705` completed with exit `0:0` in about 65 seconds;
its qualification wall times are the per-endpoint runner times shown above.

## Structural and energy boundary

Both structures remain Fe56C24.  Relative to the common qualified initial
energy `-682.125901399781 eV`, the saved PQC candidates were high by
`+13.7426174509 eV` (seed 101) and `+7.9533534413 eV` (seed 7).  Their initial
minimum-image distances (C-C / Fe-C / Fe-Fe, in A) were respectively
`1.38892737 / 1.82076556 / 2.26773793` and
`2.21957567 / 1.73969889 / 2.31346863`.  These are structural diagnostics of
the retained candidates; the first has a particularly short C-C contact.
The strict quench and positive finite-difference spectra show stationary
numerical endpoints under this model, while the MC decisions rejected both.
They do not establish a distinct stable Fe7C3 phase, chemical validity,
magnetism, or transfer beyond this supplied structure and backend.

The comparison denominator and its corrected accounting remain in
[fe7c3-80-vc-comparison.md](fe7c3-80-vc-comparison.md).  In that summary, the
8 requested outer attempts all entered; 7 were paid, 1 was an uncharged
entered attempt, and the true not-started count is 0.  Primary outcomes give
one maxiter stage, four reached budget/request-limit stages, and no backend
failure.
