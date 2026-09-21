# XXXII table0 erfc derivative floor audit

## Source facts

`/tmp/expiry-stock/pair-pinned.cpp:139-144` evaluates the real-space screened
Coulomb energy as `A(x)/r`, where `x=G r` and

```
A(x) = exp(-x^2) t [a1+t(a2+t(a3+t(a4+t a5)))],
t = 1/(1+p x).
```

The five coefficients are the standard LAMMPS Ewald constants (`ewald_const.h`)
and the force branch uses `A(x)+F x exp(-x^2)`.  The latter is the exact-erfc
derivative identity, although `A` itself is the five-coefficient approximation.
The table0 path is active for the saved run (`engine-0.log:53-54`).  Its
measured `G` is `0.47570069` in the origin log (`engine-0.log:73`).

## Derivation

Let `Q=C q_i q_j`, with `C=332.06371 kcal A mol^-1 e^-2` converted to eV.
The derivative of the returned energy is

```
F_energy = Q/r^2 [ A(x) - x A'(x) ].
```

The code returns

```
F_code = Q/r^2 [ A(x) + F x exp(-x^2) ].
```

Therefore the code-minus-energy force error is

```
Delta F = Q/r^2 x [ A'(x) + F exp(-x^2) ].
```

For `t'=-p t^2`, `P(t)=a1+t(a2+t(a3+t(a4+t a5)))`,

```
A'(x)=exp(-x^2) { t' [P(t)+t P'(t)] - 2x tP(t) }.
```

This isolates the finite polynomial inconsistency without changing any model
parameter or calling a backend.

## Saved-coordinate estimate

`research/ga_ssw/audit_xxxii_erfc_floor.py` reads only the saved
`type2_xxxii.extxyz`, `lmp.data`, `plan.json`, and `engine-0.log`. It checks
atom IDs are exactly 1--172 before sorting charges, parses the first measured
`G` from the log (`0.47570069`; the 36 saved log values range only from
`0.47570059` to `0.47570080`), and enumerates 31,484 periodic atom-image
pairs below 10 Å.  The integer image bounds are derived per pair from its
fractional displacement and `10*||inv(cell)[:,k]||`, with a one-cell safety
padding; no fixed image range is assumed.  The result is saved at
`research/ga_ssw/evidence/xxxii-erfc-floor-audit/result.json` and reports:

```
sum absolute pair errors       8.36436e-05 eV/A
max per-atom vector error      2.43534e-06 eV/A
RMS per-atom vector error      4.97860e-07 eV/A
atomic projection (code-energy) -3.54944e-08 eV/A
predicted FD - analytic        -3.54944e-08 eV/A
```

Correction on 2026-09-11: the original report reversed the last sign. Since
`gradient_code - gradient_energy = -Delta F`, atomic `FD - analytic` is
`+sum(Delta F * direction)`. The raw old artifact is retained; the corrected
image-resolved derivation is in `xxxii-erfc-cell-rc-audit/result.json`.

The last value agrees in sign and magnitude with the saved table0/Ewald1e-12 atomic
signed residuals, `-3.41205e-08` and `-3.46801e-08 eV/A` in
`research/ga_ssw/evidence/xxxii-lammps-qualification-table0-ewald12/result.json`.
This is strong quantitative support that the polynomial/force mismatch can
explain the atomic residual floor.  It is an offline estimate: reciprocal-space
truncation, exact runtime shell inclusion, self-image details, and finite-step
roundoff are not separately evaluated here.

## Boundary

The result supports calling table0/Ewald1e-12 **finite-precision numerically
consistent at the observed atomic scale**, subject to the stated residual, but
not strictly energy/force conservative and not a second-order gradient pass.
The largest saved residual remains cell0, `1.69510e-06` / `1.76645e-06 eV/A`,
with other cell and RC residuals also listed in `result.json`; this atomic
calculation does not attribute those terms.  A stronger claim requires a
controlled run with measured fixed `G`, reciprocal limits/vector count, and an
independent projection of `Delta F` through the saved cell/RC Jacobians.

## Cell and rigid-body follow-up (2026-09-11)

The offline image-resolved extension now projects the same force and virial
error through both implemented coordinate Jacobians, without a PES call. It
predicts cell0 `-1.80647e-6` and RC `+7.64942e-7` in the generalized gradient
units. The remaining RC error decreases from `-8.3706e-7` to `-2.4086e-7`
when the difference step halves; a two-point Richardson extrapolation leaves
`-4.2120e-8`. These are diagnostic predictions, not a proof from two steps.
The next bounded check fixes the already measured screening parameter and
adds a third step, retaining table0/Ewald1e-12 and the physical potential.
The initial attempt selected an unrelated LAMMPS installation and failed
before any PES evaluation; it is retained as a failed environment preflight.

## Three-step fixed-G result (executed)

`xxxii-lammps-qualification-fixedG-three-step-v2` completed 52 EFS in 2.294 s
with the original successful LAMMPS 22Jul2025 Update4 wheel and existing MPICH.
The failed predecessor made one API attempt but zero physical evaluations.
The saved launcher filters only unrelated broken DeepMD plugin discovery and
restores metadata afterward; no installed package, physical force or stress was
modified. Package/library provenance is in `environment.json`.

All measured G values are 0.47570069 and all printed reciprocal-vector counts
are 2196. Equal counts do not prove equal vector membership. Translation and
rotation invariance and the fresh engine reproduce energies to 2.4e-14 eV and
forces to 4e-13 eV/Angstrom; fresh origin is exactly equal.

Cell0 signed errors at h=(1e-4,5e-5,2.5e-5) are
(-1.70974,-1.78547,-1.79460)e-6 versus predicted -1.80647e-6.
The last remainder is 1.19e-8. RC signed errors are
(-0.06990,0.52660,0.59458)e-6 versus predicted 0.76494e-6. Its last remainder
is -1.70e-7; the smaller-step Richardson remainder is worse than the first
pair (-1.48e-7 versus -3.95e-8). Thus the third point does **not** establish a
pure second-order RC remainder. Cell2/3 also retain order 1e-7 residuals.
No additional h/accuracy sweep or force correction is justified by these data.

Decision: the dominant cell error has a quantitative backend explanation;
strict conservativity and complete residual attribution remain unproved. An
experimental bounded whole RC-VC step may now test the implemented lifecycle
at its existing 0.005 generalized-gradient / 0.01 atomic-force / 0.001 stress
thresholds, with the residual explicitly recorded. This does not lower those
thresholds or establish scientific production qualification. Failure, fresh
certification and geometry remain separate outputs; successful optimization
alone cannot qualify a new molecular-crystal phase or native engine parity.
