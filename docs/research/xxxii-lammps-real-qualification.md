# XXXII stock-LAMMPS ASE adapter: actual 172-atom qualification

The adapter runs the converted original molecular-crystal model with preserved atom ID/type/charge/topology and the actual ARC-derived cell. It is **not yet certified as an energy/force-consistent oracle at the requested finite-difference resolution**. No quench or search was run. This is a backend numerical qualification, not molecular-crystal stability evidence or whole-binary parity.

## Frozen runs and full denominator

All four experiments use identical seed17 directions, geometry, point order and 36 E/F/stress calls: origin; Cartesian direction ±h at h=1e-4/5e-5 Å; six strain coordinates at the same steps with strain_length=5 Å; RC55 direction at the same steps with rotation/torsion_length=1 Å/rad; complete-molecule lattice translation; whole-system rotation including cell; fresh-engine origin. Each directory contains plan, source/input snapshots, calls and results.

| Evidence directory under research/ga_ssw/evidence | Table bits | Ewald accuracy | EFS | seconds | largest derivative error |
|---|---:|---:|---:|---:|---:|
| xxxii-lammps-qualification-first | 12 (original) | 1e-6 | 36 | 1.094 | 0.1078 |
| xxxii-lammps-qualification-table0 | 0 | 1e-6 | 36 | 1.119 | 0.1085 |
| xxxii-lammps-qualification-table0-ewald10 | 0 | 1e-10 | 36 | 1.134 | 8.19e-6 |
| xxxii-lammps-qualification-table0-ewald12 | 0 | 1e-12 | 36 | 1.213 | 1.77e-6 |

Errors are derivatives with respect to explicitly scaled Å-valued coordinates, in eV/Å. All **144 development EFS** are retained separately from future search costs. Closing the Coulomb table and changing Ewald precision were sequential, explicitly authorized numerical-control experiments. They did not change any force-field coefficient, cutoff, topology or geometry, but they are not the original numerical configuration. Neither diagnostic was installed as a public default.

The fresh-engine origin reproduced energy/force/stress exactly in all runs; translating a complete molecule by a lattice vector reproduced energy to about 2e-14 eV and forces to 4e-13 eV/Å. Rotation energy discrepancy fell from 1.6414e-5 eV at 1e-6 to 9.8794e-10 eV at 1e-10 and zero at 1e-12. The last run's rotation force discrepancy is 3.59e-13 eV/Å. These results support the coordinate/ID mapping and identify a numerical-precision issue; they do not certify every derivative.

## Identified mechanisms and evidence boundary

1. Stock pair tables independently interpolate Coulomb energy and force, using reduced-precision distance lookup. Native also uses table12. Disabling tables does **not** repair the dominant first-run discrepancy.
2. The initial crystal contains ordinary periodic atom pairs extremely close to the 10 Å real-space Coulomb cutoff: zero-based (11,55), 10.000001884 Å; (43,131), 9.999977148 Å; (116,130), 10.000058739 Å. The finite-step probes cross that cutoff. Unshifted truncated erfc(G r)/r has a nonzero energy jump at the cutoff. `analyze_xxxii_cutoff_crossings.py` reconstructs all signed crossing pairs with image offsets and base/plus/minus distances, without any new EFS. Its output is `xxxii-lammps-qualification-table0/cutoff-crossings.json`.
3. Using the source formula for the original Ewald G (not a measured runtime extraction) and exact mathematical erfc, the predicted Cartesian cutoff contribution is 0.00181760 / 0.00363521 eV/Å versus observed signed errors 0.00181877 / 0.00363750. Most cell directions show similarly dominant cutoff jumps. The RC h=5e-5 discrepancy is **not** explained by that calculation: residual 0.1641 eV/Å. Reciprocal-set changes and the approximate erfc are additional mechanisms, not proven exclusive explanations. No analytic correction was applied to engine results.
4. Ewald automatically estimates G and chooses a finite reciprocal set from cell and accuracy; its finite set can change discretely. The precision-run engine logs preserve actual printed G and reciprocal-vector counts. Increasing accuracy greatly reduces the initial errors and restores rotation invariance at numerical precision.
5. Even table0 is not an exact analytical erfc implementation: `pair-pinned.cpp:139–144` constructs a five-coefficient approximation to erfc, then computes force using the identity for the derivative of exact erfc. Energy at line190 uses the approximate value. Therefore energy and force are not literally derivatives of one finite approximation. A subsequent zero-PES calculation projects the derivative mismatch over all cutoff-bounded periodic image pairs at the saved geometry and measured origin G. It predicts atomic FD-minus-analytic error -3.54944e-8 eV/Å, versus the observed -3.41205e-8 / -3.46801e-8 (sign corrected on 2026-09-11). This quantitatively accounts for the atomic residual scale; it does not attribute the full cell/RC residual matrix. See `2026-09-10-xxxii-erfc-floor-audit.md` and `evidence/xxxii-erfc-floor-audit/result.json`.

At Ewald1e-12 the Cartesian errors are 3.41e-8 / 3.47e-8 eV/Å; RC errors are 7.21e-8 / 5.24e-7; worst cell errors 1.70e-6 / 1.77e-6. Most errors do not fall fourfold when h is halved. Thus the requested second-order finite-difference consistency remains unclosed. Do not silently label the full matrix passed or use optimizer convergence as a replacement.

Source evidence: the official LAMMPS implementation archived in `/tmp/expiry-stock/{pair-pinned.cpp,ewald.cpp,kspace.cpp}`; stable source commit `9c5ab448c78a14fd534619622162ba418d6a1fb1`, with actual wheel/version recorded separately in each plan. The source algorithm is consistent with the runtime trends, but a pinned source excerpt is not itself a binary-level proof.

Decision: preserve all failed/partial numerical qualifications; stop additional PES probes pending a focused resolution of the finite-approximation derivative floor. Do not shift Coulomb energy, replace forces, relax the scientific claim, or begin RC-VC search on the strength of these checks alone.

## Updated decision after fixed-G three-step probe

A further52EFS run has completed, bringing this derivative qualification series
to196 actualEFS plus one failed zero-PES environment preflight. Dominant cell0
residual agrees with the erfc derivative mismatch; small RC/cell residuals remain
unattributed and the three-step RC result does not establish pure second-order
convergence. See `2026-09-10-xxxii-erfc-floor-audit.md` for signed values and
`xxxii-lammps-qualification-fixedG-three-step-v2` for complete artifacts.
The previous stop-on-unexplained-dominant-error decision is superseded by a
bounded experimental RC-VC lifecycle test, retaining all known residuals and
existing optimization tolerances. Exact energy/force conservativity and native
whole-engine parity remain unclaimed. No public defaults or forces are changed.
