# XXXII: actual RC-VC failure identifies periodic special-pair discontinuity

The primitive stock-LAMMPS converted-GAFF backend is not a generally valid
oracle along the explored sheared cells. A full complex RC-VC test reached a
finite energy jump that cannot be fixed by changing LBFGS history, the dimer,
convergence tolerances or iteration budget. No native LASP whole-engine parity
or physical search success is inferred from the earlier local qualifications.

## Actual trigger and attribution

At Gaussian index5 of `xxxii-rc-vc-central-ritz-completion`, Safe-total stops
with line_search_failed after65 accepted iterations and90 rejected trials;
the biased gradient norm is1.307817. Thirteen fresh directional checks reproduce
the saved energy to1e-13 and gradient to6e-15, but a1e-6 generalized-coordinate
step along the reconstructed LBFGS direction increases the bare energy by
0.234615137eV. The jump remains approximately constant as h decreases; the
finite-difference derivative therefore diverges as1/h.

Two explicit EFS decompositions attribute the jump to:

| Contribution | Delta energy, eV |
|---|---:|
| Coulomb real-space | +0.24037692095 |
| LJ | -0.00576176549 |
| bonded/angles/improper | below3e-14 |
| dihedral and reciprocal | about1e-7 each |

The specific source mechanism is the restricted Cartesian half-box check in
[LAMMPS stable22Jul2025_update4 domain.h](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/domain.h#L175):
`minimum_image_check` decides whether a topologically special neighbor is a
far periodic image, using each Cartesian component versus half a box dimension.
It is not a general closest-lattice-vector computation in a skew cell.

For zero-based pair(129,134), connected by topology path129–130–133–134,
restricted z changes from-3.38623824 to-3.38623835A while half-z is about
3.38623825A. This crosses that predicate. The true ASE `find_mic` distance is
continuous,3.63680852358→3.63680853477A; an earlier offline fractional-rounding
calculation incorrectly called its discontinuous folded vector a MIC result.
The corrected source and geometry evidence must be used.

The source charges are-0.48053 and-0.12634. The restored ordinary Coulomb term
`332.06371*(kcal/mol in eV)*qi*qj/r` is0.24037690186eV, matching the observed
jump to the smooth background change. Its ordinary mixed LJ contribution
accounts for the opposite-signed LJ jump. All180 explicit bond lengths remain
unchanged to1.4e-14A under the rigid coordinate mapping. This localizes the
failure to periodic special-pair treatment, not a broken covalent bond or a
large error in the RC analytic coordinate gradient.

## Equivalent representations, not a physical supercell search

Four evaluations compare the identical primitive configurations represented
with1x1x1 or1x1x2 replication. No new degrees of freedom are searched; topology
and coordinates are repeated, energy is divided by the replica count, forces
are averaged over identical images and stress is intensive.

The1x1x2 representation removes the discontinuity: the same perturbation gives
-2.4430840e-7eV. However, its center energy per primitive cell is0.627956eV above
the primitive result and its forces differ, so the primitive representation
already has additional incorrect exclusions at that point. It is not safe to
silently splice a new oracle into the old trajectory and retain its claims.

Six further evaluations compare1x1x2,1x1x3 and2x2x2. They agree at the center
within2e-13eV per cell,6.1e-13eV/A for averaged forces and9e-16eV/A^3 stress;
the small perturbation energy changes agree within2e-13eV. Replica force
images agree at about1e-12. This supports a repaired representation at these
specific two geometries, not all possible VC cells or a complete new search.

## Decision and next bounded work

Stop optimizer tuning against the discontinuous primitive trajectory. Retain
all outcomes, including the successful frozen central-Ritz directions and the
failed full walks. A research-only replicated ASE adapter must first enforce
that every intended1–2/1–3/1–4 topological displacement fits the enlarged
restricted half-box domain, preserve original per-ID types/charges, and expose
replica-dependent cost. Its unit-cell input state and search DOFs remain172
atoms; internal replication is an oracle representation only. No unqualified
runtime switch or empirical cell restraint is introduced.

The original LASP binary contains both standard Neighbor::half_bin_newton_tri
and custom half_bin_newton_tri_molcry_only. Their dispatch and exclusion
semantics are now being examined; the stock failure is not automatically
attributed to native LASP. The converted-GAFF whole-engine parity boundary
remains explicit until that evidence is available.

Artifacts: `research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion/`
contains the full trajectory, line-search-gradient, energy-jump-components,
supercell-representation and supercell-cross-representation. The complete
listed converted-GAFF diagnostic series currently costs7676API calls,
including failed whole steps and enlarged-representation tests; internal atom
counts differ, so these API totals are not interchangeable measures of runtime
or force-field arithmetic cost. See the saved current-cost-ledger.json.
