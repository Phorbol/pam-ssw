# XXXII stock CHARMM conversion: source-level algebraic conditions

The proposed conversion is supported by the official stock **pair, dihedral and neighbor source**, conditional on the separately recovered native angular convention and actual topology audit. It changes the representation of the same model, not fitted parameters. No LAMMPS installation, backend implementation or PES calculation was performed.

Official LAMMPS stable resolved to commit `9c5ab448c78a14fd534619622162ba418d6a1fb1` on 2026-09-10. Each downloaded stable file was compared byte-for-byte with its commit-pinned URL:

- [dihedral_charmm.cpp](https://github.com/lammps/lammps/blob/9c5ab448c78a14fd534619622162ba418d6a1fb1/src/MOLECULE/dihedral_charmm.cpp)
- [pair_lj_charmm_coul_long.cpp](https://github.com/lammps/lammps/blob/9c5ab448c78a14fd534619622162ba418d6a1fb1/src/KSPACE/pair_lj_charmm_coul_long.cpp)
- [neighbor.cpp](https://github.com/lammps/lammps/blob/9c5ab448c78a14fd534619622162ba418d6a1fb1/src/neighbor.cpp)

## Exact parameter map for the audited XXXII weights

Keep each atom ID, force-field type, charge, bond/angle/improper and dihedral incidence. Change only the following model representation:

```
pair_style lj/charmm/coul/long 9 10
pair_modify mix arithmetic
dihedral_style charmm
special_bonds lj 0 0 0 coul 0 0 0
kspace_style ewald 1e-6
```

For every original pair type, use `epsilon, R/2^(1/6), 0.6*epsilon14, R14/2^(1/6)` as stock epsilon/sigma/epsilon14/sigma14. If the native optional14 coefficients are absent, epsilon14=epsilon and R14=R before conversion. For explicit cross-pair overrides transform each row directly; otherwise keep arithmetic mixing.

For every dihedral row `(K,n,phase,wLJ,wC)`, the audited rows with `(wLJ,wC)=(.5,5/6)` become stock `(K,n,phase,5/6)`; rows `(0,0)` become weight0. Native's near-.833 input snap must be applied as **exact5/6**, not its printed decimal. Do not delete repeated zero-weight angular rows: they still contribute their K term. This constant-.6 map is specific to the audited weights (ratio wLJ/wC=.6), not a universal arbitrary-independent-weight converter.

The equivalence proof is per dihedral occurrence:

`(5/6) * (0.6 U14_LJ) = 0.5 U14_LJ`,
`(5/6) * U14_C = (5/6) U14_C`.

Both terms are unswitched endpoint interactions. The ordinary intermolecular LJ table is unaffected by epsilon14 scaling. Under arithmetic mixing, sqrt[(.6 epsilon14_i)(.6 epsilon14_j)]=.6 sqrt(epsilon14_i epsilon14_j), and uniform R conversion commutes with the arithmetic distance mean. Therefore this is coordinate-independent algebraic equivalence of these terms, not equivalence restricted to rigid molecules. Actual one-weighted-row-per-endpoint evidence belongs to the separate topology audit; the transformation itself preserves every dihedral occurrence.

## Stock source checks

Stock dihedral computes endpoint LJ/Coulomb then multiplies both by weight (lines246–267). It extracts the four 14 coefficient arrays from the pair style. It neither truncates nor switches these endpoint terms. Stock pair init_one mixes ordinary and14 parameters independently and creates standard4epsilon LJ coefficient tables (lines747–766). Its ordinary pair loop multiplies LJ by special_lj and subtracts `(1-special_coul)*qqrd2e*qi*qj/r` from screened Coulomb, matching the recovered native pair expressions after R conversion and its forced-zero factors.

The suspected zero-special neighbor-list problem is specifically resolved in the inspected stock source: neighbor.cpp lines531–563 sets **all special_flag[1..3]=2 when force->kspace exists**, even when both physical pair factors are zero. Those neighbors remain tagged for pairwise long-range subtraction. Thus replacing the native input's0.1 with stock0 does not by itself remove required Ewald corrections on this source path. Do not use neigh_modify exclusions as an alternative.

Stock angular energy follows `K[1+cos(n*phi-phase)]`, with phase in integer degrees (dihedral source152–173 and314–339). This proves the stock convention only. Matching native's K, multiplicity, plane orientation and phase convention requires the separate native angular audit. For phases0/180 the sign of phase is irrelevant; a pi offset in phi is still material for odd multiplicity and must not be assumed away.

## Remaining checks before an equivalence claim for the full calculator

1. Confirm native angular construction versus the stock bond/plane convention and all actual phase values; preserve complete dihedral multiplicity and weighted-endpoint occurrences.
2. Preserve the native units-real Coulomb prefactor. These inspected stock classes use `force->qqrd2e`; no class-local constant replacement occurs in them. Compare the actual engine value to the native value rather than assuming version identity.
3. Match Ewald accuracy, Coulomb tabulation, boundary/image conventions and bonded ghost extent. Both endpoint implementations use topology-connected images; an atom wrapped independently without correct images can invalidate the comparison.
4. Match normal pair cutoffs/switch, bonded coefficient units and stress sign/conversion. Do not replace old energy-switch CHARMM with charmmfsw.
5. Qualify the resulting model with actual172-atom energies, atom forces and six cell derivatives, ideally against bounded native primitive references first. Source algebra proves potential terms under these conditions, not successful engine integration or scientific search performance.

This closes the stock special-neighbor bookkeeping uncertainty for the pinned version and provides an executable conversion specification. It does not imply that LASP must be run, modified, or that license/protection paths need inspection.

## Concurrent native angular audit closes condition1

The independent native-dihedral audit now reports 12 noncoplanar four-atom original-instruction cases for the geometry slice `0x12d9c4b–0x12d9f1d`: c and s agree with the stock expressions using `a=v1 cross(-v2), b=v3 cross(-v2)`, including orientation, with no pi offset. Its 12 endpoint-tail cases include distances1.7,3,9.5,12 Å and zero/nonzero weights; the unswitched formula remains active at12 Å. The actual topology contains392 weighted endpoint pairs, each with exactly one(.5,5/6) occurrence, plus40 zero-weight dihedral entries. Those reported primitive/topology results close the first pending condition; their durable artifacts are maintained by the native-dihedral audit rather than duplicated here. Engine Coulomb constant, image/topology transport, Ewald settings and full172-atom E/F/stress verification remain pending.

## Engine constants and unit-consistent stress (static follow-up)

Native `Update::set_units` at0x12c7ee0 matches the literal `real` string at0x4a2ffe0, then writes these exact IEEE-double constants:

| Quantity | Native instruction/value | Stock update.cpp real branch |
|---|---|---|
| nktv2p | 0x12c80bd, bits0x40f0bd86a3d70a3d =68568.415 | 68568.415 |
| qqr2e | 0x12c80d1, bits0x4074c104f4c6e6da =332.06371 | 332.06371 |
| qe2f | 0x12c80e5, bits0x40370f8023a6ce36 =23.060549 | 23.060549 |

Both the previously pinned official commit and [stable_22Jul2025_update4 update.cpp](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/update.cpp#L170) agree. The expected wheel name alone does not establish build provenance; record its actual version and, when possible, inspect its exported qqrd2e/nktv2p. `qqrd2e` is the dielectric-adjusted prefactor used by pair and dihedral; explicitly keep dielectric1.0, consistent with the supplied input. No Coulomb rescaling or fitted compensation is indicated.

The real-unit PES is in kcal/mol and Å. Convert both E and F by the same explicitly recorded energy factor `cE = ase.units.kcal / ase.units.mol`; record ASE's CODATA convention/version. `qe2f=23.060549` is an electric-field unit factor; using its reciprocal as an exact kcal/mol-to-eV conversion is not justified.

For stress prefer a dedicated configurational pressure compute:

```
dielectric 1.0
compute pam_virial all pressure NULL virial
```

Stock [compute_pressure.cpp](https://github.com/lammps/lammps/blob/stable_22Jul2025_update4/src/compute_pressure.cpp#L299) returns `P_component = virial_component / V * nktv2p` without kinetic energy when its kinetic flag is off. Therefore use **sigma_ASE = -P_component * cE / 68568.415**, with LAMMPS `[xx,yy,zz,xy,xz,yz]` permuted to ASE `[xx,yy,zz,yz,xz,xy]`. This exactly reverses the engine's rounded pressure unit factor before applying the same energy conversion as E/F. Direct use of a modern physical atm conversion can produce a small avoidable mismatch against the derivative of converted E; this is a unit-convention effect, not evidence for an algorithmic stress correction. Include kspace, bonded and pair virials; do not use a kinetic pressure tensor or omit Ewald virial.

All-coordinate/over-cutoff scope audit: the recovered14 endpoint terms remain active beyond the LJ/Coulomb real-space cutoff in both native and stock dihedral routines. Ordinary Ewald special corrections retain the same finite real-space cutoff structure; this statement is equality of the represented implementations, not proof that every arbitrarily stretched topology is a physically intended molecular force field. Preserve ghost bonded connectivity and images so that the same endpoint vector is used. The uniform.6 epsilon14 map remains algebraic under mixed types, exact5/6 snap and every preserved dihedral occurrence. Differences from finite double evaluation of `2**(1/6)` or `.6*(5/6)` are roundoff, not reasons to tune coefficients. Nonfinite/colliding geometry, collapsed cells and missing bonded ghost atoms are outside the valid evaluation domain.

No single universal numerical tolerance is derived from this static audit. Roundoff-level primitive checks should be scaled to term magnitudes; full E/F/stress differences must additionally account for the explicitly requested Ewald1e-6 accuracy, Coulomb tabulation and finite-difference truncation/roundoff. Do not relax a tolerance after observing a model mismatch or absorb it into fitted parameters.
