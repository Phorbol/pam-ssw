# XXXII custom AMBER pair: bounded formula recovery

The custom pair has a plausible algebraic translation to stock **lj/charmm/coul/long**, but a simple style rename is wrong. Its distance parameter is R_min, not standard LJ sigma. More importantly its compute routine overrides every special-neighbor LJ/Coulomb weight to zero, despite the explicit 0.1 input values. The matching single routine does not apply that override. Complete model portability still requires the separate DihedralAmber endpoint terms and engine/topology qualification. No backend was implemented and no PES was run.

## Source and bounded inspection

Uploaded LASP SHA256 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`. Read coeff 0x1117670, init_one 0x111adc0, single 0x11181e0; additionally the narrow necessary compute special-neighbor consumer, settings 0x11175b0, and init_style cutoff prefix. Excerpts are under `amber-pair-evidence/`. They are Intel objdump output from the uploaded ELF, not original source files. Inspection was static, not a full Ewald/engine audit.

## LJ coefficients and mixing

coeff parses ordinary epsilon and R into object arrays +0x368 and +0x370. Optional epsilon14 and R14 go to +0x378 and +0x380; absent optional inputs copy the ordinary values (0x111775d–760). No factor of two or sixth-root conversion occurs in coeff.

init_one constructs ordinary tables:

| object offset | stored value |
|---|---|
| +0x388 | 48 epsilon R^12 |
| +0x390 | 24 epsilon R^6 |
| +0x398 | 4 epsilon R^12 |
| +0x3a0 | 4 epsilon R^6 |

Corresponding 14 tables +0x3b0/+0x3b8/+0x3c0/+0x3c8 have the same expressions with epsilon14/R14. The literal constants are 48,24,4 at ELF 0x4aa6420/428/430. Both single and compute subsequently multiply the attraction table by 2 and the whole LJ expression by 0.25 (0x4aa63c8 and 0x4aa63c0). Therefore

`U0(r) = epsilon [(R/r)^12 - 2(R/r)^6]`,

`r F0(r) = 12 epsilon [(R/r)^12 - (R/r)^6]`.

The exact ordinary LJ conversion is **epsilon_stock=epsilon; sigma_stock=R/2^(1/6)**, also for the optional 14 parameters. R is the pair minimum distance, not its half-radius. Multiplying R by 2 would be wrong.

Unset cross coefficients use Pair::mix_energy and mix_distance. Their arithmetic branch is sqrt(epsilon_i epsilon_j) and (R_i+R_j)/2 (0x10cbb13–17 and 0x10cbb99–a5). Constant rescaling of R commutes with this mixing. Thus the input's arithmetic mix does not obstruct the sigma conversion. Other mix policies were not needed for XXXII.

## Switch and cutoff

settings stores first argument at +0x328, second +0x330, optional third Coulomb cutoff +0x348; with two arguments the second is also Coulomb cutoff. init_style squares the first two into a=+0x338 and b=+0x340, squares Coulcut into +0x350 and stores D=(b-a)^3 at +0x360. The supplied `9 10` therefore means LJ inner9, outer10 and Coulcut10 Å.

For a<r²<b, single computes

`S(r) = (b-r²)^2 (b+2r²-3a) / (b-a)^3`.

It uses U_LJ=S U0 and force numerator S(r)*rF0 + T(r)*U0, where
`T(r)=12r²(b-r²)(r²-a)/(b-a)^3 = -r S'(r)`.
Below inner use S=1,T=0; at/outside outer LJ is zero. This is energy switching, not force switching. The explicit derivative term means unshifted lj/cut or an energy-only multiplier is insufficient.

Stock [LAMMPS CHARMM pair documentation](https://docs.lammps.org/pair_charmm.html), checked 2026-09-10, distinguishes original energy-switched lj/charmm/coul/long from newer charmmfsw styles, documents standard zero-crossing sigma, optional 14 coefficients and the two-/three-cutoff syntax. Those are the appropriate stock comparison targets. The present static recovery does not substitute a version-pinned stock source comparison and numerical qualification.

## Decisive special-neighbor override

At compute 0x1116db9–de9:

```
tag = encoded_neighbor >> 30
factor_lj = special_lj[tag]
if unsigned(tag - 1) <= 2:   # tag 1,2,3
    factor_lj = 0
    factor_coul = 0
else:
    factor_coul = special_coul[tag]
neighbor_index = encoded_neighbor & 0x3fffffff
```

Thus the custom pair's actual force computation suppresses direct LJ for all marked 1–2/1–3/1–4 neighbors. The Coulomb short-range expression subtracts `(1-factor_coul)*C*q_i*q_j/r` from the screened real-space term: this is the long-range special correction, **not simply skipping every Coulomb operation**. Its nominal factor is zero for those three tags. The input's 0.1 cannot be interpreted as the surviving pairwise contribution or added to the dihedral weights.

single accepts explicit factor_coul and factor_lj; it directly multiplies LJ by the latter and uses the former for the Coulomb correction. It never reads the neighbor tag. Therefore single(...,.1,.1,...) is **not** a faithful reference for what compute does to marked special neighbors. Any isolated oracle must supply factors0 for that case or test the actual compute consumer.

## Exact-equivalence conditions and unresolved model boundary

A stock candidate needs all of the following, not merely a name change:

- ordinary energy-switched lj/charmm/coul/long, inner9 outer10 Coulcut10, converted ordinary and 14 sigma, unchanged epsilon, arithmetic mixing;
- effective zero pair factors for tags1/2/3, with topology and neighbor construction retaining all corrections needed by Ewald; blindly changing 0.1 to0 can change neighbor-list inclusion, so check the target LAMMPS version's long-range special-neighbor handling;
- the exact custom dihedral angular and unswitched endpoint terms, with their independently specified LJ and Coulomb weights. The pair alone does not restore 1–4 interactions;
- matching real-unit Coulomb conversion constant, Ewald accuracy/implementation and chosen Coulomb table settings. single shows the usual screened real-space polynomial in exp[-(g*r)^2] plus the special correction; the reciprocal solver and tabulation were not reconstructed here;
- unchanged atom-ID/type/charge/topology/image mapping and correctly converted virial, followed by actual172-atom E/F/stress qualification.

Consequently **pair formula conversion is feasible in principle**, while whole-model equivalence remains conditional. This finding narrows what is needed from missing source to the custom dihedral/topology and engine conventions; it does not certify a production backend or authorize a force-field substitution.
