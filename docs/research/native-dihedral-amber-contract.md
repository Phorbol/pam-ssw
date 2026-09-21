# Native DihedralAmber and exact stock representation for XXXII

**The custom five-coefficient dihedral can be expressed by stock CHARMM for the actual XXXII parameter set without replacing the physical model.** The required representation uses a common dihedral weight of 5/6 and rescales only the special 1–4 epsilon by 0.6. It does not send 1–4 interactions through the switched ordinary pair potential. This closes the dihedral algebra; whole-engine E/F/stress comparison remains required before declaring a working equivalent backend.

## Native evidence

Uploaded ELF `GA-SSW_program/lasp`: coeff 0x12daa10–0x12dad5f; compute 0x12d9b10–0x12daa0f; init_style 0x12db3f0. Disassembly snapshots are in `research/ga_ssw/evidence/native-dihedral-amber/`.

`coeff` requires six arguments including type selection: `(type,K,n,delta,w_LJ,w_C)`. K is stored through object+0x148, LJ weight through +0x150, Coulomb weight through +0x158, multiplicity through +0x170 and phase through +0x178. Multiplicity and phase are parsed as integers; n must be nonnegative and each weight must lie in [0,1]. Either positive weight enables endpoint interactions. The source has an additional exact rule: **0.83 < w_C < 0.84 snaps to binary double 5/6** (0x12dac97–0x12dacae). Thus the input 0.8333333333 is stored as 5/6, while zero remains zero.

Angles use `v1=r1-r2`, `v2=r3-r2`, `v3=r4-r3`, `a=v1×(-v2)`, `b=v3×(-v2)`, then `cos(phi)=a·b/(|a||b|)` and `sin(phi)=|v2| a·v3/(|a||b|)`. The cosine/sine multiplicity recurrence and phase shift give

`E_torsion = K[1+cos(n phi-delta)]`, with `dE/dphi=-Kn sin(n phi-delta)`.

These are the same signed-angle convention and torsion formula as stock CHARMM. The zero-multiplicity branch has constant `K(1+cos(delta))` and zero torque. Native XYZ derivative contractions follow the same cross-product construction; this review's instruction oracle checks the signed angle itself and the separate endpoint forces, not every torsional Cartesian force instruction.

After each dihedral's torsional contribution, compute reads its two weights and, if either is positive, acts directly on atoms 1 and 4. There is **no endpoint-pair deduplication, special-neighbor query, cutoff or switching test in this tail**. Energy and virial are tallied into the pair contribution. `init_style` extracts `lj14_1...lj14_4` from the pair object, along with `implicit`; the specified long-range pair has implicit=0. With the pair producer's recovered arrays (48 epsilon R^12, 24 epsilon R^6, 4 epsilon R^12, 4 epsilon R^6), native compute's extra 1/4 and factor-2 attraction give:

`E14 = w_LJ epsilon14[(R14/r)^12-2(R14/r)^6] + w_C C q1 q4/r`.

The force on atom 1 is `(r1-r4)/r²` times

`12 w_LJ epsilon14[(R14/r)^12-(R14/r)^6] + w_C C q1 q4/r`,

with the opposite force on atom 4. No 9–10 Å switching is applied here, even though it is applied to ordinary LJ pairs. The two zero-weight copies still contribute their torsional energy; they suppress only duplicate 1–4 pair corrections.

## Actual topology, not an assumption about AMBER

`audit_xxxii_dihedral_pairs.py` reads all 432 dihedrals and 180 bonds from the original lmp.data. It finds **392 unique unordered terminal pairs**. All have shortest bond-graph distance exactly three; all 392 graph-distance-three pairs occur in the dihedral list; none overlap graph-distance-one/two pairs. Every endpoint pair has exactly one nonzero weighted entry summing to `(0.5,5/6)`. The remaining 40 dihedral entries have zero endpoint weights. Full rows, source IDs and multiplicities are retained in `pair-audit.json`. The maximum endpoint distance in the supplied crystal is 4.02608 Å, but the conversion below does not rely on it remaining below cutoff.

## Exact stock transformation

Keep every atom ID, molecular ID, type, charge, bond, angle, improper and dihedral row. Preserve each K, n and integer phase. With standard `pair_style lj/charmm/coul/long 9 10` (the energy-switch version, not fsw), set for each type:

- normal epsilon unchanged; normal sigma = native R / 2^(1/6);
- epsilon14 = **0.6 times native epsilon14**, sigma14 = native R14 / 2^(1/6);
- stock dihedral CHARMM weight = **5/6** for original (.5,5/6) types and **0** for original (0,0) types;
- ordinary `special_bonds lj/coul 0 0 0`, consistent with the recovered native compute exclusions; preserve arithmetic mixing and Ewald settings.

Then `(5/6)*0.6=0.5` gives the native 1–4 LJ prefactor, while the same stock dihedral weight gives the native 5/6 Coulomb prefactor. Both stock and native compute these endpoint interactions without switching/cutoff. Uniformly scaling every type's epsilon14 by 0.6 preserves this factor under geometric epsilon mixing; sigma conversion is linear and commutes with arithmetic mixing. No change to the torsional K is required. Using conventional nonzero global 1–4 special_bonds instead would route endpoint interactions through a different cutoff path and is not the general-coordinate proof obtained here.

Stock source explicitly accepts any CHARMM weight in [0,1], so 5/6 is legal despite documentation examples emphasizing 0, 0.5 and 1. It also requires zero global 1–4 special factors when weighted CHARMM dihedrals are active. Primary source: [LAMMPS dihedral_charmm.cpp](https://github.com/lammps/lammps/blob/9c5ab448c78a14fd534619622162ba418d6a1fb1/src/MOLECULE/dihedral_charmm.cpp), coefficient validation and compute endpoint block; [pair documentation](https://docs.lammps.org/pair_charmm.html). The independent pair audit establishes the ordinary-pair switch/exclusion correspondence; this report does not rederive all Ewald engine internals.

This transformation is specific to the available uniform ratio w_LJ/w_C=0.6. Arbitrary custom inputs with incompatible ratios for the same atom-type pair would require a different representation; do not export this as universal DihedralAmber conversion.

## Bounded instruction verification

`probe_dihedral_amber_tail.py` executes the original isolated compute tail (0x12da756 through 0x12da8db, or zero-weight skip at 0x12da9d1) for **12 cases**: r=1.7,3,9.5,12 Å, and three weight pairs including both zero. Explicit fake object/coordinate/pair-array inputs are supplied; no external call is emulated, no original main process is executed, and no expiry path is touched. Native energy and radial force match the formula to floating-point tolerance, including a nonzero endpoint correction at 12 Å.

A separate slice 0x12d9c4b–0x12d9f1d checks **12 noncoplanar geometries**, recovering native sine/cosine and matching the stock signed-angle construction to 1e-14 tolerance. It rules out a pi offset or opposite signed-angle convention in the recovered geometry. `tail-oracle.json` retains all 24 cases. All are controlled numerical instruction tests, not a physical 172-atom engine run. **0 PES calls, no installation.**

The next necessary validation is a stock-engine fixed-geometry comparison of the converted actual system against a native reference including force and virial contributions. The present conclusion is an algebraically supported viable conversion, not yet an installed or production-qualified calculator.

## Prepared converter and actual output

`research/ga_ssw/convert_xxxii_amber.py` now implements this restricted conversion. It requires the exact audited SHA256 of both original files, verifies the 392-pair topology and paired coefficient contract, and refuses nonempty output directories. It preserves all non-pair/non-dihedral coefficient sections verbatim. Outputs `lmp.data`, `in.simple`, and a manifest identifying the model-preserving algebra and the still-unreplaced template box. Actual output: `research/ga_ssw/evidence/xxxii-stock-charmm-converted/`.

`tests/research/test_convert_xxxii_amber.py`: **3 passed**. Tests check all unchanged topology/type/charge/mass sections, each dihedral row, modified-source rejection, duplicate-weight rejection, and normal/special LJ energy/force equivalence for every 11×11 type pair at four distances including 9.5 and 12 Å. These remain pure numerical checks with 0 PES calls, not a full-engine validation.
