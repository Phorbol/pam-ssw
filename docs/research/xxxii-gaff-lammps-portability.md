# XXXII original GAFF/LAMMPS backend portability preflight

**The uploaded model is not currently a drop-in stock LAMMPS/ASE calculation.** Its topology and parameters are available, but two custom force styles are embedded in LASP without their source/library distribution. Installing ordinary LAMMPS alone would not close this gap. No installation, model substitution, binary execution, expiry modification or PES calculation was performed.

## What is actually supplied

Source directory: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc/`.

`in.simple` explicitly requests `units real`, `atom_style full`, full PBC, harmonic bonds/angles, `dihedral_style amber`, `improper_style cvff`, `pair_style lj/amber/coul/long 9 10`, arithmetic mixing, `kspace_style ewald 1e-6`, and **`special_bonds lj 0.1 0.1 0.1 coul 0.1 0.1 0.1`**. The decimal 0.1 values must not be silently interpreted as exclusions or conventional AMBER weights.

`lmp.data` supplies 172 atoms, 180 bonds, 292 angles, 432 dihedrals and 80 impropers; 11 atom types, 12 bond types, 16 angle types, 16 dihedral types and 2 improper types. The first eight dihedral coefficient rows have five values `(K,n,phase,0.50,0.8333333333)`; types 9–16 repeat the first three coefficients with the final two zero. This strongly suggests separately represented nonbonded weights, but their exact meaning and interaction with special_bonds remain unproved until the custom implementation is read. It is not safe to infer the net 1–4 interaction by adding these numbers to 0.1.

The actual structure is C84H68Cl8N4O8 (four 43-atom molecules), with a triclinic cell saved in `tests/standalone/fixtures/type2_xxxii.extxyz`. The data-file 41×32×51 orthogonal box is a template, not that crystal cell. Any backend initialization must preserve topology/type/charge IDs while replacing coordinates and cell from the real crystal with the correct image convention.

## Stock compatibility is not established by the word AMBER

Current official LAMMPS [pair-style list](https://docs.lammps.org/pairs.html) and [dihedral-style list](https://docs.lammps.org/dihedrals.html) do not document the two uploaded `amber` names. Stock [CHARMM dihedral documentation](https://docs.lammps.org/dihedral_charmm.html) specifies four coefficients and explains its own 1–4 bookkeeping. Its documented AMBER usage uses zero dihedral pair weights and conventional 1–4 factors; that is different from this five-coefficient file. Renaming `amber` to `charmm` therefore is neither a parser-compatible nor a demonstrated mathematical translation.

The [special_bonds documentation](https://docs.lammps.org/special_bonds.html) defines separate 1–2/1–3/1–4 pair weights. The six explicit 0.1 entries are part of this input's model. Standard `lj/cut/coul/long` and conventional AMBER special-bonds settings cannot replace the custom pair interaction without proof of equivalence. In particular the custom pair's two cutoff arguments, LJ parameter convention, cutoff switching, Coulomb long-range corrections and treatment of special neighbors need recovery; naming alone proves none of those details.

## The missing implementation exists inside the uploaded binary

Read-only `nm -C` finds:

- `LAMMPS_NS::DihedralAmber::compute(int,int)` at **0x12d9b10** and `coeff(int,char**)` at **0x12daa10**;
- `LAMMPS_NS::PairLJAmberCoulLong::compute(int,int)` at **0x1116bc0**, `single(...)` at **0x11181e0**, `settings(int,char**)` at **0x11175b0**, `coeff(int,char**)` at **0x1117670**, and `init_one(int,int)` at **0x111adc0**.

Binary strings identify `../dihedral_amber.cpp/.h` and `../pair_lj_amber_coul_long.cpp/.h`. These are compilation metadata, **not available source files**. The extracted program directory contains only README.md, check_env.sh, lasp and sgn.jar. A bounded filename scan of the program, full example and preview trees found none of the four custom source/header files. No separately supplied custom shared/static library was found there. This gives a concrete bounded reverse-engineering target rather than an unknowable force-field name. Formula recovery was not attempted in this preflight.

## ASE interface boundary and executable route

Local ASE `ase/calculators/lammpslib.py` is installed and was inspected read-only. Its standard type dictionary maps chemical symbol to LAMMPS type, and its rebuilding path can issue `set atom ... type ...` from that dictionary. XXXII has **11 force-field types for only five elements**; allowing that default path would collapse distinct C/O/H environments. An adapter must preserve each atom's original type, charge, molecular ID and complete bonded topology from `lmp.data`. `create_atoms=False`/`create_box=False` are available options, but their existence alone is not proof that the default lifecycle will preserve this molecular data. No working configured adapter is claimed.

Once the custom styles are available or faithfully independently restored, the concrete route is:

1. Build a compatible LAMMPS shared library with the required bonded styles, Ewald support and the exact custom pair/dihedral implementations. Record its version and build options.
2. Load `lmp.data` and unchanged force-field commands once into that engine. Maintain a fixed atom-ID/type/charge/topology mapping; replace positions and triclinic cell from ASE without applying an additional unintended affine move. Never regenerate force-field types from elements.
3. Implement a narrow ASE Calculator adapter for `run 0` E/F/virial evaluations. Convert real-unit energy/force to eV/eV Å⁻¹ and pressure to ASE tensile-positive stress. Use configurational virial stress with no kinetic term. Preserve/rebuild long-range solver and neighbor data consistently on each cell change.
4. Before RC-VC search, verify one actual 172-atom crystal E/F/stress point against the original model, plus finite-difference atomic and six cell components with consistent topology and images. Only then run the already implemented RC-VC driver and independent final qualification.

Two viable ways to close step 1: obtain the four custom files plus any modifications to the LAMMPS core they depend on and a compatible version identifier; or recover the relevant coefficient/init/compute routines from the existing binary and validate the independent styles against original-instruction reference cases. The latter may also need to inspect how special-neighbor tags, dihedral-endpoint weights and long-range exclusions are consumed. Neither route requires replacing the intended model with ordinary GAFF defaults.

The precise remaining missing evidence is therefore **custom pair and dihedral semantics, their special-neighbor interaction, a compatible executable/library, and an ID-preserving ASE adapter qualification**. Existing topology/parameter files need not be requested again. Native binary symbols are available; that makes bounded recovery plausible, but not yet an implemented production backend.

Evidence: `research/ga_ssw/evidence/xxxii-gaff-portability/preflight.json` and `embedded-symbols.txt`. All searches were read-only. External exact-style web searches did not locate an author-distributed implementation; unrelated AMBER/LAMMPS examples were not used as substitutions.
