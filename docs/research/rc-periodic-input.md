# Explicit bond-based periodic molecular image lift

2026-09-10. `rc_periodic_input.unwrap_rigid_molecules(atoms,bonds)` now returns
lifted atoms, integer images, connected-component atom indices and provenance.
It preserves input cell/PBC/species and uses only supplied zero-based blist bonds.
No chemical cutoff, covalent radius or guessed molecular connectivity is used.

For each bond, Minkowski-reduced ASE lattice geometry supplies candidate closest
images. The two shortest distances are compared at a stated floating-point error
scale (128 eps times coordinate/cell scale and cell condition number); equal
images are rejected, including half-cell ambiguity. This tolerance is numerical,
not chemical. BFS assigns integer images with the lowest-index root unchanged.
A nonzero image sum around a bond cycle is rejected as winding connectivity,
which cannot define one finite unwrapped molecule. Output obeys
`positions_out = positions_in + images @ cell`.

Tests use wrapped S22 water dimers with triclinic cells and explicit lattice
translations, half-cell and winding-cycle negative controls, and actual XXXII
frame zero from `TYPE2-XXXII/addition/add.arc` plus `mc/rigidbody` and `mc/blist`.
The supplied XXXII frame is already molecularly unwrapped (zero needed shifts);
an additionally wrapped copy is lifted back to identical component-relative
geometry. It contains 172 atoms, 180 blist bonds and four 43-atom molecules.
The explicit rigid forest has 46 internal/relative coordinates and a finite
zero-coordinate Jacobian. Maximum supplied bonded distance is 1.73174 A.
No PES was evaluated. Evidence: `research/ga_ssw/evidence/rc-xxxii-image-lift/result.json`.

The separate optional Pymatgen archive matcher was exercised in mace_env with
PYTHONNOUSERSITE=1 on actual AlOH26. Explicit ltol=.2/stol=.3/angle_tol=5 controls
pass rotation+translation+permutation/image shifts, unimodular basis change and
a 2x supercell; 1.5x isotropic dilation and changed composition are rejected.
scale=False, primitive_cell=True and attempt_supercell=True were retained.
These are representation controls, not calibrated phase or basin tolerances.

Validation: image-lift tests 3 passed; optional Pymatgen control 1 passed. No
MACE model, GPU job, calculator evaluation or environment modification was used.
The forest's current internal reference view explicitly sets PBC=False on an
owned copy; the lift itself retains PBC. VC-RC wrapper integration must retain
physical periodic E/F/stress on the outer supplied structure.
