# VC-RC geometry: exact rigid interiors and cell enthalpy derivative

2026-09-10. Implemented **geometry and E/F/stress pullback only** in `pamssw/standalone/rc_vc_geometry.py`. There is no VC-RC walker, no native Kabsch/lambda parity, and no periodic molecular physics validation from this change. It follows the independent map explicitly requested by the parent after the isolated forest implementation.

## State, coordinates and boundary

`RigidForestCellChart(atoms, trees, anchor=0, rotation_length=..., torsion_length=..., strain_length=...)` requires fully periodic, unconstrained atoms and a positive-determinant cell. `trees` uses the same explicit component dictionaries as `RigidForestChart` and the rigidbody/blist parser. All root bodies must be noncollinear. Each molecule must already be supplied in one consistent **lifted Cartesian representation**: no per-atom wrapping, bond inference or periodic image reconstruction is attempted. Molecular trees can lie across the cell boundary provided their coordinates explicitly retain that continuous lift. Oracle atoms preserve the original PBC and current cell.

All root rotations and internal torsions are retained. Only the three translation coordinates of the selected anchor root are fixed, removing one globally redundant translation. Root orientations relative to the cell can affect the PES and are not removed as isolated global rotations. The symmetric cell chart already omits a redundant global cell orientation. With K components and m joints, dimension is `6K-3+m+6`. The initial generalized coordinate is all zeros.

Scaled coordinates contain translations in Å, root rotations `Lrot*p`, torsions `Ltor*theta`, and six strain coordinates `Lcell*s`. The positive metric lengths are mandatory explicit inputs in Å/radian for rotations/torsions and Å for dimensionless strain. Strain uses the existing orthonormal symmetric basis `(xx,yy,zz,yz,xz,xy)` with off-diagonal basis values `1/sqrt(2)`. There are no inferred optimal defaults.

## Exact finite map

Use the row-cell convention, with fixed reference L0 and fixed lifted rigid-body reference coordinates. Let `D=exp(S)` and `L=L0 D`. For each component let `u_i(q)` be the existing exact articulated-chain Cartesian map before cell deformation, `c0` its root-body geometric center and `t` its root translation. Set

```
c = c0 + t,
R_i = u_i(q) + c (D-I).
```

Equivalently, `R_i = c D + (u_i(q)-c)`. Thus the molecular center moves affinely with the cell, while every internal rotated/torsioned vector stays Cartesian and rigid. This differs from `R_i=u_i D`, which would stretch bonds and violate the declared constraint. Shared joint endpoints stay shared because all atoms in a component receive the same center correction.

Root and torsion derivatives are those of the exact finite exponential/product map. A root translation derivative gains the row `e_j(D-I)`, so its total Cartesian derivative is `e_j D`. Cell derivatives are

```
dD = Dexp(S)[dS],
dL = L0 dD,
dR_i = c dD  (cell coordinate, rigid pose held fixed).
```

The matrix-exponential Frechet derivative evaluates finite nonzero strain exactly, including noncommuting shear variations. `geometry(q)` returns independent Atoms, `dR/dq` with shape `(N,3,Dof)` and `dL/dq` with shape `(3,3,Dof)`. No small-strain substitution is used as a finite derivative.

## Work-conjugate E+pV gradient

An ASE stress oracle supplies Cartesian force F and tensile-positive symmetric stress sigma at the same physical geometry. Stress differentiates an **affine** atom-plus-cell variation. Our molecular cell map is nonaffine, so using stress alone would differentiate the wrong coordinate map.

For any generalized displacement let `A=L^-1 dL`. The affine Cartesian displacement corresponding to that cell change would be `R A`; the remaining physical displacement is `dR-R A`. Therefore

```
d(E+pV) = -F:(dR-R A) + V (sigma+p I):A.
```

Here `:` is the Frobenius contraction, pressure is a scalar in eV/Å³ positive for compression, and forces have the standard `F=-dE/dR` sign. The pressure contribution uses `dV=V tr(A)`. Each Jacobian column is substituted directly into this identity; root translations/rotations/torsions have `dL=0`, and cell columns include the nonaffine force correction. The implementation returns the existing `VCEvaluation`, keeping full physical forces/stress separately from the generalized gradient.

`evaluate(q, callback, pressure=0)` expects `callback(atoms)->(E,F,full_3x3_stress)`. `ASEStressSurface` is an existing counted adapter for ASE calculators with stress. The module itself performs no hidden oracle calls, projections, quenches or reference resets.

## Verification and limits

`tests/standalone/test_rc_vc_geometry.py` has four checks:

- Two nonlinear Cu3 bodies in a nonorthogonal periodic cell: all 15 coordinate derivatives of positions/cell agree with central differences at finite root rotations and nonzero full strain; all internal distances stay rigid.
- On that same geometry, actual ASE EMT E/F/stress gives all 15 E+pV derivatives within 3e-7 eV/Å of central differences at positive pressure (31 combined E/F/stress calls per invocation). Stress-only cell gradients demonstrably differ from the exact answer, so the nonaffine correction is tested nontrivially.
- A shared-joint Cu4 articulated chain retains both bodies' internal distances while undergoing finite torsion, rotation and strain. All ten actual EMT enthalpy derivatives at negative pressure match within 3e-7 eV/Å (21 calls per invocation).
- Lifted coordinates outside the primitive cell are preserved at zero coordinates; nonperiodic input is rejected.

The complete single-chain/forest/VC geometry selection runs **15 passed**, with existing ASE/NumPy deprecation warnings only. EMT here is a consistent differentiable atomistic oracle for mathematical checking; these artificial rigid Cu fragments are not a model of molecular-crystal chemistry. No walk, minima discovery, molecular-crystal performance, Hessian stability or MACE calculation was performed.

The archived RC paper §2.3 and native symbols motivate preserving rigid interiors under cell changes, but the native procedure involves Kabsch fitting and its full coordinate/force/lambda contract remains unclosed. This center-affine exact map is an explicitly independent replacement, not that native algorithm. It has no inverse packing, image-lift parser, loop closure, linear-root support or explicit angle-axis rank-loss rejection yet. A future biased trajectory must freeze this entire chart while Gaussian history is live; changing reference/lift mid-proposal would alter the objective. A future final-quench policy must separately declare whether it releases the rigid constraint and permits cell relaxation; neither is provided by this geometry module.
