# Symmetric logarithmic strain coordinates for independent VC research

Date: 2026-09-10. This is a geometry/oracle implementation, not yet a VC walker,
quench, original LASP reproduction, or globally validated crystal search.

## Coordinates and objective

ASE uses row atomic coordinates and row cell vectors. Keep one reference H0
fixed throughout a biased trajectory and write

```
S = sum_a s_a B_a,  F = exp(S),  H = H0 F,  R = X F,
q = (vec(X), L s),  objective = E(R,H) + p det(H).
```

B uses ordered components (xx, yy, zz, yz, xz, xy). Offdiagonal matrices have
both symmetric entries 1/sqrt(2), so B_a:B_b=delta_ab. Unlike engineering
Voigt strain this orthonormal basis needs no hidden factor two in the metric.
L is an explicitly supplied positive length in Angstrom. Every q component
then has length units, and every gradient has eV/Angstrom units. Choosing L
sets the relative atom/cell proposal metric; it is not a fitted physical
constant or a universally optimal default. The code requires it explicitly.

Symmetric S makes F positive definite and det(H)>0 for positive det(H0).
There are six strain degrees and no cell rotation coordinate. General cell
orientations outside this fixed chart are rejected by pack, not silently
rotated. Treating orientations as redundant scientifically requires rotational
invariance of the underlying problem. Extremely large strains can still be
numerically singular or physically invalid; exp does not certify a sensible
crystal. Finite/positive-determinant checks reject invalid numerical results.

X and R are unwrapped coordinates. No cell image is chosen by this module.
A path bias and optimizer state must stay on this same continuous lift; cell
basis changes, wrapping or reference resets would require transforming all
stored state. We do not implement such resets.

## Exact gradient in the row convention

At fixed F, dR=dX F, hence G_X=G_R F^T=-forces F^T.

At fixed X and changing F, let A=F^{-1}dF. Then dH=H A and dR=R A.
ASE tensile-positive symmetric stress is the derivative with respect to this
affine strain: dE=V sigma:A. Pressure adds p V I:A. Under the Frobenius
inner product this gives

```
G_F = F^{-T} V (sigma + p I).
```

There is no extra atomic virial correction here: the physical stress already
includes the affine atomic motion at fixed scaled positions. Adding one would
double count that derivative. Atomic gradients are a separate variation of X.

For the matrix exponential, Dexp(S)^*[G]=Dexp(S^T)[G]. Therefore

```
G_S = expm_frechet(S.T, G_F, compute_expm=False)
g_cell[a] = (B_a:G_S)/L.
```

This adjoint formula avoids assuming that S commutes with the stress; that
assumption would fail precisely for finite offdiagonal strain. Symmetric-basis
contraction automatically selects the symmetric variation space without
inventing rotational gradient components.

Cross-check: official [ASE FrechetCellFilter source](https://docs.ase-lib.org/_modules/ase/filters.html)
and the installed ASE utility `get_forces_frechet` use the same deformation
chain rule and Frechet directional contraction in ASE's transposed deformation
convention. Our code has no pseudo-atom representation and does not call an ASE
cell filter. SciPy supplies only 3x3 matrix exponential/Frechet linear algebra.

## Interface

`SymmetricLogStrainChart(atoms, strain_length=L)` provides pack(atoms),
unpack(q), split(q), project(vector), and evaluate(q, callback, pressure=p).
The callback returns physical (E, Cartesian forces, full symmetric stress).
VCEvaluation retains both the physical results and transformed objective/gradient.
`ASEStressSurface` owns one serial calculator and increments requests even if a
calculator property fails. Energy/free_energy choice is explicit; no fallback
or zero stress is substituted for unavailable stress.

The optional project operation only removes the three atomic uniform
translations and retains all six cell components. It is not applied silently
to the reported gradient. A caller using that quotient must ensure translation
invariance, and a single-atom cell then has only the six strain directions.

## Numerical evidence and limits

`tests/standalone/test_vc_geometry.py`: 11 tests passed. The physical oracle is
ASE EMT for Cu4, cubic fcc starting a=3.6 Angstrom, reshaped to triclinic
H0=[[3.6,.18,-.07],[.06,3.7,.13],[-.11,.04,3.55]], then seed12 atomic Gaussian
perturbations of standard deviation .027 Angstrom. Two nonzero symmetric
strain coordinates (including shear), L=1.4 and 5.2 Angstrom, and p=0 and .013
eV/Angstrom^3 yield eight real-oracle combinations. Central differences at
h=2e-6 Angstrom check three atomic and every strain derivative. They pass
2e-7 absolute/relative derivative tolerance. The 72 checked derivatives have
maximum absolute error 4.4545e-9 eV/Angstrom across 152 E/F/stress oracle
requests. These are consistency checks, not
claims of strain search efficiency. Exact tests also cover volume pressure
terms, metric rescaling, positive determinant, unwrapped pack/unpack, absence
of rotation DOF and translation projection.

An end-to-end joint atom/cell escape plus true enthalpy quench, separate physical
force/stress convergence, model-domain diagnostics and cross-system search
comparisons remain necessary before any algorithmic claim.
