# Independent dimer rotation for the SSW paper path

Status: experimental numerical solver, not a complete CBD implementation or
scientific search-efficiency result. Module: `pamssw/standalone/dimer.py`.

## Scientific question and source boundary

Can a direction solver using local plane rotations replace the growing Ritz
subspace while keeping the SSW anchor bias, surface, Cartesian metric, separation
and residual criterion fixed? Native BRZERO4 recovery need not block this test.
We seek a stationary low-curvature direction for nonperiodic unconstrained atoms,
not a TS, equilibrium sample or certificate of global minimum curvature.

Source checked 2026-09-09: ASE official `DimerEigenmodeSearch` source,
https://docs.ase-lib.org/_modules/ase/mep/dimer.html, specifically
`converge_to_eigenmode`. It rotates in the orientation/rotational-force plane,
fits angular curvature and distinguishes a minimum from a maximum. Its source
lists Henkelman and Jonsson, JCP 111, 7010 (1999), and later dimer improvements.
The current implementation does **not** copy ASE's trial-angle Fourier fit,
force extrapolation, tolerances or translations. It is an independently derived
plane-minimizing dimer variant, not exact reproduction of that implementation.

BP-CBD 2012, DOI 10.1021/ct300250h, eqs4–9 (uploaded full text reviewed in
`bp-cbd-upload-review.md`), establishes the rotation-only quadratic anchor bias.
CBD 2010, DOI 10.1021/ct9005147, sec2.1 explicitly cautions that rotation
stationarity need not select the global lowest mode. These source conclusions
apply to our interpretation, not a claim of native iteration parity.

## Local derivation and finite-separation approximation

Normalize the original anchor once, n0. For unit orientation n, evaluate

    A(n) = [F(R)-F(R+h*n)]/h - a*(n0·n)*n0
    c = n·A(n)
    r = A(n)-c*n.

For a quadratic PES, A is the symmetric linear operator H-a*n0*n0.T and r is
half the constrained Rayleigh gradient. For nonzero residual, set t=-r/||r||,
remove floating-point leakage along n, and sample A(t). Construct the matrix of
A restricted to span(n,t), symmetrize it, and choose its lower eigenvector.
This is algebraically the minimum of

    c(theta)=C00*cos(theta)^2+2*C01*cos(theta)*sin(theta)+C11*sin(theta)^2.

No trial-angle or learning-rate parameter is needed. Only this two-dimensional
plane is retained between iterations. Orient the result toward n0, then **always
request its actual endpoint force again**. Never accept an interpolated residual.
If the remaining budget cannot cover both the tangent probe and final fresh
check, return the current directly evaluated direction as unconverged unless its
residual is already small.

At finite h on an anharmonic PES, A(n) is only approximately linear; changing
n's sign also changes one-sided secant error. The projected antisymmetry is
reported (largest Frobenius norm over sampled planes), and the actual final
residual determines convergence. This does not guarantee monotonically decreasing
true finite-separation curvature, nor global lowest-mode convergence. No random
restart, clamp, alternate backend, conjugate history or noise-masking fallback
is introduced. Persistent residual noise is a failed rotation for the outer
walker, not permission to relax the threshold silently.

## Parameter and cost contract

- h / `fd_step`: Angstrom, numerical force-difference separation, truncation
  O(h). Needs backend force-noise and h sensitivity checks.
- a / `rotation_bias`: eV/Angstrom², nonnegative search-direction curvature
  bias, explicit caller value. No newly claimed universal optimum.
- `tol`: eV/Angstrom², norm of tangent secant residual. In BP-CBD's linearized
  one-sided convention, its force residual is 2*h times this value.
- `max_hvp`: positive integer number of permitted endpoint evaluations. One
  center request + one initial endpoint + two endpoints per completed rotation.
  At most 1+max_hvp calls. Even budgets may leave one unused endpoint allowance.

User atoms/anchor remain unchanged; no overall translation/rotation projection,
PBC or constraints. Returning a small residual at an initial excited eigenvector
is allowed and deliberately tested: convergence is not a lowest-spectrum proof.

## Validation and keep/delete decision

Numerical tests compare the biased solution with an independently assembled
dense Hessian, verify Cartesian rotation equivariance, every small force budget,
nonlinear direct terminal residual, and the excited-eigenvector limitation.
They are formula/interface checks, not scientific validation. Test-first initial
run failed on missing module; after implementation all ten checks passed.

The parent experiment should compare against existing Ritz on identical real
geometries, anchors, PES precision, h, a and call budgets, then paired full SSW/LS
escapes with actual total costs and independently verified endpoints. Keep this
backend experimental until it supplies useful reliable escapes per cost across
relevant systems. Repeated failures or no demonstrated advantage justify leaving
it optional or deleting it; no extra heuristic should be added just to rescue a
single example. Native CBD root selection is a separate comparator.
