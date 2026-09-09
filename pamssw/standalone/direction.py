"""Experimental reference soft-mode solver, independent of PAM policies.

The native SSW evidence establishes biased/unbiased dimer rotation, not this
solver's equivalence. This implementation instead minimizes a Rayleigh quotient
in a bounded Krylov subspace using symmetric Ritz projection. General symmetric
operator background: Lehoucq, Sorensen & Yang, ARPACK Users Guide (1998), also
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html.
It is a fully reorthogonalized, unrestarted reference solver, not ARPACK itself.
No production or scientific search-efficiency validation is implied.
"""
from dataclasses import dataclass
import numpy as np
from ase.constraints import FixAtoms


@dataclass(frozen=True)
class SoftModeResult:
    direction: np.ndarray
    curvature: float
    residual_norm: float
    hvp_calls: int
    force_calls: int
    converged: bool
    projected_symmetry_error: float


def reference_soft_mode(atoms, initial_direction, *, fd_step, max_hvp,
                        residual_tol, active_mask=None, evaluate=None,
                        finite_difference="central"):
    """Return a low-curvature direction with an explicit finite force budget.

    ``evaluate(atoms) -> (energy, forces)`` uses eV and eV/Angstrom; default
    evaluation uses the attached ASE calculator. Fixed-cell Cartesian metric:
    H v = -[F(R+h*v)-F(R-h*v)]/(2*h) for unit v. fd_step is Angstrom,
    residual_tol eV/Angstrom²; neither has a universal fitted default. Caller
    must establish h sensitivity against force noise and anharmonic truncation.
    At most max_hvp HVPs (2*max_hvp force requests) are performed, with one HVP
    reserved to measure the returned vector's actual finite-difference residual.

    finite_difference="forward" instead uses a shared center force and one
    endpoint per HVP (1+max_hvp force requests, first-order truncation).

    Only FixAtoms and an optional boolean (N,3) active_mask are supported.
    Translations/rotations are not silently projected out; external potentials
    may break those symmetries. Initial direction determines the reachable
    subspace: convergence to an eigenvector does not certify the global lowest
    eigenvalue, physical stability, or native LASP dimer behavior. The returned
    residual is for the finite-difference operator, not an exact Hessian.
    """
    if not np.isfinite(fd_step) or fd_step<=0:
        raise ValueError('fd_step must be finite and positive')
    if not np.isfinite(residual_tol) or residual_tol<=0:
        raise ValueError('residual_tol must be finite and positive')
    if isinstance(max_hvp,bool) or int(max_hvp)!=max_hvp or max_hvp<2:
        raise ValueError('max_hvp must be an integer of at least 2')
    shape=atoms.positions.shape
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError('requires finite nonempty atom positions')
    mask=np.ones(shape,dtype=bool)
    if active_mask is not None:
        supplied=np.asarray(active_mask)
        if supplied.shape!=shape or supplied.dtype!=bool:
            raise ValueError('active_mask must be boolean with shape (N, 3)')
        mask &= supplied
    for constraint in atoms.constraints:
        if not isinstance(constraint,FixAtoms):
            raise ValueError('only FixAtoms constraints are supported')
        mask[constraint.get_indices()]=False
    initial=np.asarray(initial_direction,dtype=float)
    if initial.shape!=shape or not np.isfinite(initial).all():
        raise ValueError('initial_direction requires finite shape (N, 3)')
    initial=(initial*mask).ravel()
    norm=np.linalg.norm(initial)
    if norm==0 or not np.isfinite(norm):
        raise ValueError('initial_direction must have nonzero active norm')
    initial=initial/norm
    if evaluate is None:
        if atoms.calc is None:
            raise ValueError('ASE calculator or evaluate callback required')
        def evaluate(candidate):
            return candidate.get_potential_energy(), candidate.get_forces(apply_constraint=False)
    if finite_difference not in ("central", "forward"):
        raise ValueError("finite_difference must be central or forward")
    force_calls=0
    def checked_force(trial):
        nonlocal force_calls
        energy,force=evaluate(trial)
        force_calls+=1
        force=np.asarray(force,dtype=float)
        if force.shape!=shape or not np.isfinite(force).all() or not np.isfinite(energy):
            raise ValueError("evaluator returned invalid energy/forces")
        return force.copy()
    base=atoms.copy();base.calc=atoms.calc
    center_force=checked_force(base) if finite_difference=="forward" else None
    hvp_calls=0
    def hvp(vector):
        nonlocal hvp_calls
        forces=[]
        for sign in ((1.,-1.) if finite_difference=="central" else (1.,)):
            trial=atoms.copy();trial.calc=atoms.calc
            trial.set_positions(atoms.positions+sign*fd_step*vector.reshape(shape),apply_constraint=False)
            forces.append(checked_force(trial))
        hvp_calls+=1
        other=forces[1] if finite_difference=="central" else center_force
        denominator=2*fd_step if finite_difference=="central" else fd_step
        return (-(forces[0]-other)/denominator*mask).ravel()

    basis=[];images=[];vector=initial.copy();symmetry_error=0.
    for _ in range(min(int(max_hvp)-1,int(mask.sum()))):
        basis.append(vector.copy());images.append(hvp(vector))
        q=np.column_stack(basis);hq=np.column_stack(images)
        projected=q.T@hq
        symmetry_error=float(np.linalg.norm(projected-projected.T))
        values,coefficients=np.linalg.eigh((projected+projected.T)/2)
        direction=q@coefficients[:,0]
        surrogate_residual=hq@coefficients[:,0]-values[0]*direction
        if np.linalg.norm(surrogate_residual)<=residual_tol:
            break
        # Twice modified Gram-Schmidt controls loss of orthogonality.
        vector=images[-1].copy()
        for _ in range(2):
            for previous in basis:
                vector-=np.dot(previous,vector)*previous
        norm=np.linalg.norm(vector)
        if norm<=np.finfo(float).eps*max(1.,np.linalg.norm(images[-1])):
            break
        vector/=norm
    direction/=np.linalg.norm(direction)
    if np.dot(direction,initial)<0:
        direction=-direction
    hd=hvp(direction)
    curvature=float(direction@hd)
    residual=float(np.linalg.norm(hd-curvature*direction))
    return SoftModeResult(direction.reshape(shape),curvature,residual,hvp_calls,
                          force_calls,residual<=residual_tol,symmetry_error)


def paper_biased_direction(atoms, anchor, *, rotation_bias, fd_step, max_hvp,
                           tol, evaluate):
    """SSW2013 biased direction with an explicit alternative numerical solver.

    Shang & Liu, JCTC 2013 9,1838, DOI 10.1021/ct301010b, equations 3-6
    and Overall Algorithm step 2. ``anchor`` is the ORIGINAL random direction
    for this MC move, reused at each Gaussian insertion, normalized by caller.
    ``evaluate`` is the real PES (or explicitly declared modified surface);
    the rotation-only quadratic bias is added here, not deposited Gaussians.
    rotation_bias=a is an explicit nonnegative curvature in eV/Angstrom².
    The paper supplies no universal a; the caller must choose and record it.

    One center force plus one endpoint force per HVP implements the one-sided
    dimer secant. A symmetric Ritz solve replaces the paper's Broyden dimer
    iterations. Finite separation makes this operator only approximately
    linear/symmetric; projected_symmetry_error and final direct residual
    expose that approximation. This is paper-level numerical substitution,
    NOT native rotation parity. Sign follows anchor; inaccessible subspaces
    are not searched. tol uses curvature units, not native rotation ftol.
    No energy or force calls beyond 1+max_hvp; no cell degrees of freedom.
    """
    from .native_rotation import RotationQuadraticBias
    bias=RotationQuadraticBias(atoms.positions,anchor,rotation_bias)
    def biased(trial):
        energy,force=evaluate(trial)
        be,bf=bias.evaluate(trial)
        return energy+be,np.asarray(force)+bf
    return reference_soft_mode(atoms,anchor,fd_step=fd_step,max_hvp=max_hvp,
                               residual_tol=tol,evaluate=biased,
                               finite_difference="forward")
