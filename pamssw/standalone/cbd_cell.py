"""2014 CBD-cell geometry and bounded unbiased cell-mode direction.

ASE row lattice L, fixed unwrapped fractional positions s, R=sL. The objective
is E+pV and its full lattice gradient is V L^-T (stress+pI), where stress is
ASE tensile-positive. Cell rotations L A (A antisymmetric) are projected out
only for direction search, using one projector fixed at the dimer center.
This is a cell block, not the joint log-strain walker or an outer SSW scheme.

Paper: Shang, Zhang, Liu, PCCP 2014, DOI 10.1039/c4cp01485e, Eqs. (3)-(7).
The returned curvature uses the conventional positive Hessian sign; printed
Eq. (7) has an unresolved sign mismatch with its gradient definition.
"""
import numpy as np
from .generalized_numerics import generalized_dimer
from .vc_geometry import VCEvaluation, _valid_atoms


def _cell(q):
    value = np.asarray(q, dtype=float)
    if value.shape != (9,) or not np.isfinite(value).all():
        raise ValueError('cell coordinates must be nine finite entries in Angstrom')
    lattice = value.reshape(3, 3)
    if np.linalg.det(lattice) <= 0:
        raise ValueError('positive cell determinant required')
    return lattice


def _rotation_basis(lattice):
    tangents = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a = np.zeros((3, 3)); a[i, j] = 1.; a[j, i] = -1.
        tangents.append((lattice @ a).ravel())
    return np.linalg.qr(np.column_stack(tangents), mode='reduced')[0]


class CellChart:
    """Nine Cartesian lattice entries at fixed fractional atom coordinates.

    The chart does not wrap atoms, relax atoms, limit a finite cell step, or
    silently replace a nonpositive determinant. Energies and physical forces
    remain unprojected. Positive external pressure is in eV/Angstrom^3.
    """
    ndof = 9

    def __init__(self, atoms):
        _valid_atoms(atoms)
        self.reference = atoms.copy()
        self.reference.calc = None
        self.reference_cell = atoms.cell.array.copy()
        self.reference_cell.setflags(write=False)
        self.fractional = atoms.get_scaled_positions(wrap=False).copy()
        self.fractional.setflags(write=False)

    def pack(self, atoms):
        _valid_atoms(atoms)
        if not np.array_equal(atoms.numbers, self.reference.numbers):
            raise ValueError('cell chart composition and atom order must remain fixed')
        if not np.allclose(atoms.get_scaled_positions(wrap=False), self.fractional,
                           rtol=0., atol=1e-12):
            raise ValueError('cell chart requires fixed unwrapped fractional positions')
        return atoms.cell.array.ravel().copy()

    def unpack(self, q):
        lattice = _cell(q)
        atoms = self.reference.copy()
        atoms.cell = lattice.copy()
        atoms.positions = self.fractional @ lattice
        _valid_atoms(atoms)
        return atoms

    def project(self, vector, *, center=None):
        """Orthogonal rotation removal at center (default: reference lattice).

        Fix center across HVP evaluations. Recomputing it at displaced images
        differentiates the projector too and changes the intended Hessian.
        """
        value = np.asarray(vector, dtype=float)
        if value.shape != (9,) or not np.isfinite(value).all():
            raise ValueError('vector requires nine finite entries')
        lattice = self.reference_cell if center is None else _cell(center)
        rotation = _rotation_basis(lattice)
        return value - rotation @ (rotation.T @ value)

    def evaluate(self, q, evaluate, *, pressure=0.):
        """Callback(atoms)->(E, Cartesian forces, full symmetric ASE stress)."""
        if not np.isscalar(pressure) or not np.isfinite(pressure):
            raise ValueError('pressure must be a finite scalar in eV/Angstrom^3')
        lattice = _cell(q)
        atoms = self.unpack(q)
        energy, forces, stress = evaluate(atoms)
        energy = float(energy)
        forces = np.asarray(forces, dtype=float)
        stress = np.asarray(stress, dtype=float)
        if (not np.isfinite(energy) or forces.shape != atoms.positions.shape
                or stress.shape != (3, 3) or not np.isfinite(forces).all()
                or not np.isfinite(stress).all()):
            raise ValueError('invalid physical energy, forces or full stress')
        tolerance = 64 * np.finfo(float).eps * max(1., np.linalg.norm(stress))
        if np.linalg.norm(stress - stress.T) > tolerance:
            raise ValueError('symmetric ASE stress required')
        volume = atoms.get_volume()
        gradient = np.linalg.solve(lattice.T, volume * (stress + pressure * np.eye(3))).ravel()
        objective = energy + float(pressure) * volume
        if not np.isfinite(gradient).all() or not np.isfinite(objective):
            raise ValueError('nonfinite cell objective or gradient')
        return VCEvaluation(objective, gradient, atoms, energy, forces.copy(), stress.copy(), volume)


def cell_direction(chart, q0, anchor, *, evaluate, pressure=0., fd_step=.005,
                   max_hvp=5, rotation_force_tol=.1):
    """Budget-limited plane-dimer cell direction; no anchor bias.

    Default at most six E/F/stress requests (one center + five HVP probes);
    callers may explicitly supply a larger numerical budget.
    fd_step=.005 Angstrom and Frot tolerance=.1 eV/Angstrom are the paper's
    examples, not optimal defaults. Eq. (5) gives Frot norm = 2*fd_step times
    the gradient-HVP residual norm, hence tol=rotation_force_tol/(2*fd_step).
    Return SoftModeResult including convergence/cost; do not treat an exhausted
    budget as converged. This reuses the Python plane dimer, not native CBD's
    unrecovered complete Broyden update sequence.
    """
    center = _cell(q0).ravel().copy()
    if (isinstance(max_hvp, (bool, np.bool_)) or not isinstance(max_hvp, (int, np.integer))
            or max_hvp < 1):
        raise ValueError('max_hvp must be a positive integer')
    for value, name in ((fd_step, 'fd_step'), (rotation_force_tol, 'rotation_force_tol')):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    raw_anchor = np.asarray(anchor, dtype=float)
    direction = chart.project(raw_anchor, center=center)
    if np.linalg.norm(direction) <= 64 * np.finfo(float).eps * max(1., np.linalg.norm(raw_anchor)):
        raise ValueError('anchor requires a nonzero nonrotational component')
    rotation = _rotation_basis(center.reshape(3, 3))

    def projected_evaluate(q):
        result = chart.evaluate(q, evaluate, pressure=pressure)
        gradient = result.gradient - rotation @ (rotation.T @ result.gradient)
        return result.objective, gradient

    return generalized_dimer(center, direction, rotation_bias=0., fd_step=fd_step,
                             max_hvp=max_hvp, tol=rotation_force_tol / (2 * fd_step),
                             evaluate=projected_evaluate)
