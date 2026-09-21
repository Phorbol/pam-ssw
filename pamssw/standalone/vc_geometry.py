"""Explicit 3N+6 coordinates and exact E+pV gradient for variable-cell research.

Row convention H=H0 exp(S), R=X exp(S), S symmetric. q=(vec(X), L*s6),
with an orthonormal symmetric basis (xx,yy,zz,yz,xz,xy). No synthetic atoms,
cell rotations, wrapping, quenching or SSW policy are introduced here.
See docs/research/vc-logstrain-chart.md for the derivation and metric caveat.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, expm_frechet


SYMMETRIC_BASIS = np.zeros((6, 3, 3))
for _i in range(3):
    SYMMETRIC_BASIS[_i, _i, _i] = 1.
for _i, (_a, _b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
    SYMMETRIC_BASIS[_i, _a, _b] = SYMMETRIC_BASIS[_i, _b, _a] = 1. / np.sqrt(2.)
SYMMETRIC_BASIS.setflags(write=False)


def _valid_atoms(atoms):
    if not len(atoms) or not atoms.pbc.all() or atoms.constraints:
        raise ValueError('VC chart requires nonempty unconstrained fully periodic atoms')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('finite positions and cell required')
    if np.linalg.det(atoms.cell.array) <= 0:
        raise ValueError('positive reference/current cell determinant required')


@dataclass(frozen=True)
class VCEvaluation:
    objective: float  # eV: energy + pressure*volume
    gradient: np.ndarray  # eV/Angstrom, derivative with respect to packed q
    atoms: object
    energy: float
    forces: np.ndarray  # Physical Cartesian eV/Angstrom; not projected
    stress: np.ndarray  # Physical ASE tensile-positive eV/Angstrom^3
    volume: float


class ASEStressSurface:
    """Serial ASE E/F/stress oracle with explicit objective convention and cost.

    `evaluate` returns energy, Cartesian forces, full 3x3 stress. One request is
    one combined API evaluation, not one SCF iteration. Failed requests count;
    missing calculator stress propagates. Own a separate calculator per walker.
    """
    def __init__(self, calculator, *, force_consistent=False):
        if calculator is None:
            raise ValueError('ASE calculator required')
        self.calculator = calculator
        self.force_consistent = bool(force_consistent)
        self.requests = 0

    def evaluate(self, atoms):
        _valid_atoms(atoms)
        trial = atoms.copy()
        trial.calc = self.calculator
        self.requests += 1
        energy = float(trial.get_potential_energy(force_consistent=self.force_consistent))
        forces = np.array(trial.get_forces(), copy=True)
        stress = np.array(trial.get_stress(voigt=False), copy=True)
        if (not np.isfinite(energy) or forces.shape != trial.positions.shape
                or stress.shape != (3, 3) or not np.isfinite(forces).all()
                or not np.isfinite(stress).all()):
            raise ValueError('invalid calculator energy, forces or stress')
        return energy, forces, stress


class SymmetricLogStrainChart:
    """Fixed reference chart with explicit atomic/strain length metric.

    strain_length L is in Angstrom and MUST be supplied. It sets relative
    atomic/cell search geometry, not a change in physical energy. No universal
    optimal L is asserted. A frame remains fixed for an entire biased path.
    pack rejects cell rotation relative to H0 instead of silently rotating its
    atoms or altering stored Gaussian directions/centers.
    """
    def __init__(self, atoms, *, strain_length):
        _valid_atoms(atoms)
        if not np.isscalar(strain_length) or not np.isfinite(strain_length) or strain_length <= 0:
            raise ValueError('strain_length must be finite and positive in Angstrom')
        self.strain_length = float(strain_length)
        self.reference = atoms.copy()
        self.reference.calc = None
        self.reference_cell = atoms.cell.array.copy()
        self.reference_cell.setflags(write=False)
        self.natoms = len(atoms)
        self.ndof = 3 * self.natoms + 6

    def _q(self, q):
        value = np.asarray(q, dtype=float)
        if value.shape != (self.ndof,) or not np.isfinite(value).all():
            raise ValueError('q must be finite flat 3N+6 coordinates')
        return value

    def split(self, q):
        value = self._q(q)
        x = value[:-6].reshape(self.natoms, 3)
        strain = np.einsum('i,ijk->jk', value[-6:] / self.strain_length, SYMMETRIC_BASIS)
        return x.copy(), strain

    def pack(self, atoms):
        _valid_atoms(atoms)
        if not np.array_equal(atoms.numbers, self.reference.numbers):
            raise ValueError('ordered species must match the reference')
        deformation = np.linalg.solve(self.reference_cell, atoms.cell.array)
        tolerance = 64 * np.finfo(float).eps * max(1., np.linalg.norm(deformation))
        if np.linalg.norm(deformation - deformation.T) > tolerance:
            raise ValueError('cell is outside symmetric log-strain chart (rotation/shear convention)')
        deformation = .5 * (deformation + deformation.T)
        values, vectors = np.linalg.eigh(deformation)
        if np.any(values <= 0):
            raise ValueError('deformation must be symmetric positive definite')
        strain = (vectors * np.log(values)) @ vectors.T
        x = np.linalg.solve(deformation.T, atoms.positions.T).T
        return np.concatenate((x.ravel(), self.strain_length * np.einsum('ijk,jk->i', SYMMETRIC_BASIS, strain)))

    def unpack(self, q):
        x, strain = self.split(q)
        deformation = expm(strain)
        trial = self.reference.copy()
        trial.cell = self.reference_cell @ deformation
        trial.positions = x @ deformation
        _valid_atoms(trial)
        return trial

    def project(self, vector):
        """Remove atomic uniform translations only; all six strain DOF survive.

        The caller must establish global translation invariance of the PES.
        This operation is not silently applied by evaluate: full objective
        gradients remain available and the direction solver selects its domain.
        """
        result = self._q(vector).copy()
        atomic = result[:-6].reshape(self.natoms, 3)
        atomic -= atomic.mean(axis=0)
        return result

    def evaluate(self, q, evaluate, *, pressure=0.):
        """Return objective and full gradient from callback(atoms)->(E,F,stress).

        Pressure is scalar in eV/Angstrom^3, positive for compression. Stress
        must be ASE tensile-positive full symmetric 3x3. No bias is added here.
        E/F/stress must differentiate the same physical objective.
        """
        if not np.isscalar(pressure) or not np.isfinite(pressure):
            raise ValueError('pressure must be a finite scalar in eV/Angstrom^3')
        x, strain = self.split(q)
        deformation = expm(strain)
        trial = self.unpack(q)
        energy, forces, stress = evaluate(trial)
        energy = float(energy)
        forces, stress = np.asarray(forces, dtype=float), np.asarray(stress, dtype=float)
        if (not np.isfinite(energy) or forces.shape != x.shape or stress.shape != (3, 3)
                or not np.isfinite(forces).all() or not np.isfinite(stress).all()):
            raise ValueError('invalid oracle energy, forces or full stress')
        tolerance = 64 * np.finfo(float).eps * max(1., np.linalg.norm(stress))
        if np.linalg.norm(stress - stress.T) > tolerance:
            raise ValueError('symmetric ASE stress required')
        volume = trial.get_volume()
        gx = -forces @ deformation.T
        gf = np.linalg.solve(deformation.T, volume * (stress + pressure * np.eye(3)))
        # Adjoint Dexp(S)^*[G] = Dexp(S.T)[G] in the Frobenius metric.
        gs = expm_frechet(strain.T, gf, compute_expm=False)
        gcell = np.einsum('ijk,jk->i', SYMMETRIC_BASIS, gs) / self.strain_length
        gradient = np.concatenate((gx.ravel(), gcell))
        objective = energy + float(pressure) * volume
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise ValueError('nonfinite transformed objective or gradient')
        return VCEvaluation(objective, gradient, trial, energy, forces.copy(), stress.copy(), volume)
