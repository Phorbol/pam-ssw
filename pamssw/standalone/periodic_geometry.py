"""Fixed-cell periodic translation quotient for Cartesian SSW directions.

Experimental geometry, not variable-cell SSW or native LASP parity. The caller
must declare global translation invariance of its oracle; periodic boundary
conditions alone do not imply that an external field is translation invariant.
No lattice images are selected here. The walk is a continuous Cartesian lift,
so stored Gaussian centers, directions and optimizer coordinates stay unwrapped.
ASE coordinates/PBC semantics: https://docs.ase-lib.org/ase/atoms.html.
"""
import numpy as np


class FixedCellTranslationFrame:
    """X(z)=X0+P(z-X0), P v = v-mean_atoms(v), in the Euclidean metric.

    P is symmetric/idempotent and removes precisely the three uniform
    translations. P retains rotational displacements: at fixed periodic cell,
    rotating only atoms is generally a physical deformation, not a symmetry.
    Neither masses nor a fitted tolerance enter the projector. The affine
    section has no orientation-chart singularity, unlike cluster Eckart frames.
    Use this for direction/HVP evaluation; ordinary true/biased quenches may
    remain unrestricted Cartesian. Complete gradients must match the evaluated
    objective; `evaluate` provides the corresponding exact pullback.
    """
    def __init__(self, atoms):
        if len(atoms) < 2 or not atoms.pbc.any() or atoms.constraints:
            raise ValueError('periodic translation frame needs >=2 unconstrained periodic atoms')
        self.reference = atoms.positions.copy()
        self.cell = atoms.cell.array.copy()
        self.pbc = atoms.pbc.copy()
        self.numbers = atoms.numbers.copy()
        if not np.isfinite(self.reference).all() or not np.isfinite(self.cell).all():
            raise ValueError('finite periodic positions and cell required')
        if np.linalg.matrix_rank(self.cell) != 3:
            raise ValueError('periodic translation frame requires a full-rank fixed cell')
        for array in (self.reference, self.cell, self.pbc, self.numbers):
            array.setflags(write=False)

    def project(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != self.reference.shape or not np.isfinite(vector).all():
            raise ValueError('finite frame-shaped vector required')
        return vector - vector.mean(axis=0)

    def positions(self, candidate):
        return self.reference + self.project(np.asarray(candidate) - self.reference)

    def validate(self, atoms):
        """Reject changed state-space metadata before any calculator request."""
        if (atoms.constraints or not np.array_equal(atoms.cell.array, self.cell)
                or not np.array_equal(atoms.pbc, self.pbc)
                or not np.array_equal(atoms.numbers, self.numbers)):
            raise ValueError('translation frame requires unchanged cell, PBC, ordered species and constraints')
        if atoms.positions.shape != self.reference.shape or not np.isfinite(atoms.positions).all():
            raise ValueError('finite frame-shaped coordinates required')

    def evaluate(self, atoms, evaluate):
        """Evaluate complete E(X(z)), return complete force P F(X(z)).

        `evaluate` must include every term whose derivative is requested. The
        input Atoms and evaluator are not mutated; ASE calculator ownership is
        the caller's responsibility. This method never wraps trial positions.
        """
        self.validate(atoms)
        trial = atoms.copy()
        trial.positions = self.positions(atoms.positions)
        energy, forces = evaluate(trial)
        if not np.isfinite(energy):
            raise ValueError('evaluator returned nonfinite energy')
        return float(energy), self.project(forces)


def global_translation_free_direction(atoms, rng):
    """Explicit global Maxwell direction projected into fixed-cell internal DOF.

    Draw Cartesian normals / sqrt(mass), project with the Euclidean P, then
    normalize. This is NOT the paper's local-pair mixture. No MIC pair selection,
    silent fallback or rotation removal is included. A one-atom primitive cell
    has no nontranslation atomic DOF and is rejected; use a physical supercell.
    """
    frame = FixedCellTranslationFrame(atoms)
    masses = atoms.get_masses()
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('finite positive masses required')
    direction = frame.project(rng.normal(size=frame.reference.shape) / np.sqrt(masses[:, None]))
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('sampled direction has no finite nontranslation component')
    return direction / norm
