"""Research-only, approximate nonperiodic geometry matching for the PAM pool."""
import numpy as np
from ase import Atoms
from ase.geometry import distance

from pamssw.archive import MinimaArchive


class ASEPermutationArchive(MinimaArchive):
    """Reuse archive bookkeeping with ASE's heuristic permutation RMSD.

    A returned small distance witnesses one alignment, not the global optimum
    over rotations and atom assignments. Stored states are never reordered.
    """

    @staticmethod
    def validate_state(state):
        if any(state.pbc) or np.any(state.fixed_mask):
            raise ValueError('ASE pool identity requires nonperiodic unconstrained states')

    def find_match(self, state, energy):
        self.validate_state(state)
        return super().find_match(state, energy)

    @staticmethod
    def _rmsd(lhs, rhs):
        if lhs.n_atoms != rhs.n_atoms or not np.array_equal(
                np.sort(lhs.numbers), np.sort(rhs.numbers)):
            return float('inf')
        # For one atom, removing translation leaves no internal geometry.
        if lhs.n_atoms <= 1:
            return 0.
        left = Atoms(numbers=lhs.numbers, positions=lhs.positions)
        right = Atoms(numbers=rhs.numbers, positions=rhs.positions)
        return float(distance(left, right, permute=True) / np.sqrt(lhs.n_atoms))
