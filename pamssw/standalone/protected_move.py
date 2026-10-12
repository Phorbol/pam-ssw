"""Opt-in scalar backtracking for fixed-cell SSW Gaussian moves.

The thresholds reproduce the recovered native scalar guards as explicit
heuristics. The periodic distance calculation is an ASE geometry choice, not
a claim of native image-enumeration parity.
"""
from dataclasses import dataclass
import numbers

import numpy as np
from ase.geometry import find_mic, minkowski_reduce


@dataclass(frozen=True)
class ScalarMovePolicy:
    """Native-derived scalar retry values (Angstrom where dimensional)."""
    displacement_max: float = 2.0
    contraction_max: float = 0.25
    shrink_factor: float = 0.95
    minimum_width: float = 0.1
    max_trials: int = 50

    def __post_init__(self):
        vals = (self.displacement_max, self.contraction_max,
                self.shrink_factor, self.minimum_width)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
               for x in vals) or not all(np.isfinite(x) for x in vals):
            raise ValueError('move policy values must be finite')
        if self.displacement_max <= 0 or self.minimum_width <= 0:
            raise ValueError('displacement_max and minimum_width must be positive')
        if not 0 <= self.contraction_max < 1:
            raise ValueError('contraction_max must be in [0, 1)')
        if not 0 < self.shrink_factor < 1:
            raise ValueError('shrink_factor must be in (0, 1)')
        if (isinstance(self.max_trials, bool) or
                not isinstance(self.max_trials, (int, np.integer)) or self.max_trials < 1):
            raise ValueError('max_trials must be a positive integer')


@dataclass(frozen=True)
class ProtectedMoveResult:
    accepted: bool
    atoms: object
    direction: np.ndarray
    width: float
    diagnostics: dict


def _minimum_separation(atoms):
    """Minimum distinct-atom MIC distance and shortest nonzero periodic image."""
    positions = np.asarray(atoms.positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError('move geometry must have finite positions')
    best = np.inf
    if len(atoms) > 1:
        raw = positions[:, None, :] - positions[None, :, :]
        mic, _ = find_mic(raw.reshape((-1, 3)), atoms.cell, pbc=atoms.pbc)
        distances = np.linalg.norm(mic.reshape(raw.shape), axis=-1)
        upper = distances[np.triu_indices(len(atoms), 1)]
        if upper.size:
            best = float(np.min(upper))
    if np.any(atoms.pbc):
        cell = np.asarray(atoms.cell.array, dtype=float)
        if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) == 0.0:
            raise ValueError('periodic move geometry requires a finite nonsingular cell')
        reduced, _ = minkowski_reduce(atoms.cell, pbc=atoms.pbc)
        vectors = np.asarray(reduced)[np.asarray(atoms.pbc, dtype=bool)]
        if len(vectors):
            best = min(best, float(np.min(np.linalg.norm(vectors, axis=1))))
    return best


def validate_move_geometry(atoms):
    """Fail early for geometry outside the protected fixed-cell domain."""
    if atoms.constraints:
        raise NotImplementedError('protected moves require unconstrained atoms')
    distance = _minimum_separation(atoms)
    if distance <= 0.0:
        raise ValueError('protected move requires positive initial minimum separation')
    return distance


def protected_move(atoms, direction, width, policy):
    """Return the first scalar-guarded trial and its actual direction/width.

    At most ``max_trials`` trial geometries are considered. The recovered
    native loop evaluates 50 trials total; the 0.1 A floor is checked only
    after a rejection, so the initially requested width is always considered.
    No calculator or random state is accessed.
    """
    if not isinstance(policy, ScalarMovePolicy):
        raise TypeError('policy must be ScalarMovePolicy')
    d = np.asarray(direction, dtype=float)
    if d.shape != atoms.positions.shape or not np.isfinite(d).all():
        raise ValueError('direction must be finite and match the atom geometry')
    if (isinstance(width, (bool, np.bool_)) or
            not isinstance(width, numbers.Real) or not np.isfinite(width) or width <= 0):
        raise ValueError('width must be finite and positive')
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError('direction must be nonzero')
    before = validate_move_geometry(atoms)
    trial_width = float(width)
    attempts = []
    for retry in range(int(policy.max_trials)):
        displacement = trial_width * d
        candidate = atoms.copy()
        candidate.positions = atoms.positions + displacement
        actual = candidate.positions - atoms.positions
        max_atom = float(np.max(np.linalg.norm(actual, axis=1)))
        after = _minimum_separation(candidate)
        short = (np.isfinite(before) and np.isfinite(after) and
                 before * (1.0 - policy.contraction_max) > after)
        accepted = max_atom <= policy.displacement_max and not short
        attempts.append({'retry': retry, 'width': trial_width,
                         'max_atom_displacement': max_atom,
                         'minimum_separation_before': before,
                         'minimum_separation_after': after,
                         'accepted': bool(accepted)})
        if accepted:
            actual_width = float(np.linalg.norm(actual))
            if actual_width == 0.0:
                raise ValueError('trial displacement is not representable')
            return ProtectedMoveResult(True, candidate,
                actual / actual_width, actual_width,
                {'attempts': tuple(attempts), 'retry_count': retry,
                 'requested_width': float(width), 'actual_width': actual_width})
        next_width = trial_width * policy.shrink_factor
        if retry + 1 >= policy.max_trials or next_width < policy.minimum_width:
            break
        trial_width = next_width
    return ProtectedMoveResult(False, atoms.copy(), np.zeros_like(d), 0.0,
        {'attempts': tuple(attempts), 'retry_count': len(attempts) - 1,
         'requested_width': float(width), 'actual_width': 0.0,
         'reason': 'move_retry_exhausted'})
