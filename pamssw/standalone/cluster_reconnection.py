"""Experimental isolated-cluster reconnection geometry.

This is a calculator-free translation of the recovered finite-coordinate
routine.  It is intentionally optional; the paper SSW entry point can invoke
it only at its landing boundary. Distances are Cartesian Angstroms; the quadratic distance table is
the reference implementation's O(N^2) storage and can lead to O(N^3) work in
the sequential repair path.
"""
from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class ClusterTranslation:
    """One component translation selected by the reconnection procedure."""

    members: np.ndarray
    separation: float
    translation: np.ndarray


@dataclass(frozen=True)
class ClusterReconnectionResult:
    """Copy of the input and the finite geometric procedure's trace."""

    atoms: Atoms
    medoid: int
    initial_component: np.ndarray
    moves: tuple[ClusterTranslation, ...]
    last_separation: float


def reconnect_clusters(atoms: Atoms, criterion: float, *, repair: bool = False) -> ClusterReconnectionResult:
    """Optionally translate disconnected components toward the medoid component.

    ``atoms`` must be an isolated, unconstrained, nonperiodic structure with at
    least two finite Cartesian positions.  The input is never modified and no
    calculator is accessed.  With ``repair=False`` the returned copy is
    unchanged and ``last_separation`` is the first attachment distance; with
    ``repair=True`` it is the last pretranslation attachment distance.  The
    native displacement fraction is retained as the empirical ``0.7`` rule.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError('atoms must be an ASE Atoms object')
    if len(atoms) < 2:
        raise ValueError('at least two finite isolated Cartesian positions required')
    if bool(np.asarray(atoms.pbc).any()) or atoms.constraints:
        raise ValueError('cluster reconnection requires nonperiodic unconstrained atoms')
    x = np.array(atoms.positions, dtype=float, copy=True)
    if x.shape != (len(atoms), 3) or not np.isfinite(x).all():
        raise ValueError('at least two finite isolated Cartesian positions required')
    if isinstance(criterion, (bool, np.bool_)):
        raise ValueError('positive finite distance criterion required')
    try:
        criterion = float(criterion)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('positive finite distance criterion required') from exc
    if not np.isfinite(criterion) or criterion <= 0:
        raise ValueError('positive finite distance criterion required')
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    scores = distances[:, ::2].sum(axis=1) + distances[:, 1::2].sum(axis=1)
    medoid = int(np.argmin(scores))
    order = list(range(len(x)))
    order[0], order[medoid] = order[medoid], order[0]

    def expand(start: int, end: int) -> int:
        cursor = start
        while cursor < end:
            atom = order[cursor]
            for j in range(end, len(x)):
                if np.linalg.norm(x[atom] - x[order[j]]) < criterion:
                    order[end], order[j] = order[j], order[end]
                    end += 1
            cursor += 1
        return end

    end = expand(0, 1)
    initial_component = np.asarray(order[:end], dtype=int)
    moves = []
    last = 0.0
    while end < len(x):
        best = float('inf')
        chosen = None
        displacement = None
        for j in range(end, len(x)):
            for i in range(end):
                vector = x[order[i]] - x[order[j]]
                distance = float(np.linalg.norm(vector))
                if distance < best:
                    best, chosen, displacement = distance, j, vector
        last = best
        if not repair:
            break
        order[end], order[chosen] = order[chosen], order[end]
        new_end = expand(end, end + 1)
        translation = displacement * ((best - 0.7 * criterion) / best)
        members = np.asarray(order[end:new_end], dtype=int)
        x[members] += translation
        moves.append(ClusterTranslation(members.copy(), best, np.asarray(translation, dtype=float).copy()))
        end = new_end
    result_atoms = atoms.copy()
    result_atoms.set_positions(x)
    return ClusterReconnectionResult(result_atoms, medoid, initial_component.copy(), tuple(moves), float(last))
