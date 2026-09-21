"""Pure geometry recovered from LASP ``localatomgroup_mode``.

The native helper receives an explicitly selected 1-based pair and a raw
integer per atom group mask.  It forms the unnormalised cross product
``(x[k]-x[i]) x (x[k]-x[j])`` for entries whose group value is exactly one,
then applies the Cartesian freedom mask.  Pair selection and group creation
remain caller responsibilities.
"""
from dataclasses import dataclass

import numpy as np

from .native_local_pair import _next_random


# Empirical constants recovered from find_leastmoveatoms_ for Run_type=5.
# They are reproduction parameters, not recommended general search scales.
_NEAR_CENTER_A = 2.0
_OUTER_BAND_A = 3.0


@dataclass(frozen=True)
class LocalGroupSelection:
    pair: tuple[int, int | None]
    group_mask: np.ndarray
    draw_count: int


def select_native_local_group(reference_positions, atoms, rng, freedom_mask=None):
    """Recover the Run_type=5 Cartesian axis/group selection (no PES calls).

    ``reference_positions`` is the caller's saved geometry, not automatically
    the previous Gaussian center. Only the first Cartesian freedom flag of
    each atom controls eligibility in the executable. This is intentionally
    distinct from a complete ASE constraint projection.

    The native one-based zero second index becomes ``None`` when there is no
    candidate. No random draw or replacement axis is invented in that case.
    This component is not yet a complete native direction controller.
    """
    if len(atoms) < 2 or atoms.pbc.any():
        raise ValueError('selection requires at least two nonperiodic atoms')
    if atoms.constraints:
        raise ValueError('ASE constraints are unsupported; pass an explicit mask')
    current = np.asarray(atoms.positions, dtype=float)
    reference = np.asarray(reference_positions, dtype=float)
    if (reference.shape != current.shape or not np.isfinite(reference).all()
            or not np.isfinite(current).all()):
        raise ValueError('reference and current positions require finite matching (N, 3) arrays')
    mask = np.ones_like(current, dtype=bool) if freedom_mask is None else np.asarray(freedom_mask)
    if mask.dtype != np.bool_ or mask.size != current.size:
        raise ValueError('freedom_mask must be a 3N boolean array')
    active = np.flatnonzero(mask.reshape(current.shape)[:, 0])
    if not active.size:
        raise ValueError('selection requires at least one active atom')

    movement = np.linalg.norm(current - reference, axis=1)
    first = active[np.argmin(movement[active])]
    first_distance = np.linalg.norm(current - current[first], axis=1)
    movement[first_distance < _NEAR_CENTER_A] = 0.0
    second = active[np.argmin(movement[active])]
    score = first_distance + np.linalg.norm(current - current[second], axis=1)
    axis_first = int(active[np.argmax(score[active])])
    threshold = max(_OUTER_BAND_A, float(score[axis_first]) - _OUTER_BAND_A)
    candidates = active[(score[active] > threshold) & (active != axis_first)]
    group = np.zeros(len(atoms), dtype=np.int32)
    group[candidates] = 1
    axis_second = (int(candidates[int(_next_random(rng) * len(candidates))])
                   if len(candidates) else None)
    return LocalGroupSelection((axis_first, axis_second), group, int(bool(len(candidates))))


def native_local_group(atoms, pair, group_mask, freedom_mask=None):
    """Return the recovered local-group direction for an explicit pair.

    ``pair`` is a two element zero-based atom-index sequence.  ``group_mask``
    is a length-N integer array; only values equal to one are selected, as in
    the archived Fortran helper.  The result is deliberately unnormalised.
    """
    n = len(atoms)
    if n < 2:
        raise ValueError("local group requires at least two atoms")
    if np.any(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError("native local group requires isolated nonperiodic atoms")
    if len(getattr(atoms, "constraints", ())) != 0:
        raise ValueError("ASE constraints are unsupported; pass an explicit mask")
    try:
        values = tuple(pair)
    except TypeError as exc:
        raise ValueError("pair must contain two distinct integer zero-based indices") from exc
    if (len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) for v in values)):
        raise ValueError("pair must contain two distinct integer zero-based indices")
    i, j = (int(v) for v in values)
    if i == j or not (0 <= i < n) or not (0 <= j < n):
        raise ValueError("pair must contain two distinct integer zero-based indices")
    groups = np.asarray(group_mask)
    if groups.dtype.kind not in "iu" or groups.size != n:
        raise ValueError("group_mask must be an integer array of length N")
    groups = groups.reshape(n)
    mask = np.ones((n, 3), dtype=bool) if freedom_mask is None else np.asarray(freedom_mask)
    if mask.dtype != np.bool_ or mask.size != 3 * n:
        raise ValueError("freedom_mask must be a 3N boolean array")
    mask = mask.reshape(n, 3)
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (n, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite Cartesian coordinates")
    result = np.zeros_like(positions)
    for k in range(n):
        if groups[k] == 1:
            result[k] = np.cross(positions[k] - positions[i], positions[k] - positions[j])
    return result * mask
