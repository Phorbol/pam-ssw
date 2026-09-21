"""Native local atom pair-group direction producer recovered from LASP."""
from __future__ import annotations

import numpy as np


def native_local_pair_group(atoms, pair, mask_a, mask_b, output=None):
    """Build the native pair-group vector for an explicit selected pair.

    ``pair`` uses zero-based atom indices.  ``mask_a`` and ``mask_b`` are
    integer per-atom masks: A-members receive ``+u`` first, then B-members
    receive ``-u`` and therefore overwrite on overlap.  The two pair endpoints
    are finally forced to ``+1.2*u`` and ``-1.2*u``.  Non-members retain
    ``output`` when supplied, otherwise zero.
    """
    positions = np.asarray(atoms.positions, dtype=float)
    n = len(positions)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError("positions must be finite Cartesian coordinates")
    if np.any(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError("native pair-group requires nonperiodic atoms")
    if len(getattr(atoms, "constraints", ())) != 0:
        raise ValueError("native pair-group does not support ASE constraints")
    try:
        values = tuple(pair)
    except TypeError as exc:
        raise ValueError("pair must contain two integer atom indices") from exc
    if (len(values) != 2 or any(isinstance(x, (bool, np.bool_)) or
            not isinstance(x, (int, np.integer)) for x in values)):
        raise ValueError("pair must contain two integer atom indices")
    i, j = (int(x) for x in values)
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError("pair must contain two distinct valid atom indices")
    a = np.asarray(mask_a)
    b = np.asarray(mask_b)
    if a.size != n or b.size != n or a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
        raise ValueError("mask_a and mask_b must be integer arrays of length N")
    if output is None:
        result = np.zeros((n, 3), dtype=float)
    else:
        result = np.asarray(output, dtype=float).copy()
        if result.shape != (n, 3) or not np.isfinite(result).all():
            raise ValueError("output must be a finite (N, 3) array")
    delta = positions[j] - positions[i]
    norm = float(np.linalg.norm(delta))
    if norm == 0.0:
        raise ValueError("pair endpoints must have nonzero separation")
    u = delta / norm
    for k in np.flatnonzero(a.reshape(n) == 1):
        result[k] = u
    for k in np.flatnonzero(b.reshape(n) == 1):
        result[k] = -u
    result[i] = 1.2 * u
    result[j] = -1.2 * u
    return result
