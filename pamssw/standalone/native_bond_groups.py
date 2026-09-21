"""Bounded nonperiodic reference for LASP's C4 endpoint bond groups.

The constants are empirical values recovered from the archived executable.
The full initialized 53x53 fastbond table is represented by its 50 nonzero
entries; missing entries and elements above Z=53 use the recovered radius
fallback. This helper is not yet connected to a walker.
"""

from dataclasses import dataclass

import numpy as np
from .native_local_pair import _radius


_FASTBOND = {
    (1, 1): 0.800000011920929,
    (1, 5): 1.0839999914169312,
    (1, 6): 1.0839999914169312,
    (1, 7): 1.0010000467300415,
    (1, 8): 0.9470000267028809,
    (1, 9): 0.9200000166893005,
    (1, 14): 1.4800000190734863,
    (1, 15): 1.4149999618530273,
    (1, 16): 1.3259999752044678,
    (1, 17): 1.2799999713897705,
    (1, 35): 1.409999966621399,
    (1, 53): 1.600000023841858,
    (5, 5): 1.7000000476837158,
    (5, 8): 1.5119999647140503,
    (6, 6): 1.5119999647140503,
    (6, 7): 1.4390000104904175,
    (6, 8): 1.3930000066757202,
    (6, 9): 1.3530000448226929,
    (6, 14): 1.8600000143051147,
    (6, 15): 1.840000033378601,
    (6, 16): 1.812000036239624,
    (6, 17): 1.781000018119812,
    (6, 35): 1.940000057220459,
    (6, 53): 2.1600000858306885,
    (7, 7): 1.2829999923706055,
    (7, 8): 1.3329999446868896,
    (7, 9): 1.3600000143051147,
    (7, 14): 1.7400000095367432,
    (7, 15): 1.649999976158142,
    (7, 16): 1.6740000247955322,
    (7, 17): 1.75,
    (7, 35): 1.899999976158142,
    (7, 53): 2.0999999046325684,
    (8, 8): 1.4500000476837158,
    (8, 9): 1.4199999570846558,
    (8, 14): 2.0,
    (8, 15): 1.659999966621399,
    (8, 16): 1.4700000286102295,
    (8, 17): 1.7000000476837158,
    (8, 22): 2.0999999046325684,
    (8, 35): 1.850000023841858,
    (8, 53): 2.049999952316284,
    (9, 14): 1.5700000524520874,
    (9, 15): 1.5399999618530273,
    (9, 16): 1.5499999523162842,
    (16, 35): 2.240000009536743,
    (16, 53): 2.4000000953674316,
    (17, 17): 1.9900000095367432,
    (35, 35): 2.2799999713897705,
    (53, 53): 2.6700000762939453,
}
_MISSING_BOND_THRESHOLD = 0.05
_GROUP_SCALE = 1.3


@dataclass(frozen=True)
class NativeBondGroups:
    first_group: np.ndarray
    second_group: np.ndarray
    status: str
    cutoff_sources: tuple[str, ...]


def _validated_pair(pair, n):
    try:
        values = tuple(pair)
    except TypeError as exc:
        raise ValueError("pair must contain two distinct zero-based indices") from exc
    if (len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) for v in values)):
        raise ValueError("pair must contain two distinct zero-based indices")
    i, j = (int(v) for v in values)
    if i == j or not (0 <= i < n) or not (0 <= j < n):
        raise ValueError("pair must contain two distinct zero-based indices")
    return i, j


def native_bond_groups(atoms, pair):
    """Return the two native C4 endpoint masks or a connected-pair fallback.

    This bounded implementation accepts isolated, unconstrained geometries.
    Masks exclude their seed endpoint, matching ``group_atoms`` writes.
    """
    n = len(atoms)
    if n < 2:
        raise ValueError("bond groups require at least two atoms")
    if np.any(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError("native bond groups currently require nonperiodic atoms")
    if len(getattr(atoms, "constraints", ())) != 0:
        raise ValueError("ASE constraints are unsupported")
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError("positions must be finite Cartesian coordinates")
    endpoints = _validated_pair(pair, n)
    numbers = np.asarray(atoms.numbers, dtype=int)
    if np.any((numbers < 1) | (numbers > 118)):
        raise ValueError("requires physical atomic numbers 1..118")

    cutoffs = np.zeros((n, n), dtype=float)
    sources = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = tuple(sorted((int(numbers[i]), int(numbers[j]))))
            cutoff = _FASTBOND.get(key, 0.0)
            source = "fastbond"
            if cutoff < _MISSING_BOND_THRESHOLD:
                cutoff = _radius(key[0]) + _radius(key[1])
                source = "radius_fallback"
            cutoffs[i, j] = cutoffs[j, i] = cutoff
            sources.add(source)

    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)

    def component(seed):
        visited = np.zeros(n, dtype=bool)
        output = np.zeros(n, dtype=np.int32)
        visited[seed] = True

        def grow(current):
            while True:
                candidates = np.flatnonzero(
                    (~visited) & (distances[current] < _GROUP_SCALE * cutoffs[current])
                )
                if not candidates.size:
                    return
                selected = int(candidates[np.argmin(distances[current, candidates])])
                visited[selected] = True
                output[selected] = 1
                grow(selected)

        grow(seed)
        return output

    first = component(endpoints[0])
    if first[endpoints[1]] == 1:
        second = np.zeros(n, dtype=np.int32)
        status = "connected_pair_fallback"
    else:
        second = component(endpoints[1])
        status = "separate_groups"
    return NativeBondGroups(first, second, status, tuple(sorted(sources)))
