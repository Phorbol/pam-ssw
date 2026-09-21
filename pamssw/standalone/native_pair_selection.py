"""Experimental, calculator-free Run_type=5 pair geometry reconstruction.

The distance/angle rules are empirical constants in the uploaded executable,
not recommended universal search parameters. This module is not enabled by
the default walker and does not establish a complete native direction policy.
"""
from dataclasses import dataclass
import numpy as np

from .native_local_pair import _next_random, _radius


def _coordinates(atoms, pair):
    x = np.asarray(atoms.positions, dtype=float)
    if len(x) < 2 or atoms.pbc.any() or atoms.constraints:
        raise ValueError('requires at least two unconstrained nonperiodic atoms')
    if not np.isfinite(x).all():
        raise ValueError('finite coordinates required')
    pair = tuple(pair)
    if (len(pair) != 2 or any(isinstance(i, (bool, np.bool_)) or
            not isinstance(i, (int, np.integer)) for i in pair)
            or min(pair) < 0 or max(pair) >= len(x) or pair[0] == pair[1]):
        raise ValueError('requires two distinct zero-based atom indices')
    return x, pair


def native_pair_allowed(atoms, pair):
    """Recovered ``check_forbiden`` for isolated geometry and native flag=+1.

    Neighbors within 2.2 Å of either endpoint are checked against the pair
    direction. Reject angle <10 degrees, or angle <60 with distance <1.25 Å,
    or angle <30 with distance <1.5 Å. The original caller can still return
    a pair rejected here when its separate rejection counter is exhausted.

    Native periodic-image selection is outside this nonperiodic contract.
    """
    x, pair = _coordinates(atoms, pair)
    for endpoint, other in (pair, pair[::-1]):
        axis = x[endpoint] - x[other]
        axis_norm = np.linalg.norm(axis)
        for k in range(len(x)):
            if k in pair:
                continue
            neighbor = x[endpoint] - x[k]
            distance = np.linalg.norm(neighbor)
            if distance >= 2.2:
                continue
            if axis_norm < 1e-10 or distance < 1e-10:
                return False
            # Native acos has no clipping. Preserve its floating-point
            # boundary behavior instead of inventing an angular tolerance.
            with np.errstate(invalid='ignore'):
                angle = np.arccos(np.dot(axis, neighbor) / axis_norm / distance)*180/np.pi
            if angle < 10 or (angle < 60 and distance < 1.25) or (angle < 30 and distance < 1.5):
                return False
    return True


@dataclass(frozen=True)
class PairRefreshResult:
    pair: tuple[int, int | None]
    geometry_accepted: bool
    stop_reason: str
    draw_count: int
    distance_or_fixatom_rejections: int
    forbidden_rejections: int
    element_rejections: int


def refresh_native_pair(atoms, pair, rng, *, fixatom=None):
    """Run_type=5 pair refresh, with explicit nonperiodic/free-atom geometry.

    Native ``fixatom`` values >=0.5 exclude second candidates; these values
    are not ASE constraints. First-atom selection follows the native separate
    species/neighbor rules. The count limit 150 applies ONLY to distance and
    fixatom rejections; no extra retry cap or repair is introduced here.
    A finite RNG iterator can impose a caller budget (StopIteration propagates).

    ``pair[1]=None`` represents the native zero index. On rejection-limit
    exit the last written pair is retained, even when not geometry-accepted.
    Coordinates are physical isolated Cartesian coordinates; no artificial
    periodic-image chart is introduced into the Python/ASE interface.
    """
    x = np.asarray(atoms.positions, dtype=float)
    n = len(x)
    if n < 2 or atoms.pbc.any() or atoms.constraints or not np.isfinite(x).all():
        raise ValueError('requires finite unconstrained nonperiodic atoms')
    pair = tuple(pair)
    if len(pair) != 2:
        raise ValueError('pair must contain two indices; second may be None')
    for k, atom in enumerate(pair):
        if k == 1 and atom is None:
            continue
        if (isinstance(atom, (bool, np.bool_)) or not isinstance(atom, (int, np.integer))
                or not 0 <= atom < n):
            raise ValueError('invalid zero-based pair index')
    fixed = np.zeros(n) if fixatom is None else np.asarray(fixatom, dtype=float)
    if fixed.shape != (n,) or not np.isfinite(fixed).all():
        raise ValueError('fixatom requires N finite values')
    z = atoms.numbers
    first, second = pair
    draws = 0

    def draw():
        nonlocal draws
        value = _next_random(rng)
        draws += 1
        return value

    u, v = draw(), draw()
    if min(z) < 10 and max(z) < 10 and u > 0.20000000298023224:
        first = int(n*v)
    light_selected = False
    if min(z) < 10 and z[first] > 10 and draw() > 0.10000000149011612:
        distances = np.linalg.norm(x-x[first], axis=1)
        for _ in range(min(n, 100)):
            candidates = np.flatnonzero(distances > 0.10000000149011612)
            chosen = int(candidates[np.argmin(distances[candidates])]) if len(candidates) else 0
            if z[chosen] <= 10:
                first, light_selected = chosen, True
                break
            distances[chosen] = 10.0
    if draw() > .5 and not light_selected:
        distances = np.linalg.norm(x-x[first], axis=1)
        neighbors = [k for k in range(n) if k != first and
                     distances[k] < _radius(z[first]) + _radius(z[k]) + .5]
        if len(neighbors) > 100:
            raise ValueError('native neighbor buffer is limited to 100 entries')
        if neighbors:
            first = neighbors[int(draw()*len(neighbors))]
    distances = np.linalg.norm(x-x[first], axis=1)
    rejected = forbidden = element = 0
    while rejected < 150:
        candidate = int(draw()*n)
        if distances[candidate] <= 2.0 or fixed[candidate] >= .5:
            rejected += 1
            continue
        second = candidate
        u = draw()
        if z[second] > 20 and u > 0.6000000238418579:
            element += 1
            continue
        if native_pair_allowed(atoms, (first, second)):
            return PairRefreshResult((first, second), True, 'geometry_accepted',
                                     draws, rejected, forbidden, element)
        forbidden += 1
    return PairRefreshResult((first, second), False, 'rejection_limit',
                             draws, rejected, forbidden, element)
