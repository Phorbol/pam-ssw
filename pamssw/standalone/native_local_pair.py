"""Pure geometry reference for LASP's local atom-pair mode.

This isolated helper is not connected to a walker or calculator. Radii are
Angstrom values extracted from the archived ELF species-radius table in
``research/ga_ssw/evidence/native-cluster-control-generator``; uncovered
atomic numbers use the archived 1.25 Angstrom fallback.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np

_SPECIES_RADIUS = {1:.31,2:.28,3:1.28,4:.96,5:.85,6:.76,7:.71,8:.66,9:.57,10:.58,11:.96,12:.91,13:1.01,14:.91,15:.97,16:.95,17:.92,18:1.06,19:.93,20:1.20,21:1.20,73:1.70}
_FALLBACK_RADIUS = 1.25
_NEIGHBOR_MARGIN, _NEIGHBOR_DISTANCE, _NEIGHBOR_SCALE = .5, 3.0, .8

@dataclass(frozen=True)
class LocalPairResult:
    raw_direction: np.ndarray
    selections: tuple[dict[str, Any], ...]
    draw_count: int
    accepted_count: int
    neighbor_lists: tuple[tuple[int, ...], tuple[int, ...]]

def _next_random(rng):
    if callable(rng): value = rng()
    elif hasattr(rng, "random") and not hasattr(rng, "__next__"): value = rng.random()
    else: value = next(rng)
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("rng draws must be finite and satisfy 0 <= u < 1")
    return value

def _mask(atoms, freedom_mask):
    n = len(atoms)
    if freedom_mask is None: return np.ones((n, 3), dtype=bool)
    a = np.asarray(freedom_mask)
    if a.dtype != np.bool_ or a.size != 3*n: raise ValueError("freedom_mask must be a 3N boolean array")
    return a.reshape(n, 3).copy()

def _radius(z): return _SPECIES_RADIUS.get(int(z), _FALLBACK_RADIUS)

def native_local_pair(atoms, pair, rng, marker=1, freedom_mask=None):
    """Return the archived local-pair direction without PES evaluation.

    ``pair`` contains two distinct zero-based atom indices. ``rng`` is a
    callable, NumPy generator, or iterator yielding explicit uniform draws.
    ``marker`` is ``+1`` or ``-1`` and flips the complete raw direction.
    ``freedom_mask`` is an optional 3N boolean Cartesian mask. The returned
    ``raw_direction`` is not unit-normalized after neighbor accumulation.
    The radius and pair/neighbor gates are empirical archived geometry gates,
    with distances measured in Angstrom.
    """
    if len(atoms) < 2: raise ValueError("local pair requires at least two atoms")
    if np.any(np.asarray(atoms.pbc, dtype=bool)): raise ValueError("native local pair requires isolated nonperiodic atoms")
    if len(getattr(atoms, "constraints", ())) != 0: raise ValueError("ASE constraints are unsupported; pass an explicit mask")
    try:
        pair_values = tuple(pair)
    except TypeError as exc:
        raise ValueError("pair must contain two distinct integer zero-based indices") from exc
    if (len(pair_values) != 2 or any(
            isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) for v in pair_values)):
        raise ValueError("pair must contain two distinct integer zero-based indices")
    i, j = int(pair_values[0]), int(pair_values[1])
    if i == j or not (0 <= i < len(atoms)) or not (0 <= j < len(atoms)): raise ValueError("pair must contain two distinct integer zero-based indices")
    if marker not in (-1, 1): raise ValueError("marker must be 1 or -1")
    mask = _mask(atoms, freedom_mask)
    positions = np.asarray(atoms.positions, dtype=float).copy()
    if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)): raise ValueError("positions must be finite Cartesian coordinates")
    numbers = np.asarray(atoms.numbers, dtype=int)
    pair_delta = positions[i] - positions[j]; pair_norm = float(np.linalg.norm(pair_delta))
    result = np.zeros_like(positions)
    if pair_norm < max(.6 * (_radius(numbers[i]) + _radius(numbers[j])), .7):
        return LocalPairResult(result, (), 0, 0, ((), ()))
    pair_unit = pair_delta / pair_norm
    result[i], result[j] = -marker * pair_unit, marker * pair_unit
    lists = []
    for center in (i, j):
        selected = tuple(k for k in range(len(positions)) if k != center and mask[k, 0] and np.linalg.norm(positions[k] - positions[center]) < _radius(numbers[center]) + _radius(numbers[k]) + _NEIGHBOR_MARGIN)
        if len(selected) > 100: raise ValueError("neighbor list exceeds the 100-slot reference domain")
        lists.append(selected)
    selections=[]; draws=accepted=0
    for endpoint, slots_in in ((j, lists[0]), (i, lists[1])):
        slots=list(slots_in); endpoint_accepted=0
        for attempt in range(len(slots)):
            u=_next_random(rng); draws += 1; slot=int(len(slots)*u); atom=slots[slot]
            event={"endpoint":endpoint,"attempt":attempt,"u":u,"slot":slot,"atom":atom}
            if atom is None:
                event["status"]="cleared"; selections.append(event); continue
            k=atom
            if k in (i,j): event["status"]="pair_atom"; selections.append(event); continue
            delta=positions[k]-positions[endpoint]; norm=float(np.linalg.norm(delta))
            if norm <= _NEIGHBOR_DISTANCE: event["status"]="distance_skip"; selections.append(event); continue
            slots[slot]=None; result[k] += -marker * _NEIGHBOR_SCALE * delta/norm
            accepted += 1; endpoint_accepted += 1; event["status"]="accepted"; selections.append(event)
            if endpoint_accepted >= 4: break
    result *= mask
    return LocalPairResult(result, tuple(selections), draws, accepted, (tuple(lists[0]),tuple(lists[1])))
