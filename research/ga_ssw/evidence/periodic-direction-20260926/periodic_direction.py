"""Independent fixed-cell periodic adaptation of Run5 local directions.

This module defines periodic geometry for an ASE SSW caller; it is not a claim
of LASP periodic-trajectory parity. Positions and caller-supplied history seeds
remain in their continuous coordinate chart. Only local geometry is imaged.
"""
from __future__ import annotations

import numpy as np
from ase.geometry import find_mic

from .native_direction_control import LocalDirectionResult, _native_normalize, _validated_active_mask
from .native_local_group import LocalGroupSelection
from .native_local_pair import _next_random, _radius
from .native_random import native_vmb2
from .native_bond_groups import _FASTBOND, _GROUP_SCALE, _MISSING_BOND_THRESHOLD

_PAIR_MIN = 0.7
_PAIR_SCALE = 0.6
_NEIGHBOR_MARGIN = 0.5
_NEIGHBOR_DISTANCE = 3.0
_NEIGHBOR_SCALE = 0.8
_FORBIDDEN_NEIGHBOR_RANGE = 2.2


def _validate_atoms(atoms):
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.ndim != 2 or positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        raise ValueError("atoms require finite (N, 3) positions")
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.shape != (3,) or not np.any(pbc):
        raise ValueError("periodic direction geometry requires at least one periodic axis")
    if np.linalg.matrix_rank(np.asarray(atoms.cell, dtype=float)) < 3:
        raise ValueError("periodic direction geometry requires a nonsingular cell")
    if getattr(atoms, "constraints", ()):
        raise ValueError("pass an explicit active_mask instead of ASE constraints")
    if len(atoms) < 2:
        raise ValueError("periodic direction geometry requires at least two atoms")
    return positions, np.asarray(atoms.cell, dtype=float), pbc


def _mic(vectors, cell, pbc):
    result, _ = find_mic(np.asarray(vectors, dtype=float), cell=cell, pbc=pbc)
    return np.asarray(result, dtype=float)


def _stable_argmax(values, candidates):
    """Resolve machine-roundoff ties by the lowest eligible atom index."""
    maximum = float(np.max(values[candidates]))
    scale = max(1.0, abs(maximum), float(np.max(np.abs(values[candidates]))))
    tied = candidates[values[candidates] >= maximum - 8 * np.finfo(float).eps * scale]
    return int(tied[0])


def _stage_movement(current, reference):
    """Stage-history magnitude in the continuous coordinate chart."""
    return np.linalg.norm(np.asarray(current, dtype=float) - np.asarray(reference, dtype=float), axis=1)


def _validated_pair(pair, n):
    try:
        values = tuple(pair)
    except TypeError as exc:
        raise ValueError("pair must contain two distinct integer zero-based indices") from exc
    if (len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) for v in values)):
        raise ValueError("pair must contain two distinct integer zero-based indices")
    i, j = (int(v) for v in values)
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError("pair must contain two distinct integer zero-based indices")
    return i, j


def _component(seed, distances, cutoffs):
    n = len(distances)
    visited = np.zeros(n, dtype=bool)
    output = np.zeros(n, dtype=np.int32)
    visited[seed] = True
    stack = [seed]
    while stack:
        current = stack.pop()
        candidates = np.flatnonzero((~visited) & (distances[current] < _GROUP_SCALE * cutoffs[current]))
        if candidates.size:
            candidates = candidates[np.argsort(distances[current, candidates], kind="stable")]
            for candidate in candidates:
                visited[candidate] = True
                output[candidate] = 1
                stack.append(int(candidate))
    return output


def _periodic_groups(atoms, pair, positions, cell, pbc):
    i, j = pair
    numbers = np.asarray(atoms.numbers, dtype=int)
    n = len(numbers)
    cutoffs = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(a + 1, n):
            key = tuple(sorted((int(numbers[a]), int(numbers[b]))))
            cutoff = _FASTBOND.get(key, 0.0)
            if cutoff < _MISSING_BOND_THRESHOLD:
                cutoff = _radius(key[0]) + _radius(key[1])
            cutoffs[a, b] = cutoffs[b, a] = cutoff
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(_mic(delta.reshape(-1, 3), cell, pbc).reshape(n, n, 3), axis=2)
    first = _component(i, distances, cutoffs)
    if first[j] == 1:
        return first, np.zeros(n, dtype=np.int32), "connected_pair_fallback"
    return first, _component(j, distances, cutoffs), "separate_groups"


def select_periodic_direction_group(reference_positions, atoms, rng, active_mask=None):
    """Select the Run5 local group using minimum-image distances in fixed PBC."""
    positions, cell, pbc = _validate_atoms(atoms)
    n = len(positions)
    active_mask = _validated_active_mask(active_mask, n)
    if active_mask is None:
        active = np.arange(n)
    else:
        active = np.flatnonzero(active_mask)
    reference = np.asarray(reference_positions, dtype=float)
    if reference.shape != positions.shape or not np.isfinite(reference).all():
        raise ValueError("reference_positions must be finite and match (N, 3)")
    # Unlike local geometry, the stage reference is a caller-owned continuous
    # lift; minimum imaging here would alias large valid stage displacements.
    movement = _stage_movement(positions, reference)
    first = int(active[np.argmin(movement[active])])
    from_first = np.linalg.norm(_mic(positions - positions[first], cell, pbc), axis=1)
    movement[from_first < 2.0] = 0.0
    second = int(np.argmin(movement))
    from_second = np.linalg.norm(_mic(positions - positions[second], cell, pbc), axis=1)
    score = from_first + from_second
    axis_first = _stable_argmax(score, active)
    threshold = max(3.0, float(score[axis_first]) - 3.0)
    score_candidates = np.flatnonzero((score > threshold) & (np.arange(n) != axis_first))
    candidates = score_candidates[active_mask[score_candidates]] if active_mask is not None else score_candidates
    group = np.zeros(n, dtype=np.int32)
    group[candidates] = 1
    axis_second = (int(score_candidates[int(_next_random(rng) * len(score_candidates))])
                   if len(score_candidates) else None)
    return LocalGroupSelection((axis_first, axis_second), group, int(bool(len(candidates))))


def _periodic_pair_direction(atoms, pair, rng, positions, cell, pbc, active_mask):
    i, j = pair
    numbers = np.asarray(atoms.numbers, dtype=int)
    pair_delta = _mic(positions[i] - positions[j], cell, pbc)
    pair_norm = float(np.linalg.norm(pair_delta))
    result = np.zeros_like(positions)
    if pair_norm < max(_PAIR_SCALE * (_radius(numbers[i]) + _radius(numbers[j])), _PAIR_MIN):
        return result, False
    pair_unit = pair_delta / pair_norm
    result[i], result[j] = -pair_unit, pair_unit
    mask = np.ones(len(positions), dtype=bool) if active_mask is None else active_mask
    for endpoint, other in ((i, j), (j, i)):
        neighbor_vectors = _mic(positions - positions[endpoint], cell, pbc)
        distances = np.linalg.norm(neighbor_vectors, axis=1)
        slots = [k for k in range(len(positions)) if k != endpoint and mask[k]
                 and distances[k] < _radius(numbers[endpoint]) + _radius(numbers[k]) + _NEIGHBOR_MARGIN]
        endpoint_accepted = 0
        for _ in range(len(slots)):
            slot = int(len(slots) * _next_random(rng))
            atom = slots[slot]
            if atom is None:
                continue
            if atom in pair:
                continue
            delta = _mic(positions[atom] - positions[other], cell, pbc)
            norm = float(np.linalg.norm(delta))
            if norm <= _NEIGHBOR_DISTANCE:
                continue
            slots[slot] = None
            result[atom] += -_NEIGHBOR_SCALE * delta / norm
            endpoint_accepted += 1
            if endpoint_accepted >= 4:
                break
    if active_mask is not None:
        result[~active_mask] = 0.0
    return result, True


def _pair_allowed_periodic(atoms, pair, positions, cell, pbc):
    for endpoint, other in (pair, pair[::-1]):
        axis = _mic(positions[endpoint] - positions[other], cell, pbc)
        axis_norm = float(np.linalg.norm(axis))
        neighbors = _mic(positions[endpoint] - positions, cell, pbc)
        distances = np.linalg.norm(neighbors, axis=1)
        for k in range(len(positions)):
            if k in pair or distances[k] >= _FORBIDDEN_NEIGHBOR_RANGE:
                continue
            if axis_norm < 1e-10 or distances[k] < 1e-10:
                return False
            with np.errstate(invalid="ignore"):
                angle = np.arccos(np.dot(axis, neighbors[k]) / axis_norm / distances[k]) * 180 / np.pi
            if angle < 10 or (angle < 60 and distances[k] < 1.25) or (angle < 30 and distances[k] < 1.5):
                return False
    return True


def _project(direction, active_mask):
    result = np.asarray(direction, dtype=float).copy()
    if active_mask is not None:
        result[~active_mask] = 0.0
    else:
        result -= np.mean(result, axis=0, keepdims=True)
    return result


def generate_periodic_direction(atoms, seed, coefficients, pair, group, rng, *,
                                group_marker, active_mask=None,
                                c1_radius_policy="restricted"):
    """Generate a fixed-cell periodic c1/c4/c6/c9 direction adaptation."""
    positions, cell, pbc = _validate_atoms(atoms)
    n = len(positions)
    pair = _validated_pair(pair, n)
    active_mask = _validated_active_mask(active_mask, n)
    vector = np.asarray(seed, dtype=float).copy()
    coeff = np.asarray(coefficients, dtype=float)
    groups = np.asarray(group)
    if vector.shape != positions.shape or not np.isfinite(vector).all():
        raise ValueError("seed must be finite with shape (N, 3)")
    if active_mask is not None:
        vector[~active_mask] = 0.0
    if coeff.shape != (10,) or not np.isfinite(coeff).all() or np.any(coeff < 0):
        raise ValueError("coefficients require ten finite nonnegative values")
    if groups.size != n or groups.dtype.kind not in "iu":
        raise ValueError("group must be an integer vector of length N")
    groups = groups.reshape(n)
    if c1_radius_policy not in ("restricted", "per_atom"):
        raise ValueError("c1_radius_policy must be restricted or per_atom")
    if np.any(coeff[[0, 2, 3, 5, 7, 8]] != 0):
        raise NotImplementedError("only c1/c4/c6/c9 composition is currently closed")
    if group_marker not in (0, -1):
        raise ValueError("group_marker must be native 0 or -1")
    uniform = _next_random(rng)  # Native consumes this draw even when c1 is off.
    if coeff[9] > 1e-6:
        vector *= coeff[9]
    if coeff[1] > 1e-6:
        near = np.linalg.norm(_mic(positions - positions[pair[0]], cell, pbc), axis=1) <= 12.0
        if c1_radius_policy == "restricted" and not np.all(near):
            raise NotImplementedError("c1 reference currently requires the all-near radius domain")
        random = native_vmb2(np.zeros_like(positions), near[:, None] * np.ones((1, 3), dtype=bool), uniform)
        vector += coeff[1] * _native_normalize(_project(random, active_mask))
    route = "none"
    if coeff[4] > 1e-6:
        if _pair_allowed_periodic(atoms, pair, positions, cell, pbc):
            if group_marker:
                first, second, status = _periodic_groups(atoms, pair, positions, cell, pbc)
                if status == "connected_pair_fallback":
                    group_marker = 0
                    local, _ = _periodic_pair_direction(atoms, pair, rng, positions, cell, pbc, active_mask)
                    route = "pair_fallback"
                else:
                    # Put all atoms in one consistent local chart rooted at p0.
                    chart = _mic(positions - positions[pair[0]], cell, pbc)
                    i, j = pair
                    axis = chart[j] - chart[i]
                    length = float(np.linalg.norm(axis))
                    if length == 0.0:
                        raise ValueError("periodic pair endpoints have zero separation")
                    unit = axis / length
                    local = np.zeros_like(positions)
                    local[first == 1] = unit
                    local[second == 1] = -unit
                    local[i], local[j] = 1.2 * unit, -1.2 * unit
                    route = "pair_group"
            else:
                local, _ = _periodic_pair_direction(atoms, pair, rng, positions, cell, pbc, active_mask)
                route = "pair"
            vector += coeff[4] * _native_normalize(_project(local, active_mask))
        else:
            route = "forbidden"
    if coeff[6] > 1e-6:
        chart = _mic(positions - positions[pair[0]], cell, pbc)
        i, j = pair
        local = np.zeros_like(positions)
        for k in range(n):
            if groups[k] == 1:
                local[k] = np.cross(chart[k] - chart[i], chart[k] - chart[j])
        if active_mask is not None:
            local[~active_mask] = 0.0
        vector += coeff[6] * _native_normalize(_project(local, active_mask))
        route = "torsion"
    direction = _native_normalize(vector)
    return LocalDirectionResult(direction, not np.any(direction), route, group_marker)
