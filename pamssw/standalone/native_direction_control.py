"""Restricted Run5/Q-off direction control recovered from the uploaded ELF.

Independent reference components; not a complete generator or a walker mode.
Coefficient thresholds are explicit empirical native inputs, not new defaults.
"""
from dataclasses import dataclass
import numpy as np
from .native_local_pair import _next_random


@dataclass(frozen=True)
class LocalCoefficients:
    coefficients: np.ndarray
    group_marker: int | None  # None means the caller's existing value is retained.


def select_local_coefficients(group, rng, *, ratio_local, local_probability,
                              group_threshold):
    """Ordinary Run5 modelevel=0, Q disabled, compression disabled.

    Consume the native five uniform draws, including the two unused in this
    restricted branch. c4 selects pair/pair-group motion; c6 selects torsional
    group motion. This selector does not generate either direction itself.
    """
    group = np.asarray(group)
    if group.ndim != 1 or group.dtype.kind not in 'iu' or np.any((group != 0) & (group != 1)):
        raise ValueError('group requires a one-dimensional binary integer mask')
    if (isinstance(ratio_local, (bool,np.bool_)) or not isinstance(ratio_local,(int,np.integer))
            or ratio_local < 0):
        raise ValueError('ratio_local must be a nonnegative integer')
    for value in (local_probability, group_threshold):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('probability thresholds must lie in [0,1]')
    strength = .1 + (.1 * _next_random(rng)) * ratio_local
    _next_random(rng)  # Earlier group/Q selection, inactive here.
    _next_random(rng)  # Q probability still consumes a draw with Q disabled.
    choice, marker = _next_random(rng), _next_random(rng)
    coefficients = np.zeros(10)
    coefficients[1] = 1.
    if local_probability > choice or not np.any(group):
        coefficients[4] = strength
        group_marker = -1 if marker > group_threshold else 0
    else:
        coefficients[6] = strength
        group_marker = None
    return LocalCoefficients(coefficients, group_marker)


@dataclass(frozen=True)
class LocalDirectionResult:
    direction: np.ndarray
    release_all: bool
    local_route: str
    group_marker: int


def _native_normalize(vector):
    square = float(np.vdot(vector, vector))
    return np.zeros_like(vector) if square <= 1e-6 else vector/np.sqrt(square)


class LocalDirectionState:
    """One escape's selected pair, group, coefficients and Gaussian snapshot.

    The caller supplies the post-Allopt selection and explicitly records each
    Gaussian center. This object does not choose an outer reference or perform
    Metropolis selection. It owns copies, never retains Broyden history, and
    invokes only the independent Python generator.
    """

    def __init__(self, pair, group, coefficients, *, group_marker,
                 c1_radius_policy='restricted', active_mask=None, geometry='nonperiodic'):
        self.geometry = geometry
        self.pair = tuple(pair)
        self.group = np.asarray(group).copy()
        self.coefficients = np.asarray(coefficients, dtype=float).copy()
        if self.coefficients.shape != (10,):
            raise ValueError('coefficients require ten entries')
        self.group_marker = group_marker
        self.active_mask = _validated_active_mask(active_mask, len(self.group))
        if c1_radius_policy not in ('restricted', 'per_atom'):
            raise ValueError('c1_radius_policy must be restricted or per_atom')
        self.c1_radius_policy = c1_radius_policy
        self.gaussian_center = None
        self.last_coefficients = None

    def save_gaussian_center(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (len(self.group), 3) or not np.isfinite(positions).all():
            raise ValueError('Gaussian center requires finite matching (N,3) positions')
        self.gaussian_center = positions.copy()

    def initial(self, atoms, rng):
        return self._generate(atoms, np.zeros_like(atoms.positions), self.coefficients, rng)

    def update(self, atoms, rng):
        if self.gaussian_center is None:
            raise ValueError('save the actual Gaussian center before updating direction')
        if atoms.positions.shape != self.gaussian_center.shape:
            raise ValueError('atom count changed within escape')
        displacement = atoms.positions-self.gaussian_center
        if self.active_mask is not None:
            displacement = displacement.copy()
            displacement[~self.active_mask] = 0.
        seed = _native_normalize(displacement)
        coefficients = np.zeros(10)
        coefficients[4:7] = self.coefficients[4:7]
        coefficients[9] = 1.2*np.sum(coefficients[:9])
        return self._generate(atoms, seed, coefficients, rng)

    def _generate(self, atoms, seed, coefficients, rng):
        self.last_coefficients = np.asarray(coefficients).copy()
        generator = generate_local_direction
        if self.geometry == 'periodic_local':
            from .periodic_direction import generate_periodic_direction
            generator = generate_periodic_direction
        elif self.geometry != 'nonperiodic':
            raise ValueError('unknown direction geometry')
        result = generator(atoms, seed, coefficients, self.pair,
                           self.group, rng, group_marker=self.group_marker,
                           c1_radius_policy=self.c1_radius_policy,
                           active_mask=self.active_mask)
        self.group_marker = result.group_marker
        return result


def generate_local_direction(atoms, seed, coefficients, pair, group, rng, *,
                             group_marker, c1_radius_policy='restricted',
                             active_mask=None):
    """Recovered free-cluster c1/c4/c6 generator composition (experimental).

    ``seed`` is the caller-owned accumulator, normally zero initially and a
    normalized stage displacement on update. c9 scales it only when >1e-6.
    The native final zero result requests true-surface relaxation; it is not
    an accepted minimum. Q/compression are not enabled here.
    ``c1_radius_policy='restricted'`` retains the all-near domain guard.
    ``'per_atom'`` is an explicit scientific correction variant: it masks each
    atom independently at 12 A from the selected atom, then applies the
    existing rigid-frame projection (P M). Projection may introduce compensating
    displacement on masked-out atoms; strict mask support is not the objective.
    The 12 A value is an empirical native-derived scale, not a universal
    physical length. No ASE calculator is invoked here.
    """
    from .cluster_frame import ClusterFrame
    from .native_local_group import native_local_group
    from .native_random import native_vmb2
    from .native_pair_selection import _coordinates, native_pair_allowed
    from .native_local_pair import native_local_pair

    positions, pair = _coordinates(atoms, pair)
    active_mask = _validated_active_mask(active_mask, len(atoms))
    frame = None if active_mask is not None else ClusterFrame(atoms)
    vector = np.asarray(seed, dtype=float).copy()
    coeff = np.asarray(coefficients, dtype=float)
    if vector.shape != positions.shape or not np.isfinite(vector).all():
        raise ValueError('seed requires finite (N,3) coordinates')
    if active_mask is not None:
        vector[~active_mask] = 0.
    if coeff.shape != (10,) or not np.isfinite(coeff).all() or np.any(coeff < 0):
        raise ValueError('coefficients require ten finite nonnegative values')
    if c1_radius_policy not in ('restricted', 'per_atom'):
        raise ValueError('c1_radius_policy must be restricted or per_atom')
    if np.any(coeff[[0,2,3,5,7,8]] != 0):
        raise NotImplementedError('only c1/c4/c6/c9 composition is currently closed')
    if group_marker not in (0,-1):
        raise ValueError('group_marker must be native 0 or -1')
    uniform = _next_random(rng)  # Draw occurs even when c1 is inactive.
    if coeff[9] > 1e-6:
        vector *= coeff[9]
    if coeff[1] > 1e-6:
        near = np.linalg.norm(positions-positions[pair[0]], axis=1) <= 12.
        if c1_radius_policy == 'restricted' and not np.all(near):
            raise NotImplementedError('c1 reference currently requires the all-near radius domain')
        random = native_vmb2(np.zeros_like(positions), near[:, None] * np.ones((1, 3), dtype=bool), uniform)
        vector += coeff[1]*_native_normalize(_project_direction(random, frame, active_mask))
    route = 'none'
    if coeff[4] > 1e-6:
        if native_pair_allowed(atoms, pair):
            if group_marker:
                from .native_bond_groups import native_bond_groups
                from .native_pair_group import native_local_pair_group
                groups = native_bond_groups(atoms, pair)
                if groups.status == 'connected_pair_fallback':
                    group_marker = 0
                    local = native_local_pair(atoms, pair, rng).raw_direction
                    route = 'pair_fallback'
                else:
                    local = native_local_pair_group(atoms, pair,
                        groups.first_group, groups.second_group)
                    route = 'pair_group'
            else:
                local = native_local_pair(atoms,pair,rng).raw_direction
                route = 'pair'
            vector += coeff[4]*_native_normalize(_project_direction(local, frame, active_mask))
        else:
            route = 'forbidden'
    if coeff[6] > 1e-6:
        local = native_local_group(atoms,pair,group)
        vector += coeff[6]*_native_normalize(_project_direction(local, frame, active_mask))
        route = 'torsion'
    direction = _native_normalize(vector)
    return LocalDirectionResult(direction,not np.any(direction),route,group_marker)


def _validated_active_mask(active_mask, atom_count):
    if active_mask is None:
        return None
    mask = np.asarray(active_mask)
    if mask.dtype != np.bool_ or mask.shape != (atom_count,):
        raise ValueError('active_mask must be a boolean N-vector')
    if not np.any(mask):
        raise ValueError('active_mask must contain at least one active atom')
    return mask.copy()


def _project_direction(vector, frame, active_mask):
    if active_mask is None:
        return frame.project(vector)
    result = np.asarray(vector, dtype=float).copy()
    result[~active_mask] = 0.
    return result
