"""Experimental outer-state controller for recovered fixed-cell directions.

This module joins the calculator-free Run_type 5 selection and direction
helpers.  Startup is an explicit Python contract; it is not evidence for the
native program's complete startup trajectory.  CBD rotation and walker
checkpointing remain caller responsibilities.
"""
from dataclasses import dataclass, replace

import numpy as np

from .native_direction_control import LocalDirectionState, select_local_coefficients
from .native_local_group import LocalGroupSelection, select_native_local_group
from .native_pair_selection import PairRefreshResult, refresh_native_pair


@dataclass(frozen=True)
class RecoveredDirectionSettings:
    ratio_local: int
    local_probability: float
    group_threshold: float
    pre_rotmax: int
    rotmax: int
    pre_ftol: float
    ftol: float
    metric: str
    max_force_calls: int
    c1_radius_policy: str = 'restricted'
    startup_order: str = 'legacy'

    def __post_init__(self):
        if (isinstance(self.ratio_local, (bool, np.bool_)) or
                not isinstance(self.ratio_local, (int, np.integer)) or
                self.ratio_local < 0):
            raise ValueError('ratio_local must be a nonnegative integer')
        for name in ('local_probability', 'group_threshold'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        for name in ('pre_rotmax', 'rotmax'):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('pre_ftol', 'ftol'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.metric not in ('euclidean', 'native_block_sum'):
            raise ValueError('metric must be euclidean or native_block_sum')
        if (isinstance(self.max_force_calls, (bool, np.bool_)) or
                not isinstance(self.max_force_calls, (int, np.integer)) or
                self.max_force_calls < 2):
            raise ValueError('max_force_calls must be an integer >=2')
        if self.c1_radius_policy not in ('restricted', 'per_atom'):
            raise ValueError('c1_radius_policy must be restricted or per_atom')
        if self.startup_order not in ('legacy', 'randomized'):
            raise ValueError('startup_order must be legacy or randomized')


@dataclass(frozen=True)
class RecoveredDirectionCheckpointState:
    """Small outer-boundary state needed to resume recovered directions."""
    settings: RecoveredDirectionSettings
    pair: tuple
    group: np.ndarray
    group_marker: int | None
    selection: LocalGroupSelection | None = None
    refresh: PairRefreshResult | None = None

    def __post_init__(self):
        if not isinstance(self.settings, RecoveredDirectionSettings):
            raise TypeError('checkpoint settings must be RecoveredDirectionSettings')
        if (len(self.pair) != 2 or any(isinstance(i, (bool, np.bool_)) or
                                       not isinstance(i, (int, np.integer)) for i in self.pair)):
            raise ValueError('checkpoint pair must contain two integer indices')
        group = np.asarray(self.group)
        if group.ndim != 1 or group.dtype.kind not in 'iu' or np.any((group != 0) & (group != 1)):
            raise ValueError('checkpoint group must be a binary one-dimensional mask')
        if (self.group_marker is not None and
                (isinstance(self.group_marker, (bool, np.bool_)) or
                 not isinstance(self.group_marker, (int, np.integer)) or
                 self.group_marker not in (-1, 0))):
            raise ValueError('checkpoint group_marker must be None, -1, or 0')
        if self.selection is not None and not isinstance(self.selection, LocalGroupSelection):
            raise TypeError('checkpoint selection must be LocalGroupSelection or None')
        if self.refresh is not None and not isinstance(self.refresh, PairRefreshResult):
            raise TypeError('checkpoint refresh must be PairRefreshResult or None')
        group = group.copy()
        group.setflags(write=False)
        object.__setattr__(self, 'pair', tuple(int(i) for i in self.pair))
        object.__setattr__(self, 'group', group)

    def validate_for_atom_count(self, count):
        if (self.group_marker is not None and
                (isinstance(self.group_marker, (bool, np.bool_)) or
                 not isinstance(self.group_marker, (int, np.integer)) or
                 self.group_marker not in (-1, 0))):
            raise ValueError('checkpoint group_marker must be None, -1, or 0')
        if len(self.group) != count:
            raise ValueError('checkpoint group does not match atom count')
        if any(i < 0 or i >= count for i in self.pair):
            raise ValueError('checkpoint pair does not match atom count')
        if self.selection is not None:
            if len(self.selection.group_mask) != count:
                raise ValueError('checkpoint selection group does not match atom count')
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                   or i < 0 or i >= count for i in self.selection.pair if i is not None):
                raise ValueError('checkpoint selection pair is invalid')
        if self.refresh is not None:
            if any(i is not None and (isinstance(i, (bool, np.bool_)) or
                                      not isinstance(i, (int, np.integer)) or
                                      i < 0 or i >= count) for i in self.refresh.pair):
                raise ValueError('checkpoint refresh pair is invalid')


class RecoveredDirectionController:
    """Own pair/group state across escapes and local state within one escape."""

    def __init__(self, settings):
        if not isinstance(settings, RecoveredDirectionSettings):
            raise TypeError('settings must be RecoveredDirectionSettings')
        self.settings = settings
        self._pair = None
        self._group = None
        self._outer_reference = None
        self._state = None
        self._selection = None
        self._refresh = None
        self._coefficients = None
        self._group_marker = None

    def checkpoint_state(self):
        if self._pair is None or self._group is None:
            raise ValueError('cannot checkpoint an uninitialized recovered direction controller')
        return RecoveredDirectionCheckpointState(
            self.settings, self._pair, self._group, self._group_marker,
            self._selection, self._refresh)

    def restore_checkpoint_state(self, state):
        if not isinstance(state, RecoveredDirectionCheckpointState):
            raise TypeError('recovered direction checkpoint state required')
        if state.settings != self.settings:
            raise ValueError('recovered direction settings do not match checkpoint')
        state.validate_for_atom_count(len(state.group))
        self._pair = state.pair
        self._group = state.group.copy()
        self._group_marker = state.group_marker
        self._outer_reference = None
        self._state = None
        self._selection = state.selection
        self._refresh = state.refresh
        self._coefficients = None

    @staticmethod
    def _positions(atoms, name):
        positions = np.asarray(atoms.positions, dtype=float)
        if (len(atoms) < 2 or atoms.pbc.any() or atoms.constraints or
                positions.shape != (len(atoms), 3) or not np.isfinite(positions).all()):
            raise ValueError(f'{name} requires at least two finite unconstrained nonperiodic atoms')
        return positions

    @staticmethod
    def _require_pair(pair):
        if len(pair) != 2 or pair[1] is None:
            raise ValueError('native selection did not produce a complete atom pair')
        return tuple(int(value) for value in pair)

    @staticmethod
    def _map_pair(pair, order):
        return tuple(None if value is None else int(order[value]) for value in pair)

    def _select_and_refresh(self, reference, atoms, rng):
        selected = select_native_local_group(reference, atoms, rng)
        refreshed = refresh_native_pair(atoms, selected.pair, rng)
        pair = self._require_pair(refreshed.pair)
        self._pair = pair
        self._group = selected.group_mask.copy()
        self._selection = selected
        self._refresh = refreshed

    def initialize(self, input_atoms, initial_quenched, rng):
        """Create the first pair/group using the Python startup reference."""
        reference = self._positions(input_atoms, 'input_atoms').copy()
        current = self._positions(initial_quenched, 'initial_quenched')
        if reference.shape != current.shape or not np.array_equal(input_atoms.numbers,
                                                                   initial_quenched.numbers):
            raise ValueError('startup structures require matching atoms')
        if self.settings.startup_order == 'legacy':
            self._select_and_refresh(reference, initial_quenched, rng)
            return
        if not callable(getattr(rng, 'permutation', None)):
            raise TypeError('randomized startup_order requires an RNG with permutation()')
        order = np.asarray(rng.permutation(len(initial_quenched)), dtype=int)
        ordered_atoms = initial_quenched[order]
        selected = select_native_local_group(reference[order], ordered_atoms, rng)
        refreshed = refresh_native_pair(
            ordered_atoms, selected.pair, rng)
        selected = replace(
            selected,
            pair=self._map_pair(selected.pair, order),
            group_mask=np.asarray(selected.group_mask)[np.argsort(order)].copy(),
        )
        refreshed = replace(
            refreshed, pair=self._map_pair(refreshed.pair, order))
        self._pair = self._require_pair(refreshed.pair)
        self._group = selected.group_mask.copy()
        self._selection = selected
        self._refresh = refreshed

    def begin_escape(self, current, work, rng):
        """Start after optional LS prequench while retaining the pre-LS reference."""
        reference = self._positions(current, 'current').copy()
        work_positions = self._positions(work, 'work')
        if reference.shape != work_positions.shape or not np.array_equal(current.numbers, work.numbers):
            raise ValueError('current and work require matching atoms')
        if self._pair is None or self._group is None:
            raise ValueError('initialize or observe_landing must select pair/group first')
        selected = select_local_coefficients(
            self._group, rng, ratio_local=self.settings.ratio_local,
            local_probability=self.settings.local_probability,
            group_threshold=self.settings.group_threshold)
        marker = ((0 if self._group_marker is None else self._group_marker)
                  if selected.group_marker is None else selected.group_marker)
        self._coefficients = selected
        self._outer_reference = reference
        self._state = LocalDirectionState(self._pair, self._group,
                                          selected.coefficients, group_marker=marker,
                                          c1_radius_policy=self.settings.c1_radius_policy)
        result = self._state.initial(work, rng)
        self._group_marker = result.group_marker
        return result

    def save_gaussian_center(self, center):
        if self._state is None:
            raise ValueError('begin_escape must precede a Gaussian center')
        positions = center.positions if hasattr(center, 'positions') else center
        self._state.save_gaussian_center(positions)

    def update_direction(self, work, rng):
        if self._state is None:
            raise ValueError('begin_escape must precede direction updates')
        result = self._state.update(work, rng)
        self._group_marker = result.group_marker
        return result

    def observe_landing(self, landing_atoms, rng):
        """Select the next pair/group before any caller-owned MC decision."""
        if self._outer_reference is None:
            raise ValueError('begin_escape must precede landing observation')
        landing = self._positions(landing_atoms, 'landing_atoms')
        if landing.shape != self._outer_reference.shape:
            raise ValueError('landing atom count changed within escape')
        self._select_and_refresh(self._outer_reference, landing_atoms, rng)
        self._state = None

    @property
    def diagnostics(self):
        coefficients = (None if self._coefficients is None else
                        self._coefficients.coefficients.tolist())
        if self._state is not None and self._state.last_coefficients is not None:
            coefficients = self._state.last_coefficients.tolist()
        selection = None
        if self._selection is not None:
            selection = {
                'pair': list(self._selection.pair),
                'group_mask': self._selection.group_mask.tolist(),
                'draw_count': int(self._selection.draw_count),
            }
        refresh = None
        if self._refresh is not None:
            refresh = {
                'pair': list(self._refresh.pair),
                'geometry_accepted': bool(self._refresh.geometry_accepted),
                'stop_reason': self._refresh.stop_reason,
                'draw_count': int(self._refresh.draw_count),
                'distance_or_fixatom_rejections': int(
                    self._refresh.distance_or_fixatom_rejections),
                'forbidden_rejections': int(self._refresh.forbidden_rejections),
                'element_rejections': int(self._refresh.element_rejections),
            }
        return {
            'pair': None if self._pair is None else list(self._pair),
            'group': None if self._group is None else self._group.tolist(),
            'coefficients': coefficients,
            'group_marker': (None if self._group_marker is None else int(self._group_marker)),
            'outer_reference': (None if self._outer_reference is None
                                else self._outer_reference.tolist()),
            'selection': selection,
            'refresh': refresh,
            'initialized': self._pair is not None,
            'escape_active': self._state is not None,
        }
