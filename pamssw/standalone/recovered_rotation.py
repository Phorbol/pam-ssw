"""Typed stateless settings for the recovered CBD rotation controller."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RecoveredRotationSettings:
    """Explicit recovered-CBD settings without native direction generation.

    ``pre_rotmax`` and ``rotmax`` use the recovered strict ``rotnum > limit``
    stage gates, not HVP-call budgets. ``max_force_calls`` includes the center
    and all endpoint requests across stages. ``pre_ftol`` and ``ftol`` are
    thresholds on ``10 * fd_step * ||HVP residual||`` in eV/Angstrom;
    they are not directly comparable to SSWConfig.rotation_tol (eV/Angstrom²).
    ``fd_step`` and the frame remain owned by SSWConfig. Every parameter is
    explicit; no native defaults or scientific superiority are implied.
    """

    pre_rotmax: int
    rotmax: int
    pre_ftol: float
    ftol: float
    metric: str
    max_force_calls: int

    def __post_init__(self):
        for name in ('pre_rotmax', 'rotmax', 'max_force_calls'):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.max_force_calls < 2:
            raise ValueError('max_force_calls must be at least 2')
        for name in ('pre_ftol', 'ftol'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.metric not in ('euclidean', 'native_block_sum'):
            raise ValueError('metric must be euclidean or native_block_sum')
