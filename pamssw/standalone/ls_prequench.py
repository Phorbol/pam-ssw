"""Fixed-cell LS limits and opt-in normal iteration-cap release policy."""
from dataclasses import dataclass, is_dataclass, replace
import numpy as np


@dataclass(frozen=True)
class LSPrequenchSettings:
    fmax: float  # Softened per-atom force norm, eV/Angstrom.
    steps: int  # Optimizer iterations, not physical E/F requests.
    exit_policy: str = 'force'

    def __post_init__(self):
        if isinstance(self.fmax, (bool, np.bool_)) or not np.isscalar(self.fmax) or not np.isfinite(self.fmax) or self.fmax <= 0:
            raise ValueError('LS prequench fmax must be positive finite')
        if isinstance(self.steps, (bool, np.bool_)) or not isinstance(self.steps, (int, np.integer)) or self.steps < 0:
            raise ValueError('LS prequench steps must be a nonnegative integer')
        if self.exit_policy not in ('force', 'force_or_step_limit'):
            raise ValueError("LS prequench exit_policy must be 'force' or 'force_or_step_limit'")


def validate_prequench(value):
    if value is not None and not isinstance(value, LSPrequenchSettings):
        raise TypeError('prequench must be LSPrequenchSettings or None')


def normalize_prequench_settings(value):
    """Canonicalize old pickles lacking prequench or its policy field."""
    if is_dataclass(value) and 'prequench' in value.__dataclass_fields__:
        prequench = getattr(value, 'prequench', None)
        if prequench is not None and is_dataclass(prequench):
            fields = prequench.__dataclass_fields__
            if 'exit_policy' not in fields:
                prequench = LSPrequenchSettings(prequench.fmax, prequench.steps)
            else:
                prequench = replace(prequench, exit_policy=getattr(prequench, 'exit_policy', 'force'))
        return replace(value, prequench=prequench)
    return value


def validate_prequench_exit_policy(prequench, *, optimizer):
    """Validate the optional iteration-cap release before any PES request."""
    validate_prequench(prequench)
    if prequench is None or prequench.exit_policy == 'force':
        return
    if optimizer != 'safe-lbfgs-total':
        raise ValueError("exit_policy='force_or_step_limit' requires safe-lbfgs-total telemetry")
