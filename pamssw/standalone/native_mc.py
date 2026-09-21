"""Independent translation of the uploaded ELF's stateful MC acceptance.

See docs/research/native-mc-contract.md for instruction provenance and domains.
This is opt-in, not the paper-reference driver's ordinary Metropolis policy.
"""
from dataclasses import dataclass
import math
from numbers import Integral


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if not 0 <= value <= 2**31 - 1:
        raise ValueError(f'{name} must be a nonnegative signed 32-bit integer')
    return value


def native_power10(exponent: int) -> int:
    """ELF __powi4i4(10, exponent): truncate negative powers, wrap int32.

The wrap is a recovered implementation defect, not a useful heating strategy.
"""
    if isinstance(exponent, bool) or not isinstance(exponent, Integral):
        raise TypeError('exponent must be an integer')
    if not -(2**31) <= exponent < 2**31:
        raise ValueError('exponent must fit signed 32-bit')
    if exponent < 0:
        return 0
    unsigned = pow(10, int(exponent), 2**32)
    return unsigned if unsigned < 2**31 else unsigned - 2**32


@dataclass(frozen=True)
class NativeMCState:
    """Per-walker replacement for the ELF's process-static NSAME counter."""
    nsame: int = 0

    def __post_init__(self):
        _integer(self.nsame, 'nsame')


@dataclass(frozen=True)
class NativeMCSettings:
    """Explicit settings for the recovered native acceptance policy."""
    energy_tol: float
    maxtrap: int

    def __post_init__(self):
        if not math.isfinite(self.energy_tol) or self.energy_tol < 0:
            raise ValueError('energy_tol must be finite and nonnegative')
        _integer(self.maxtrap, 'maxtrap')


@dataclass(frozen=True)
class NativeMCDecision:
    accepted: bool
    state: NativeMCState
    nsame_for_acceptance: int
    near_equal_energy: bool
    delta_energy_eV: float
    temperature_increment_K: int
    effective_temperature_K: float
    acceptance_probability: float
    uniform: float


def native_metropolis(energy1: float, energy2: float, temperature_K: float, *,
                      energy_tol: float, maxtrap: int, state: NativeMCState,
                      uniform: float) -> NativeMCDecision:
    """Reproduce the finite, positive-effective-temperature native MC domain.

    Energies and tolerance use eV, temperature K. ``uniform`` must be supplied on
    EVERY call, including downhill moves: native rd_numb is called unconditionally.
    RNG sequence generation is deliberately outside this deterministic function.
    The recovered fixed divisor 20 is retained, not replaced by modern ASE kB.
    """
    maxtrap = _integer(maxtrap, 'maxtrap')
    if not isinstance(state, NativeMCState):
        raise TypeError('state must be NativeMCState')
    if not all(math.isfinite(v) for v in (energy1, energy2, temperature_K,
                                         energy_tol, uniform)):
        raise ValueError('MC inputs must be finite')
    if temperature_K <= 0 or energy_tol < 0 or not 0 <= uniform <= 1:
        raise ValueError('require positive temperature, nonnegative tolerance, uniform in [0,1]')
    delta = energy2 - energy1
    if not math.isfinite(delta):
        raise ValueError('energy difference must be finite')
    near = abs(delta) < energy_tol
    count = int(state.nsame) + int(near)
    if count > 2**31 - 1:
        raise OverflowError('native NSAME counter overflow is outside the supported domain')
    increment = native_power10(count - maxtrap)
    effective = temperature_K + increment
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError('native integer overflow produced nonpositive effective temperature')
    # Preserve the ELF operation order and constants. For downhill moves the
    # native exp can overflow but acceptance is unconditional; no exp is needed.
    probability = 1. if delta <= 0 else math.exp(((delta / 20.) * 96485. / -8.314) / effective)
    accepted = delta <= 0 or uniform <= probability
    next_count = 0 if accepted and not near else count
    return NativeMCDecision(bool(accepted), NativeMCState(next_count), count,
                            bool(near), float(delta), increment, float(effective),
                            float(probability), float(uniform))
