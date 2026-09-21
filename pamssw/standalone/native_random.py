"""Isolated executable-reference random primitives, not the default RNG.

Recovered VMB2/VELO_LOC/RAN3 preserve seed quantization and the promoted
single-precision uniform scale. They must not be described as an isotropic
continuous sampler when the caller supplies one uniform scalar in [0,1).
"""
import math
import numpy as np


def native_ran3(seed):
    """Instance-local stream matching the executable's negative-seed reset."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not -100000000 < seed < 0:
        raise ValueError('requires a negative integer seed in the recovered range')
    modulus = 1000000000
    table = [0]*56  # Retain native 1-based indexing.
    mj = (161803398-abs(int(seed))) % modulus
    table[55] = mj
    mk = 1
    for i in range(1, 55):
        index = (21*i) % 55
        table[index] = mk
        mk = (mj-mk) % modulus
        mj = table[index]
    for _ in range(4):
        for i in range(1, 56):
            table[i] = (table[i]-table[1+(i+30) % 55]) % modulus
    index, partner = 0, 31
    while True:
        index = index % 55+1
        partner = partner % 55+1
        value = (table[index]-table[partner]) % modulus
        table[index] = value
        yield value*9.999999717180685e-10


def native_vmb2(initial, mask, seed_input, *, temperature=300.):
    """Unnormalised reference random component; disabled entries are retained.

    The scalar ``seed_input`` is quantized as ``-int((seed_input+1)*10)``.
    There are ten primary bins (-10 through -19). At the floating-point
    upper edge, addition can round to 2 and produce seed -20.
    The recovered scale is 0.00172309*sqrt(T); no atomic masses enter.
    Mean subtraction includes retained entries and all atoms.
    """
    result = np.array(initial, dtype=float, copy=True)
    flags = np.asarray(mask)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result) or not np.isfinite(result).all():
        raise ValueError('initial must be a finite nonempty (N,3) array')
    if flags.shape != result.shape or flags.dtype != np.bool_:
        raise ValueError('mask must be a shape-matched boolean array')
    seed_input, temperature = float(seed_input), float(temperature)
    if not np.isfinite(seed_input) or not 0 <= seed_input < 1:
        raise ValueError('seed_input must lie in [0,1)')
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    draws = native_ran3(-int((seed_input+1)*10))
    scale = .00172309*math.sqrt(temperature)
    for i in range(len(result)):
        for j in range(3):
            if flags[i, j]:
                u, v = next(draws), next(draws)
                result[i,j] = scale*math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)
    result -= result.mean(axis=0)
    return result
