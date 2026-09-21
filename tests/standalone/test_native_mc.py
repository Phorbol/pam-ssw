"""Recovered MC state boundaries; these are not search efficacy tests."""
import math

import pytest

from pamssw.standalone.native_mc import NativeMCState, native_metropolis, native_power10


def test_native_integer_power_including_observed_wrap():
    assert [native_power10(k) for k in (-2, -1, 0, 1, 9, 10, 12, 32)] == [
        0, 0, 1, 10, 1000000000, 1410065408, -727379968, 0]


def test_repeated_near_energies_increment_before_temperature_and_not_reset():
    state = NativeMCState()
    temperatures = []
    for _ in range(4):
        decision = native_metropolis(0., 0., 300., energy_tol=.001,
                                     maxtrap=2, state=state, uniform=.99)
        assert decision.accepted
        state = decision.state
        temperatures.append(decision.effective_temperature_K)
    assert state.nsame == 4
    assert temperatures == [300., 301., 310., 400.]


def test_twenty_factor_rejection_preserves_state_acceptance_resets():
    state = NativeMCState(3)
    rejected = native_metropolis(0., 1., 300., energy_tol=.001,
                                maxtrap=5, state=state, uniform=.9)
    assert rejected.acceptance_probability == pytest.approx(math.exp(-1/20*96485/8.314/300))
    assert not rejected.accepted
    assert rejected.state == state
    accepted = native_metropolis(0., 1., 300., energy_tol=.001,
                                maxtrap=5, state=state, uniform=.1)
    assert accepted.accepted
    assert accepted.state.nsame == 0


def test_exact_tolerance_is_different_and_probability_equality_accepts():
    first = native_metropolis(0., .125, 300., energy_tol=.125,
                             maxtrap=5, state=NativeMCState(2), uniform=0.)
    assert not first.near_equal_energy
    equal = native_metropolis(0., .125, 300., energy_tol=.125,
                             maxtrap=5, state=NativeMCState(2),
                             uniform=first.acceptance_probability)
    assert equal.accepted and equal.state.nsame == 0


def test_downhill_is_accepted_even_if_exponential_overflows():
    d = native_metropolis(0., -1e6, 1., energy_tol=.001,
                          maxtrap=5, state=NativeMCState(2), uniform=.999)
    assert d.accepted and d.acceptance_probability == 1. and d.state.nsame == 0


def test_native_overflow_cannot_silently_make_negative_temperature():
    with pytest.raises(ValueError, match='effective temperature'):
        native_metropolis(0., 0., 300., energy_tol=.001, maxtrap=1,
                          state=NativeMCState(12), uniform=.5)


@pytest.mark.parametrize('kwargs', [{'uniform': float('nan')}, {'energy_tol': -1},
                                    {'temperature_K': 0}, {'maxtrap': 1.5}])
def test_invalid_inputs_fail_explicitly(kwargs):
    args = dict(energy1=0., energy2=.01, temperature_K=300., energy_tol=.001,
                maxtrap=5, state=NativeMCState(), uniform=.5)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        native_metropolis(**args)


def test_relocated_original_mc_machine_code_oracle():
    """Opt-in instruction-level comparison; no LASP job or calculator is run.

    The original MC and integer-power instructions are copied; RIP references
    are relocated, RNG replaced by a supplied scalar, exp bound to host libm.
    Consequently this checks control flow, not the original RNG/libm bitstream.
    """
    import ctypes
    import hashlib
    import mmap
    import os
    import platform
    import struct
    from pathlib import Path

    filename = os.environ.get('PAMSSW_NATIVE_MC_ELF')
    if filename is None:
        pytest.skip('set PAMSSW_NATIVE_MC_ELF for the uploaded ELF slice oracle')
    if platform.machine() != 'x86_64':
        pytest.skip('uploaded ELF is x86_64')
    elf = Path(filename).read_bytes()
    assert elf[:6] == b'\x7fELF\x02\x01'
    phoff = struct.unpack_from('<Q', elf, 32)[0]
    entsize, count = struct.unpack_from('<HH', elf, 54)
    segments = [struct.unpack_from('<IIQQQQQQ', elf, phoff + i*entsize)
                for i in range(count)]

    def extract(address, size):
        for kind, flags, offset, va, pa, filesz, memsz, align in segments:
            if kind == 1 and va <= address and address + size <= va + filesz:
                return elf[offset + address-va:offset + address-va + size]
        raise AssertionError(f'ELF address unavailable: {address:x}')

    mc_address = 0x57e820
    code = bytearray(extract(mc_address, 0x112))
    # Fail rather than execute an unknown ELF build.
    assert hashlib.sha256(code).hexdigest() == (
        '822853160c04e2a39345f1e9451e8a798ccac5a9655f4015536a06e842b3d1ab')
    assert hashlib.sha256(extract(0x4923ec0, 0x78)).hexdigest() == (
        '271faa222e48f38ec4df327a2b8d0d457897393be2cf163b21373cc56a9920f6')
    assert extract(0x4a43848, 24) == struct.pack('<ddd', 20., 96485., -8.314)
    memory = mmap.mmap(-1, 0x31000,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    nsame_offset, para_offset = 0x400, 0x1000
    targets = {
        0x57e841: (1, 5, 0x300),  # rd_numb
        0x57e846: (3, 7, para_offset),
        0x57e86b: (3, 7, 0x410),  # absolute-value mask
        0x57e87d: (2, 6, nsame_offset),
        0x57e885: (2, 6, nsame_offset),
        0x57e88d: (2, 6, nsame_offset),
        0x57e89e: (1, 5, 0x200),  # __powi4i4, base always ten
        0x57e8ac: (4, 8, 0x420),
        0x57e8b4: (4, 8, 0x428),
        0x57e8c0: (4, 8, 0x430),
        0x57e8d2: (1, 5, 0x320),  # exp
        0x57e919: (2, 10, nsame_offset),
    }
    for address, (disp, length, target) in targets.items():
        position = address - mc_address
        struct.pack_into('<i', code, position + disp, target - position - length)
    memory[:len(code)] = code
    memory[0x200:0x278] = extract(0x4923ec0, 0x78)
    memory[0x410:0x420] = extract(0x4a43740, 16)
    memory[0x420:0x438] = extract(0x4a43848, 24)
    draw = [0.]
    draws = [0]
    scalar_pointer = ctypes.POINTER(ctypes.c_double)

    @ctypes.CFUNCTYPE(None, scalar_pointer)
    def fixed_random(output):
        output[0] = draw[0]
        draws[0] += 1

    libm = ctypes.CDLL('libm.so.6')
    for at, target in [(0x300, fixed_random), (0x320, libm.exp)]:
        pointer = ctypes.cast(target, ctypes.c_void_p).value
        memory[at:at+12] = b'\x48\xb8' + struct.pack('<Q', pointer) + b'\xff\xe0'
    function = ctypes.CFUNCTYPE(None, scalar_pointer, scalar_pointer,
                               scalar_pointer, ctypes.POINTER(ctypes.c_int32))(base)
    tested = 0
    try:
        for same in (0, 1, 2, 3, 10, 11, 34):
            for delta in (-10., -.125, -.01, 0., .01, .125, 1., 10.):
                for uniform in (0., .1, .5, 1.):
                    struct.pack_into('<i', memory, nsame_offset, same)
                    struct.pack_into('<i', memory, para_offset + 0x12c, 2)
                    struct.pack_into('<d', memory, para_offset + 0x2de98, .125)
                    draw[0] = uniform
                    e1, e2, temperature = map(ctypes.c_double, (0., delta, 300.))
                    accepted = ctypes.c_int32()
                    function(ctypes.byref(e1), ctypes.byref(e2),
                             ctypes.byref(temperature), ctypes.byref(accepted))
                    actual_same = struct.unpack_from('<i', memory, nsame_offset)[0]
                    expected = native_metropolis(0., delta, 300., energy_tol=.125,
                                                 maxtrap=2, state=NativeMCState(same),
                                                 uniform=uniform)
                    assert bool(accepted.value) == expected.accepted
                    assert actual_same == expected.state.nsame
                    tested += 1
        assert tested == draws[0] == 224
    finally:
        memory.close()
