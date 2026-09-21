"""Isolate VC judge prefix and pressure diagnostic; no main/PES/optimizer calls."""
import hashlib
import itertools
import json
import struct
from pathlib import Path

import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import *
from research.ga_ssw.probe_native_weight_emulated import load_elf, DATA, STACK

ELF = '/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'
PARA, CONTROL = 0x53ed7a0, 0x53ed5c0


def machine(segments):
    u = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, ms, data in segments:
        start = va & ~4095
        u.mem_map(start, ((va + ms + 4095) & ~4095) - start)
        u.mem_write(va, data)
    u.mem_map(DATA, 0x100000)
    u.mem_map(STACK, 0x100000)
    u.reg_write(UC_X86_REG_RSP, STACK + 0x80008)
    return u


def scalar(u, address, value):
    u.mem_write(address, struct.pack('<d', value))


def execute(u, start, stop, extra=()):
    def guard(engine, pc, size, data):
        if not (start <= pc < stop or any(a <= pc < b for a, b in extra)):
            raise RuntimeError(f'outside isolated arithmetic: {pc:x}')
    u.hook_add(UC_HOOK_CODE, guard)
    u.emu_start(start, stop, timeout=1000000, count=10000)
    assert u.reg_read(UC_X86_REG_RIP) == stop


def main():
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
    rows = []
    # Include exact threshold: native uses strict <, not <=.
    for sfa_max, fa_max, pressure_residual in itertools.product([.04, .05, .06], repeat=3):
        u = machine(segments)
        obj, arr = DATA + 0x1000, DATA + 0x5000
        u.mem_write(DATA, struct.pack('<Q', obj))
        u.mem_write(obj + 0x4d8, struct.pack('<Q', arr))
        # Fortran rank-two array, 3 components x 2 columns, column stride 24.
        for offset, value in [(0x508, 3), (0x520, 2), (0x528, 24), (0x530, 1)]:
            u.mem_write(obj + offset, struct.pack('<q', value))
        u.mem_write(arr, np.array([-.5*sfa_max, sfa_max, 0., 0., 0., 0.], dtype='<f8').tobytes())
        scalar(u, PARA + 0x2db28, .05)
        scalar(u, PARA + 0x2db30, .05)
        scalar(u, CONTROL + 0x18, fa_max)
        scalar(u, CONTROL + 0x28, pressure_residual)
        u.reg_write(UC_X86_REG_RDI, DATA)
        u.reg_write(UC_X86_REG_RSI, DATA + 0x8000)
        execute(u, 0x5f6950, 0x5f6bbb, [(0x5f6eae, 0x5f6eba)])
        observed = bool(struct.unpack('<i', u.mem_read(CONTROL + 0xc0, 4))[0])
        expected = sfa_max < .05 or (fa_max < .05 and pressure_residual < .05)
        assert observed == expected
        rows.append(dict(sfa_max=sfa_max, fa_max=fa_max, pressure_residual=pressure_residual, converged=observed))
    pressure_rows = []
    for name, stress, pressure in [
        ('hydrostatic', np.eye(3)*.01, 0.),
        ('deviatoric', np.diag([.2, -.2, 0.]), 0.),
        ('shear', np.array([[0., .3, 0.], [.3, 0., 0.], [0., 0., 0.]]), 0.),
        ('external_pressure_cancel', -np.eye(3)*.02, .02),
    ]:
        u = machine(segments)
        obj = DATA + 0x1000
        u.mem_write(obj + 0x128, stress.astype('<f8').tobytes(order='F'))
        scalar(u, PARA + 0x2db38, pressure)
        u.reg_write(UC_X86_REG_R15, obj)
        u.reg_write(UC_X86_REG_R12, PARA)
        u.reg_write(UC_X86_REG_RBP, STACK + 0x80000)
        factor = struct.unpack('<d', u.mem_read(0x54c0420, 8))[0]
        divisor = struct.unpack('<d', u.mem_read(0x4a46cf8, 8))[0]
        execute(u, 0x5e451b, 0x5e459c)
        observed = struct.unpack('<d', u.mem_read(CONTROL + 0x28, 8))[0]
        expected = abs(np.trace(stress)/3 + pressure)*factor
        assert divisor == 3. and abs(observed-expected) < 1e-12
        pressure_rows.append(dict(name=name, stress=stress.tolist(), pressure=pressure, residual_gpa=observed, conversion=factor))
    out = dict(elf_sha256=hashlib.sha256(blob).hexdigest(), scope='isolated arithmetic only; no full stop/release trajectory', new_PES=0, judge_rows=rows, pressure_rows=pressure_rows)
    path = Path('research/ga_ssw/evidence/native-stress-producer-review/vc-convergence-prefix.json')
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(judge_cases=len(rows), pressure_cases=pressure_rows), indent=2))


if __name__ == '__main__':
    main()
