"""Bounded original-instruction oracle for saved local-cell modes (flag=0).

No PES, LASP initialization, RNG, external calls or replacement stubs. Inputs
are explicit emulated static state, not recovered state from a native search.
Run with python -m research.ga_ssw.probe_native_cell_saved_mode --elf PATH.
"""
import argparse
import hashlib
import json
import math
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import (UC_X86_REG_RDI, UC_X86_REG_RSI,
    UC_X86_REG_RDX, UC_X86_REG_RCX, UC_X86_REG_R8,
    UC_X86_REG_RSP, UC_X86_REG_RIP)
from research.ga_ssw.probe_native_weight_emulated import load_elf

ENTRY = 0x6e4fe0
SAVED_END = 0x6e5381
STATE_Y, STATE_I, STATE_MAX, STATE_MIN = 0x791b8a8, 0x791b8b0, 0x791b8b4, 0x791b8b8
DATA, STACK = 0x70000000, 0x71000000
STOP = STACK + 0x1000
SHA256 = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
# Distinct, nonorthogonal ASE lattice rows a,b,c; no model length scale implied.
CELL = ((2.0, 0.3, 0.1), (0.4, 3.0, 0.2), (0.6, 0.7, 4.0))


def expected_mode(index, y, maximum):
    """Independent row formula; deliberately does not use oracle output."""
    if index == 5:
        v = CELL[maximum - 1]
        rows = [tuple((-1.0 if j == maximum - 1 else 1.0)*x for x in v)
                for j in range(3)]
    else:
        order = {0: (0, 2, 1), 1: (2, 1, 0), 2: (1, 0, 2)}[index]
        sign = -1.0 if y > 0.2 else 1.0
        rows = [tuple(sign*x for x in CELL[j]) for j in order]
    flat = [x for row in rows for x in row]
    norm = math.sqrt(sum(x*x for x in flat))
    return [x/norm for x in flat]


def emulate(segments, index, y, maximum, minimum):
    def original_bytes(address, size):
        for start, memory_size, content in segments:
            if start <= address and address + size <= start + memory_size:
                offset = address - start
                return content[offset:offset+size].ljust(size, b'\0')
        raise ValueError(f'ELF segment does not cover {address:#x}+{size:#x}')

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    pages = ((0x6e4000, 0x2000), (0x4a4c000, 0x1000), (0x791b000, 0x1000))
    for address, size in pages:
        uc.mem_map(address, size)
        uc.mem_write(address, original_bytes(address, size))
    for address in (DATA, STACK):
        uc.mem_map(address, 0x2000)
    uc.mem_write(STATE_Y, struct.pack('<d', y))
    for address, value in ((STATE_I, index), (STATE_MAX, maximum), (STATE_MIN, minimum)):
        uc.mem_write(address, struct.pack('<i', value))
    # Native Fortran columns a,b,c are contiguous, matching flattened ASE rows.
    uc.mem_write(DATA + 0x100, struct.pack('<9d', *(x for row in CELL for x in row)))
    uc.mem_write(DATA + 0x300, struct.pack('<i', 0))
    # rdi/rcx are unused on this branch; provide mapped storage, never callbacks.
    for register, pointer in ((UC_X86_REG_RDI, DATA), (UC_X86_REG_RSI, DATA+0x100),
                              (UC_X86_REG_RDX, DATA+0x200), (UC_X86_REG_RCX, DATA+0x280),
                              (UC_X86_REG_R8, DATA+0x300)):
        uc.reg_write(register, pointer)
    sp = STACK + 0x1808
    uc.mem_write(sp, struct.pack('<Q', STOP))
    uc.reg_write(UC_X86_REG_RSP, sp)
    visited = []

    def check_instruction(machine, address, size, unused):
        if not ENTRY <= address < SAVED_END:
            raise RuntimeError(f'execution escaped saved-mode branch at {address:#x}')
        code = bytes(machine.mem_read(address, size))
        # Branch contains no calls: reject direct/indirect calls before execution.
        if code[0] in (0xe8, 0x9a) or (code[0] == 0xff and (code[1] >> 3) & 7 in (2, 3)):
            raise RuntimeError(f'forbidden call at {address:#x}')
        visited.append(address)

    uc.hook_add(UC_HOOK_CODE, check_instruction)
    uc.emu_start(ENTRY, STOP, count=1000, timeout=1000000)
    if uc.reg_read(UC_X86_REG_RIP) != STOP:
        raise RuntimeError('instruction/time cap reached before normal return')
    actual = list(struct.unpack('<9d', bytes(uc.mem_read(DATA+0x200, 72))))
    expected = expected_mode(index, y, maximum)
    difference = [a-b for a, b in zip(actual, expected)]
    error = max(abs(x) for x in difference)
    return dict(index=index, y=y, max_row=maximum, min_row=minimum, flag=0,
                executed_addresses=[hex(x) for x in visited], actual=actual,
                expected=expected, difference=difference, max_abs_error=error,
                passed=all(math.isfinite(x) for x in actual) and error <= 1e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elf', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        raise ValueError('unsupported ELF: addresses are version-specific')
    cases = [(i, y, 1, 2) for i in (0, 1, 2) for y in (0.1, 0.3)]
    cases += [(5, y, maximum, minimum) for maximum in (1, 2, 3)
              for minimum in (1, 2, 3) if minimum != maximum for y in (0.1, 0.3)]
    rows = [emulate(segments, *case) for case in cases]
    passed = all(row['passed'] for row in rows)
    print(json.dumps(dict(elf_sha256=digest, entry=hex(ENTRY),
          mapped_original_pages=['0x6e4000+0x2000', '0x4a4c000+0x1000', '0x791b000+0x1000'],
          input_ase_cell_rows=CELL, rows=rows, passed=passed,
          scope='saved-state local-cell algebra only; no PES or search-effect evidence'),
          indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
