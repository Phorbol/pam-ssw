"""Read-only bounded execution of uploaded native H/C pair lookup functions.

This is a reference oracle, never a production runtime dependency. It executes
only mapped original code/rodata pages, has no external call stubs, and cannot
run LASP initialization, calculator or its main program. The resulting raw
lookup numbers are NOT initialized LS amplitudes or final neighbor cutoffs.
"""
import argparse
import hashlib
import json
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RSP, UC_X86_REG_XMM0, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elf', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704':
        raise ValueError('unsupported ELF; entry addresses are version-specific')

    def read(address, size):
        for start, _, data in segments:
            if start <= address and address + size <= start + len(data):
                return data[address - start:address - start + size]
        raise ValueError(f'unmapped ELF bytes at {address:#x}')

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for address, size in [(0x6cb000, 0x2000), (0x4a4c000, 0x1000), (0x70000000, 0x2000)]:
        uc.mem_map(address, size)
        if address != 0x70000000:
            uc.mem_write(address, read(address, size))
    rows = []
    for name, entry in [('bondeneval_', 0x6cb660), ('bondlenval_', 0x6cc0b0)]:
        for a, b in [(1, 1), (1, 6), (6, 1), (6, 6)]:
            uc.mem_write(0x70000000, struct.pack('<ii', a, b))
            uc.reg_write(UC_X86_REG_RDI, 0x70000000)
            uc.reg_write(UC_X86_REG_RSI, 0x70000004)
            sp, stop = 0x70001808, 0x70001000
            uc.mem_write(sp, struct.pack('<Q', stop))
            uc.reg_write(UC_X86_REG_RSP, sp)
            uc.reg_write(UC_X86_REG_XMM0, 0)
            uc.emu_start(entry, stop, count=10000, timeout=1000000)
            if uc.reg_read(UC_X86_REG_RIP) != stop:
                raise RuntimeError('instruction budget exhausted before normal return')
            value = struct.unpack('<d', struct.pack('<Q', uc.reg_read(UC_X86_REG_XMM0) & ((1 << 64) - 1)))[0]
            rows.append(dict(function=name, entry=hex(entry), pair=[a, b], raw_return=value))
    print(json.dumps(dict(elf_sha256=digest, rows=rows,
        static_len_toller=struct.unpack('<d', read(0x5520568, 8))[0],
        interpretation='raw lookup functions only; caller scaling/filters require separate audit'), indent=2))


if __name__ == '__main__':
    main()
