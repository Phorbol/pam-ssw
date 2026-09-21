"""Bounded H/C/O execution of the original pair lookup functions.

This maps only the two leaf lookups; it does not initialize LASP or invoke any
external call. Values are raw lookup returns, before caller scaling/filters.
"""
import argparse
import hashlib
import json
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RSP, UC_X86_REG_XMM0, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf


ELF_SHA256 = 'bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704'
LOOKUPS = {
    'bondeneval_': (0x6cb660, 'ELF objdump lookup body; symbol referenced by analysis/ls-bond-info-init.asm; size 0xa50'),
    'bondlenval_': (0x6cc0b0, 'ELF objdump lookup body; symbol referenced by analysis/ls-bond-info-init.asm; size 0xad0'),
}
ELEMENTS = (1, 6, 8)  # H, C, O; atomic-number arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elf', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256:
        raise ValueError('unsupported ELF; entry addresses are version-specific')

    def read(address, size):
        for start, _, data in segments:
            if start <= address and address + size <= start + len(data):
                return data[address - start:address - start + size]
        raise ValueError(f'unmapped ELF bytes at {address:#x}')

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for address, size in [(0x6cb000, 0x3000), (0x4a4c000, 0x1000), (0x70000000, 0x2000)]:
        uc.mem_map(address, size)
        if address != 0x70000000:
            uc.mem_write(address, read(address, size))
    rows = []
    for name, (entry, source) in LOOKUPS.items():
        for a in ELEMENTS:
            for b in ELEMENTS:
                uc.mem_write(0x70000000, struct.pack('<ii', a, b))
                uc.reg_write(UC_X86_REG_RDI, 0x70000000)
                uc.reg_write(UC_X86_REG_RSI, 0x70000004)
                stop = 0x70001000
                uc.reg_write(UC_X86_REG_RSP, 0x70001808)
                uc.mem_write(0x70001808, struct.pack('<Q', stop))
                uc.reg_write(UC_X86_REG_XMM0, 0)
                uc.emu_start(entry, stop, count=10000, timeout=1000000)
                if uc.reg_read(UC_X86_REG_RIP) != stop:
                    raise RuntimeError('instruction budget exhausted before normal return')
                value = struct.unpack('<d', struct.pack('<Q', uc.reg_read(UC_X86_REG_XMM0) & ((1 << 64) - 1)))[0]
                rows.append(dict(function=name, entry=hex(entry), pair=[a, b], raw_return=value, source=source))
    output = dict(elf_sha256=digest, elements={'H': 1, 'C': 6, 'O': 8}, rows=rows,
                  assembly={
                      'source': 'ELF objdump of the lookup bodies; callers/symbol references in analysis/ls-bond-info-init.asm',
                      'functions': {
                          'bondeneval_': {'entry': '0x6cb660', 'end_exclusive': '0x6cc0b0',
                                          'rodata_refs': ['0x4a4c288-0x4a4c468']},
                          'bondlenval_': {'entry': '0x6cc0b0', 'end_exclusive': '0x6cc3f1',
                                          'rodata_refs': ['0x4a4c288-0x4a4c338', '0x4a4c470-0x4a4c638']},
                      },
                      'method': 'objdump -d --start-address/--stop-address; no caller or external call mapped',
                  },
                  static_len_toller=struct.unpack('<d', read(0x5520568, 8))[0],
                  interpretation='raw leaf lookup only; no caller scaling, filters, initialization, or neighbor cutoff')
    with open(args.output, 'w') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
