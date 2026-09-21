"""Verify the post-update CBD guard using the uploaded ELF instructions only.

Stops before either branch performs mutations/callbacks; no LASP main or PES.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_R13, UC_X86_REG_RSP, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256

DATA = 0x720000010000
STACK = 0x720000020000
ENTRY = 0x5ccba4
REENTER = 0x5ccbcf
STOP_CLIMB = 0x5ccbf8


def run(segments, status):
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, size, chunk in segments:
        low = va & ~4095
        uc.mem_map(low, ((va + size + 4095) & ~4095) - low)
        uc.mem_write(va, chunk)
    for address in (DATA, STACK):
        uc.mem_map(address, 0x10000)
    obj = DATA + 0x1000
    uc.mem_write(DATA, struct.pack('<Q', obj))
    uc.mem_write(obj + 0x1b34, status.encode().ljust(30, b' '))
    uc.reg_write(UC_X86_REG_R13, DATA)
    uc.reg_write(UC_X86_REG_RSP, STACK + 0x8000)
    calls = []
    def hook(machine, address, size, context):
        if address in (REENTER, STOP_CLIMB):
            machine.emu_stop()
        elif address == 0x49a8420:
            calls.append(address)
        elif not 0x400000 <= address < 0x8000000:
            raise RuntimeError(f'unexpected instruction {address:#x}')
    uc.hook_add(UC_HOOK_CODE, hook)
    uc.emu_start(ENTRY, STOP_CLIMB + 16, timeout=1_000_000, count=100_000)
    end = uc.reg_read(UC_X86_REG_RIP)
    expected = STOP_CLIMB if status.rstrip() == 'Allopt' else REENTER
    return dict(status=status, stop=hex(end), reenters_cbd=end == REENTER,
                runtime_comparisons=len(calls), passed=end == expected and len(calls) == 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elf', default=ELF_DEFAULT)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256:
        raise ValueError('unsupported ELF')
    cases = [run(segments, status) for status in ('Allopt', 'climb', 'CBD', 'Allopt_softPES')]
    report = dict(elf=args.elf, sha256=digest, entry=hex(ENTRY), cases=cases,
                  scope='post-update status predicate only; run_type=5 reachability and update_mode0 not executed; real for_cpstr; zero PES calls',
                  passed=all(case['passed'] for case in cases))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
