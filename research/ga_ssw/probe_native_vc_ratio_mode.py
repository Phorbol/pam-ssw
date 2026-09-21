"""Isolated native get_random_mode0 ratio gate; no PES or LASP main."""

import hashlib
import json
import struct
from pathlib import Path

from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64, Uc
from unicorn.x86_const import (
    UC_X86_REG_RDI,
    UC_X86_REG_RIP,
    UC_X86_REG_RSP,
)

from research.ga_ssw.probe_native_weight_emulated import load_elf


ELF = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
ENTRY = 0x5E80D0
END = 0x5E830F
PARA = 0x53ED7A0
STOP = 0x700000000000
STACK = 0x710000000000
DATA = 0x720000000000
DESCRIPTOR = DATA
OBJECT = DATA + 0x1000
VTABLE = DATA + 0x2000
STUB_BASE = DATA + 0x3000
STUBS = {
    0x210: STUB_BASE + 0x00,
    0x218: STUB_BASE + 0x10,
    0x1B0: STUB_BASE + 0x20,
    0x190: STUB_BASE + 0x30,
}


def put_i32(uc, address, value):
    uc.mem_write(address, struct.pack("<i", value))


def get_i32(uc, address):
    return struct.unpack("<i", uc.mem_read(address, 4))[0]


def run_case(ratio, nsswstep):
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == SHA256
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, memsz, chunk in segments:
        start = va & ~0xFFF
        size = ((va + memsz + 0xFFF) & ~0xFFF) - start
        uc.mem_map(start, size)
        if chunk:
            uc.mem_write(va, chunk)
    for address, size in ((STOP, 0x1000), (STACK, 0x20000), (DATA, 0x10000)):
        uc.mem_map(address, size)

    uc.mem_write(DESCRIPTOR, struct.pack("<Q", OBJECT))
    uc.mem_write(DESCRIPTOR + 0x38, struct.pack("<Q", VTABLE))
    for slot, target in STUBS.items():
        uc.mem_write(VTABLE + slot, struct.pack("<Q", target))
        uc.mem_write(target, b"\xc3")

    # Disable diagnostic writes and leave the mode vector writable.  These are
    # input state bits only; all ratio arithmetic below executes native bytes.
    put_i32(uc, PARA + 0x2DDEC, 0)
    put_i32(uc, PARA + 0x130, 0)
    put_i32(uc, PARA + 0x2DB58, ratio)
    put_i32(uc, OBJECT + 0x2A74, nsswstep)
    put_i32(uc, OBJECT + 0x2260, 123456)

    return_address = STOP
    stack = STACK + 0x1FFF0
    uc.mem_write(stack, struct.pack("<Q", return_address))
    uc.reg_write(UC_X86_REG_RSP, stack)
    uc.reg_write(UC_X86_REG_RDI, DESCRIPTOR)

    trace = []

    def hook(machine, address, size, user):
        trace.append(address)
        if ENTRY <= address < END:
            return
        if address in STUBS.values():
            return
        if address == STOP:
            return
        raise RuntimeError(f"unexpected PC {address:#x}")

    uc.hook_add(UC_HOOK_CODE, hook)
    uc.emu_start(ENTRY, STOP, count=1000)
    assert uc.reg_read(UC_X86_REG_RIP) == STOP
    assert any(pc == 0x5E8112 for pc in trace)
    assert any(pc == 0x5E814B for pc in trace) == (ratio > 0)
    assert any(pc == 0x5E8152 for pc in trace) == (ratio > 0)
    assert any(pc == 0x5E82EA for pc in trace) == (ratio < 0)

    remainder = nsswstep % abs(ratio) if ratio else None
    cell = (remainder == 1) if ratio > 0 else (ratio < 0 and remainder != 1)
    expected = -1 if cell else 0
    actual = get_i32(uc, OBJECT + 0x2260)
    assert actual == expected
    expected_slot = 0x210 if expected == -1 else 0x218
    assert STUBS[expected_slot] in trace
    return {
        "ratio_atomcell": ratio,
        "object_nsswstep": nsswstep,
        "expected_lcellmove": expected,
        "actual_lcellmove": actual,
        "selected_slot": hex(expected_slot),
        "trace": [hex(pc) for pc in trace],
    }


def main():
    rows = [run_case(ratio, nsswstep) for ratio in (0, 1, 2, 5, -1, -2, -5) for nsswstep in (0, 1, 2, 5, 6)]
    report = {
        "elf": ELF,
        "elf_sha256": SHA256,
        "entry": hex(ENTRY),
        "scope": "native ratio/lcellmove gate only; parameter/object memory stubbed; no PES, main loop, or production strategy",
        "formula": "For nonnegative nsswstep: ratio=0 false; ratio>0 true iff nsswstep%ratio==1; ratio<0 true iff nsswstep%abs(ratio)!=1",
        "cases": rows,
    }
    output = Path("research/ga_ssw/evidence/native-vc-ratio-mode.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases":len(rows),"all_passed":True,"formula":report["formula"]}))


if __name__ == "__main__":
    main()
