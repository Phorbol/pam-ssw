"""Isolated native fixcell_climb force-mask branch; no BFGS/PES/main."""

import hashlib
import json
import struct
from pathlib import Path

from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64, Uc
from unicorn.x86_const import (
    UC_X86_REG_R13,
    UC_X86_REG_R14,
    UC_X86_REG_R15,
    UC_X86_REG_RBP,
    UC_X86_REG_RBX,
    UC_X86_REG_RIP,
)

from research.ga_ssw.probe_native_weight_emulated import load_elf


ELF = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
ENTRY = 0x5F25A2
STOP = 0x5F25B9
EXHAUST = 0x5F41BC
STACK = 0x710000000000
DATA = 0x720000000000
PARA = DATA
OBJECT = DATA + 0x1000
FA = DATA + 0x3000
DESCRIPTOR = DATA + 0x4000


def put_i32(uc, address, value):
    uc.mem_write(address, struct.pack("<i", value))


def put_i64(uc, address, value):
    uc.mem_write(address, struct.pack("<Q", value))


def read_qwords(uc, address, count):
    return list(struct.unpack("<" + "Q" * count, uc.mem_read(address, 8 * count)))


def run_case(fixcell):
    blob, segments = load_elf(ELF)
    assert hashlib.sha256(blob).hexdigest() == SHA256
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, memsz, chunk in segments:
        start = va & ~0xFFF
        size = ((va + memsz + 0xFFF) & ~0xFFF) - start
        uc.mem_map(start, size)
        if chunk:
            uc.mem_write(va, chunk)
    uc.mem_map(STACK, 0x20000)
    uc.mem_map(DATA, 0x400000)

    # Synthetic structure layout: five rows of three doubles: two atomic
    # rows followed by three cell rows.  The native address formula uses the
    # byte stride at +0x220 and lower bound at +0x228, so rows 3,4,5 map to
    # FA+48, FA+72, and FA+96 respectively.
    put_i32(uc, PARA + 0x2DBE0, fixcell)
    put_i64(uc, OBJECT, 2)
    put_i64(uc, OBJECT + 0x200, 3)
    put_i64(uc, OBJECT + 0x220, 24)
    put_i64(uc, OBJECT + 0x228, 1)
    put_i64(uc, OBJECT + 0x1D0, FA)
    put_i64(uc, DESCRIPTOR, OBJECT)
    original = [0x1111111111111111 + i for i in range(15)]
    uc.mem_write(FA, struct.pack("<" + "Q" * len(original), *original))

    uc.reg_write(UC_X86_REG_RBX, PARA)
    uc.reg_write(UC_X86_REG_R15, OBJECT)
    uc.reg_write(UC_X86_REG_R14, 0)
    uc.reg_write(UC_X86_REG_R13, DESCRIPTOR)
    uc.reg_write(UC_X86_REG_RBP, STACK + 0x10000)
    trace = []

    def hook(machine, address, size, user):
        trace.append(address)
        if ENTRY <= address < EXHAUST or address == STOP:
            return
        raise RuntimeError(f"unexpected PC {address:#x}")

    uc.hook_add(UC_HOOK_CODE, hook)
    try:
        uc.emu_start(ENTRY, STOP, count=500)
    except Exception as error:
        raise RuntimeError(f"native probe failed at {[hex(pc) for pc in trace]}") from error
    assert uc.reg_read(UC_X86_REG_RIP) == STOP
    got = read_qwords(uc, FA, len(original))
    if fixcell:
        assert got[:6] == original[:6]
        assert got[6:] == [0] * 9
        assert all(pc in trace for pc in (0x5F404E, 0x5F419A, 0x5F4195))
        assert 0x5F40D1 not in trace
    else:
        assert got == original
        assert 0x5F404E not in trace
    return {
        "fixcell_climb": fixcell,
        "fa_before": [hex(v) for v in original],
        "fa_after": [hex(v) for v in got],
        "trace": [hex(pc) for pc in trace],
        "stop": hex(uc.reg_read(UC_X86_REG_RIP)),
    }


def main():
    rows = [run_case(0), run_case(1)]
    report = {
        "elf": ELF,
        "elf_sha256": SHA256,
        "entry": hex(ENTRY),
        "stop": hex(STOP),
        "exhaust_guard": f"{ENTRY:#x} <= pc < {EXHAUST:#x}, or pc == {STOP:#x}",
        "synthetic_layout": {
            "fa_shape": [5, 3],
            "atomic_rows": 2,
            "cell_rows": 3,
            "object_plus_0x200": 3,
            "object_plus_0x220_byte_stride": 24,
            "object_plus_0x228_lower_bound": 1,
            "cell_fa_indices": [6, 7, 8, 9, 10, 11, 12, 13, 14],
            "cell_fa_addresses": ["FA+48..FA+120 (24-byte row stride)"],
        },
        "expected_branch_checks": ["0x5f404e", "0x5f419a", "0x5f4195"],
        "memset_call": "not executed: fa dimension is 3, short path",
        "scope": "native fixcell_climb force-input branch only; synthetic object/Fortran-like storage; no BFGS, PES, LASP main, or production strategy",
        "claim": "fixcell_climb=true zeros the synthetic three cell-force entries while atomic fa entries remain unchanged; this does not prove coordinate freezing",
        "cases": rows,
    }
    output = Path("research/ga_ssw/evidence/native-fixcell-climb-mask.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "all_passed": True}))


if __name__ == "__main__":
    main()
