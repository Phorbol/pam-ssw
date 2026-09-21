"""Bounded Unicorn probe for LASP localatomgroup_mode (no PES or caller state).

The direct caller at 0x5d8255 establishes this ABI:
  rdi=&natoms, rsi=coords (Fortran 3-by-N doubles), rdx=unused here,
  rcx=integer freedom mask, r8=output (3-by-N doubles), r9=&[i,j],
  stack arg 7=&group membership mask.
"""
import hashlib
import json
import struct

import numpy as np
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RCX, UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_RSP

from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
ENTRY = 0x6e4c00
STOP = 0x700000000000
STACK = 0x710000000000
DATA = 0x720000000000
RET = 0x700000000100


def _ret(u):
    sp = u.reg_read(UC_X86_REG_RSP)
    u.reg_write(UC_X86_REG_RIP, struct.unpack("<Q", u.mem_read(sp, 8))[0])
    u.reg_write(UC_X86_REG_RSP, sp + 8)


def run(coords, pair=(1, 2), freedom=None, groups=None):
    blob, segments = load_elf(ELF)
    u = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, memsz, chunk in segments:
        lo = va & ~4095
        u.mem_map(lo, ((va + memsz + 4095) & ~4095) - lo)
        if chunk:
            u.mem_write(va, chunk)
    for base, size in ((STOP, 0x2000), (STACK, 0x200000), (DATA, 0x200000)):
        u.mem_map(base, size)

    x = np.asarray(coords, dtype="<f8")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("coords must have shape (N,3)")
    n = len(x)
    if freedom is None:
        freedom = np.ones(3 * n, dtype="<i4")
    if groups is None:
        groups = np.ones(n, dtype="<i4")
    freedom = np.asarray(freedom, dtype="<i4").reshape(-1)
    groups = np.asarray(groups, dtype="<i4").reshape(-1)
    if len(freedom) != 3 * n or len(groups) != n:
        raise ValueError("mask lengths do not match coords")

    count = DATA
    pos = DATA + 0x1000
    out = DATA + 0x4000
    mask = DATA + 0x8000
    desc = DATA + 0x9000
    group_mask = DATA + 0xa000
    u.mem_write(count, struct.pack("<i", n))
    u.mem_write(pos, x.reshape(-1).tobytes())
    u.mem_write(out, bytes(8 * 3 * n))
    u.mem_write(mask, freedom.tobytes())
    u.mem_write(desc, struct.pack("<ii", *pair))
    u.mem_write(group_mask, groups.tobytes())

    # At callee entry rbp+0x10 aliases the caller's first stack argument.
    sp = STACK + 0x1ff00
    u.mem_write(sp, struct.pack("<QQ", STOP, group_mask))
    u.reg_write(UC_X86_REG_RSP, sp)
    for reg, val in ((UC_X86_REG_RDI, count), (UC_X86_REG_RSI, pos),
                     (UC_X86_REG_RCX, mask), (UC_X86_REG_R8, out),
                     (UC_X86_REG_R9, desc)):
        u.reg_write(reg, val)

    def hook(uc, address, size, user):
        if address == RET:
            return
        if address < ENTRY or address >= ENTRY + 0x250:
            raise RuntimeError(f"unexpected external {address:#x}")

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(ENTRY, STOP, count=100000)
    got = np.frombuffer(u.mem_read(out, 8 * 3 * n), dtype="<f8").reshape(n, 3).copy()
    return {"output": got.tolist(), "elf_sha256": hashlib.sha256(blob).hexdigest(),
            "pair": list(pair), "freedom": freedom.tolist(), "groups": groups.tolist()}


def expected(coords, pair=(1, 2), freedom=None, groups=None):
    x = np.asarray(coords, dtype=float)
    n = len(x)
    if freedom is None:
        freedom = np.ones(3 * n, dtype=int)
    if groups is None:
        groups = np.ones(n, dtype=int)
    i, j = np.asarray(pair, dtype=int) - 1
    d = x[i] - x[j]
    out = np.zeros_like(x)
    for k in range(n):
        if groups[k] == 1:
            out[k] = np.cross(x[k] - x[i], x[k] - x[j])
    out *= np.asarray(freedom).reshape(n, 3)
    return out


if __name__ == "__main__":
    cases = [
        ("all", [[0, 0, 0], [3, 4, 0], [9, 0, 0]], (1, 2), [1] * 9, [1, 1, 1]),
        ("group_filter", [[0, 0, 0], [3, 4, 0], [9, 0, 0]], (1, 2), [1] * 9, [1, 0, 1]),
        ("freedom_filter", [[0, 0, 0], [3, 4, 0], [9, 0, 0]], (1, 2), [1, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]),
        ("non_endpoint_group", [[0, 0, 0], [3, 4, 0], [9, 0, 0]], (1, 2), [1] * 9, [1, 1, 0]),
    ]
    rows = []
    for label, coords, pair, freedom, groups in cases:
        got = run(coords, pair, freedom, groups)
        ref = expected(coords, pair, freedom, groups).tolist()
        err = float(np.max(np.abs(np.asarray(got["output"]) - ref)))
        rows.append({"case": label, **got, "expected": ref, "max_error": err, "passed": err < 1e-14})
    result = {"entry": hex(ENTRY), "scope": "isolated helper; no PES", "cases": rows,
              "passed": all(row["passed"] for row in rows)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
