"""Isolated ELF probe for the native ``moveds`` retry decision slice.

Only instructions at 0x5c7452--0x5c74cb are executed.  The preceding
``present_tooshort`` result and the local scalar are synthetic inputs; the
branch arithmetic and ds*=0.95 instructions are uploaded ELF bytes.  No
LASP entry point, PES, allocator, or protection path is entered.
"""
from __future__ import annotations

import argparse, hashlib, json, struct
from pathlib import Path

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RBP, UC_X86_REG_RIP
from research.ga_ssw.probe_native_weight_emulated import load_elf

ELF_DEFAULT = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
ENTRY, ACCEPT, RETRY, EXHAUST = 0x5C7452, 0x5CA780, 0x5C6DDC, 0x5C74D1
DATA, STACK, STOP = 0x720000000000, 0x710000000000, 0x700000000000


def putd(u, address, value):
    u.mem_write(address, struct.pack("<d", float(value)))


def getd(u, address):
    return struct.unpack("<d", u.mem_read(address, 8))[0]


def run(segments, name, flag, candidate_scalar, retry_local_scalar,
        returned_scalar, ds, retry_count, disp_perstep=1.0,
        bonddisp_perstep=0.1):
    u = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, size, chunk in segments:
        lo = va & ~4095
        u.mem_map(lo, ((va + size + 4095) & ~4095) - lo)
        if chunk:
            u.mem_write(va, chunk)
    u.mem_map(DATA, 0x100000)
    u.mem_map(STACK, 0x20000)
    u.mem_map(STOP, 0x1000)
    rbp, para = STACK + 0x10000, DATA
    # Inputs at the exact local/object operands used by the branch slice.
    putd(u, para + 0x2E1F0, disp_perstep)
    putd(u, para + 0x2E1F8, bonddisp_perstep)
    u.mem_write(rbp - 0xDC, struct.pack("<i", int(flag)))
    # Distinct operands in the original slice: -0x40 is the preceding
    # distance reduction; -0xd8 is a pre-existing local used for the
    # bonddisp expression; -0xd0 is present_tooshort's returned scalar.
    putd(u, rbp - 0xD8, retry_local_scalar)
    putd(u, rbp - 0xD0, returned_scalar)
    putd(u, rbp - 0x40, candidate_scalar)
    putd(u, rbp - 0x308, ds)
    u.mem_write(rbp - 0x348, struct.pack("<i", int(retry_count)))
    u.mem_write(rbp - 0x50, struct.pack("<Q", para))
    u.reg_write(UC_X86_REG_RBP, rbp)
    u.reg_write(UC_X86_REG_RAX, para)
    trace = []

    def hook(machine, address, size, user):
        trace.append(hex(address))
        if not (ENTRY <= address <= EXHAUST or address in (ACCEPT, RETRY, EXHAUST)):
            raise RuntimeError(f"PC escaped guarded retry slice at {address:#x}")
        if address in (ACCEPT, RETRY, EXHAUST):
            machine.reg_write(UC_X86_REG_RIP, STOP)

    u.hook_add(UC_HOOK_CODE, hook)
    u.reg_write(UC_X86_REG_RIP, ENTRY)
    try:
        u.emu_start(ENTRY, STOP, count=1000)
        status = "ok"
    except Exception as exc:  # retain exact blocked address as evidence
        status = "blocked"
        error = f"{type(exc).__name__}: {exc}"
    row = dict(name=name, synthetic_inputs=dict(present_tooshort_flag=flag,
              candidate_scalar= candidate_scalar,
              retry_local_scalar=retry_local_scalar,
              returned_scalar=returned_scalar, ds_initial=ds,
              retry_count=retry_count,
              disp_perstep=disp_perstep, bonddisp_perstep=bonddisp_perstep),
               status=status, trace=trace,
               ds_final=getd(u, rbp - 0x308),
               retry_final=struct.unpack("<i", u.mem_read(rbp - 0x348, 4))[0],
               branch_target=(trace[-1] if trace else None),
               executed_elf_range="0x5c7452-0x5c74cb")
    if status == "blocked":
        row["error"] = error
        row["rip"] = hex(u.reg_read(UC_X86_REG_RIP))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", default=ELF_DEFAULT)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    blob, segments = load_elf(args.elf)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        raise ValueError("unsupported ELF digest")
    rows = [
        run(segments, "accepted", 0, 0.5, 0.25, 0.5, 0.5, 10),
        run(segments, "shorten_then_retry", 1, 9.0, 8.0, 7.0, 0.5, 10),
        run(segments, "retry_exhausted_by_ds", 1, 9.0, 8.0, 7.0, 0.05, 10),
        run(segments, "retry_exhausted_by_count", 1, 9.0, 8.0, 7.0, 0.2, 51),
    ]
    expected = {
        "accepted": (hex(ACCEPT), 0.5),
        "shorten_then_retry": (hex(RETRY), 0.475),
        "retry_exhausted_by_ds": (hex(EXHAUST), 0.0475),
        "retry_exhausted_by_count": (hex(EXHAUST), 0.19),
    }
    for row in rows:
        if row["status"] != "ok":
            raise AssertionError(f"probe did not stop cleanly: {row}")
        target, ds_expected = expected[row["name"]]
        if row["branch_target"] != target or abs(row["ds_final"] - ds_expected) > 1e-15:
            raise AssertionError(f"unexpected native branch result: {row}")
    payload = dict(elf=args.elf, elf_sha256=digest, unicorn="instruction-level",
                   entry=hex(ENTRY), guarded_pc_range=[hex(ENTRY), hex(EXHAUST)],
                   stop_targets=[hex(ACCEPT), hex(RETRY), hex(EXHAUST)],
                   scope="moveds retry decision slice only",
                   synthetic_dependencies=["present_tooshort flag/scalar",
                       "rbp locals", "para disp/bonddisp fields"],
                   native_executed=["test flag", "disp/scalar comparisons",
                       "ds *= 0.95", "ds/retry loop bounds"], rows=rows)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "targets": [r["branch_target"] for r in rows]}))


if __name__ == "__main__":
    main()
