"""Isolated execution of native fixed-cell climb_convg_ scalar gates."""

import hashlib
import json
import math
import struct
from pathlib import Path

from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64, Uc
from unicorn.x86_const import UC_X86_REG_RBP, UC_X86_REG_RDI, UC_X86_REG_RIP, UC_X86_REG_RSI, UC_X86_REG_RSP

from research.ga_ssw.probe_native_weight_emulated import load_elf
from research.ga_ssw.native_stage_predicate import closed_native_stage_predicate

ELF = "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
ENTRY, RET = 0x5CD130, 0x710000000100
PARA, CONTROL = 0x53ED7A0, 0x53ED5C0
STACK, DATA = 0x710000000000, 0x720000000000
OBJECT, DESC, FA, TRAJ, OUT = DATA + 0x1000, DATA + 0x2000, DATA + 0x3000, DATA + 0x4000, DATA + 0x5000


def w_i32(u, a, x):
    u.mem_write(a, struct.pack("<i", x))


def w_i64(u, a, x):
    u.mem_write(a, struct.pack("<Q", x))


def w_f64(u, a, x):
    u.mem_write(a, struct.pack("<d", x))


def run_case(ng, *, low=False, multi=False, force_limit=False, saved=False,
             force_value=None, height_value=0.0, gm_value=0.0,
             climbstep=10, para_ng=99):
    blob, segs = load_elf(ELF)
    u = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, memsz, chunk in segs:
        start = va & ~0xFFF
        size = ((va + memsz + 0xFFF) & ~0xFFF) - start
        u.mem_map(start, size)
        if chunk:
            u.mem_write(va, chunk)
    u.mem_map(STACK, 0x20000); u.mem_map(DATA, 0x10000)
    # Disable formatted diagnostic calls and supply a 1x3 Fortran-like fa.
    w_i32(u, PARA + 0x2DDEC, 0)
    w_i32(u, PARA + 0xF0, 0); w_i32(u, PARA + 0xF4, para_ng)
    w_i32(u, PARA + 0x2DD20, 15); w_i32(u, PARA + 0x2DD24, 10)
    w_f64(u, PARA + 0x2DD30, 0.15); w_f64(u, PARA + 0x2DD38, 1.0)
    w_f64(u, PARA + 0x2DD40, 1.0); w_f64(u, PARA + 0x2DD48, 1.0)
    w_i32(u, CONTROL + 0x08, climbstep); w_i32(u, CONTROL + 0x1BC, -1 if multi else 0)
    w_f64(u, CONTROL + 0x58, height_value); w_f64(u, CONTROL + 0x60, gm_value)
    w_i32(u, CONTROL + 0x78, 0); w_i32(u, CONTROL + 0x7C, 0)
    w_i64(u, OBJECT + 0x200, 3); w_i64(u, OBJECT + 0x218, 1); w_i64(u, OBJECT + 0x220, 3)
    w_i64(u, OBJECT + 0x228, 1); w_i64(u, OBJECT + 0x1D0, FA)
    w_i64(u, OBJECT + 0x1668, TRAJ); w_i64(u, OBJECT + 0x16A8, 1)
    w_i32(u, OBJECT + 0x1660, ng)
    w_f64(u, OBJECT + 0x1AC0, 0.0); w_f64(u, OBJECT + 0x1AC8, -0.11 if low else 0.0)
    w_f64(u, TRAJ + (0x690 * max(0, ng - 1)) + 0x230, 1.1 if saved else 0.0)
    force_value = (2.0 if force_limit else 0.0) if force_value is None else force_value
    vals = [force_value, 0.0, 0.0]
    for i, x in enumerate(vals): w_f64(u, FA + 8 * i, x)
    w_i64(u, DESC, OBJECT)
    u.mem_write(STACK + 0x1000, struct.pack("<Q", RET))
    u.reg_write(UC_X86_REG_RSP, STACK + 0x1000); u.reg_write(UC_X86_REG_RBP, STACK + 0x8000)
    u.reg_write(UC_X86_REG_RDI, DESC); u.reg_write(UC_X86_REG_RSI, OUT)
    trace = []
    def hook(_, address, size, __):
        trace.append(address)
        if address == RET: _.emu_stop()
        elif not (ENTRY <= address < 0x5CDA6E): raise RuntimeError(f"unexpected PC {address:#x}")
    u.hook_add(UC_HOOK_CODE, hook)
    u.reg_write(UC_X86_REG_RIP, ENTRY)
    try:
        u.emu_start(ENTRY, RET, count=20000)
    except Exception as error:
        raise RuntimeError(f"native callee stopped at {[hex(pc) for pc in trace[-12:]]}") from error
    got = {"lclimbstop": struct.unpack("<i", u.mem_read(CONTROL + 0x7C, 4))[0],
           "lclimb_allstop": struct.unpack("<i", u.mem_read(CONTROL + 0x78, 4))[0]}
    current = -0.11 if low else 0.0
    max_force = abs(force_value)
    expected = closed_native_stage_predicate(
        max_force=max_force, climb_stopf=0.15,
        base_energy=current, initial_energy=0.0,
        saved_gaussian_energy=1.1 if saved else 0.0,
        maxe_height=height_value, maxe_height_gm=gm_value,
        e_maxlimit=1.0, f_maxlimit=1.0, e_maxlimit_gm=1.0,
        para_ng=para_ng, ng=ng, climbstep=climbstep,
        ngaus_relax=15, ngaus_relax_ini=10, multi_pes=multi)
    expected_lsb = int(expected.known_stop)
    assert (got["lclimb_allstop"] & 1) == int(expected.allstop), (got, expected.__dict__)
    assert (got["lclimbstop"] & 1) == expected_lsb, (got, expected.__dict__)
    return {"inputs": {"ng": ng, "low": low, "multi_pes": multi, "force_limit": force_limit, "saved": saved, "force_value": force_value, "height_value": height_value, "gm_value": gm_value, "climbstep": climbstep, "para_ng": para_ng}, "native": got, "closed_scalar_predicate": expected.__dict__, "trace_tail": [hex(x) for x in trace[-8:]]}


def main():
    cases = [
        run_case(1), run_case(1, climbstep=11), run_case(2),
        run_case(2, low=True), run_case(2, low=True, multi=True),
        run_case(2, force_limit=True), run_case(2, saved=True),
        run_case(2, height_value=1.0), run_case(2, gm_value=1.0),
        run_case(2, force_value=0.15),
        run_case(2, force_value=math.nextafter(0.15, math.inf)),
        run_case(2, height_value=math.nextafter(1.0, -math.inf)),
        run_case(2, height_value=math.nextafter(1.0, math.inf)),
        run_case(2, gm_value=math.nextafter(1.0, -math.inf)),
        run_case(2, gm_value=math.nextafter(1.0, math.inf)),
        run_case(2, para_ng=2),
    ]
    out = Path("research/ga_ssw/evidence/native-climb-convg-scalar-20260912-v2/result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"elf": ELF, "sha256": hashlib.sha256(load_elf(ELF)[0]).hexdigest(), "cases": cases}, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(out)}))


if __name__ == "__main__":
    main()
