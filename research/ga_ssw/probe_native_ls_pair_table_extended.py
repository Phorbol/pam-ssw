"""Read-only native LS raw pair lookup for selected TYPE4/CuO elements.

This is an isolated Unicorn probe of the two release lookup functions.  It
does not initialize LS, count neighbors, run a calculator, or enter LASP.
Raw returns are not standard chemical bond constants and are not final LS
amplitudes or neighbor cutoffs.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_CODE
from unicorn.x86_const import (
    UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RSP, UC_X86_REG_XMM0,
    UC_X86_REG_RIP,
)

from research.ga_ssw.probe_native_weight_emulated import load_elf


ELF_SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
PAIRS = ((1, 1), (1, 6), (6, 1), (6, 6),
         (8, 8), (8, 22), (22, 8), (22, 22),
         (8, 29), (29, 8), (22, 29), (29, 22),
         (8, 79), (79, 8), (22, 79), (79, 22),
         (29, 29), (79, 79))
ENTRIES = (("bondeneval_", 0x6CB660), ("bondlenval_", 0x6CC0B0))


def run(elf: Path) -> dict:
    blob, segments = load_elf(str(elf))
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ELF_SHA256:
        raise ValueError("unsupported ELF; entry addresses are version-specific")

    def read(address, size):
        for start, _, data in segments:
            if start <= address and address + size <= start + len(data):
                return data[address - start:address - start + size]
        raise ValueError(f"unmapped ELF bytes at {address:#x}")

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    # The selected-element dispatch reads the full 1..96 jump/constant table;
    # map a bounded page window around both entry regions rather than only the
    # H/C rows used by the original probe.
    for address, size in ((0x6CB000, 0x10000), (0x50A000, 0x3000),
                          (0x4A4C000, 0x10000), (0x70000000, 0x2000)):
        uc.mem_map(address, size)
        if address != 0x70000000:
            uc.mem_write(address, read(address, size))
    # bondlenval_ falls back to the release species_radius_ helper for pairs
    # without a dedicated bond-length branch.  Emulate only that documented
    # helper ABI (Z pointer in RDI, output double pointer in RSI); no PES or
    # caller state is entered.
    radius_path = Path(__file__).resolve().parent / "evidence/native-cluster-control-generator/species-radius-table.json"
    radius_rows = json.loads(radius_path.read_text())["rows"]
    radii = {int(row["atomic_number"]): float(row["radius"]) for row in radius_rows}
    def hook(emu, address, _size, _user):
        if address != 0x50AC90:
            return
        z = struct.unpack("<i", emu.mem_read(emu.reg_read(UC_X86_REG_RDI), 4))[0]
        if z not in radii:
            raise ValueError(f"species radius missing for Z={z}")
        out = emu.reg_read(UC_X86_REG_RSI)
        emu.mem_write(out, struct.pack("<d", radii[z]))
        rsp = emu.reg_read(UC_X86_REG_RSP)
        ret = struct.unpack("<Q", emu.mem_read(rsp, 8))[0]
        emu.reg_write(UC_X86_REG_RSP, rsp + 8)
        emu.reg_write(UC_X86_REG_RIP, ret)
    uc.hook_add(UC_HOOK_CODE, hook)
    rows = []
    for name, entry in ENTRIES:
        for a, b in PAIRS:
            uc.mem_write(0x70000000, struct.pack("<ii", a, b))
            uc.reg_write(UC_X86_REG_RDI, 0x70000000)
            uc.reg_write(UC_X86_REG_RSI, 0x70000004)
            stop, stack = 0x70001000, 0x70001808
            uc.mem_write(stack, struct.pack("<Q", stop))
            uc.reg_write(UC_X86_REG_RSP, stack)
            uc.reg_write(UC_X86_REG_XMM0, 0)
            try:
                uc.emu_start(entry, stop, count=10000, timeout=1000000)
                if uc.reg_read(UC_X86_REG_RIP) != stop:
                    raise RuntimeError("instruction budget exhausted before normal return")
                value = struct.unpack("<d", struct.pack("<Q", uc.reg_read(UC_X86_REG_XMM0) & ((1 << 64) - 1)))[0]
                rows.append({"function": name, "pair": [a, b], "status": "returned", "raw_return": value})
            except Exception as exc:
                rows.append({"function": name, "pair": [a, b], "status": "unsupported_or_faulted",
                             "error": type(exc).__name__ + ": " + str(exc),
                             "rip": hex(uc.reg_read(UC_X86_REG_RIP))})
    return {
        "elf_sha256": digest,
        "elements": {"H": 1, "C": 6, "O": 8, "Ti": 22, "Cu": 29, "Au": 79},
        "pairs": [[a, b] for a, b in PAIRS],
        "rows": rows,
        "static_len_toller": struct.unpack("<d", read(0x5520568, 8))[0],
        "interpretation": "raw lookup only; caller scaling, filters, bond counting and LS target are separate contracts",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args.elf), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
