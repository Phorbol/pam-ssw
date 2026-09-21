"""Bounded C4 bond-group oracle: native fastbond lookup plus pure group route."""

import hashlib
import json
import struct
from pathlib import Path

from ase import Atoms
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RDX, UC_X86_REG_RSI, UC_X86_REG_RSP

from pamssw.standalone.native_bond_groups import native_bond_groups
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


def native_fastbond(segments, pairs):
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    for va, memsz, chunk in segments:
        lo = va & ~4095
        uc.mem_map(lo, ((va + memsz + 4095) & ~4095) - lo)
        if chunk:
            uc.mem_write(va, chunk)
    data = 0x700000000000
    uc.mem_map(data, 0x3000)
    rows = []
    for first, second in pairs:
        uc.mem_write(data, struct.pack("<ii", first, second) + bytes(8))
        stack, stop = data + 0x1000, data + 0x2000
        uc.mem_write(stack, struct.pack("<Q", stop))
        for register, value in ((UC_X86_REG_RDI, data), (UC_X86_REG_RSI, data + 4),
                                (UC_X86_REG_RDX, data + 8), (UC_X86_REG_RSP, stack)):
            uc.reg_write(register, value)
        uc.emu_start(0x58F2A0, stop, count=100_000)
        rows.append({"pair": [first, second],
                     "value": struct.unpack("<d", uc.mem_read(data + 8, 8))[0]})
    return rows


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    pairs = ((1, 1), (1, 6), (1, 8), (6, 6), (6, 8), (8, 8), (29, 79))
    lookup = native_fastbond(segments, pairs)
    expected = [0.800000011920929, 1.0839999914169312, 0.9470000267028809,
                1.5119999647140503, 1.3930000066757202, 1.4500000476837158, 0.0]
    assert [row["value"] for row in lookup] == expected

    cases = []
    for label, atoms, pair, expected_status, expected_groups in (
        ("two_ch_fragments", Atoms("CHCH", positions=[[0, 0, 0], [1, 0, 0],
                                                       [6, 0, 0], [7, 0, 0]]),
         (0, 2), "separate_groups", ([0, 1, 0, 0], [0, 0, 0, 1])),
        ("connected_ch", Atoms("CHH", positions=[[0, 0, 0], [1, 0, 0], [4, 0, 0]]),
         (0, 1), "connected_pair_fallback", ([0, 1, 0], [0, 0, 0])),
        ("cu_au_radius_fallback", Atoms("CuAu", positions=[[0, 0, 0], [3, 0, 0]]),
         (0, 1), "connected_pair_fallback", ([0, 1], [0, 0])),
    ):
        result = native_bond_groups(atoms, pair)
        passed = (result.status == expected_status and
                  result.first_group.tolist() == expected_groups[0] and
                  result.second_group.tolist() == expected_groups[1])
        cases.append({"case": label, "numbers": atoms.numbers.tolist(),
                      "positions": atoms.positions.tolist(), "pair": list(pair),
                      "status": result.status, "first_group": result.first_group.tolist(),
                      "second_group": result.second_group.tolist(),
                      "cutoff_sources": list(result.cutoff_sources), "passed": passed})
    assert all(case["passed"] for case in cases)
    report = {"elf_sha256": ELF_SHA256, "fastbond_entry": "0x58f2a0",
              "scope": "native fastbond instructions; nonperiodic unconstrained pure group route; no PES",
              "fastbond": lookup, "cases": cases, "passed": True}
    path = Path("research/ga_ssw/evidence/native-bond-groups-20260917.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "lookups": len(lookup), "cases": len(cases)}))


if __name__ == "__main__":
    main()
