"""Full native C4 group selection, producer and mixing on two small clusters."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from ase import Atoms
from unicorn.x86_const import (UC_X86_REG_RAX, UC_X86_REG_R14, UC_X86_REG_R9,
                               UC_X86_REG_RBP, UC_X86_REG_RIP)

from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_bond_groups import native_bond_groups
from pamssw.standalone.native_local_pair import native_local_pair
from pamssw.standalone.native_pair_group import native_local_pair_group
from research.ga_ssw.probe_addgaussian_emulated import DATA
from research.ga_ssw.probe_native_group_mixture import normalized
from research.ga_ssw.probe_native_pair_generator import C4Oracle
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


class C4GroupOracle(C4Oracle):
    def hook(self, uc, address, size, user):
        if address == 0x5D5C50:
            super().hook(uc, address, size, user)
            obj = self.readq(DATA)
            self.group_marker = -1
            uc.mem_write(obj + 0x2204, struct.pack("<i", -1))
            numbers = self.alloc(np.asarray(self.atoms.numbers, dtype="<i4").tobytes())
            self.descriptor(obj + 0x8, numbers, (self.n,), 4)
            first = self.alloc(bytes(4 * self.n))
            second = self.alloc(bytes(4 * self.n))
            visited = self.alloc(bytes(4 * self.n))
            matrix = self.alloc(bytes(8 * self.n * self.n))
            fractional = self.arr(self.positions / 40.0)
            coordinate_copy = self.arr(self.positions)
            self.descriptor(obj + 0x50, first, (self.n,), 4)
            self.descriptor(obj + 0x98, second, (self.n,), 4)
            self.descriptor(obj + 0x598, fractional, (3, self.n))
            self.descriptor(obj + 0x538, coordinate_copy, (3, self.n))
            self.descriptor(obj + 0xAE8, matrix, (self.n, self.n))
            self.descriptor(obj + 0xB48, visited, (self.n,), 4)
            self.first_mask_pointer = first
            self.second_mask_pointer = second
        elif address == 0x5D784A:
            self.after_first = np.frombuffer(
                uc.mem_read(self.first_mask_pointer, 4 * self.n), dtype="<i4"
            ).copy()
        elif address == 0x5D77EF:
            self.forbidden_gate = struct.unpack("<i", uc.mem_read(
                uc.reg_read(UC_X86_REG_RBP) - 0x50, 4))[0]
        elif address == 0x5D79B3:
            self.after_second = np.frombuffer(
                uc.mem_read(self.first_mask_pointer, 4 * self.n), dtype="<i4"
            ).copy()
        elif address == 0x6E4AA0:
            self.group_calls += 1
        elif address == 0x4970360:
            # Probe preallocates every destination; native realloc metadata
            # differs only in compiler flags after find_atom_in_group.
            self.ret()
        elif (0x5A61A0 <= address < 0x5A6B70 or
              0x5A4300 <= address < 0x5A4500 or
              0x58F2A0 <= address < 0x58F680 or
              0x6E4AA0 < address < 0x6E4C00):
            pass
        else:
            super().hook(uc, address, size, user)


def main():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    rows = []
    cases = (
        ("separate_ch", Atoms("CHCH", positions=[[14.7, 15.1, 15.0],
                                                   [13.9, 14.8, 15.2],
                                                   [17.2, 15.8, 14.8],
                                                   [18.0, 16.3, 15.1]]), (0, 2)),
        ("connected_ch", Atoms("CHCH", positions=[[14.7, 15.1, 15.0],
                                                    [13.9, 14.8, 15.2],
                                                    [17.2, 15.8, 14.8],
                                                    [18.0, 16.3, 15.1]]), (0, 1)),
    )
    for label, atoms, pair in cases:
        reference = native_bond_groups(atoms, pair)
        frame = ClusterFrame(atoms)
        seed = normalized(frame.project(np.arange(len(atoms) * 3).reshape(-1, 3) * 0.013))
        oracle = C4GroupOracle(segments)
        oracle.atoms = atoms
        oracle.positions = np.asarray(atoms.positions, dtype="<f8")
        oracle.axis = pair
        oracle.group = np.zeros(len(atoms), np.int32)
        oracle.uniform = 0.17
        oracle.group_marker = -1
        oracle.pair_calls = oracle.group_calls = 0
        oracle.after_first = oracle.after_second = None
        oracle.forbidden_gate = None
        oracle.instruction_limit = 8_000_000
        oracle.timeout_us = 30_000_000
        coefficients = np.zeros(10)
        coefficients[4] = 0.6
        coefficients[9] = 0.72
        oracle.coefficients = coefficients
        try:
            got = oracle.run_geometry(atoms.positions, seed, np.zeros_like(seed))
        except Exception as exc:
            registers = {name: oracle.uc.reg_read(register) for name, register in
                         (("rax", UC_X86_REG_RAX), ("r14", UC_X86_REG_R14),
                          ("r9", UC_X86_REG_R9))}
            raise RuntimeError(f"{label} failed at {oracle.uc.reg_read(UC_X86_REG_RIP):#x}: {registers}") from exc
        if reference.status == "separate_groups":
            raw = native_local_pair_group(atoms, pair, reference.first_group,
                                          reference.second_group)
            expected_route = "group"
        else:
            raw = native_local_pair(atoms, pair, lambda: 0.17).raw_direction
            expected_route = "pair_fallback"
        expected = normalized(0.72 * seed + 0.6 * normalized(frame.project(raw)))
        error = float(np.max(np.abs(got - expected)))
        first_ok = np.array_equal(oracle.after_first, reference.first_group)
        second_ok = (reference.status != "separate_groups" or
                     np.array_equal(oracle.after_second, reference.second_group))
        route_ok = ((expected_route == "group" and oracle.group_calls == 1 and oracle.pair_calls == 0) or
                    (expected_route == "pair_fallback" and oracle.group_calls == 0 and oracle.pair_calls == 1))
        passed = first_ok and second_ok and route_ok and error < 1e-12
        rows.append({"case": label, "positions": atoms.positions.tolist(),
                     "numbers": atoms.numbers.tolist(), "pair": list(pair),
                     "reference_status": reference.status,
                     "forbidden_gate": oracle.forbidden_gate,
                     "native_first_group": (None if oracle.after_first is None
                                            else oracle.after_first.tolist()),
                     "native_second_group": (None if oracle.after_second is None
                                             else oracle.after_second.tolist()),
                     "group_calls": oracle.group_calls, "pair_calls": oracle.pair_calls,
                     "output": got.tolist(), "output_error": error, "passed": passed})
    report = {"elf_sha256": ELF_SHA256, "scope": __doc__, "cases": rows,
              "passed": all(row["passed"] for row in rows)}
    assert report["passed"], report
    path = Path("research/ga_ssw/evidence/native-c4-group-generator-20260917.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(rows),
                      "max_error": max(row["output_error"] for row in rows)}))


if __name__ == "__main__":
    main()
