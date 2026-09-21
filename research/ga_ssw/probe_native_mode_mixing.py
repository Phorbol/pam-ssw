"""Audit the bounded native direction/mode mixing evidence.

This probe consumes the already executed update_mode0 instruction oracle.  It
does not execute LASP main, PES, protection, or the native RNG.  The purpose
is to make the c4..c6 -> c9 relation and the direction consequence explicit.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RIP, UC_X86_REG_RSI
from research.ga_ssw.probe_native_direction_update import UpdateOracle
from research.ga_ssw.probe_native_rotation_weight_branch import ELF_DEFAULT, ELF_SHA256
from research.ga_ssw.probe_native_weight_emulated import load_elf


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/direction-update.json"
OUTPUT = ROOT / "research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/mode-mixing.json"


class GeneratorProbe(UpdateOracle):
    """Run update and the c9-positive generator prefix, stopping at scalar tail."""
    def hook(self, uc, address, size, user):
        if address == 0x580640:
            # Explicit deterministic substitute for rd_numb; this is not native RNG.
            self.d(uc.reg_read(UC_X86_REG_RDI), 0.5)
            self.ret()
            return
        if address == 0x5d5ff9:
            # Rejoined scalar/vector path follows c9 multiplication.
            obj = self.readq(0x720000000000)
            ptr = self.readq(obj + 0x1788)
            self.generator_n0 = self.readarr(ptr).tolist()
            uc.emu_stop()
            return
        if address == 0x5d5c50:
            pointer = uc.reg_read(UC_X86_REG_RSI)
            self.captured = {"coefficients": np.frombuffer(uc.mem_read(pointer, 80), dtype="<f8").tolist(),
                             "normalized_displacement": self.readarr(self.direction).tolist()}
            return
        if 0x5d5c50 <= address < 0x5d5ff9 or 0x5d874b <= address < 0x5d8768:
            return
        super().hook(uc, address, size, user)


def execute_c9_prefix():
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    probe = GeneratorProbe(segments)
    current = np.arange(15, dtype=float).reshape(5, 3) * 0.1 + 0.3
    reference = np.zeros((5, 3))
    old = np.full((5, 3), -1.0 / np.sqrt(15.0))
    try:
        probe.probe(current, reference, old, [10.0, 0.0, 0.0])
    except AssertionError:
        pass
    captured = probe.captured
    assert probe.uc.reg_read(UC_X86_REG_RIP) == 0x5d5ff9
    c9 = captured["coefficients"][9]
    n0 = np.asarray(captured["normalized_displacement"])
    got = np.asarray(probe.generator_n0)
    expected = c9 * n0
    return {"coefficients": captured["coefficients"], "c9": c9,
            "input_n0": n0.tolist(), "generator_n0": got.tolist(),
            "expected_c9_n0": expected.tolist(),
            "max_error": float(np.max(np.abs(got - expected))),
            "passed": bool(np.max(np.abs(got - expected)) < 1e-14),
            "stop": "0x5d5ff9", "rng": "hooked rd_numb=0.5 deterministic; not native RNG"}


def main():
    source = json.loads(INPUT.read_text())
    cases = []
    for case in source["cases"]:
        c = np.asarray(case["coefficients"], dtype=float)
        d = np.asarray(case["normalized_displacement"], dtype=float)
        # update_mode0's recorded input is s/||s||, where s=x_current-x_record.
        expected_c9 = 1.2 * float(c[4] + c[5] + c[6])
        cases.append({
            "n": case["n"],
            "old_sign": case["old_sign"],
            "c4_c6": c[4:7].tolist(),
            "c9": float(c[9]),
            "c9_formula": "c9 = 1.2 * (c4 + c5 + c6)",
            "c9_expected": expected_c9,
            "c9_error": abs(float(c[9]) - expected_c9),
            "direction_is_normalized_displacement": bool(
                np.linalg.norm(d.ravel()) - 1.0 < 1e-14
            ),
            "direction_independent_of_old_sign": None,
        })
    by_n = {}
    for c in cases:
        by_n.setdefault(c["n"], []).append(c)
    source_by_n = {}
    for case in source["cases"]:
        source_by_n.setdefault(case["n"], {})[case["old_sign"]] = np.asarray(
            case["normalized_displacement"], dtype=float
        )
    for group in by_n.values():
        if len(group) == 2:
            same = bool(np.max(np.abs(source_by_n[group[0]["n"]][-1] - source_by_n[group[0]["n"]][1])) < 1e-14)
            group[0]["direction_independent_of_old_sign"] = same
            group[1]["direction_independent_of_old_sign"] = same
    c9_prefix = execute_c9_prefix()
    report = {
        "source": str(INPUT.relative_to(ROOT)),
        "cases": cases,
        "c9_prefix_execution": c9_prefix,
        "passed": all(c["c9_error"] < 1e-14 and c["direction_is_normalized_displacement"] for c in cases) and c9_prefix["passed"],
        "scope": "bounded update_mode0 plus native gen_randommode c9-positive prefix; no main/PES/protection; no RNG hook",
        "static_addresses": {
            "update_mode0_entry": "0x5d55f0",
            "update_normalization_call": "0x5d5bf4-0x5d5c06",
            "gen_randommode_entry": "0x5d5c50",
            "gen_randommode_n_normal": "0x5d6724-0x5d6733",
            "c4_write": "0x5c0493-0x5c049b",
            "c5_write": "0x5c0385-0x5c0398",
            "c6_write": "0x5c04c4-0x5c04cc",
            "c9_mix": "0x5d5997-0x5d59d0",
            "c9_scale_n0": "0x5d5eb6-0x5d5fc4",
            "c3_reads": "0x5d6ed8, 0x5d7376",
            "c4_read": "0x5d7634",
            "c5_read": "0x5da284",
            "c6_read": "0x5d82a1",
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(cases), "output": str(OUTPUT)}))
    assert report["passed"]


if __name__ == "__main__":
    main()
