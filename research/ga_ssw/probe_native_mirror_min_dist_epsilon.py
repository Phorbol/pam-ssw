"""Isolated boundary check for mirror_min_dist_'s squared-distance margin."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np


ELF = Path(
    "/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp"
)
EPSILON_ADDRESS = 0x4A4C158
EPSILON = 0.001


def load_probe():
    path = Path(__file__).with_name("probe_native_mirror_min_dist.py")
    spec = importlib.util.spec_from_file_location("mirror_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_virtual(blob, segments, address, size):
    for start, mem_size, data in segments:
        if start <= address and address + size <= start + len(data):
            return data[address - start : address - start + size]
    raise ValueError(f"ELF address is not file-backed: {address:#x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe = load_probe()
    blob, segments = probe.load_elf(ELF)
    raw = read_virtual(blob, segments, EPSILON_ADDRESS, 8)
    epsilon = struct.unpack("<d", raw)[0]
    assert epsilon == EPSILON
    cell = np.diag([4.0, 4.0, 4.0])
    source_segments = segments
    rows = []
    for separation in (2.00010, 2.000125, 2.00015):
        first = np.zeros(3)
        second = np.array([separation, 0.0, 0.0])
        frac1 = first @ np.linalg.inv(cell)
        frac2 = second @ np.linalg.inv(cell)
        vector, distance, lattice = probe.run(
            source_segments,
            first,
            second,
            frac1,
            frac2,
            cell,
            use_reclat=False,
        )
        direct = -separation
        wrapped = 4.0 - separation
        direct_sq = direct * direct
        wrapped_sq = wrapped * wrapped
        expected_shift = 1 if direct_sq - wrapped_sq > epsilon else 0
        actual_shift = 1 if vector[0] > 0 else 0
        assert actual_shift == expected_shift
        rows.append(
            {
                "separation": separation,
                "direct_squared": direct_sq,
                "wrapped_squared": wrapped_sq,
                "improvement": direct_sq - wrapped_sq,
                "expected_shift": expected_shift,
                "native_vector": vector.tolist(),
                "native_distance": float(distance),
                "native_shift": actual_shift,
                "cell_c_order": lattice.tolist(),
            }
        )
    report = {
        "elf": str(ELF),
        "elf_sha256": hashlib.sha256(blob).hexdigest(),
        "entry": hex(probe.ENTRY),
        "epsilon_address": hex(EPSILON_ADDRESS),
        "epsilon_squared_distance": epsilon,
        "cases": rows,
        "evidence_scope": "isolated mirror_min_dist_ only; no main/PES/protection",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
