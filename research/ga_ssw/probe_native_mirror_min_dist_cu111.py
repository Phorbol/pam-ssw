"""Replay two saved Cu111 LS ledger geometries through mirror_min_dist_.

This is an isolated callee probe: it reads the existing audit ledger and runs
no calculator, LASP main path, or protection logic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic


def load_probe(path: Path):
    spec = importlib.util.spec_from_file_location("mirror_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expected(cart1, cart2, cell):
    cell = np.asarray(cell, dtype=float)
    delta = np.asarray(cart1, dtype=float) - np.asarray(cart2, dtype=float)
    candidates = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                vector = delta + np.array([i, j, k]) @ cell
                candidates.append((float(vector @ vector), vector, (i, j, k)))
    return min(candidates, key=lambda item: item[0])


def native_candidate(cart1, cart2, cell, vector):
    cell = np.asarray(cell, dtype=float)
    delta = np.asarray(cart1, dtype=float) - np.asarray(cart2, dtype=float)
    candidates = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shift = (i, j, k)
                candidate = delta + np.array(shift) @ cell
                candidates.append((float(np.linalg.norm(candidate - vector)), shift, candidate))
    return min(candidates, key=lambda item: item[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case",
        default="research/ga_ssw/evidence/constrained-gaussian-reference-20260912/cu111-ls_native",
    )
    parser.add_argument(
        "--probe",
        default="research/ga_ssw/probe_native_mirror_min_dist.py",
    )
    args = parser.parse_args()
    case = Path(args.case)
    probe = load_probe(Path(args.probe))
    ledger = [json.loads(line) for line in (case / "evaluations.jsonl").read_text().splitlines()]
    entries = {
        request: next(entry for entry in ledger if entry["request"] == request)
        for request in (132, 152)
    }
    cell = np.asarray(entries[132]["atoms"]["cell"], dtype=float)
    rows = []
    elf = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
    blob, segments = probe.load_elf(elf)
    assert hashlib.sha256(blob).hexdigest() == probe.ELF_SHA256
    for request, entry in entries.items():
        positions = np.asarray(entry["atoms"]["positions"], dtype=float)
        cart1, cart2 = positions[8], positions[11]
        xfrac1 = cart1 @ np.linalg.inv(cell)
        xfrac2 = cart2 @ np.linalg.inv(cell)
        vector, distance, lattice = probe.run(
            segments,
            cart1,
            cart2,
            xfrac1,
            xfrac2,
            cell,
            use_reclat=False,
        )
        expected_sq, expected_vector, shift = expected(cart1, cart2, cell)
        mic_vector, mic_distance = find_mic(
            cart1 - cart2, cell, pbc=[True, True, False]
        )
        unit = vector / distance
        mic_unit = np.asarray(mic_vector) / mic_distance
        native_error, native_shift, native_candidate_vector = native_candidate(
            cart1, cart2, cell, vector
        )
        assert native_error < 1e-12
        rows.append(
            {
                "request": request,
                "pair": [8, 11],
                "cart1": cart1.tolist(),
                "cart2": cart2.tolist(),
                "native_vector": vector.tolist(),
                "native_distance": float(distance),
                "native_unit_vector": unit.tolist(),
                "native_shift_27": list(native_shift),
                "native_candidate_vector_error": float(native_error),
                "native_is_27_candidate": bool(native_error < 1e-12),
                "explicit_27_vector": expected_vector.tolist(),
                "explicit_27_distance": float(np.sqrt(expected_sq)),
                "ase_mic_vector": np.asarray(mic_vector).tolist(),
                "ase_mic_distance": float(mic_distance),
                "ase_mic_unit_vector": mic_unit.tolist(),
                "native_matches_27": bool(
                    np.allclose(vector, expected_vector, atol=1e-12, rtol=0)
                    and abs(distance - np.sqrt(expected_sq)) < 1e-12
                ),
                "native_distance_minus_global_27_min": float(
                    distance - np.sqrt(expected_sq)
                ),
                "native_matches_ase_mic": bool(
                    np.allclose(vector, mic_vector, atol=1e-12, rtol=0)
                ),
                "cell": cell.tolist(),
            }
        )
    vector_delta = np.asarray(rows[1]["native_vector"]) - np.asarray(rows[0]["native_vector"])
    unit_delta = np.asarray(rows[1]["native_unit_vector"]) - np.asarray(rows[0]["native_unit_vector"])
    report = {
        "case": str(case),
        "source_ledger": str(case / "evaluations.jsonl"),
        "elf": str(elf),
        "elf_sha256": hashlib.sha256(blob).hexdigest(),
        "entry": hex(probe.ENTRY),
        "pair": [8, 11],
        "rows": rows,
        "native_vector_delta_152_minus_132": vector_delta.tolist(),
        "native_unit_vector_delta_152_minus_132": unit_delta.tolist(),
        "native_direction_changed": bool(np.linalg.norm(unit_delta) > 1e-12),
        "evidence_scope": "saved ledger geometry plus isolated mirror_min_dist_ only; no PES/main/protection",
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
