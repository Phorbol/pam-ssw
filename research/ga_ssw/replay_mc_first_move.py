#!/usr/bin/env python3
"""Replay the first C60 outer move against an archived E/F ledger.

The frozen source tree is prepended before importing PAM-SSW.  The surface
returns only the next archived energy/force row after a strict 1e-10 A
position check, so no calculator, MACE model, GPU, or new PES call is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "research/ga_ssw/evidence/ssw-allocation-factorial-20260918/c60"
MATCHED = ROOT / "research/ga_ssw/evidence/mc-scale-matched-20260918/c60"
TEMPERATURES = (150.0, 2999.844423801854)
POSITION_TOL = 1.0e-10


def atoms_from(payload):
    from ase import Atoms

    return Atoms(numbers=payload["numbers"], positions=payload["positions"],
                 cell=payload["cell"], pbc=payload["pbc"])


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


class ReplaySurface:
    def __init__(self, rows, stop_after):
        self.rows = rows
        self.stop_after = stop_after
        self.index = 0
        self.requests = 0
        self.failure = None

    def evaluate(self, atoms):
        if self.index >= self.stop_after:
            self.failure = {"kind": "stop_after_first_record", "request": self.requests}
            raise RuntimeError("stop_after_first_record")
        if self.index >= len(self.rows):
            self.failure = {"kind": "ledger_exhausted", "request": self.requests}
            raise RuntimeError("replay ledger exhausted")
        row = self.rows[self.index]
        expected = atoms_from(row["atoms"])
        position_error = float(np.max(np.abs(atoms.positions - expected.positions)))
        if position_error > POSITION_TOL:
            self.failure = {"kind": "position_mismatch", "request": self.requests + 1, "max_error_A": position_error}
            raise RuntimeError(
                f"position mismatch at replay row {self.index + 1}: "
                f"{position_error:.17g} A > {POSITION_TOL:.17g} A"
            )
        if not np.array_equal(atoms.numbers, expected.numbers):
            self.failure = {"kind": "number_mismatch", "request": self.requests + 1}
            raise RuntimeError(f"atomic-number mismatch at replay row {self.index + 1}")
        if not np.array_equal(atoms.cell.array, expected.cell.array) or not np.array_equal(atoms.pbc, expected.pbc):
            self.failure = {"kind": "cell_pbc_mismatch", "request": self.requests + 1}
            raise RuntimeError(f"cell/PBC mismatch at replay row {self.index + 1}")
        self.index += 1
        self.requests += 1
        return float(row["energy"]), np.asarray(row["forces"], dtype=float)


def load_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "search":
            rows.append(row)
    if not rows:
        raise ValueError(f"no search rows in {path}")
    return rows


def run_case(source_root: Path, case: str, temperature: float) -> dict:
    # This must precede all PAM imports; the result records source provenance.
    sys.path.insert(0, str(source_root))
    from ase.io import read
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw

    plan = json.loads((PARENT / "plan.json").read_text())
    input_path = PARENT / "inputs" / f"{case}.traj"
    ledger_path = PARENT / f"{case}-strict" / "requests.jsonl"
    rows = load_rows(ledger_path)
    parent_result = json.loads((PARENT / f"{case}-strict" / "result.json").read_text())
    target_requests = parent_result["initial"]["evaluation_requests"] + parent_result["records"][0]["evaluation_requests"]
    atoms = read(input_path)
    config = SSWConfig(**plan["configs"][case])
    config = replace(config, temperature_K=temperature)
    surface = ReplaySurface(rows, target_requests)
    result = run_ssw(atoms.copy(), surface, steps=1, config=config,
                     rng=np.random.default_rng(plan["seeds"][case]))
    first = result.records[0] if result.records else None
    return {
        "case": case,
        "temperature_K": temperature,
        "source_root": str(source_root),
        "input_path": str(input_path),
        "ledger_path": str(ledger_path),
        "ledger_rows_available": len(rows),
        "expected_first_record_requests": target_requests,
        "successful_replay": bool(first is not None and first.landing is not None and surface.failure is None and surface.requests == target_requests),
        "replay_requests_consumed": surface.requests,
        "result_evaluation_requests": result.evaluation_requests,
        "result_status": result.status,
        "record_count": len(result.records),
        "first_record": None if first is None else {
            "status": first.status,
            "evaluation_requests": first.evaluation_requests,
            "landing_energy_eV": None if first.landing is None else first.landing.energy,
            "landing_max_force_eV_per_A": None if first.landing is None else first.landing.max_force,
            "error": first.error,
        },
        "replay_failure": surface.failure,
        "position_tolerance_A": POSITION_TOL,
        "position_checks": "passed" if surface.failure is None else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=MATCHED / "source")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/ga_ssw/evidence/mc-scale-matched-20260918/replay-first-move.json")
    args = parser.parse_args()
    source_manifest = args.source_root.parent / "source-manifest.json"
    cases = ["c60_17093", "c60_17094"]
    rows = []
    for case in cases:
        for temperature in TEMPERATURES:
            rows.append(run_case(args.source_root, case, temperature))
    output = {
        "scope": "deterministic first-outer-move replay; no new PES evaluation",
        "source_root": str(args.source_root),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "parent_plan": str(PARENT / "plan.json"),
        "parent_arm": "strict",
        "steps": 1,
        "temperatures_K": list(TEMPERATURES),
        "position_tolerance_A": POSITION_TOL,
        "interpretation": "Temperature is changed only in the frozen SSW config; the replay oracle returns the archived parent E/F rows sequentially.",
        "results": rows,
        "conclusion_boundary": "If all four rows pass and first records match, temperature cannot explain a first-move difference under this exact ledger and source replay. This does not test later MC decisions or establish a search-wide causal claim.",
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(encode(output), indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(rows),
                      "all_ok": all(row["successful_replay"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
