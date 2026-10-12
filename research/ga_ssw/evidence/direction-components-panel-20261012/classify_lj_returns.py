#!/usr/bin/env python3
"""Zero-PES initial-versus-landing geometry readout for LJ direction panel runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMPARATOR = REPO / "research/ga_ssw/evidence/ssw-native-lj-panel-20261007/analyze_panel.py"


def load_compare():
    spec = importlib.util.spec_from_file_location("existing_lj_panel_geometry", COMPARATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing geometry comparator: {COMPARATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_geometry


def ledger_cost(path: Path):
    if not path.is_file():
        return {"present": False, "charged_requests": None, "actual_calculator_calls": None}
    charged = calls = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            charged += int(bool(row.get("charged")))
            calls += int(row.get("actual_calculator_calls", int(bool(row.get("actual_calculator_called")))))
    return {"present": True, "charged_requests": charged, "actual_calculator_calls": calls}


def role_frame(path: Path, role: str):
    if not path.is_file():
        return None
    for atoms in read(path, index=":"):
        if atoms.info.get("role") == role:
            return atoms
    return None


def one_slot(seed_dir: Path, compare):
    result_path = seed_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    structures = seed_dir / "structures.extxyz"
    initial = role_frame(structures, "initial")
    landing = role_frame(structures, "landing")
    geometry = compare(initial, landing)
    event = next((row for row in result.get("outer_events", [])
                  if row.get("kind") == "outer_step" and row.get("index") == 0), None)
    search = ledger_cost(seed_dir / "search-ef-ledger.jsonl")
    fresh = ledger_cost(seed_dir / "fresh-ef-ledger.jsonl")
    # Preserve both the runner's recorded values and independent ledger recounts.
    return {
        "case": result.get("case"), "method": result.get("method"), "seed": result.get("seed", seed_dir.name),
        "run_dir": str(seed_dir), "status": result.get("status", "missing_result"),
        "error": result.get("error"), "execution_outcome": result.get("execution_outcome"),
        "landing_geometry_vs_same_run_initial": geometry,
        "initial_present": initial is not None, "landing_present": landing is not None,
        "landing_accepted": None if event is None else event.get("accepted"),
        "landing_status": None if event is None else event.get("status"),
        "landing_energy_eV": None if event is None else event.get("landing_energy_eV"),
        "initial_energy_eV": (result.get("initial") or {}).get("energy_eV"),
        "energy_delta_landing_minus_initial_eV": (
            None if event is None or (result.get("initial") or {}).get("energy_eV") is None
            else event.get("landing_energy_eV") - result["initial"]["energy_eV"]),
        "costs": {
            "runner_search_requests": result.get("search_requests"),
            "runner_actual_calculator_calls": result.get("actual_calculator_calls"),
            "runner_denied_requests": result.get("denied_requests"),
            "runner_fresh": result.get("fresh"),
            "search_ledger_recount": search, "fresh_ledger_recount": fresh,
            "additional_pes_calls_for_this_readout": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs=2, type=Path, help="two panel run roots, e.g. runs-1756504 runs-1756505")
    parser.add_argument("--out", type=Path, default=HERE / "lj-return-classification.json")
    args = parser.parse_args()
    compare = load_compare()
    slots = []
    for root in args.run_roots:
        root = root.resolve()
        for seed_dir in sorted(root.glob("*/*/seed-*")):
            slots.append(one_slot(seed_dir, compare))
    payload = {
        "purpose": "zero-PES return-to-initial geometry classification; not basin-coverage inference",
        "geometry_comparator": str(COMPARATOR),
        "comparison": "existing compare_geometry(initial, landing): graph isomorphism plus proper-rotation RMS",
        "run_roots": [str(p.resolve()) for p in args.run_roots],
        "slot_count": len(slots), "slots": slots,
    }
    with args.out.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    same = sum(row["landing_geometry_vs_same_run_initial"].get("classification") == "same" for row in slots)
    print(json.dumps({"output": str(args.out), "slots": len(slots), "same_as_initial": same,
                      "additional_pes_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
