"""Offline geometry checks for saved SiO2 panel endpoints; never loads a calculator."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

CUTOFFS = (1.7, 1.8, 1.9, 2.0)


def describe(data: dict) -> dict:
    atoms = Atoms(**{key: data[key] for key in ("numbers", "positions", "cell", "pbc")})
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, d, shifts = neighbor_list("ijdS", atoms, 6.0, self_interaction=False)
    pairs = {}
    for left, right in (("Si", "O"), ("O", "O"), ("Si", "Si")):
        mask = (symbols[i] == left) & (symbols[j] == right)
        if left == right:
            mask &= i < j
        pairs[f"{left}-{right}"] = float(d[mask].min()) if mask.any() else None
    si_indices = np.flatnonzero(symbols == "Si")
    o_indices = np.flatnonzero(symbols == "O")
    coordination = {"cutoffs_A": list(CUTOFFS), "Si_to_O": {}, "O_to_Si": {}}
    for label, centers, center_symbol, neighbor_symbol in (
        ("Si_to_O", si_indices, "Si", "O"), ("O_to_Si", o_indices, "O", "Si")
    ):
        counts_at_cutoff = {}
        for cutoff in CUTOFFS:
            counts = [int(np.sum((i == center) & (symbols[j] == neighbor_symbol) & (d < cutoff)))
                      for center in centers]
            counts_at_cutoff[str(cutoff)] = {
                "per_center": counts,
                "histogram": {str(k): v for k, v in sorted(collections.Counter(counts).items())},
            }
        coordination[label] = counts_at_cutoff
    return {
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
        "volume_A3": float(atoms.get_volume()),
        "cell_A": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "species_pair_minimum_A": pairs,
        "coordination_sensitivity": coordination,
        "note": "Pair distances use periodic images within a 6 A neighbor search; coordination bands are descriptive sensitivity checks, not bond/phase assignments.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for checks_path in sorted(root.glob("arms-150*/arm-*/case*/fresh-checks.json")):
        run_dir = checks_path.parent
        summary = json.loads((run_dir / "summary.json").read_text())
        result = json.loads((run_dir / "search-result.json").read_text())["result"]
        outer = result["records"][1:]
        checks = json.loads(checks_path.read_text())["checks"]
        for check in checks:
            if check.get("source") == "initial":
                record_status, accepted = "initial", None
            else:
                idx = int(check["record_index"])
                if idx < 0 or idx >= len(outer):
                    raise ValueError(f"bad landing record_index {idx} in {checks_path}")
                record_status = outer[idx]["status"]
                accepted = outer[idx].get("accepted")
            geom = describe(check["atoms"])
            geom.update({
                "arm": run_dir.name,
                "method": summary["method"],
                "seed": summary["seed"],
                "source": check["source"],
                "record_index": check.get("record_index"),
                "record_status": record_status,
                "accepted": accepted,
                "fresh_certified": check["physical_certificate"]["certified"],
                "fresh_fmax_eV_A": check["physical_certificate"]["fmax_eV_A"],
                "fresh_stress_residual_max_eV_A3": check["physical_certificate"]["stress_residual_max_eV_A3"],
                "energy_eV": check["energy_eV"],
                "objective_eV": check["objective_eV"],
            })
            rows.append(geom)
    if len(rows) != 23:
        raise ValueError(f"expected 23 fresh-checked initial/landing geometries, got {len(rows)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scope": "offline geometry audit of saved fresh-check atoms; no PES/calculator calls",
        "pair_search_cutoff_A": 6.0,
        "coordination_cutoffs_A": list(CUTOFFS),
        "endpoint_count": len(rows),
        "endpoints": rows,
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "endpoint_count": len(rows), "calculator_calls": 0}))


if __name__ == "__main__":
    main()
