#!/usr/bin/env python3
"""Small read-only follow-up for final geometries and saved Gaussian paths."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
ADAPTOR = AseAtomsAdaptor()
TOLS = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}


def read_result(case, method):
    path = HERE / f"{case}-{method}-seed41" / "result.json"
    with path.open() as stream:
        return json.load(stream)


def best_match(case):
    paths = [HERE / f"{case}-{method}-seed41" / "best.extxyz" for method in ("ritz", "recovered")]
    if not all(path.exists() for path in paths):
        return {"status": "missing_best_geometry", "missing": [str(p) for p in paths if not p.exists()]}
    atoms = [read(str(path)) for path in paths]
    if not all(a.pbc.all() for a in atoms) or not np.array_equal(atoms[0].numbers, atoms[1].numbers):
        return {"status": "geometry_contract_mismatch", "pbc": [a.pbc.tolist() for a in atoms]}
    structures = [ADAPTOR.get_structure(a) for a in atoms]
    return {
        "status": "compared",
        "ritz_vs_recovered_matches": {
            key: bool(StructureMatcher(**tol, primitive_cell=False, scale=False,
                                      attempt_supercell=False, comparator=ElementComparator()).fit(*structures))
            for key, tol in TOLS.items()
        },
        "interpretation": "pairwise final-best geometry only; not strict basin or phase identity",
    }


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def analyze_arm(case, method):
    result = read_result(case, method)
    initial = result.get("initial", {})
    initial_energy = initial.get("energy")
    initial_cost = initial.get("evaluation_requests")
    initial_cost_valid = isinstance(initial_cost, int) and not isinstance(initial_cost, bool) and initial_cost >= 0
    initial_ok = initial.get("converged") is True and finite(initial_energy) and initial_cost_valid
    best = float(initial_energy) if initial_ok else None
    cumulative = initial_cost if initial_cost_valid else 0
    points = [{"record_index": None, "cost_requests": cumulative, "best_so_far_energy_eV": best,
               "event": "initial", "qualified": bool(initial_ok)}]
    paths = []
    records = result.get("records", [])
    for index, record in enumerate(records):
        cost = record.get("evaluation_requests")
        if isinstance(cost, int) and not isinstance(cost, bool) and cost >= 0:
            cumulative += cost
        centers = [np.asarray(event["center"], dtype=float) for event in record.get("climb", [])
                   if isinstance(event, dict) and event.get("center") is not None]
        step_lengths = [float(np.linalg.norm(b - a)) for a, b in zip(centers, centers[1:])]
        path_length = sum(step_lengths)
        endpoint = float(np.linalg.norm(centers[-1] - centers[0])) if len(centers) >= 2 else (0.0 if centers else None)
        climb = record.get("climb", [])
        final_has_center = bool(climb and isinstance(climb[-1], dict) and climb[-1].get("center") is not None)
        paths.append({
            "record_index": index, "record_status": record.get("status"),
            "record_requests": cost, "cumulative_requests": cumulative,
            "gaussian_center_count": len(centers), "path_length_A_cartesian": path_length,
            "first_to_last_center_A_cartesian": endpoint,
            "first_last_endpoint_complete": final_has_center,
            "missing_final_endpoint_not_inferred": not final_has_center,
        })
        landing = record.get("landing")
        qualified = isinstance(landing, dict) and landing.get("converged") is True and finite(landing.get("energy"))
        if qualified and (best is None or float(landing["energy"]) < best):
            best = float(landing["energy"])
        points.append({"record_index": index, "cost_requests": cumulative,
                       "best_so_far_energy_eV": best, "event": record.get("status"),
                       "qualified_landing": bool(qualified), "record_error": record.get("error")})
    final_best = min((float(m["energy"]) for m in result.get("minima", [])
                      if isinstance(m, dict) and finite(m.get("energy")) and m.get("converged") is True),
                     default=(best if best is not None else None))
    earliest = None
    if final_best is not None:
        for index, record in enumerate(records):
            landing = record.get("landing")
            if isinstance(landing, dict) and landing.get("converged") is True and finite(landing.get("energy")) and abs(float(landing["energy"]) - final_best) <= 1e-8:
                earliest = index
                break
    return {
        "method": method, "status": result.get("status"),
        "initial_energy_eV": initial_energy if initial_ok else None,
        "final_best_energy_eV": final_best,
        "earliest_completed_record_attaining_final_best": earliest,
        "record_energy_tolerance_eV_for_index_lookup_only": 1e-8,
        "best_so_far_curve": points,
        "gaussian_attempt_paths": paths,
    }


def write_report(data):
    lines = ["# Rotation-priority follow-up", "",
             "Best-so-far curves use the initial true minimum and converged true landings; failed records retain their cumulative request cost without adding an energy. Gaussian-center distances are direct Cartesian distances in the saved fixed cell, without minimum-image wrapping. Missing terminal centers are not inferred.", "",
             "The 1e-8 eV equality is used only to map the final stored best back to its earliest record index. It is not an energy-ranking or faster-discovery criterion; energy differences below MLIP accuracy are not interpreted as meaningful.", "",
             "| Case | Policy | Records | Final best (eV) | Earliest record index |", "|---|---|---:|---:|---:|"]
    for case, row in data["cases"].items():
        for method, arm in row.get("arms", {}).items():
            lines.append(f"| {case} | {method} | {len(arm.get('gaussian_attempt_paths', []))} | {arm.get('final_best_energy_eV')} | {arm.get('earliest_completed_record_attaining_final_best')} |")
        lines.append(f"| {case} | Ritz~recovered best geometry | — | — | {row.get('best_geometry', {}).get('ritz_vs_recovered_matches', 'unavailable')} |")
        lines.append("")
        for method, arm in row.get("arms", {}).items():
            lines.append(f"### {case} / {method}: cost-energy curve")
            lines.extend(["", "| Record | Cumulative requests | Best-so-far energy (eV) | Event | Qualified landing |", "|---:|---:|---:|---|---|"])
            for point in arm.get("best_so_far_curve", []):
                lines.append(f"| {point.get('record_index', 'initial')} | {point['cost_requests']} | {point.get('best_so_far_energy_eV')} | {point.get('event')} | {point.get('qualified_landing', point.get('qualified'))} |")
            lines.append("")
    (HERE / "followup.md").write_text("\n".join(lines) + "\n")


def main():
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "calculator_or_PES_calls": 0,
            "geometry_matcher": {"tolerances": TOLS, "primitive_cell": False, "scale": False,
                                 "attempt_supercell": False, "maximum_fits": 4}, "cases": {}}
    for case in ("aloh3", "brookite48"):
        row = {"best_geometry": best_match(case), "arms": {}}
        for method in ("ritz", "recovered"):
            try:
                row["arms"][method] = analyze_arm(case, method)
            except Exception as error:
                row["arms"][method] = {"status": "analysis_error", "error": f"{type(error).__name__}: {error}"}
        data["cases"][case] = row
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (HERE / "followup.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        write_report(data)
    return 1 if any(a.get("status") == "analysis_error" for r in data["cases"].values() for a in r["arms"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
