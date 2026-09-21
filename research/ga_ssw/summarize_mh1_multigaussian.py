"""Offline summary for the frozen MH1 multi-Gaussian rotation experiment.

The script consumes JSON result/summary files only.  It reuses the established
1.64/1.70/1.80 A graph predicates and does not infer basin identity from them.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_c60_random_development import graph_row


CUTS = (1.64, 1.7, 1.8)


def _edges(atoms, cutoff):
    if not atoms or len(atoms.get("numbers", [])) != 60:
        return frozenset()
    positions = np.asarray(atoms["positions"], dtype=float)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    return frozenset((int(i), int(j)) for i, j in zip(*np.where(np.triu((distances < cutoff) & (distances > 0), 1))))


def _endpoint_change(initial_atoms, landing_atoms):
    if not initial_atoms or not landing_atoms:
        return None
    initial_positions = np.asarray(initial_atoms.get("positions", []), dtype=float)
    landing_positions = np.asarray(landing_atoms.get("positions", []), dtype=float)
    if initial_positions.shape != landing_positions.shape or initial_positions.ndim != 2:
        return None
    pair_initial = np.linalg.norm(initial_positions[:, None] - initial_positions[None, :], axis=2)
    pair_landing = np.linalg.norm(landing_positions[:, None] - landing_positions[None, :], axis=2)
    upper = np.triu(np.ones(pair_initial.shape, dtype=bool), 1)
    pair_rms = float(np.sqrt(np.mean((pair_landing[upper] - pair_initial[upper]) ** 2)))
    return {
        str(cut): {
            "initial_edges": len(_edges(initial_atoms, cut)),
            "landing_edges": len(_edges(landing_atoms, cut)),
            "changed_labeled_edges": len(_edges(initial_atoms, cut) ^ _edges(landing_atoms, cut)),
        }
        for cut in CUTS
    } | {"pair_distance_rms_A": pair_rms}


def _geometry(atoms):
    if not atoms or len(atoms.get("numbers", [])) != 60:
        return None
    return {str(cut): graph_row(atoms["numbers"], atoms["positions"], cut)
            for cut in CUTS}


def _stage_record(record):
    climb = record.get("climb") or []
    return {
        "index": record.get("index"),
        "status": record.get("status"),
        "accepted": record.get("accepted"),
        "evaluation_requests": record.get("evaluation_requests"),
        "gaussian_count": len(climb),
        "climb": [
            {
                key: event.get(key)
                for key in ("index", "status", "rotation_solver", "rotation_force_requests",
                            "rotation_stop_reason", "rotation_converged", "weight", "width",
                            "quench_requests", "true_energy", "biased_energy", "error")
                if key in event
            }
            for event in climb if isinstance(event, dict)
        ],
    }


def _arm_summary(folder, root_rows, state, arm, source_energy):
    row = next((item for item in root_rows
                if item.get("state") == state and item.get("arm") == arm), None)
    row = row or {"status": "missing_summary", "state": state, "arm": arm}
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    checks = {int(item["index"]): item for item in row.get("fresh_checks", [])
              if isinstance(item, dict) and "index" in item}
    minima = [] if result is None else result.get("minima", [])
    minima_out = []
    for index, minimum in enumerate(minima):
        atoms = minimum.get("atoms", {})
        energy = minimum.get("energy")
        check = checks.get(index, {})
        minima_out.append({
            "index": index,
            "evaluation_requests": minimum.get("evaluation_requests"),
            "energy_eV": energy,
            "delta_energy_from_input_eV": None if energy is None else energy - source_energy,
            "max_force_eV_A": minimum.get("max_force"),
            "fresh_check": check,
            "fresh_qualified": bool(check.get("qualified", False)),
            "geometry": _geometry(atoms),
        })
    raw_records = [] if result is None else result.get("records", [])
    records = [_stage_record(record) for record in raw_records]
    initial_atoms = None if result is None else (result.get("initial") or {}).get("atoms")
    landing_atoms = None
    for record in raw_records:
        if isinstance(record.get("landing"), dict) and record["landing"].get("atoms"):
            landing_atoms = record["landing"]["atoms"]
    rotation_calls = sum(int(event.get("rotation_force_requests", 0) or 0)
                        for record in records for event in record["climb"])
    biased_quench_calls = sum(int(event.get("quench_requests", 0) or 0)
                              for record in records for event in record["climb"])
    initial_requests = None if result is None else (result.get("initial") or {}).get("evaluation_requests")
    record_requests = [record.get("evaluation_requests") for record in raw_records]
    return {
        "state": state,
        "arm": arm,
        "status": row.get("status"),
        "error": row.get("error"),
        "search_calls": row.get("search_calls"),
        "fresh_calls": row.get("fresh_calls"),
        "boundary": row.get("boundary"),
        "denials": row.get("denials"),
        "record_count": len(records),
        "gaussian_counts": row.get("gaussian_counts", [r["gaussian_count"] for r in records]),
        "record_statuses": row.get("record_statuses", [r["status"] for r in records]),
        "costs": {
            "initial_requests": initial_requests,
            "record_requests": record_requests,
            "total_requests": None if result is None else result.get("evaluation_requests"),
            "rotation_calls": rotation_calls,
            "biased_quench_calls": biased_quench_calls,
        },
        "endpoint_change": _endpoint_change(initial_atoms, landing_atoms),
        "records": records,
        "minima": minima_out,
        "fresh_denominator": sum(bool(item.get("fresh_check")) for item in minima_out),
        "fresh_qualified": sum(bool(item.get("fresh_qualified")) for item in minima_out),
        "cage_candidates_by_cutoff": {
            str(cut): sum(bool((item.get("geometry") or {}).get(str(cut), {}).get("graph_cage_candidate"))
                          for item in minima_out if item.get("fresh_qualified"))
            for cut in CUTS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.input / "plan.json").read_text())
    summary_path = args.input / "summary.json"
    root_rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    output = {
        "source": str(args.input),
        "scope": plan.get("scope"),
        "budgets": {key: plan.get(key) for key in ("search_cap_per_arm", "fresh_cap_per_arm", "total_cap_per_arm", "wall_seconds_per_arm")},
        "reference": {"type": "input-relative energy only", "energy_success_not_evaluated": True},
        "same_basin_claim": False,
        "cases": {},
        "missing_or_failed": [],
    }
    for state, metadata in plan.get("inputs", {}).items():
        state_dir_rows = {}
        for arm in plan.get("arms", []):
            folder = args.input / f"{state}-{arm}"
            if not folder.exists():
                output["missing_or_failed"].append({"state": state, "arm": arm, "reason": "missing_result_directory"})
                continue
            state_dir_rows[arm] = _arm_summary(folder, root_rows, state, arm, metadata["source_energy_eV"])
            if state_dir_rows[arm]["status"] in ("exception", "failed", "missing_summary"):
                output["missing_or_failed"].append({"state": state, "arm": arm, "status": state_dir_rows[arm]["status"], "error": state_dir_rows[arm]["error"]})
        output["cases"][state] = state_dir_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
