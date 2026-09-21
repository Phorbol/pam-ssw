"""Offline budget-slice analysis for the MH1 equal-budget C60 runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_c60_random_development import graph_row


BUDGETS = (3000, 6000, 12000)
REFERENCE_ENERGY = -62215.39337032347
ENERGY_TOLERANCE = 0.01
CUTS = (1.64, 1.7, 1.8)


def _edge_signature(atoms, cutoff):
    if not atoms or len(atoms.get("numbers", [])) != 60:
        return None
    positions = np.asarray(atoms["positions"], dtype=float)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    return tuple((int(i), int(j)) for i, j in zip(*np.where(np.triu((distances < cutoff) & (distances > 0), 1))))


def _geometry(atoms):
    if not atoms or len(atoms.get("numbers", [])) != 60:
        return None
    return {str(cut): graph_row(atoms["numbers"], atoms["positions"], cut)
            for cut in CUTS}


def _fresh_checks(summary_row):
    return {int(item["index"]): item for item in summary_row.get("fresh_checks", [])
            if isinstance(item, dict) and "index" in item}


def _minimum(index, cumulative, landing, check, accepted=None):
    atoms = landing.get("atoms", {})
    stored_energy = landing.get("energy")
    fresh_energy = check.get("energy_eV")
    energy = fresh_energy if fresh_energy is not None else stored_energy
    return {
        "index": index,
        "cumulative_search_calls": cumulative,
        "accepted": accepted,
        "stored_energy_eV": stored_energy,
        "fresh_energy_eV": fresh_energy,
        "energy_eV": energy,
        "stored_vs_fresh_energy_delta_eV": (None if fresh_energy is None or stored_energy is None
                                             else stored_energy - fresh_energy),
        "fmax_eV_A": landing.get("max_force"),
        "fresh_check": check,
        "fresh_qualified": bool(check.get("qualified", False)),
        "energy_target": bool(check.get("qualified", False) and energy is not None and
                                energy <= REFERENCE_ENERGY + ENERGY_TOLERANCE),
        "geometry": _geometry(atoms),
        "edge_signatures": {str(cut): _edge_signature(atoms, cut) for cut in CUTS},
    }


def _slice(rows, budget, initial, source_energy):
    reached = [row for row in rows if row["cumulative_search_calls"] <= budget]
    qualified = [row for row in reached if row["fresh_qualified"]]
    best_pool = qualified + ([initial] if initial["fresh_qualified"] and initial["cumulative_search_calls"] <= budget else [])
    best = min((row for row in best_pool if row["energy_eV"] is not None),
               key=lambda row: row["energy_eV"], default=None)
    result = {
        "budget_search_calls": budget,
        "available": True,
        "minimum_count": len(reached),
        "fresh_qualified_count": len(qualified),
        "completed_landing_count": len(reached),
        "accepted_count": sum(bool(row.get("accepted")) for row in reached),
        "acceptance_rate": (sum(bool(row.get("accepted")) for row in reached) / len(reached)
                            if reached else None),
        "best_qualified_energy_eV": None if best is None else best["energy_eV"],
        "best_qualified_delta_from_input_eV": None if best is None else best["energy_eV"] - source_energy,
        "best_minimum_index": None if best is None else best["index"],
        "best_energy_includes_initial": True,
        "qualification_scope": "fresh force threshold; connectivity reported separately",
        "best_connected_energy_by_cutoff_eV": {},
        "best_qualified_delta_from_reference_eV": None if best is None else best["energy_eV"] - REFERENCE_ENERGY,
        "energy_target_count": sum(bool(row["energy_target"]) for row in qualified),
        "cage_candidates_by_cutoff": {},
        "connected_by_cutoff": {},
        "energy_cage_intersection_by_cutoff": {},
        "distinct_labelled_bondgraph_count_by_cutoff": {},
    }
    for cut in CUTS:
        key = str(cut)
        cage = [row for row in qualified if (row.get("geometry") or {}).get(key, {}).get("graph_cage_candidate")]
        connected = [row for row in qualified if (row.get("geometry") or {}).get(key, {}).get("components") == 1]
        connected_best_pool = [row for row in best_pool if (row.get("geometry") or {}).get(key, {}).get("components") == 1]
        result["best_connected_energy_by_cutoff_eV"][key] = min((row["energy_eV"] for row in connected_best_pool), default=None)
        result["cage_candidates_by_cutoff"][key] = len(cage)
        result["connected_by_cutoff"][key] = len(connected)
        result["energy_cage_intersection_by_cutoff"][key] = sum(row["energy_target"] for row in cage)
        result["distinct_labelled_bondgraph_count_by_cutoff"][key] = len({row["edge_signatures"].get(key) for row in qualified if row["edge_signatures"].get(key) is not None})
    result["minimum_indices"] = [row["index"] for row in reached]
    return result


def _analyze_arm(folder, summary_row, source_energy):
    result_path = folder / "result.json"
    if not result_path.exists():
        return {"status": "missing_result", "summary": summary_row, "budgets": [], "minimum_records": []}
    result = json.loads(result_path.read_text())
    initial = result.get("initial") or {}
    records = result.get("records", [])
    initial_requests = int(initial.get("evaluation_requests", 0))
    cumulative = initial_requests
    checks = _fresh_checks(summary_row)
    initial_minimum = _minimum(0, cumulative, initial, checks.get(0, {}))
    minima = []
    next_index = 1
    record_costs = []
    for record in records:
        cost = int(record.get("evaluation_requests", 0))
        cumulative += cost
        landing = record.get("landing")
        record_costs.append({"index": record.get("index"), "evaluation_requests": cost,
                             "cumulative_search_calls": cumulative,
                             "status": record.get("status"), "accepted": record.get("accepted")})
        if isinstance(landing, dict) and landing.get("converged"):
            minima.append(_minimum(next_index, cumulative, landing, checks.get(next_index, {}),
                                   accepted=record.get("accepted")))
            next_index += 1
    accounting = {
        "result_evaluation_requests": result.get("evaluation_requests"),
        "reconstructed_initial_plus_records": cumulative,
        "accounting_matches": result.get("evaluation_requests") == cumulative,
        "summary_search_calls": summary_row.get("search_calls"),
        "summary_matches": summary_row.get("search_calls") == cumulative,
    }
    if not accounting["accounting_matches"] or not accounting["summary_matches"]:
        raise ValueError(f"cost accounting mismatch: {folder}")
    if next_index != len(result.get("minima", [])):
        raise ValueError(f"landing/minima mapping mismatch: {folder}")
    actual = summary_row.get("search_calls")
    return {
        "status": summary_row.get("status"),
        "error": summary_row.get("error"),
        "boundary": summary_row.get("boundary"),
        "denials": summary_row.get("denials"),
        "actual_search_calls": actual,
        "fresh_calls": summary_row.get("fresh_calls"),
        "source_energy_eV": source_energy,
        "initial_converged": bool(initial.get("converged", False)),
        "initial_failure": (None if initial.get("converged", False)
                            else {"converged": initial.get("converged"),
                                  "energy_eV": initial.get("energy"),
                                  "max_force_eV_A": initial.get("max_force")}),
        "accounting": accounting,
        "record_costs": record_costs,
        "initial_minimum": initial_minimum,
        "minimum_records": minima,
        "failed_or_partial_records": [row for row in record_costs if row["status"] not in ("completed", "lower_true_energy", "gaussian_limit", "stage_release")],
        "budgets_reached": [budget for budget in BUDGETS if actual is not None and actual >= budget],
        "budgets": [_slice(minima, budget, initial_minimum, source_energy) if actual is not None and actual >= budget else {
            "budget_search_calls": budget, "available": False, "reason": "actual_search_calls_below_budget",
            "actual_search_calls": actual,
            "minimum_indices": [row["index"] for row in minima if row["cumulative_search_calls"] <= budget]
        } for budget in BUDGETS],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.input / "plan.json").read_text())
    summary_rows = json.loads((args.input / "summary.json").read_text()) if (args.input / "summary.json").exists() else []
    output = {
        "source": str(args.input),
        "scope": plan.get("scope"),
        "budgets": list(BUDGETS),
        "reference": {"energy_eV": REFERENCE_ENERGY, "tolerance_eV": ENERGY_TOLERANCE},
        "graph_cutoffs_A": list(CUTS),
        "same_basin_claim": False,
        "cases": {},
        "missing_or_failed": [],
    }
    for state, metadata in plan.get("inputs", {}).items():
        output["cases"][state] = {}
        for arm in plan.get("arms", []):
            folder = args.input / f"{state}-{arm}"
            summary_row = next((row for row in summary_rows
                                if row.get("state") == state and row.get("arm") == arm),
                               {"state": state, "arm": arm, "status": "missing_summary"})
            if not folder.exists():
                output["missing_or_failed"].append({"state": state, "arm": arm, "reason": "missing_result_directory"})
            arm_result = _analyze_arm(folder, summary_row, metadata["source_energy_eV"])
            output["cases"][state][arm] = arm_result
            if arm_result.get("status") in ("missing_result", "missing_summary", "exception", "failed"):
                output["missing_or_failed"].append({"state": state, "arm": arm, "status": arm_result.get("status"), "error": arm_result.get("error")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
