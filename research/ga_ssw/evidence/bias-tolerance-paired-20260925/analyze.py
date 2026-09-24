#!/usr/bin/env python3
"""CPU-side graph/permutation-aware endpoint comparison for the tolerance probe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEPTH_ANALYZER = HERE.parent / "climb-depth-ablation-20260925" / "analyze.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("bias_tolerance_depth_analysis", DEPTH_ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def atom_record(atoms):
    return {"numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist()}


def main():
    from ase.io import read

    analyzer = load_analyzer()
    plan = load_json(HERE / "plan.json")
    run_summary = load_json(HERE / "runs-summary.json")
    if run_summary is None:
        raise FileNotFoundError("runs-summary.json is absent; no execution evidence to analyze")
    output_rows = []
    pairs = []
    for name, spec in plan["cases"].items():
        by_bias = {}
        structures = {}
        for bias in plan["arms"]:
            arm = HERE / "runs" / f"{name}-bias{int(round(100*bias)):03d}"
            summary = load_json(arm / "summary.json")
            if summary is None:
                summary = next((row for row in run_summary["rows"]
                                if row.get("case") == name and row.get("bias_fmax") == bias),
                               {"status": "missing_arm_summary"})
            initial_path, landing_path = arm / "post-initial-quench.traj", arm / "outer-landing.traj"
            initial = atom_record(read(initial_path)) if initial_path.is_file() else None
            landing = atom_record(read(landing_path)) if landing_path.is_file() else None
            checks = {item.get("label"): item for item in summary.get("fresh_checks", [])}
            initial_ok = bool(checks.get("post-initial-quench", {}).get("force_qualified")
                              and checks["post-initial-quench"].get("same_numbers")
                              and checks["post-initial-quench"].get("same_cell")
                              and checks["post-initial-quench"].get("same_pbc"))
            landing_ok = bool(summary.get("landing_fresh_qualified"))
            comparison = (analyzer.structure_comparison(initial, landing, spec["system"])
                          if initial is not None and landing is not None else
                          {"status": "missing_initial_or_landing_geometry"})
            row = {"case": name, "system": spec["system"], "seed": spec["seed"],
                   "bias_fmax": bias, "execution_status": summary.get("status"),
                   "outer_records": summary.get("result_records"),
                   "completed_gaussian_stages": summary.get("completed_gaussian_stages"),
                   "search_requests": summary.get("search_requests"),
                   "fresh_requests": summary.get("fresh_requests"),
                   "total_requests_including_fresh": summary.get("total_requests_including_fresh"),
                   "initial_fresh_qualified": initial_ok,
                   "landing_fresh_qualified": landing_ok,
                   "qualifying_landing_total_cost": (summary.get("total_requests_including_fresh")
                                                     if landing_ok else None),
                   "landing_vs_post_initial_structure": comparison,
                   "fresh_start_energy_eV": checks.get("post-initial-quench", {}).get("energy_eV"),
                   "fresh_landing_energy_eV": checks.get("outer-landing", {}).get("energy_eV"),
                   "fresh_landing_minus_start_energy_eV": (
                       checks["outer-landing"]["energy_eV"] - checks["post-initial-quench"]["energy_eV"]
                       if checks.get("outer-landing", {}).get("status") == "completed"
                       and checks.get("post-initial-quench", {}).get("status") == "completed" else None),
                   "search_ledger": analyzer.stream_ledger(arm / "requests.jsonl"),
                   "fresh_ledger": analyzer.stream_ledger(arm / "fresh-requests.jsonl"),
                   "graph_equality_is_not_a_basin_test": True}
            row["search_ledger_matches_summary"] = (
                row["search_ledger"].get("status") == "valid" and
                row["search_ledger"].get("charged_calls") == summary.get("search_requests"))
            row["fresh_ledger_matches_summary"] = (
                row["fresh_ledger"].get("status") == "valid" and
                row["fresh_ledger"].get("charged_calls") == summary.get("fresh_requests"))
            output_rows.append(row)
            by_bias[str(bias)] = row
            structures[str(bias)] = landing
        low, high = by_bias.get("0.1"), by_bias.get("0.2")
        pair = {"case": name, "seed": spec["seed"],
                "both_ledgers_valid_and_closed": bool(low and high and all(
                    row[flag] for row in (low, high) for flag in
                    ("search_ledger_matches_summary", "fresh_ledger_matches_summary"))),
                "both_landing_force_qualified": bool(low and high and
                    low["landing_fresh_qualified"] and high["landing_fresh_qualified"]),
                "total_cost_0.1": None if low is None else low["total_requests_including_fresh"],
                "total_cost_0.2": None if high is None else high["total_requests_including_fresh"],
                "delta_total_cost_0.2_minus_0.1": (
                    high["total_requests_including_fresh"] - low["total_requests_including_fresh"]
                    if low and high and isinstance(low["total_requests_including_fresh"], int)
                    and isinstance(high["total_requests_including_fresh"], int) else None),
                "landing_vs_landing_structure": analyzer.structure_comparison(
                    structures.get("0.1"), structures.get("0.2"), spec["system"])
                    if structures.get("0.1") is not None and structures.get("0.2") is not None
                    else {"status": "missing_landing_geometry"},
                "fresh_landing_energy_0.2_minus_0.1_eV": (
                    high["fresh_landing_energy_eV"] - low["fresh_landing_energy_eV"]
                    if low and high and isinstance(low.get("fresh_landing_energy_eV"), (int, float))
                    and isinstance(high.get("fresh_landing_energy_eV"), (int, float)) else None),
                "both_landings_fresh_force_qualified": bool(low and high and
                    low["landing_fresh_qualified"] and high["landing_fresh_qualified"]),
                "interpretation": "paired start and RNG seed; tolerance may alter later RNG consumption; not bitwise trajectory pairing"}
        pairs.append(pair)
    result = {"status": "analyzed_development_probe",
              "scope": "one first outer attempt per fresh start; not default selection or universal performance",
              "structural_method": "existing climb-depth analyzer: element-labeled graphs, proper Kabsch over graph-compatible mappings, C4H6 CCCC torsion; C60 graph cutoffs 1.80 and 1.64 A",
              "rows": output_rows, "paired_costs": pairs,
              "total_search_requests": run_summary.get("total_search_requests"),
              "total_fresh_requests": run_summary.get("total_fresh_requests")}
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {HERE / 'analysis.json'}")


if __name__ == "__main__":
    main()
