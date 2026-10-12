#!/usr/bin/env python3
"""Offline matched readout for calibrated shape-LS and its frozen controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NEW_DEFAULT = HERE / "run-1752156" / "summary.json"
OLD_DEFAULT = HERE.parent / "shape-single-escape-v1" / "run-1739743" / "summary.json"
STARTS, SEEDS = ("c4-gauche", "c4-trans", "c60-minimum"), (701, 702)
ARMS = ("none", "pair-0.03", "shape-0.03", "shape-linear-response")
C4_CUTOFF, C60_CUTOFFS = "hc-table-plus-0.1A", ("1.64A", "1.70A", "1.80A")


def load_helper():
    path = HERE.parent / "shape-single-escape-v1" / "analyze.py"
    spec = importlib.util.spec_from_file_location("shape_single_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_full(new, old):
    expected_new = {f"{s}-seed{n}-shape-linear-response" for s in STARTS for n in SEEDS}
    expected_old = {f"{s}-seed{n}-{a}" for s in STARTS for n in SEEDS for a in ARMS[:3]}
    for label, summary, expected in (("calibrated", new, expected_new),
                                     ("baseline", old, expected_old)):
        ids = [row.get("run_id") for row in summary.get("runs", [])]
        if summary.get("status", "").startswith("started") or not expected <= set(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{label} summary incomplete/duplicated: status={summary.get('status')}, "
                             f"expected_present={len(expected & set(ids))}/{len(expected)}")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_summary", nargs="?", type=Path, default=NEW_DEFAULT)
    parser.add_argument("baseline_summary", nargs="?", type=Path, default=OLD_DEFAULT)
    args = parser.parse_args()
    new_path, old_path = args.new_summary.resolve(), args.baseline_summary.resolve()
    new, old = json.loads(new_path.read_text()), json.loads(old_path.read_text())
    require_full(new, old)
    helper = load_helper()
    new_inputs = {r["name"]: r for r in new.get("input_qualifications", [])}
    old_inputs = {r["name"]: r for r in old.get("input_qualifications", [])}
    new_graph, old_graph = (helper.load_graph_helpers(p.parent.parent) for p in (new_path, old_path))
    rows = {}
    raw_rows = {}
    for summary_path, summary, inputs, graph_helpers in (
            (new_path, new, new_inputs, new_graph), (old_path, old, old_inputs, old_graph)):
        for r in summary["runs"]:
            rows[r["run_id"]] = helper.make_arm_record(summary_path, r, inputs, graph_helpers)
            raw_rows[r["run_id"]] = r

    matched = {(s, n): {a: f"{s}-seed{n}-{a}" for a in ARMS} for s in STARTS for n in SEEDS}

    # Initial coordinates are checked from the result artifacts, without emitting atom arrays.
    initial_identity = {}
    for key, arm_ids in matched.items():
        positions = {}
        for arm, rid in arm_ids.items():
            base = new_path.parent if arm == ARMS[-1] else old_path.parent
            result = json.loads((base / rid / "result.json").read_text())
            positions[arm] = np.asarray(result["initial"]["atoms"]["positions"], dtype=float)
        ref = positions["none"]
        initial_identity[f"{key[0]}-seed{key[1]}"] = {
            "same_positions_all_four_arms": all(np.array_equal(ref, p) for p in positions.values()),
            "max_abs_delta_A_vs_none": {a: float(np.max(np.abs(p-ref))) for a, p in positions.items()},
        }

    calibration = json.loads((HERE / "calibration.json").read_text())["starts"]
    outcomes, response, graph_pairs, torsions, paired_costs = [], [], [], [], []
    compare_graphs = new_graph[1]
    for key, arm_ids in matched.items():
        bundles = {}
        paired_costs.append({"start_seed": f"{key[0]}-seed{key[1]}", "actual_calculate_calls": {
            arm: raw_rows[rid].get("actual_calculate_calls") for arm, rid in arm_ids.items()}})
        for arm, rid in arm_ids.items():
            row, raw = rows[rid], raw_rows[rid]
            bundles[arm] = row.get("_landing_graph_bundle")
            failures = []
            if raw.get("status") != "returned": failures.append("not_returned")
            if raw.get("run_ssw_status") != "completed": failures.append("driver_incomplete")
            if raw.get("outer_records") != 1: failures.append("outer_record_count")
            if raw.get("search_denials", 0): failures.append("search_denials")
            if raw.get("landing_surface") != "true": failures.append("landing_not_true_surface")
            fresh = raw.get("fresh_candidate") or {}
            if fresh.get("status") != "completed": failures.append("fresh_check_incomplete")
            elif fresh.get("force_qualified") is not True: failures.append("fresh_force_unqualified")
            outcomes.append({"run_id": rid, "failure_flags": failures, "status": raw.get("status"),
                "driver": raw.get("run_ssw_status"), "surface": raw.get("landing_surface"),
                "fresh": fresh.get("status"), "fresh_fmax_eV_A": fresh.get("fmax_eV_A"),
                "fresh_force_qualified": fresh.get("force_qualified"), "mc_accepted": raw.get("mc_accepted"),
                "search_requests": raw.get("search_requests"), "fresh_requests": raw.get("fresh_requests"),
                "actual_calculate_calls": raw.get("actual_calculate_calls"),
                "denials": raw.get("search_denials"), "boundary": raw.get("search_boundary")})
            prep = raw.get("ls_preparation") or {}
            before, after = prep.get("true_energy_before"), prep.get("true_energy_after")
            actual = after-before if before is not None and after is not None else (0.0 if arm == "none" else None)
            cal = calibration[key[0]]
            frac = (0.0 if arm == "none" else 0.03 if arm in ("pair-0.03", "shape-0.03")
                    else raw.get("initial_fraction"))
            chi = (0.0 if arm == "none" else cal["chi_pair_eV"] if arm == "pair-0.03"
                   else cal["chi_shape_eV"])
            base_fraction = cal["base_pair_fraction"]
            prediction = 0.5 * (frac/base_fraction)**2 * chi if frac is not None else None
            response.append({"run_id": rid, "prequench_true_delta_eV": actual, "quadratic_prediction_eV": prediction,
                "actual_over_prediction": actual/prediction if prediction not in (None, 0) and actual is not None else None,
                "initial_fraction": frac, "chi_eV": chi})
            if key[0].startswith("c4-"):
                torsions.append({"run_id": rid, "raw_deg": row["c4_torsion_raw_input_deg"],
                                 "initial_deg": row["c4_torsion_actual_initial_deg"],
                                 "landing_deg": row["c4_torsion_landing_deg"]})
        cutoffs = (C4_CUTOFF,) if key[0].startswith("c4-") else C60_CUTOFFS
        for i, a in enumerate(ARMS):
            for b in ARMS[i+1:]:
                valid = all(rows[arm_ids[x]]["candidate_generated_independent_of_mc"] for x in (a,b))
                iso = {c: compare_graphs(bundles[a], bundles[b]).get(c) for c in cutoffs} if valid and bundles[a] and bundles[b] else {c: None for c in cutoffs}
                graph_pairs.append({"pair": [arm_ids[a], arm_ids[b]], "candidate_gate": valid, "isomorphic_by_cutoff": iso})

    total_new_actual = sum(int(r.get("actual_calculate_calls") or 0) for r in new["runs"])
    total_old_actual = sum(int(r.get("actual_calculate_calls") or 0) for r in old["runs"])
    report = {
        "scope": "offline summary/result/extxyz analysis; no PES calls",
        "initial_position_identity": initial_identity,
        "all_runs_and_failures": outcomes,
        "paired_costs": paired_costs,
        "prequench_response_vs_quadratic_prediction": response,
        "fresh_force_and_graph_isomorphism_across_arms": graph_pairs,
        "c4_torsions_deg": torsions,
        "costs": {
            "calibrated_six_search_plus_fresh_calculations": total_new_actual,
            "calibrated_input_preflights": new.get("input_qualification_requests"),
            "calibrated_total_including_3_preflights": total_new_actual + int(new.get("input_qualification_requests", 0)),
            "baseline_24_arms_actual_calculations": total_old_actual,
            "baseline_input_preflights": old.get("input_qualification_requests"),
            "baseline_total_including_preflights": total_old_actual + int(old.get("input_qualification_requests", 0)),
            "matched_baseline_fresh_requests": sum(int(raw_rows[f"{s}-seed{n}-{a}"].get("fresh_requests") or 0)
                for s in STARTS for n in SEEDS for a in ARMS[:3]),
            "calibrated_fresh_requests": sum(int(r.get("fresh_requests") or 0) for r in new["runs"]),
        },
        "limits": ["The quadratic prediction is the local physical-minimum energy response +a^2 chi/2, not a finite-prequench guarantee.",
                   "Graph isomorphism and C4 torsions are structural diagnostics, not basin identity or reaction certification.",
                   "Wall times come from different Slurm jobs and are not treated as paired hardware evidence."],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
