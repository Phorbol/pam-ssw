"""Terminal C60 cost-prefix and topology readout; no PES evaluations."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


def run(prepared, output):
    manifest = json.loads((prepared / "search-plan-manifest.json").read_text())
    sys.path.insert(0, str(prepared / "source"))
    from pamssw.standalone import load_ssw_checkpoint
    from ase.io import write
    spec = importlib.util.spec_from_file_location("frozen_c60_validator", prepared / "validator.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    observations_by_arm = {}
    for entry in sorted(manifest["arm_map"], key=lambda x: x["array_task_id"]):
        folder = prepared / entry["run_dir"]
        plan = json.loads((folder / "plan.json").read_text())
        row = {"case": entry["case"], "arm": entry["arm"], "path": str(folder)}
        if not (folder / "budget.json").exists():
            row.update(status="not_started", search_requests=0, fresh_requests=0)
            rows.append(row)
            continue
        budget = json.loads((folder / "budget.json").read_text())
        raw = ([json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
               if (folder / "requests.jsonl").exists() else [])
        if [x["id"] for x in raw] != list(range(1, len(raw)+1)):
            raise ValueError(f"nonsequential issued request IDs: {folder}")
        uncertain = budget.get("unconfirmed_search_reservations", 0)
        if budget["status"] == "running":
            uncertain += max(0, budget.get("search_reserved", budget["search"])-budget["search"])
        if budget["status"] != "running" and not uncertain and len(raw) != budget["search"]:
            raise ValueError(f"clean terminal raw request/budget mismatch: {folder}")
        if len(budget["fresh_checks"]) != budget["fresh"]:
            raise ValueError(f"fresh reservation/check count mismatch: {folder}")
        row.update(status=budget["status"], search_requests=budget["search"],
                   fresh_requests=budget["fresh"], raw_search_requests=len(raw),
                   unconfirmed_search_reservations=uncertain,
                   failed_pes_requests=sum("error" in x for x in raw),
                   actual_search_calculator_calls=sum(x.get("calculate_calls", 0) for x in budget["segments"]),
                   reserved_gpu_seconds=budget["reserved_seconds"],
                   fresh_checks=budget["fresh_checks"])
        checkpoint_path = folder / "last-result.pkl"
        if not checkpoint_path.exists():
            checkpoint_path = folder / "checkpoint.pkl"
        if not checkpoint_path.exists():
            row["checkpoint_status"] = "missing"
            rows.append(row)
            continue
        cp = load_ssw_checkpoint(checkpoint_path)
        cumulative = cp.initial.evaluation_requests
        observations = [(None, cumulative, cp.initial, None)]
        for step in cp.records:
            cumulative += step.evaluation_requests
            if step.landing is not None and step.landing.converged:
                observations.append((step.index, cumulative, step.landing, step.accepted))
        if cumulative != cp.evaluation_requests or cumulative > budget["search"]:
            raise ValueError(f"checkpoint lineage cost mismatch: {folder}")
        qualified = []
        structures = []
        for index, cost, minimum, accepted in observations:
            atoms = minimum.atoms.copy()
            atoms.calc = None
            graphs = {str(c): validator.graph_row(atoms.numbers, atoms.positions, c)
                      for c in (1.64, 1.7, 1.8)}
            force_pass = bool(minimum.converged and minimum.max_force <= plan["ssw_config"]["fmax"])
            cage = all(x["graph_cage_candidate"] for x in graphs.values())
            energy = minimum.energy <= plan["reference_energy_eV"]+.01
            record = dict(outer_index=index, cumulative_search_requests=cost,
                          energy_eV=minimum.energy, fmax_eV_A=minimum.max_force,
                          search_force_qualified=force_pass, graphs=graphs,
                          cage_all_cutoffs=cage, energy_target=energy,
                          graph_energy_candidate=bool(force_pass and cage and energy),
                          accepted=accepted)
            qualified.append(record)
            atoms.info.update({k: v for k, v in record.items() if k != "graphs" and v is not None})
            structures.append(atoms)
        observations_by_arm[(entry["case"], entry["arm"])] = qualified
        name = f"{entry['case']}-{entry['arm']}"
        if structures:
            write(output / f"{name}-landings.extxyz", structures)
        (output / f"{name}-observations.json").write_text(json.dumps(qualified, indent=2, allow_nan=False)+"\n")
        row.update(outer_attempts=len(cp.records), outer_statuses=dict(Counter(s.status for s in cp.records)),
                   qualified_observations=sum(x["search_force_qualified"] for x in qualified),
                   lineage_requests=cp.evaluation_requests, cost_prefixes={})
        for cap in (15000, 30000, 60000):
            valid = [x for x in qualified if x["search_force_qualified"] and x["cumulative_search_requests"] <= min(cap, budget["search"])]
            hits = [x for x in valid if x["graph_energy_candidate"]]
            row["cost_prefixes"][str(cap)] = {"horizon_reached": budget["search"] >= cap,
                "qualified_observations": len(valid),
                "best_energy_eV": min((x["energy_eV"] for x in valid), default=None),
                "first_graph_energy_candidate_cost": hits[0]["cumulative_search_requests"] if hits else None}
        row["physical_cage_review"] = "required" if any(x["search_force_qualified"] and x["cage_all_cutoffs"] for x in qualified) else "no_cage_candidate"
        rows.append(row)
    pairs = []
    for case in sorted({r["case"] for r in rows}):
        matched = [r for r in rows if r["case"] == case]
        common = min(r["search_requests"] for r in matched)
        comparisons = []
        for requested in (15000, 30000, 60000):
            horizon = min(requested, common)
            arms = {}
            for r in matched:
                valid = [x for x in observations_by_arm.get((case, r["arm"]), [])
                         if x["search_force_qualified"] and x["cumulative_search_requests"] <= horizon]
                arms[r["arm"]] = {"qualified_observations": len(valid),
                    "best_energy_eV": min((x["energy_eV"] for x in valid), default=None),
                    "graph_energy_candidate": any(x["graph_energy_candidate"] for x in valid)}
            comparisons.append({"requested_prefix": requested, "actual_common_horizon": horizon,
                                "both_horizon_reached": common >= requested, "arms": arms})
        pairs.append({"case": case, "comparisons": comparisons})
    result = {"rows": rows, "matched_cost_pairs": pairs,
              "qualification_overhead": json.loads((prepared / manifest["qualification_summary"]).read_text()),
              "scope": "Two new development inputs, MH-1 PES; graph/energy candidates need cold and 3D physical review. Observations are not deduplicated basins."}
    if sum(x["search_requests"] for x in rows) > 240000 or sum(x["fresh_requests"] for x in rows)>12:
        raise ValueError("aggregate experiment budget exceeded")
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"rows": [{k:r.get(k) for k in ("case", "arm", "status", "search_requests", "fresh_requests", "physical_cage_review")} for r in rows]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.prepared.resolve(), args.out.resolve())
