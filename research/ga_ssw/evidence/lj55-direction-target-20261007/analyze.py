#!/usr/bin/env python3
"""Offline four-slot target and cost readout; absent/failed runs stay in denominator."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lj55_target_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def ledger_cost(path):
    cost = dict(path=str(path), exists=path.is_file(), charged=0, denied=0,
                calculator_calls=0, failed=0)
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            cost["charged"] += bool(row.get("charged"))
            cost["denied"] += row.get("status") == "denied"
            cost["failed"] += row.get("status") == "failed"
            cost["calculator_calls"] += int(row.get("actual_calculator_calls", 0))
    return cost


def main():
    from ase.io import read
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runs", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    target = runner.target_atoms()
    prepare_summary = json.loads((args.prepared / "prepare-summary.json").read_text())
    prep_rows = []
    for seed in runner.SEEDS:
        folder = args.prepared / f"lj55-seed{seed}"
        prep_rows.append(dict(seed=seed,
            search=ledger_cost(folder / "prepare-ef-ledger.jsonl"),
            fresh=ledger_cost(folder / "prepare-fresh-ledger.jsonl")))
    ref_fresh = ledger_cost(args.prepared / "cambridge-reference-fresh-ledger.jsonl")
    rows = []
    for slot, folder in enumerate(args.runs):
        search = ledger_cost(folder / "search-ef-ledger.jsonl")
        fresh = ledger_cost(folder / "fresh-ef-ledger.jsonl")
        row = dict(slot=slot, seed=runner.SEEDS[slot // 2],
                   arm="rotation" if slot % 2 == 0 else "full_per_atom",
                   directory=str(folder.resolve()), search=search, fresh=fresh,
                   result_present=False, joint_target_verified=False)
        if not (folder / "result.json").is_file():
            row.update(execution_status="missing_result", budget_outcome="unresolved",
                       scientific_outcome="not_verified")
            rows.append(row)
            continue
        result = json.loads((folder / "result.json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        assert (result["slot"], result["seed"], result["arm"]) == (slot, row["seed"], row["arm"])
        assert provenance["imports"]["head_pamssw_tree"] == "c572a1cc0766f8d3ff534a467ad64613aaa78ce0"
        row.update(result_present=True, execution_status=result["result_status"],
                   reported_outcome=result["status"],
                   code_commit=provenance["imports"]["git_head"],
                   core_tree=provenance["imports"]["head_pamssw_tree"],
                   prepared_input_sha=provenance["prepared_sha256"],
                   initial_energy_eV=result["initial"]["energy_eV"] if result.get("initial") else None,
                   best_energy_eV=result["best_energy_eV"],
                   search_seconds=result["search_wall_seconds"],
                   outer_statuses=dict(Counter(e["status"] for e in result["outer_events"])),
                   outer_attempts=len(result["outer_events"]),
                   cold_fresh_certificates=result["fresh"],
                   result_request_closure=result["request_closure"])
        row["raw_cost_matches_result"] = bool(search["charged"] == result["search_requests_total"]
            and search["calculator_calls"] == result["actual_calculator_calls"]
            and search["denied"] == result["denied_requests"]
            and fresh["charged"] == result["fresh_requests"])
        assert row["raw_cost_matches_result"], f"cost mismatch: {folder}"
        common = args.prepared / f"lj55-seed{row['seed']}" / "prepared.extxyz"
        assert row["prepared_input_sha"] == runner.sha256(common)
        # The primary cold E/F certificate was evaluated before coordinate output
        # rounding. Independently replay only the saved candidate's structure gate.
        candidate_file = folder / "target-candidate.extxyz"
        candidate_fresh = next((f for f in result["fresh"] if f.get("role") == "target_candidate"), None)
        if candidate_fresh and candidate_file.is_file():
            atoms = read(candidate_file)
            check = runner.common_qualification(atoms, candidate_fresh["energy_eV"],
                candidate_fresh["fmax_eV_A"], candidate_fresh.get("passed", False), target)
            row["replayed_saved_candidate_gate"] = check
            row["joint_target_verified"] = bool(result["status"] == "first_hit"
                and result["candidate_pause_confirmed"] and result["result_status"] == "paused"
                and candidate_fresh.get("passed") and check["qualified_candidate"])
        if row["joint_target_verified"]:
            row.update(budget_outcome="stopped_at_target", scientific_outcome="joint_target_found",
                       first_hit_search_requests=search["charged"])
        else:
            reason = ("request_censored" if search["denied"] and search["charged"] == runner.SEARCH_CAP
                      else "wall_censored" if search["denied"]
                      else "outer_step_censored" if row["outer_attempts"] == runner.OUTER_CAP
                      else "execution_or_qualification_failure")
            row.update(budget_outcome=reason, scientific_outcome="joint_target_not_found",
                       first_hit_search_requests=None)
        rows.append(row)
    payload = dict(scope="two-new-start paired LJ55 development target qualification, not published success-rate replication",
                   analysis_source=dict(path=str(Path(__file__).resolve()), git_head=runner.source_info()["git_head"]),
                   preparation_directory=str(args.prepared.resolve()), prepare_summary=prepare_summary,
                   preparation_costs=prep_rows, reference_fresh_cost=ref_fresh,
                   run_denominator=4, rows=rows,
                   total_search_requests=sum(r["search"]["charged"] for r in rows),
                   total_arm_fresh_requests=sum(r["fresh"]["charged"] for r in rows),
                   total_preparation_requests=sum(r["search"]["charged"]+r["fresh"]["charged"] for r in prep_rows)+ref_fresh["charged"])
    lines = ["# LJ55 joint-target paired readout", "",
             "Current fixed configuration; two starts are not a general success-probability estimate.",
             "Absence, backend errors, qualification failures and independent budget censoring remain in the denominator.", "",
             "| Seed | Arm | Joint target | Paid search | Calculator calls | First target cost | Best E | Outer attempts | Boundary |",
             "|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['joint_target_verified']} | {r['search']['charged']} | "
                     f"{r['search']['calculator_calls']} | {r.get('first_hit_search_requests')} | "
                     f"{r.get('best_energy_eV')} | {r.get('outer_attempts')} | {r['budget_outcome']} |")
    lines += ["", f"Total search {payload['total_search_requests']}, arm fresh {payload['total_arm_fresh_requests']}, "
              f"shared preparation/reference {payload['total_preparation_requests']} requests.", "",
              "Energy target is independently qualified Cambridge LJ55 -279.248470 epsilon; SSW2013 Table1 prints -297.24847.",
              "Joint target: force-qualified true quench, energy window, connected geometry, permutation/proper-rotation match, cold E/F certificate.",
              "No extrapolated or imputed hit costs for censored runs; no causal comparison with old configurations or native periodic representation."]
    (args.output / "analysis.json").write_text(json.dumps(payload, indent=2)+"\n")
    (args.output / "README.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
