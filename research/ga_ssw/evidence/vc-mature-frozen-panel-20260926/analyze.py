"""Read completed task artifacts only; no PES calls or artifact mutation."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
rows = []
for run in a.runs:
    exits = run / "worker-exits.tsv"
    for line in exits.read_text().splitlines():
        name, rc = line.split()
        task = run / name
        row = dict(run=str(run), task=name, process_exit=int(rc))
        result = task / "result.json"
        if not result.exists():
            row.update(status="missing_result", task_prepared=(task/"task.json").exists())
            ledger = task/"stage"/"requests.jsonl"
            data = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
            row["completed_ledger_requests"] = len([x for x in data if "request" in x])
            row["attempt_records"] = len([x for x in data if x.get("event")=="attempt"])
        else:
            d = json.loads(result.read_text())
            fresh = d.get("fresh_endpoint", {})
            row.update({k:d.get(k) for k in ("status", "search_requests", "first_common_qualified_request", "stage_and_fresh_wall_seconds")})
            row["source_validation"] = d.get("source_validation", {})
            row["fresh"] = {k:fresh.get(k) for k in ("requests", "physical_certificate", "fmax_eV_A", "pressure_residual_stress_max_eV_A3", "biased_common_gradient_norm_eV_A")}
            row["native_termination"] = d.get("native_termination")
            row["total_recorded_requests"] = sum([d.get("search_requests",0), d.get("source_validation",{}).get("requests",0), fresh.get("requests",0)])
            field = "biased_common_gradient_norm_eV_A" if d.get("phase")=="biased" else "common_physical_certificate"
            limit = .05 if d.get("phase")=="biased" else 1.
            first = next((x["request"] for x in d.get("accepted_iterates",[]) if x.get(field) is not None and x[field] <= limit), None)
            assert first == d.get("first_common_qualified_request"), name
        rows.append(row)
a.output.write_text(json.dumps(dict(rows=rows),indent=2)+"\n")
for r in rows:
    print(r["task"], r["process_exit"], r["status"], r.get("search_requests"), r.get("first_common_qualified_request"))
