#!/usr/bin/env python3
"""Read existing SSW ledgers and reproduce request-count accounting; no PES calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EV = ROOT / "research/ga_ssw/evidence"
OUT = Path(__file__).resolve().parent / "cost-readout.json"


def split_record(record):
    climb = record.get("climb") or []
    rotation = sum(x.get("rotation_force_requests", 0) or 0 for x in climb)
    biased = sum(x.get("quench_requests", 0) or 0 for x in climb)
    climb_other = sum(
        x.get("requests", 0) - (x.get("rotation_force_requests", 0) or 0)
        - (x.get("quench_requests", 0) or 0) for x in climb
    )
    landing = record.get("landing") or {}
    prep = record.get("ls_preparation") or {}
    return {
        "rotation": rotation,
        "biased_quench": biased,
        "climb_unassigned": climb_other,
        "true_landing_quench": landing.get("evaluation_requests", 0) or 0,
        "ls_prequench": prep.get("evaluation_requests", 0) or 0,
        "record_requests": record.get("evaluation_requests", 0) or 0,
        "attempt_status": record.get("status"),
        "landing_converged": landing.get("converged"),
    }


def read_run(path):
    d = json.loads(path.read_text())
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    records = [split_record(r) for r in d.get("records", [])]
    fields = ("rotation", "biased_quench", "climb_unassigned",
              "true_landing_quench", "ls_prequench", "record_requests")
    totals = {key: sum(r[key] for r in records) for key in fields}
    totals["run_evaluation_requests"] = d.get("evaluation_requests")
    totals["unassigned_outside_records"] = d.get("evaluation_requests", 0) - totals["record_requests"]
    totals["wall_seconds_whole_run"] = summary.get("elapsed_seconds")
    totals["attempts"] = len(records)
    totals["attempt_status_counts"] = {}
    for r in records:
        key = r["attempt_status"]
        totals["attempt_status_counts"][key] = totals["attempt_status_counts"].get(key, 0) + 1
    totals["landing_unconverged_attempts"] = sum(r["landing_converged"] is False for r in records)
    totals["request_closure_error"] = totals["record_requests"] - sum(totals[k] for k in fields[:-1])
    return {"source": str(path.relative_to(ROOT)), "summary_source": str(summary_path.relative_to(ROOT)),
            "totals": totals, "records": records}


def main():
    selected = json.loads((EV / "climb-depth-ablation-20260925/stage-cost-audit.json").read_text())
    stage_panels = {}
    for system in ("C4H6", "C60"):
        rows = [r for r in selected["rows"] if ("c4h6" in r["source"]) == (system == "C4H6")]
        # Selected-stage ledger fields: request count, quench optimizer requests.
        req = sum(r["requests"] for r in rows)
        q = sum(r["quench_requests"] or 0 for r in rows)
        stage_panels[system] = {
            "scope": "selected correlated climb stages only",
            "source": "research/ga_ssw/evidence/climb-depth-ablation-20260925/stage-cost-audit.json",
            "stages": len(rows), "all_stage_requests": req,
            "quench_requests": q,
        }
        # Compute rotation and residual from original path records named by the audit rows.
        inputs = json.loads((EV / "climb-depth-ablation-20260925/inputs.json").read_text())
        srcs = {x["path"]: x["selected"] for x in inputs["sources"]}
        rotation = residual = 0
        for source in {r["source"] for r in rows}:
            source_path = Path(source)
            if not source_path.exists():
                # Source path is from the sibling behavior-parity worktree; resolve by checkout-relative tail.
                parts = Path(source).parts
                source_path = ROOT.parent / "ga-ssw-behavior-parity" / Path(*parts[parts.index("research"):])
            raw = json.loads(source_path.read_text())
            selected_indices = next(v for k, v in srcs.items() if k == source)
            for record in raw["records"]:
                if record["index"] not in selected_indices:
                    continue
                for event in record["climb"]:
                    rot = event.get("rotation_force_requests", 0) or 0
                    bias = event.get("quench_requests", 0) or 0
                    rotation += rot
                    residual += event["requests"] - rot - bias
        stage_panels[system]["rotation_force_requests"] = rotation
        stage_panels[system]["event_unassigned_requests"] = residual
        stage_panels[system]["closure"] = rotation + q + residual

    c60_root = EV / "c60-local-defect-20260925/direction-probe/runs"
    c60 = [read_run(p) for p in sorted(c60_root.glob("*/result.json")) if p.parent.name.endswith(("1101", "1102"))]
    periodic_root = EV / "periodic-direction-20260926/runs"
    periodic = [read_run(p) for p in sorted(periodic_root.glob("*/result.json"))]
    result = {
        "purpose": "Read-only evaluation-request cost audit; no walltime attribution by phase",
        "stage_panels": stage_panels,
        "c60_direction_probe_runs": c60,
        "periodic_direction_runs": periodic,
        "lj_diagnostic": {
            "source": "research/ga_ssw/evidence/lj-gaussian-policy-probe-20260927/report.md",
            "scope": "82 saved Gaussian stages, 12 outer attempts, after initial quench",
            "total_requests": 5678, "biased_local_optimizer_requests": 4361,
            "rotation_requests": 566, "true_quench_and_other_requests": 751,
            "closure": 5678,
            "walltime_by_phase": None,
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
