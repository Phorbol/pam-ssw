"""Zero-PES audit for the six Fe7C3 mature-baseline result directories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _eval_counts(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    charged = sum(bool(row.get("charged")) for row in rows)
    denied = sum(not bool(row.get("charged")) for row in rows)
    return {"rows": len(rows), "charged": charged, "denied": denied,
            "search": sum(row.get("stage") == "search" for row in rows),
            "fresh": sum(row.get("stage") == "fresh" for row in rows),
            "search_charged": sum(row.get("stage") == "search" and row.get("charged") for row in rows),
            "fresh_charged": sum(row.get("stage") == "fresh" and row.get("charged") for row in rows),
            "search_denied": sum(row.get("stage") == "search" and not row.get("charged") for row in rows),
            "fresh_denied": sum(row.get("stage") == "fresh" and not row.get("charged") for row in rows)}


def audit_arm(directory, requested_attempts=2, fresh_reserve=3):
    directory = Path(directory)
    result_path, eval_path = directory / "result.json", directory / "evaluations.jsonl"
    if not result_path.is_file() or not eval_path.is_file():
        return {"status": "pending", "missing": [str(p.name) for p in (result_path, eval_path) if not p.is_file()]}
    result = _read(result_path)
    counts = _eval_counts(eval_path)
    records = result.get("records") or []
    attempts = [r for r in records if isinstance(r, dict) and "index" in r]
    paid_attempts = [r for r in attempts if int(r.get("requests", 0) or 0) > 0]
    fresh = result.get("fresh") or {}
    checks = fresh.get("checks") if isinstance(fresh, dict) else []
    checks = checks or []
    initial_cert = None
    if records and isinstance(records[0], dict):
        cert = records[0].get("certificate")
        if isinstance(cert, dict):
            initial_cert = cert.get("certified")
        elif isinstance(records[0].get("quench"), dict):
            initial_cert = records[0]["quench"].get("certificate", {}).get("certified")
    native = []
    common = []
    for call in result.get("adapter_calls", []):
        native.append({"source": call.get("source"), "status": call.get("status"),
                       "native_success": call.get("native_success"),
                       "native_message": call.get("certificate", {}).get("native_message")})
        common.append(call.get("certificate", {}).get("convergence_norm"))
    statuses = [r.get("status") for r in records if isinstance(r, dict)]
    errors = [r.get("error", "") for r in records if isinstance(r, dict)]
    nested = [c.get("status") for r in records if isinstance(r, dict)
              for c in (r.get("climb") or []) if isinstance(c, dict)]
    all_statuses = statuses + nested
    maxiter_failures = sum(s == "maxiter" for s in all_statuses)
    native_stop = sum(s == "native_stop" for s in all_statuses)
    budget_failures = sum("budget" in str(s).lower() or "budget" in str(e).lower()
                          for s, e in zip(statuses, errors))
    consistency = {
        "reported_requests_eq_charged": result.get("requests") == counts["charged"],
        "search_requests_eq_search_charged": result.get("search_requests") is None or
                                             result.get("search_requests") == counts["search_charged"],
        "fresh_requests_eq_fresh_charged": (result.get("fresh", {}).get("checked") is None or
                                             result.get("fresh", {}).get("checked") <= counts["fresh_charged"]),
    }
    consistency["errors"] = [name for name, ok in consistency.items() if name != "errors" and not ok]
    landings = result.get("landings") or []
    new_landings = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_index = int(check.get("index", -1))
        if not (0 <= check_index < len(landings)):
            continue
        landing = landings[check_index]
        if int(landing.get("index", -1)) < 0:
            continue
        initial = landings[0]
        new_landings.append({
            "landing_index": int(landing["index"]),
            "fresh_check_index": check_index,
            "energy": landing.get("energy"),
            "energy_delta_from_initial": (None if landing.get("energy") is None or
                                           initial.get("energy") is None else
                                           landing["energy"] - initial["energy"]),
            "accepted": bool(landing.get("accepted", False)),
            "fresh_certified": check.get("certified") is True,
        })
    return {"status": "audited", "counts": counts,
            "reported_requests": result.get("requests"),
            "requested_attempts": requested_attempts,
            "entered_attempts": len(attempts),
            "not_entered_attempts": max(0, requested_attempts - len(attempts)),
            "initial_certified": initial_cert,
            "fresh_requested": fresh.get("requested") if isinstance(fresh, dict) else None,
            "fresh_checked": len(checks),
            "fresh_reserve": fresh_reserve,
            "new_landing_count": sum((result.get("landings", [])[int(c.get("index", -1))].get("index", -1) >= 0)
                                      for c in checks if isinstance(c, dict) and 0 <= int(c.get("index", -1)) < len(result.get("landings", []))),
            "new_landing_certified": sum((result.get("landings", [])[int(c.get("index", -1))].get("index", -1) >= 0
                                           and c.get("certified") is True)
                                          for c in checks if isinstance(c, dict) and 0 <= int(c.get("index", -1)) < len(result.get("landings", []))),
            "new_landings": new_landings,
            "record_attempts": len(attempts),
            "paid_attempts": len(paid_attempts),
            "native_statuses": native, "common_convergence_norms": common,
            "maxiter_records": maxiter_failures, "native_stop_records": native_stop,
            "failed_records": sum(s in ("evaluation_failed", "biased_quench_failed",
                                         "true_quench_failed") for s in all_statuses),
            "censored_requests": counts["denied"],
            "budget_failure_records": budget_failures,
            "consistency": consistency,
            "reported_status": result.get("status")}


def audit_campaign(root):
    root = Path(root)
    plan = _read(root / "plan.json")
    solvers = plan.get("research_solvers", ["safe_total", "scipy", "ase"])
    seeds = plan.get("research_seeds", plan.get("seeds", [7, 101]))
    rows = {}
    for solver in solvers:
        for seed in seeds:
            rows[f"{solver}-seed{seed}"] = audit_arm(
                root / "comparison" / f"{solver}-seed{seed}",
                requested_attempts=2, fresh_reserve=plan.get("fresh_reserve", 3))
    audited = [x for x in rows.values() if x["status"] == "audited"]
    return {"status": "audited" if len(audited) == len(rows) else "pending",
            "expected_arms": len(rows), "audited_arms": len(audited), "arms": rows,
            "zero_pes": True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = audit_campaign(args.root)
    output = args.output or args.root / "audit-summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
