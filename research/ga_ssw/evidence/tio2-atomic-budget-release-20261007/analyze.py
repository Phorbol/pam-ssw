"""Audit the frozen saved-stage comparison without additional PES evaluations."""
import argparse
import json
from pathlib import Path


def ledger(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    unknown = [r for r in rows if r.get("event") not in {"evaluation", "failure", "denial"}]
    if unknown:
        raise ValueError(f"non-ledger records in {path}")
    paid = [r for r in rows if r["event"] != "denial"]
    if [r["request"] for r in paid] != list(range(1, len(paid) + 1)):
        raise ValueError(f"nonsequential paid requests in {path}")
    return {"requests": len(paid),
            "calculator_calls": sum(r["calculator_calls"] for r in paid),
            "failures": sum(r["event"] == "failure" for r in paid),
            "denials": len(rows) - len(paid)}


def analyze(paths, plan):
    summary = []
    for slot, path in enumerate(paths):
        result = json.loads((path / "result.json").read_text())
        if result["slot"] != slot:
            raise ValueError(f"wrong slot order: {path}")
        for kind, name in (("search", "requests.jsonl"), ("fresh", "fresh-requests.jsonl")):
            observed = ledger(path / name)
            recorded = result[f"{kind}_cost"]
            if any(recorded[k] != v for k, v in observed.items()):
                raise ValueError(f"raw {kind} cost mismatch: {path}")
            if not result[f"{kind}_ledger_closure"]["closed"]:
                raise ValueError(f"runner {kind} closure failed: {path}")
        atomic = result["atomic_climb"]
        true = result.get("true_quench") or {}
        if atomic.get("ledger_request_delta", 0) + true.get("ledger_request_delta", 0) != result["search_cost"]["requests"]:
            raise ValueError(f"stage cost mismatch: {path}")
        if atomic.get("reported_requests", 0) != atomic.get("ledger_request_delta", 0):
            raise ValueError(f"atomic report mismatch: {path}")
        if "reported_requests" in true and true["reported_requests"] != true.get("ledger_request_delta", 0):
            raise ValueError(f"true-quench report mismatch: {path}")
        summary.append({"path": str(path), "case": result["case"], "policy": result["rotation_exit_policy"],
                        "status": result["status"], "atomic_status": atomic.get("status"),
                        "gaussians": sum("weight" in e for e in atomic.get("events", [])),
                        "released_rotations": sum(x is True for x in atomic.get("rotation_budget_released", [])),
                        "rotation_requests": atomic.get("rotation_force_requests", 0),
                        "biased_quench_requests": atomic.get("biased_quench_requests", 0),
                        "true_quench_requests": true.get("ledger_request_delta", 0),
                        "true_quench_status": true.get("status"),
                        "candidate_minimum": true.get("candidate_minimum", False),
                        "cold_checks": result["cold_checks"],
                        "cold_confirmed_target": result.get("cold_confirmed_target_candidate", False),
                        "energy_eV": true.get("energy_eV"), "search_cost": result["search_cost"],
                        "fresh_cost": result["fresh_cost"]})
    totals = {kind: {k: sum(r[f"{kind}_cost"][k] for r in summary)
                     for k in ("requests", "calculator_calls", "failures", "denials")}
              for kind in ("search", "fresh")}
    if totals["search"]["requests"] > plan["budget"]["total_search_request_cap"] or totals["fresh"]["requests"] > plan["budget"]["total_fresh_request_cap"]:
        raise ValueError("aggregate request cap exceeded")
    pairs = []
    for strict, released in zip(summary[::2], summary[1::2]):
        if strict["case"] != released["case"] or (strict["policy"], released["policy"]) != ("force", "force_or_budget"):
            raise ValueError("paired identity mismatch")
        pairs.append({"case": strict["case"], "strict_atomic_status": strict["atomic_status"],
                      "released_atomic_status": released["atomic_status"],
                      "released_certified_minimum": released["candidate_minimum"],
                      "released_cold_confirmed_target": released["cold_confirmed_target"]})
    return {"raw_cost_audit": "passed", "evidence_class": "saved-stage development probe",
            "arms": summary, "pairs": pairs, "totals": totals,
            "scope": "No global success-rate, default-policy or optimizer-advantage conclusion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((Path(__file__).with_name("plan.json")).read_text())
    result = analyze(args.runs, plan)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# TiO2 saved atomic-stage comparison", "", "Raw request and calculator-call ledgers close for all four arms.", "",
             "| Case | Policy | Atomic stop | Gaussian terms | Certified minimum | Search requests |",
             "|---|---|---|---:|---|---:|"]
    for row in result["arms"]:
        lines.append(f"| {row['case']} | {row['policy']} | {row['atomic_status']} | {row['gaussians']} | {row['candidate_minimum']} | {row['search_cost']['requests']} |")
    lines += ["", result["scope"], "", "Fresh input measurements are diagnostic; only certified candidates can be target discoveries."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"audit": result["raw_cost_audit"], "totals": result["totals"], "pairs": result["pairs"]}))


if __name__ == "__main__":
    main()
