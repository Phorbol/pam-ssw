"""Audit completed Fe7C3 frozen-quench diagnostic artifacts offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root):
    root = Path(root)
    diagnosis = root / "frozen-quench-diagnosis"
    arms = []
    for path in sorted(diagnosis.glob("seed*/**/history*/result.json")):
        result = json.loads(path.read_text())
        rows = [json.loads(line) for line in
                (path.parent / "evaluations.jsonl").read_text().splitlines()]
        arms.append(dict(path=str(path.relative_to(root)), seed=result["seed"],
                         arm=result["arm"], history=result["history"],
                         safe_status=result["safe_lbfgs"]["status"],
                         final_status=result["final_fresh"].get("status"),
                         requests=result["requests"], charged=sum(r.get("charged", False) for r in rows),
                         optimizer_requests=sum(r.get("stage", "").startswith("safe-lbfgs") for r in rows),
                         final_fresh_requests=sum(r.get("stage") == "fresh-final" for r in rows),
                         final_norm=result["final_fresh"].get("norm"),
                         final_gradient_l2=result["final_fresh"].get("gradient_l2"),
                         final_within_gtol=(result["final_fresh"].get("norm") is not None and
                                            result["final_fresh"]["norm"] <= 0.001)))
    checks = []
    for path in sorted(diagnosis.glob("failed-check-*.json")):
        value = json.loads(path.read_text())
        checks.append(dict(path=str(path.relative_to(root)), status=value.get("status"),
                           requests=value.get("requests"), source_gradient_l2=value.get("source_gradient_l2"),
                           biased_gradient_l2=value.get("biased_gradient_l2"),
                           abs_diff=value.get("gradient_l2_abs_diff"),
                           matched=(value.get("status") == "checked" and
                                    value.get("gradient_l2_abs_diff", float("inf")) <= 1e-8)))
    total = sum(item["charged"] for item in arms) + sum(item["requests"] for item in checks)
    return dict(status="audited", arms=arms, shared_failed_checks=checks,
                arm_count=len(arms), shared_check_count=len(checks), charged_total=total,
                declared_cap=2644, cap_matches=(total <= 2644 and len(arms) == 8 and len(checks) == 4),
                all_source_checks_match=all(x["matched"] for x in checks),
                note="history500 is an existing optimizer parameter tested cross-material; this is not a default change or whole-SSW gain claim")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    output = args.output or args.root / "frozen-quench-diagnosis" / "audit-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
