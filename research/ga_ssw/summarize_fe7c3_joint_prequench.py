"""Create an offline, source-linked summary of the Fe7C3 joint-LS comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def _load(path):
    return json.loads(Path(path).read_text())


def _walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _primary(record):
    if record["requests"] == 0:
        return "uncharged_entered"
    nodes = list(_walk(record))
    if any(node.get("status") == "maxiter" for node in nodes):
        return "maxiter"
    if any("budgetexhausted" in str(node.get("error", "")).lower()
           or "request limit" in str(node.get("error", "")).lower()
           for node in nodes):
        return "budget_exhausted"
    return record["status"]


def _arm(path):
    d = _load(path)
    rows = [json.loads(line) for line in (Path(path).parent / "evaluations.jsonl").read_text().splitlines()]
    prep = []
    for record in d.get("records", [])[1:]:
        ls = record.get("ls")
        if ls:
            prep.append({k: ls.get(k) for k in (
                "prequench_requests", "soft_joint_gradient_norm", "soft_fmax",
                "soft_stress_max", "energy_before", "energy_after",
                "enthalpy_before", "enthalpy_after", "enthalpy_response",
                "volume_before", "volume_after", "ls_energy_before", "ls_energy_after",
                "bond_count")})
    fresh = d.get("fresh", {})
    fresh_checks = []
    for check in fresh.get("checks", []):
        fresh_checks.append({k: check.get(k) for k in (
            "index", "status", "certified", "energy_error", "objective_error",
            "force_error", "stress_error", "fmax", "stress_max")})
    records = d.get("records", [])
    assert len(records) - 1 == d["steps_requested"] == 2
    assert all(r["requests"] > 0 for r in records[1:])
    assert len(prep) == 2
    if d.get("ls_prequench") == "joint":
        assert all(p["soft_joint_gradient_norm"] is not None for p in prep)
    assert d["requests"] == sum(bool(r.get("charged")) for r in rows)
    assert d["requests"] == d["search_requests"] + sum(bool(r.get("charged")) for r in rows if r.get("stage") == "fresh")
    assert sum(r.get("requests", 0) for r in records) == d["search_requests"]
    assert fresh.get("checked") == len(fresh_checks) == 1
    assert all(c["status"] == "checked" and c["certified"] for c in fresh_checks)
    pressure = d.get("joint_config", {}).get("pressure", 0.0)
    natoms = len(d["current"]["atoms"]["numbers"])
    for item in prep:
        if item["soft_joint_gradient_norm"] is not None:
            assert item["soft_joint_gradient_norm"] <= 0.001
            assert item["soft_fmax"] <= 0.001
            assert item["soft_stress_max"] <= 0.0001
            assert item["enthalpy_after"] == item["energy_after"] + pressure * item["volume_after"]
            assert item["enthalpy_before"] == item["energy_before"] + pressure * item["volume_before"]
            assert item["enthalpy_response"] == ((item["enthalpy_after"] - item["enthalpy_before"]) / natoms)
    outcomes = []
    for record in records[1:]:
        for climb in record.get("climb", []):
            relaxation = climb.get("relaxation", {})
            outcomes.append(dict(status=climb.get("status"), steps=relaxation.get("steps"),
                                 error=relaxation.get("error")))
    return dict(source=str(path), arm=d.get("arm"), seed=d.get("seed"),
                status=d.get("status"), steps_requested=d.get("steps_requested"),
                requests=d.get("requests"), search_requests=d.get("search_requests"),
                ledger_rows=len(rows), ledger_charged=sum(bool(r.get("charged")) for r in rows),
                ledger_uncounted=len(rows) - sum(bool(r.get("charged")) for r in rows),
                record_costs=[r.get("requests") for r in d.get("records", [])],
                record_statuses=[r.get("status") for r in records], stage_outcomes=outcomes,
                primary_outcomes=dict(Counter(_primary(r) for r in records[1:])),
                entered_attempts=len(records)-1,
                paid_attempts=sum(r["requests"] > 0 for r in records[1:]),
                landing_count=len(d.get("landings", [])), valid_proposals=d.get("valid_proposals"),
                frozen_pair_counts=[len(r["frozen_softening"]["pairs"])
                                    for r in d.get("records", [])[1:] if r.get("frozen_softening")],
                prequench=prep, fresh_status=fresh.get("status"),
                fresh_requested=fresh.get("requested"), fresh_checked=fresh.get("checked"),
                fresh_checks=fresh_checks)


def summarize(root):
    root = Path(root)
    new = root / "comparison"
    old = root.parent / "fe7c3-80-ls-filter-comparison" / "comparison"
    arms = []
    for arm in ("ls_all", "ls_filter"):
        for seed in (7, 101):
            new_path = new / f"{arm}-seed{seed}" / "result.json"
            old_path = old / f"{arm}-seed{seed}" / "result.json"
            joint_arm, fixed_arm = _arm(new_path), _arm(old_path)
            arms.append(dict(arm=arm, seed=seed, joint=joint_arm, fixed=fixed_arm,
                             total_cost_delta=joint_arm["requests"] - fixed_arm["requests"],
                             joint_prequench_requests=[x.get("prequench_requests") for x in joint_arm["prequench"]],
                             fixed_prequench_requests=[x.get("prequench_requests") for x in fixed_arm["prequench"]],
                             prequench_request_delta=[j.get("prequench_requests", 0) - f.get("prequench_requests", 0)
                                                      for j, f in zip(joint_arm["prequench"], fixed_arm["prequench"])]))
    joint_total = sum(a["joint"]["ledger_charged"] for a in arms)
    fixed_total = sum(a["fixed"]["ledger_charged"] for a in arms)
    joint_landing = sum(a["joint"]["landing_count"] for a in arms)
    fixed_landing = sum(a["fixed"]["landing_count"] for a in arms)
    denominator = sum(a["joint"]["steps_requested"] for a in arms)
    assert denominator == 8 and len(arms) == 4
    assert all(a["joint"]["valid_proposals"] == 0 and a["fixed"]["valid_proposals"] == 0 for a in arms)
    direct_prep = sum(sum(a["prequench_request_delta"]) for a in arms)
    assert direct_prep == 40
    return dict(status="audited", denominator_requested=denominator, arms=arms,
                joint_total_charged=joint_total, fixed_total_charged=fixed_total,
                total_cost_delta=joint_total-fixed_total, direct_preparation_delta=direct_prep,
                downstream_cost_delta=(joint_total-fixed_total)-direct_prep,
                joint_landing_total=joint_landing, fixed_landing_total=fixed_landing,
                joint_primary_outcomes=dict(sum((Counter(a["joint"]["primary_outcomes"]) for a in arms), Counter())),
                fixed_primary_outcomes=dict(sum((Counter(a["fixed"]["primary_outcomes"]) for a in arms), Counter())),
                prep_baseline_note="fixed-cell prep cost and joint prep cost are recorded per arm; only their direct request difference is attributable to preparation. Remaining cost delta is downstream behavior.",
                source_fresh_note="fresh checks are the original per-arm post-search calculator replacement checks; no fresh calls are inferred from search records.",
                scientific_boundary="This is an Fe7C3 fixed-parameter comparison, not evidence of a general SSW gain or default promotion.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.root)
    output = args.output or args.root / "comparison" / "joint-prequench-summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "status", "denominator_requested", "joint_total_charged", "fixed_total_charged",
        "total_cost_delta", "direct_preparation_delta", "downstream_cost_delta",
        "joint_landing_total", "fixed_landing_total")}, indent=2))


if __name__ == "__main__":
    main()
