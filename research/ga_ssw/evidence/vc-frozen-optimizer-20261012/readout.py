#!/usr/bin/env python3
"""Read-only paired summary for a frozen VC optimizer campaign directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = ("phase87", "phase139")
METHODS = ("safe_total", "scipy_lbfgsb")


def read_json(path: Path):
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"


def _trace_values(inner):
    trace = inner.get("accepted_trace") or []
    vals = [float(row["objective_eV"]) for row in trace
            if isinstance(row.get("objective_eV"), (int, float))]
    norms = [float(row["biased_block_norm_eV_A"]) for row in trace
             if isinstance(row.get("biased_block_norm_eV_A"), (int, float))]
    initial = inner.get("initial") or {}
    e0 = initial.get("objective_eV")
    e1 = vals[-1] if vals else None
    delta = None if e0 is None or e1 is None else float(e0) - e1
    return e0, e1, delta, norms[-1] if norms else None


def ledger_cost(entries, stage):
    rows = [row for row in entries if stage is None or row.get("stage") == stage]
    paid = sum(bool(row.get("paid")) for row in rows)
    denied = sum(not bool(row.get("paid")) for row in rows)
    failed = sum(bool(row.get("paid")) and row.get("status") == "error" for row in rows)
    actual_values = [row.get("actual_calculator_calls") for row in rows
                     if isinstance(row.get("actual_calculator_calls"), (int, float))]
    actual = sum(actual_values) if actual_values else None
    return {"paid_efs": paid, "actual_calculator_calls": actual,
            "paid_errors": failed, "unpaid_denials": denied}


def analyze(run_dir: Path):
    manifest, manifest_error = read_json(run_dir / "manifest.json")
    campaign_ledger, campaign_ledger_error = read_json(run_dir / "campaign-ledger.json")
    if not isinstance(campaign_ledger, list):
        campaign_ledger = []
    nested_arms = {}
    if isinstance(manifest, dict):
        for case_record in manifest.get("arms", []):
            for arm in case_record.get("arms", []):
                nested_arms[(case_record.get("case"), arm.get("method"))] = arm

    arm_rows = {}
    expected = []
    for case in CASES:
        for method in METHODS:
            key = (case, method)
            expected.append(key)
            arm_name = f"{case}-{method}"
            arm_dir = run_dir / arm_name
            result, result_error = read_json(arm_dir / "result.json")
            if result is None:
                result = nested_arms.get(key)
                if result is not None:
                    result_error = None
            if result is None:
                arm_rows[key] = {"status": "not_recorded", "artifact": result_error}
                continue

            arm_ledger, arm_ledger_error = read_json(arm_dir / "efs-ledger.json")
            if not isinstance(arm_ledger, list):
                arm_ledger = [row for row in campaign_ledger
                              if str(row.get("stage", "")).startswith(arm_name + ":")]
            inner = result.get("inner") or {}
            quench = result.get("quench") or {}
            initial_e, final_e, descent, final_norm = _trace_values(inner)
            costs = {
                "inner": ledger_cost(arm_ledger, arm_name + ":inner"),
                "quench": ledger_cost(arm_ledger, arm_name + ":unbiased_quench"),
                "certificate": ledger_cost(arm_ledger, arm_name + ":fresh_certificate"),
            }
            reported_cost = result.get("cost") or {}
            stage_paid_sum = sum(item["paid_efs"] for item in costs.values())
            stage_actual_values = [item["actual_calculator_calls"] for item in costs.values()
                                   if item["actual_calculator_calls"] is not None]
            stage_actual_sum = sum(stage_actual_values) if stage_actual_values else None
            cert = result.get("fresh_certificate") or {}
            arm_rows[key] = {
                "status": result.get("status", "unknown"),
                "inner_status": inner.get("status"),
                "inner_shared_target_reached": inner.get("shared_target_reached"),
                "inner_initial_objective_eV": initial_e,
                "inner_final_objective_eV": final_e,
                "inner_objective_descent_eV": descent,
                "inner_initial_block_norm_eV_A": (inner.get("initial") or {}).get("biased_block_norm_eV_A"),
                "inner_final_block_norm_eV_A": final_norm,
                "inner_logical_attempts": inner.get("logical_attempts"),
                "inner_elapsed_seconds": inner.get("elapsed_seconds"),
                "quench_status": quench.get("status"),
                "quench_elapsed_seconds": quench.get("elapsed_seconds"),
                "quench_energy_eV": quench.get("energy_eV"),
                "fresh_certificate": {
                    "energy_eV": cert.get("energy_eV"),
                    "max_force_eV_A": cert.get("max_force_eV_A"),
                    "max_abs_stress_eV_A3": cert.get("max_abs_stress_eV_A3"),
                    "force_qualified": cert.get("force_qualified"),
                    "stress_qualified": cert.get("stress_qualified"),
                    "jointly_qualified": cert.get("jointly_qualified"),
                } if cert else None,
                "cost_by_stage": costs,
                "reported_cost_fields": reported_cost,
                "cost_crosscheck": {
                    "ledger_arm_paid_efs": stage_paid_sum,
                    "reported_arm_paid_efs": reported_cost.get("arm_paid_efs"),
                    "arm_paid_efs_match": (None if reported_cost.get("arm_paid_efs") is None
                                            else reported_cost["arm_paid_efs"] == stage_paid_sum),
                    "ledger_arm_actual_calculator_calls": stage_actual_sum,
                    "reported_arm_actual_calculator_calls": reported_cost.get("arm_actual_calculator_calls"),
                    "inner_logical_attempts_from_result": inner.get("logical_attempts"),
                    "reported_inner_logical_requests": reported_cost.get("inner_logical_requests"),
                },
                "artifact_issues": {
                    "result": result_error,
                    "ledger": None if arm_ledger else arm_ledger_error,
                },
            }

    pairs = {}
    for case in CASES:
        left, right = arm_rows[(case, METHODS[0])], arm_rows[(case, METHODS[1])]
        pairs[case] = {
            "safe_total_status": left.get("status"),
            "scipy_lbfgsb_status": right.get("status"),
            "shared_target": {
                "safe_total": left.get("inner_shared_target_reached"),
                "scipy_lbfgsb": right.get("inner_shared_target_reached"),
            },
            "objective_descent_eV": {
                "safe_total": left.get("inner_objective_descent_eV"),
                "scipy_lbfgsb": right.get("inner_objective_descent_eV"),
            },
            "fresh_joint_qualification": {
                "safe_total": (left.get("fresh_certificate") or {}).get("jointly_qualified"),
                "scipy_lbfgsb": (right.get("fresh_certificate") or {}).get("jointly_qualified"),
            },
            "inner_cost": {
                "safe_total": left.get("cost_by_stage", {}).get("inner"),
                "scipy_lbfgsb": right.get("cost_by_stage", {}).get("inner"),
            },
            "quench_cost": {
                "safe_total": left.get("cost_by_stage", {}).get("quench"),
                "scipy_lbfgsb": right.get("cost_by_stage", {}).get("quench"),
            },
            "elapsed_seconds": {
                "safe_total": {"inner": left.get("inner_elapsed_seconds"),
                               "quench": left.get("quench_elapsed_seconds")},
                "scipy_lbfgsb": {"inner": right.get("inner_elapsed_seconds"),
                                 "quench": right.get("quench_elapsed_seconds")},
            },
        }

    status = manifest.get("status") if isinstance(manifest, dict) else None
    ledger_total = ledger_cost(campaign_ledger, None) if campaign_ledger else None
    return {
        "run_directory": str(run_dir.resolve()),
        "campaign_status": status or ("manifest_unavailable" if manifest_error else "incomplete"),
        "manifest_error": manifest_error,
        "campaign_error": manifest.get("campaign_error") if isinstance(manifest, dict) else None,
        "elapsed_seconds": manifest.get("elapsed_seconds") if isinstance(manifest, dict) else None,
        "campaign_accounting": {
            "reported_paid_efs": manifest.get("total_paid_efs") if isinstance(manifest, dict) else None,
            "reported_failed_paid_efs": manifest.get("failed_paid_efs") if isinstance(manifest, dict) else None,
            "reported_unpaid_denials": manifest.get("denied_unpaid_requests") if isinstance(manifest, dict) else None,
            "reported_actual_calculator_calls": manifest.get("actual_calculator_calls") if isinstance(manifest, dict) else None,
            "recounted_from_campaign_ledger": ledger_total,
            "campaign_ledger_rows": len(campaign_ledger),
            "campaign_ledger_error": campaign_ledger_error,
        },
        "pairs": pairs,
        "arms": {f"{case}-{method}": arm_rows[(case, method)] for case, method in expected},
        "not_run_arms": (manifest.get("not_run_arms") if isinstance(manifest, dict) and manifest.get("not_run_arms") is not None
                          else [f"{case}-{method}" for case, method in expected
                                if arm_rows[(case, method)].get("status") == "not_recorded"]),
        "interpretation_limits": [
            "Optimizer-bundle comparison is limited to the two frozen TiO2 starts.",
            "Inner biased-gradient target and unbiased force/stress qualification are separate outcomes.",
            "A censored or failed arm is not a converged or physically qualified result.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
