"""Read-only summary of PAM Gaussian policy telemetry.

The analyzer never imports a calculator or evaluates a potential.  It accepts
the completed matched-run evidence directory and keeps absent telemetry as
missing rather than reconstructing it from other fields.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _range(values):
    values = [float(value) for value in values if _finite(value)]
    return None if not values else {"min": min(values), "max": max(values), "count": len(values)}


def _bool_count(events, field):
    present = [event[field] for event in events if isinstance(event.get(field), bool)]
    return {"true": sum(present), "false": len(present) - sum(present), "missing": len(events) - len(present)}


def _policy_event_rows(result):
    rows = []
    for record_index, record in enumerate(result.get("records", [])):
        for event_index, event in enumerate(record.get("climb", [])):
            policy = event.get("gaussian_policy")
            if not isinstance(policy, dict):
                continue
            row = {
                "record_index": record_index,
                "event_index": event.get("index", event_index),
                "record_status": record.get("status"),
                "event_status": event.get("status"),
            }
            for key in (
                "width", "weight", "raw_width", "raw_weight", "k_true", "k_inner",
                "width_clamped", "weight_clamped", "parameters",
            ):
                row[key] = policy[key] if key in policy else None
            width = policy.get("width")
            weight = policy.get("weight")
            k_inner = policy.get("k_inner")
            parameters = policy.get("parameters")
            target_negative = parameters.get("target_negative_curvature") if isinstance(parameters, dict) else None
            row["target_negative_curvature"] = target_negative
            if _finite(width) and float(width) > 0 and _finite(weight) and _finite(k_inner):
                estimated = float(k_inner) - float(weight) / float(width) ** 2
                row["estimated_center_directional_curvature"] = estimated
                # Since weight = width**2 * max(k_inner + target, 0),
                # zero weight is an algebraic branch, not an error.  Width
                # clipping does not affect this identity because actual width
                # is used in the quotient; weight clipping does.
                if policy.get("weight_clamped") is False and _finite(target_negative):
                    target = min(float(k_inner), -float(target_negative))
                    row["target_identity"] = target
                    row["target_negative_curvature_residual"] = estimated - target
                else:
                    row["target_identity"] = None
                    row["target_negative_curvature_residual"] = None
            else:
                row["estimated_center_directional_curvature"] = None
                row["target_identity"] = None
                row["target_negative_curvature"] = target_negative
                row["target_negative_curvature_residual"] = None
            rows.append(row)
    return rows


def _run_summary(path):
    result_path = path / "result.json"
    summary_path = path / "summary.json"
    if not result_path.exists():
        return {"run": path.name, "state": "missing_result"}
    result = json.loads(result_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    all_events = [event for record in result.get("records", []) for event in record.get("climb", [])]
    policy_events = _policy_event_rows(result)
    event_statuses = Counter(event.get("status") for event in all_events)
    record_statuses = Counter(record.get("status") for record in result.get("records", []))
    no_landing = []
    failed_landing = []
    for index, record in enumerate(result.get("records", [])):
        landing = record.get("landing")
        if landing is None:
            no_landing.append({"record_index": index, "status": record.get("status")})
        elif isinstance(landing, dict) and landing.get("converged") is False:
            failed_landing.append({"record_index": index, "status": "landing_unconverged"})
    widths = [event.get("width") for event in policy_events]
    weights = [event.get("weight") for event in policy_events]
    k_true = [event.get("k_true") for event in policy_events]
    k_inner = [event.get("k_inner") for event in policy_events]
    estimated = [event.get("estimated_center_directional_curvature") for event in policy_events]
    residual = [event.get("target_negative_curvature_residual") for event in policy_events]
    return {
        "run": path.name,
        "state": "analyzed",
        "execution": None if summary is None else summary.get("execution"),
        "boundary": None if summary is None else summary.get("boundary"),
        "search_requests": None if summary is None else summary.get("search_requests"),
        "fresh_requests": None if summary is None else summary.get("fresh_requests"),
        "result_status": result.get("status"),
        "result_evaluation_requests": result.get("evaluation_requests"),
        "record_count": len(result.get("records", [])),
        "climb_event_count": len(all_events),
        "policy_event_count": len(policy_events),
        "event_statuses": dict(event_statuses),
        "record_statuses": dict(record_statuses),
        "stage_failures": [
            {"record_index": i, "status": status}
            for i, status in enumerate(record.get("status") for record in result.get("records", []))
            if status in {"biased_quench_failed", "evaluation_failed", "rotation_failed", "cluster_frame_failed"}
        ],
        "no_landing": no_landing,
        "failed_landing": failed_landing,
        "zero_weight_count": sum(_finite(value) and float(value) == 0.0 for value in weights),
        "width_clamps": _bool_count(policy_events, "width_clamped"),
        "weight_clamps": _bool_count(policy_events, "weight_clamped"),
        "ranges": {
            "width_A": _range(widths),
            "weight_eV": _range(weights),
            "k_true_eV_per_A2": _range(k_true),
            "k_inner_eV_per_A2": _range(k_inner),
            "estimated_center_directional_curvature_eV_per_A2": _range(estimated),
            "target_negative_curvature_residual_eV_per_A2_unclipped": _range(residual),
        },
        "policy_events": policy_events,
    }


def analyze(root):
    root = Path(root)
    runs = []
    for group in ("c60", "controls"):
        folder = root / group
        if not folder.is_dir():
            continue
        plan_path = folder / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        cases = plan.get("cases", [])
        arms = [arm.get("name") for arm in plan.get("arms", []) if isinstance(arm, dict)]
        for case in cases:
            for arm in arms:
                run = folder / f"{case}-{arm}"
                if run.is_dir():
                    runs.append(_run_summary(run))
                else:
                    runs.append({"run": run.name, "state": "missing"})
    return {
        "scope": "PAM Gaussian policy telemetry only; no efficacy, ranking, or geometry claim",
        "definitions": {
            "policy_event": "climb event containing gaussian_policy; events without that field remain counted only in climb_event_count",
            "estimated_center_directional_curvature": "k_inner - weight / width^2, an algebraic estimate from recorded finite-difference/analytic k_inner and actual policy width/weight; not an independently measured Hessian",
            "target_negative_curvature_residual": "estimated curvature - min(k_inner, -target_negative_curvature), read from each event parameters; reported only when weight_clamped is explicitly false",
            "zero_weight": "the max(..., 0) branch; a negative k_inner + target does not constitute an error",
            "requests": "copied from summary/result telemetry; no request ledger reconstruction",
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.input)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
