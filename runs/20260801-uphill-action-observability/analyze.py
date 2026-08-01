#!/usr/bin/env python3
"""Pure serialization and descriptive analysis for the U-O1 gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from pamssw.result import ActionRecord


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]


def _sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def serialize_action(action: ActionRecord) -> dict[str, Any]:
    """Serialize a typed action record without atomic coordinates or directions."""

    if not isinstance(action, ActionRecord):
        raise TypeError("action must be an ActionRecord")
    return asdict(action)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": finite[0],
        "median": float(median(finite)),
        "mean": float(mean(finite)),
        "max": finite[-1],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def analyze_case(
    summary: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not actions:
        raise ValueError("case has no completed action records")
    force_evaluations = int(summary["force_evaluations"])
    purpose_counts = {
        str(key): int(value) for key, value in summary["purpose_counts"].items()
    }
    if sum(purpose_counts.values()) != force_evaluations:
        raise ValueError("purpose ledger does not close")
    if purpose_counts.get("unattributed", -1) != 0:
        raise ValueError("unattributed force evaluations present")

    ratios = [
        float(row["walk"]["target_delivery_ratio"])
        for row in actions
        if row["walk"]["target_delivery_ratio"] is not None
    ]
    delivered = [
        row
        for row in actions
        if row["walk"]["target_delivery_ratio"] is not None
        and float(row["walk"]["target_delivery_ratio"]) >= 1.0
    ]
    unattained = [
        row
        for row in actions
        if row["walk"]["target_delivery_ratio"] is not None
        and float(row["walk"]["target_delivery_ratio"]) < 1.0
    ]
    missing_ratio = len(actions) - len(delivered) - len(unattained)
    delivered_unproductive = sum(
        row["status"] != "accepted" for row in delivered
    )
    new_count = sum(row["accepted_new_basin"] is True for row in actions)
    duplicate_count = sum(row["is_duplicate"] is True for row in actions)
    improved_count = sum(row["global_improved"] is True for row in actions)
    landing_fe = sum(int(row["landing_force_evaluations"]) for row in actions)
    direction_fe = sum(
        int(step["direction_oracle_force_evaluations"])
        for row in actions
        for step in row["walk"]["steps"]
    )
    proposal_fe = sum(
        int(step["biased_relax_force_evaluations"])
        for row in actions
        for step in row["walk"]["steps"]
    )
    true_check_fe = sum(
        int(step["true_pes_check_force_evaluations"])
        for row in actions
        for step in row["walk"]["steps"]
    )
    quench_drops = [
        float(row["escape_energy_eV"]) - float(row["landing_energy_eV"])
        for row in actions
        if row["escape_energy_eV"] is not None and row["landing_energy_eV"] is not None
    ]
    step_rows = [step for row in actions for step in row["walk"]["steps"]]
    nonproposal = {
        "direction_oracle": direction_fe,
        "true_pes_check": true_check_fe,
        "landing_true_quench": landing_fe,
    }
    return {
        "system": str(summary["system"]),
        "seed": int(summary["seed"]),
        "action_count": len(actions),
        "step_count": len(step_rows),
        "target_eV": _distribution([float(row["walk"]["target_eV"]) for row in actions]),
        "observed_max_height_eV": _distribution(
            [
                float(row["walk"]["observed_max_height_eV"])
                for row in actions
                if row["walk"]["observed_max_height_eV"] is not None
            ]
        ),
        "observed_terminal_height_eV": _distribution(
            [
                float(row["walk"]["observed_terminal_height_eV"])
                for row in actions
                if row["walk"]["observed_terminal_height_eV"] is not None
            ]
        ),
        "target_delivery_ratio": _distribution(ratios),
        "delivered_action_count": len(delivered),
        "unattained_action_count": len(unattained),
        "missing_delivery_ratio_count": missing_ratio,
        "delivered_rate": _rate(len(delivered), len(ratios)),
        "delivered_unproductive_count": delivered_unproductive,
        "delivered_unproductive_rate": _rate(delivered_unproductive, len(delivered)),
        "new_basin_count": new_count,
        "new_basin_rate": _rate(new_count, len(actions)),
        "duplicate_count": duplicate_count,
        "duplicate_rate": _rate(duplicate_count, len(actions)),
        "global_improvement_count": improved_count,
        "global_improvement_rate": _rate(improved_count, len(actions)),
        "landing_quench_drop_eV": _distribution(quench_drops),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in actions).items())),
        "walk_termination_counts": dict(
            sorted(Counter(str(row["walk"]["termination_reason"]) for row in actions).items())
        ),
        "proposal_relax_outcome_counts": dict(
            sorted(Counter(str(step["proposal_relax_outcome"]) for step in step_rows).items())
        ),
        "action_force_evaluations": {
            **nonproposal,
            "biased_proposal_relax": proposal_fe,
            "recorded_total": direction_fe + true_check_fe + proposal_fe + landing_fe,
        },
        "landing_is_largest_nonproposal_cost": bool(
            landing_fe == max(nonproposal.values())
        ),
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
    }


def analyze_cases(
    case_inputs: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], str]],
) -> dict[str, Any]:
    keys: set[tuple[str, int, int, int]] = set()
    cases = []
    for summary, actions, action_sha256 in case_inputs:
        for action in actions:
            key = (
                str(summary["system"]),
                int(summary["seed"]),
                int(action["trial_index"]),
                int(action["proposal_index"]),
            )
            if key in keys:
                raise ValueError("duplicate action key")
            keys.add(key)
        cases.append(
            {
                **analyze_case(summary, actions),
                "action_history_sha256": str(action_sha256),
            }
        )
    target_not_delivered = sum(
        row["unattained_action_count"] > row["action_count"] / 2 for row in cases
    )
    delivered_but_unproductive = sum(
        row["delivered_action_count"] > row["action_count"] / 2
        and (
            row["delivered_unproductive_count"] > row["delivered_action_count"] / 2
            or row["landing_is_largest_nonproposal_cost"]
        )
        for row in cases
    )
    diagnosis = (
        "TARGET_NOT_DELIVERED"
        if target_not_delivered >= 2
        else (
            "DELIVERED_BUT_UNPRODUCTIVE"
            if delivered_but_unproductive >= 2
            else "MIXED_SCALAR_TARGET_EVIDENCE"
        )
    )
    return {
        "schema_version": 1,
        "diagnosis": diagnosis,
        "systems_target_not_delivered": target_not_delivered,
        "systems_delivered_but_unproductive": delivered_but_unproductive,
        "unattributed_force_evaluations": sum(
            row["purpose_counts"]["unattributed"] for row in cases
        ),
        "new_force_evaluations": sum(row["force_evaluations"] for row in cases),
        "cases": cases,
        "claim_ceiling": (
            "observed true-PES micro-step endpoints for one seed per system; "
            "not barrier heights, statistical significance, or posterior-training evidence"
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-summary", action="append", type=Path, required=True)
    parser.add_argument("--action-history", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.case_summary) != len(args.action_history):
        raise SystemExit("summary and action-history counts differ")
    inputs = []
    for summary_path, action_path in zip(args.case_summary, args.action_history):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inputs.append((summary, _read_jsonl(action_path), _sha256(action_path)))
    evidence = analyze_cases(inputs)
    _write_json(args.output, evidence)
    print(json.dumps({"diagnosis": evidence["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

