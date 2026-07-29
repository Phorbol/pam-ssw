#!/usr/bin/env python3
"""Analyze the exact raw-PdO paired direction-continuation transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
STATE_ID = "raw_bootstrap"
SEEDS = (42, 43, 44)
ARMS = ("fixed_intent_ritz", "transported_direction")
MEANINGFUL_ENERGY_DROP_EV = 1.0e-3


def _purpose_total(mapping: Mapping[str, Any]) -> int:
    return sum(int(value) for value in mapping.values())


def validate_action_row(row: Mapping[str, Any]) -> None:
    purposes = row.get("purpose_counts")
    if not isinstance(purposes, Mapping):
        raise ValueError("action purpose ledger is absent")
    if int(purposes.get("bootstrap_true_quench", -1)) != 0:
        raise ValueError("bootstrap evaluations must not be hidden inside an action")
    if int(purposes.get("unattributed", -1)) != 0:
        raise ValueError("action contains unattributed evaluations")
    if _purpose_total(purposes) != int(row.get("force_evaluations", -1)):
        raise ValueError("action purpose ledger does not close")
    if (
        row.get("status") != "completed"
        or row.get("certificate") is not True
        or row.get("landing_geometry_valid") is not True
        or row.get("direction_trace_valid") is not True
        or row.get("fragmentation_applicable") is not False
        or row.get("fragmented") is not False
    ):
        raise ValueError("action lacks the strict PdO terminal certificate")


def _meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row["is_new_basin"]
        and float(row["landing_delta_eV"])
        <= -MEANINGFUL_ENERGY_DROP_EV
    )


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = raw.get("bootstrap")
    shared = raw.get("shared_initial_directions")
    rows = raw.get("cases")
    if (
        not isinstance(bootstrap, Mapping)
        or not isinstance(shared, Sequence)
        or not isinstance(rows, Sequence)
    ):
        raise ValueError("raw PdO evidence is incomplete")

    bootstrap_purposes = bootstrap.get("purpose_counts")
    if not isinstance(bootstrap_purposes, Mapping):
        raise ValueError("bootstrap purpose ledger is absent")
    if (
        bootstrap.get("certificate") is not True
        or bootstrap.get("geometry_valid") is not True
        or int(bootstrap_purposes.get("unattributed", -1)) != 0
        or _purpose_total(bootstrap_purposes)
        != int(bootstrap.get("force_evaluations", -1))
        or any(
            int(value) != 0
            for purpose, value in bootstrap_purposes.items()
            if purpose
            not in {"bootstrap_true_quench", "post_relax_validation"}
        )
    ):
        raise ValueError("bootstrap certificate or ledger does not close")

    expected = {
        (STATE_ID, seed, arm)
        for seed in SEEDS
        for arm in ARMS
    }
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("PdO transfer requires the exact six paired actions")
    for row in rows:
        validate_action_row(row)

    shared_by_seed = {int(item["seed"]): item for item in shared}
    if set(shared_by_seed) != set(SEEDS):
        raise ValueError("PdO transfer requires one shared Ritz mode per seed")
    for seed, item in shared_by_seed.items():
        purposes = item.get("purpose_counts")
        if (
            not isinstance(purposes, Mapping)
            or int(item.get("force_evaluations", -1)) != 24
            or int(purposes.get("direction_oracle", -1)) != 24
            or _purpose_total(purposes) != 24
        ):
            raise ValueError("shared initial Ritz did not consume exactly 12 HVPs")
        expected_hash = str(item["direction_sha256"])
        if any(
            row["shared_initial_direction_sha256"] != expected_hash
            for row in rows
            if int(row["seed"]) == seed
        ):
            raise ValueError("paired actions did not share the exact step-zero Ritz mode")

    arm_results: dict[str, dict[str, Any]] = {}
    meaningful_conditions: dict[str, set[int]] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        meaningful = {
            int(row["seed"]) for row in arm_rows if _meaningful(row)
        }
        meaningful_conditions[arm] = meaningful
        arm_results[arm] = {
            "completed_cases": len(arm_rows),
            "certificate_count": sum(bool(row["certificate"]) for row in arm_rows),
            "geometry_invalid_count": sum(
                not bool(row["landing_geometry_valid"]) for row in arm_rows
            ),
            "fallback_count": sum(bool(row["fallback_used"]) for row in arm_rows),
            "continuation_projection_degenerate_count": sum(
                int(row["continuation_projection_degenerate"])
                for row in arm_rows
            ),
            "new_basin_count": sum(bool(row["is_new_basin"]) for row in arm_rows),
            "meaningful_outcome_count": len(meaningful),
            "meaningful_seeds": sorted(meaningful),
            "median_landing_delta_eV": float(
                statistics.median(
                    float(row["landing_delta_eV"]) for row in arm_rows
                )
            ),
            "direction_force_evaluations": sum(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "generation_wall_time_s": sum(
                float(row["generation_wall_time_s"]) for row in arm_rows
            ),
            "quench_wall_time_s": sum(
                float(row["quench_wall_time_s"]) for row in arm_rows
            ),
        }

    control = arm_results["fixed_intent_ritz"]
    transported = arm_results["transported_direction"]
    control_events = meaningful_conditions["fixed_intent_ritz"]
    transported_events = meaningful_conditions["transported_direction"]
    no_validity_regression = bool(
        transported["certificate_count"] == control["certificate_count"]
        and transported["geometry_invalid_count"] <= control["geometry_invalid_count"]
        and transported["continuation_projection_degenerate_count"] == 0
    )
    reproduces_control_events = control_events <= transported_events
    lower_direction_cost = (
        transported["direction_force_evaluations"]
        < control["direction_force_evaluations"]
    )
    lower_total_cost = (
        transported["total_force_evaluations"]
        < control["total_force_evaluations"]
    )
    if (
        no_validity_regression
        and bool(control_events or transported_events)
        and reproduces_control_events
        and lower_direction_cost
        and lower_total_cost
    ):
        decision = "transported_direction_supported"
    elif (
        no_validity_regression
        and not control_events
        and not transported_events
        and lower_direction_cost
        and lower_total_cost
        and transported["median_landing_delta_eV"]
        <= control["median_landing_delta_eV"]
    ):
        decision = "transported_direction_cost_supported_no_terminal_event"
    else:
        decision = "transported_direction_not_supported"

    shared_cost = sum(int(item["force_evaluations"]) for item in shared)
    action_cost = sum(int(row["force_evaluations"]) for row in rows)
    bootstrap_cost = int(bootstrap["force_evaluations"])
    return {
        "schema_version": 1,
        "decision": decision,
        "mechanism_gates": {
            "no_validity_regression": no_validity_regression,
            "reproduces_control_meaningful_events": reproduces_control_events,
            "lower_direction_cost": lower_direction_cost,
            "lower_total_action_cost": lower_total_cost,
        },
        "meaningful_energy_drop_threshold_eV": MEANINGFUL_ENERGY_DROP_EV,
        "bootstrap": dict(bootstrap),
        "shared_initial_direction_force_evaluations": shared_cost,
        "action_force_evaluations": action_cost,
        "total_force_evaluations": bootstrap_cost + shared_cost + action_cost,
        "arm_results": arm_results,
        "claim_ceiling": (
            "one raw-input PdO slab, three paired seeds, one macro action per "
            "seed and arm; this tests transfer of the direction-continuation "
            "mechanism, not long-run PdO search superiority"
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Raw-PdO direction-continuation transfer",
        "",
        f"- Decision: `{evidence['decision']}`",
        (
            "- Bootstrap: "
            f"{evidence['bootstrap']['energy_drop_eV']:.6f} eV drop, "
            f"{evidence['bootstrap']['force_evaluations']} force evaluations, "
            f"{evidence['bootstrap']['wall_time_s']:.3f} s"
        ),
        (
            "- Shared initial Ritz: "
            f"{evidence['shared_initial_direction_force_evaluations']} force evaluations"
        ),
        "",
        "| arm | meaningful | median landing ΔE (eV) | direction FE | total FE | generation s | quench s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        result = evidence["arm_results"][arm]
        lines.append(
            f"| {arm} | {result['meaningful_outcome_count']} | "
            f"{result['median_landing_delta_eV']:.6f} | "
            f"{result['direction_force_evaluations']} | "
            f"{result['total_force_evaluations']} | "
            f"{result['generation_wall_time_s']:.3f} | "
            f"{result['quench_wall_time_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Claim ceiling: {evidence['claim_ceiling']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=RUN_ROOT / "pdo-raw-output" / "raw.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=RUN_ROOT / "pdo_raw_evidence.json",
    )
    parser.add_argument(
        "--conclusion",
        type=Path,
        default=RUN_ROOT / "pdo_raw_conclusion.md",
    )
    args = parser.parse_args(argv)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    evidence = analyze(raw)
    _write_json(args.evidence, evidence)
    args.conclusion.write_text(_markdown(evidence), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
