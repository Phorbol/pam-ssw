"""Fail-closed paired analysis for the C60 U2 four-arm factorial."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from typing import Iterable, Mapping


ARM_IDS = (
    "feedback_on_on",
    "feedback_on_off",
    "feedback_off_on",
    "feedback_off_off",
)
CURRENT = ARM_IDS[0]
POINT_KEYS = {
    "true_energy",
    "bias_energy",
    "softening_energy",
    "total_energy",
}
ROW_KEYS = {
    "schema_version",
    "system",
    "task_id",
    "arm_id",
    "source_task_sha256",
    "optimizer",
    "proposal_fmax",
    "proposal_maxiter",
    "last_bias_sigma",
    "last_bias_weight",
    "effective_bias_curvature",
    "certificate_satisfied",
    "force_evaluations",
    "purpose_counts",
    "wall_time_s",
    "initial",
    "final",
    "direction_progress",
    "orthogonal_displacement_norm",
    "endpoint_position_sha256",
    "observer_only_force_evaluations",
}


def analyze_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    reviewed = [dict(row) for row in rows]
    if not reviewed:
        raise ValueError("rows are empty")
    grouped: dict[
        tuple[str, str, str],
        dict[str, dict[str, object]],
    ] = defaultdict(dict)
    for row in reviewed:
        _validate_row(row)
        key = (
            str(row["system"]),
            str(row["task_id"]),
            str(row["source_task_sha256"]),
        )
        arm_id = str(row["arm_id"])
        if arm_id in grouped[key]:
            raise ValueError("duplicate task-arm row")
        grouped[key][arm_id] = row
    for key, arm_rows in grouped.items():
        if set(arm_rows) != set(ARM_IDS):
            raise ValueError(
                f"{key[0]}/{key[1]} does not close the four-arm matrix"
            )

    systems = {}
    by_system: dict[
        str,
        list[dict[str, dict[str, object]]],
    ] = defaultdict(list)
    for (system, _, _), arm_rows in grouped.items():
        by_system[system].append(arm_rows)
    for system, task_groups in sorted(by_system.items()):
        systems[system] = {
            "task_count": len(task_groups),
            "arms": {
                arm_id: _arm_summary(task_groups, arm_id)
                for arm_id in ARM_IDS
            },
            "paired_differences_vs_current": {
                arm_id: _paired_difference(task_groups, arm_id)
                for arm_id in ARM_IDS[1:]
            },
        }
    return {
        "schema_version": 1,
        "row_count": len(reviewed),
        "systems": systems,
    }


def _arm_summary(task_groups, arm_id: str) -> dict[str, float]:
    rows = [group[arm_id] for group in task_groups]
    return {
        "certificate_rate": mean(
            float(row["certificate_satisfied"]) for row in rows
        ),
        "mean_force_evaluations": mean(
            int(row["force_evaluations"]) for row in rows
        ),
        "mean_wall_time_s": mean(
            float(row["wall_time_s"]) for row in rows
        ),
        "mean_direction_progress": mean(
            float(row["direction_progress"]) for row in rows
        ),
        "mean_orthogonal_displacement_norm": mean(
            float(row["orthogonal_displacement_norm"])
            for row in rows
        ),
    }


def _paired_difference(task_groups, arm_id: str) -> dict[str, float]:
    return {
        "certificate_rate": mean(
            float(group[arm_id]["certificate_satisfied"])
            - float(group[CURRENT]["certificate_satisfied"])
            for group in task_groups
        ),
        "mean_force_evaluations": mean(
            int(group[arm_id]["force_evaluations"])
            - int(group[CURRENT]["force_evaluations"])
            for group in task_groups
        ),
        "mean_final_true_energy": mean(
            float(group[arm_id]["final"]["true_energy"])
            - float(group[CURRENT]["final"]["true_energy"])
            for group in task_groups
        ),
        "mean_direction_progress": mean(
            float(group[arm_id]["direction_progress"])
            - float(group[CURRENT]["direction_progress"])
            for group in task_groups
        ),
        "mean_orthogonal_displacement_norm": mean(
            float(group[arm_id]["orthogonal_displacement_norm"])
            - float(group[CURRENT]["orthogonal_displacement_norm"])
            for group in task_groups
        ),
    }


def _validate_row(row: Mapping[str, object]) -> None:
    if set(row) != ROW_KEYS:
        raise ValueError("row keys do not match the U2 schema")
    if row["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if row["arm_id"] not in ARM_IDS:
        raise ValueError("unknown arm_id")
    if row["observer_only_force_evaluations"] != 0:
        raise ValueError("observer-only force evaluations must be zero")
    total = row["force_evaluations"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("force_evaluations must be non-negative")
    purposes = row["purpose_counts"]
    if not isinstance(purposes, Mapping):
        raise ValueError("purpose_counts must be a mapping")
    if sum(int(value) for value in purposes.values()) != total:
        raise ValueError("purpose counts do not close")
    if int(purposes.get("unattributed", -1)) != 0:
        raise ValueError("purpose counts contain unattributed calls")
    for label in (
        "proposal_fmax",
        "last_bias_sigma",
        "last_bias_weight",
        "effective_bias_curvature",
        "wall_time_s",
        "direction_progress",
        "orthogonal_displacement_norm",
    ):
        _finite(row[label], label)
    for label in ("initial", "final"):
        point = row[label]
        if not isinstance(point, Mapping) or set(point) != POINT_KEYS:
            raise ValueError(f"{label} component keys are invalid")
        for key in POINT_KEYS:
            _finite(point[key], f"{label}.{key}")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
