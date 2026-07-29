"""Fail-closed analysis for the U0/U1 fixed-prefix three-arm matrix."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from typing import Iterable, Mapping


ARM_IDS = (
    "current_full",
    "curvature_matched_no_feedback",
    "fixed_calibrated",
)

POINT_KEYS = {
    "true_energy",
    "bias_energy",
    "softening_energy",
    "total_energy",
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

    expected = set(ARM_IDS)
    for key, arm_rows in grouped.items():
        if set(arm_rows) != expected:
            raise ValueError(
                f"{key[0]}/{key[1]} does not close the three-arm matrix"
            )

    systems: dict[str, dict[str, object]] = {}
    system_groups: dict[
        str,
        list[dict[str, dict[str, object]]],
    ] = defaultdict(list)
    for (system, _, _), arm_rows in grouped.items():
        system_groups[system].append(arm_rows)
    for system, task_groups in sorted(system_groups.items()):
        arm_summary: dict[str, dict[str, float]] = {}
        for arm_id in ARM_IDS:
            arm_rows = [group[arm_id] for group in task_groups]
            arm_summary[arm_id] = {
                "certificate_rate": mean(
                    float(row["certificate_satisfied"])
                    for row in arm_rows
                ),
                "mean_force_evaluations": mean(
                    int(row["biased_proposal_relax_force_evaluations"])
                    for row in arm_rows
                ),
                "mean_wall_time_s": mean(
                    float(row["wall_time_s"]) for row in arm_rows
                ),
                "mean_true_energy_change": mean(
                    float(row["final"]["true_energy"])
                    - float(row["initial"]["true_energy"])
                    for row in arm_rows
                ),
                "mean_direction_progress": mean(
                    float(row["direction_progress"]) for row in arm_rows
                ),
                "mean_orthogonal_displacement_norm": mean(
                    float(row["orthogonal_displacement_norm"])
                    for row in arm_rows
                ),
            }
        systems[system] = {
            "task_count": len(task_groups),
            "arms": arm_summary,
            "paired_differences_vs_current": {
                arm_id: _paired_difference(
                    task_groups,
                    arm_id=arm_id,
                )
                for arm_id in ARM_IDS[1:]
            },
        }
    return {
        "schema_version": 1,
        "row_count": len(reviewed),
        "systems": systems,
    }


def _paired_difference(
    task_groups: list[dict[str, dict[str, object]]],
    *,
    arm_id: str,
) -> dict[str, float]:
    return {
        "mean_force_evaluations": mean(
            int(group[arm_id]["biased_proposal_relax_force_evaluations"])
            - int(
                group["current_full"][
                    "biased_proposal_relax_force_evaluations"
                ]
            )
            for group in task_groups
        ),
        "mean_true_energy_change": mean(
            (
                float(group[arm_id]["final"]["true_energy"])
                - float(group[arm_id]["initial"]["true_energy"])
            )
            - (
                float(group["current_full"]["final"]["true_energy"])
                - float(group["current_full"]["initial"]["true_energy"])
            )
            for group in task_groups
        ),
        "mean_direction_progress": mean(
            float(group[arm_id]["direction_progress"])
            - float(group["current_full"]["direction_progress"])
            for group in task_groups
        ),
    }


def _validate_row(row: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "system",
        "task_id",
        "arm_id",
        "source_task_sha256",
        "certificate_satisfied",
        "biased_proposal_relax_force_evaluations",
        "wall_time_s",
        "initial",
        "final",
        "direction_progress",
        "orthogonal_displacement_norm",
        "endpoint_position_sha256",
        "observer_only_force_evaluations",
    }
    if set(row) != required:
        raise ValueError("row keys do not match the U0/U1 schema")
    if row["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    for label in ("system", "task_id"):
        if not isinstance(row[label], str) or not row[label]:
            raise ValueError(f"{label} must be a non-empty string")
    if row["arm_id"] not in ARM_IDS:
        raise ValueError("unknown arm_id")
    for label in ("source_task_sha256", "endpoint_position_sha256"):
        if not _is_sha256(row[label]):
            raise ValueError(f"{label} is invalid")
    if not isinstance(row["certificate_satisfied"], bool):
        raise ValueError("certificate_satisfied must be boolean")
    force_evaluations = row["biased_proposal_relax_force_evaluations"]
    if (
        isinstance(force_evaluations, bool)
        or not isinstance(force_evaluations, int)
        or force_evaluations < 0
    ):
        raise ValueError("force evaluations must be a non-negative integer")
    if row["observer_only_force_evaluations"] != 0:
        raise ValueError("observer-only force evaluations must be zero")
    for label in (
        "wall_time_s",
        "direction_progress",
        "orthogonal_displacement_norm",
    ):
        _finite(row[label], label)
    if float(row["wall_time_s"]) < 0.0:
        raise ValueError("wall_time_s must be non-negative")
    if float(row["orthogonal_displacement_norm"]) < 0.0:
        raise ValueError(
            "orthogonal_displacement_norm must be non-negative"
        )
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


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

