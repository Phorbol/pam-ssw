"""Fail-closed paired analysis for the U3 bias-shape ablation."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean, median
from typing import Iterable, Mapping


ARM_IDS = ("gaussian", "quadratic")
BASELINE = ARM_IDS[0]
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
    "coordinate_trust_radius",
    "last_bias_sigma",
    "last_bias_weight",
    "effective_bias_curvature",
    "certificate_satisfied",
    "gradient_norm",
    "n_iter",
    "termination_reason",
    "optimizer_success",
    "force_evaluations",
    "purpose_counts",
    "wall_time_s",
    "initial",
    "final",
    "direction_progress",
    "orthogonal_displacement_norm",
    "active_bound_fraction",
    "displacement_rms",
    "displacement_max",
    "endpoint_position_sha256",
    "observer_only_force_evaluations",
}


def analyze_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    reviewed = [dict(row) for row in rows]
    if not reviewed:
        raise ValueError("rows are empty")
    grouped = defaultdict(dict)
    for row in reviewed:
        _validate_row(row)
        key = (
            str(row["system"]),
            str(row["task_id"]),
            str(row["source_task_sha256"]),
        )
        arm = str(row["arm_id"])
        if arm in grouped[key]:
            raise ValueError("duplicate task-arm row")
        grouped[key][arm] = row
    for key, arm_rows in grouped.items():
        if set(arm_rows) != set(ARM_IDS):
            raise ValueError(
                f"{key[0]}/{key[1]} does not close the paired matrix"
            )

    by_system = defaultdict(list)
    for (system, _, _), arm_rows in grouped.items():
        by_system[system].append(arm_rows)
    systems = {}
    for system, task_groups in sorted(by_system.items()):
        systems[system] = {
            "task_count": len(task_groups),
            "arms": {
                arm: _arm_summary(task_groups, arm)
                for arm in ARM_IDS
            },
            "quadratic_minus_gaussian": _paired_difference(
                task_groups
            ),
        }
    return {
        "schema_version": 1,
        "row_count": len(reviewed),
        "systems": systems,
    }


def _arm_summary(task_groups, arm: str) -> dict[str, float]:
    rows = [group[arm] for group in task_groups]
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
        "mean_active_bound_fraction": mean(
            float(row["active_bound_fraction"]) for row in rows
        ),
        "mean_gradient_norm": mean(
            float(row["gradient_norm"]) for row in rows
        ),
        "maxiter_rate": mean(
            float(row["termination_reason"] == "maxiter")
            for row in rows
        ),
    }


def _paired_difference(task_groups) -> dict[str, float]:
    metrics = {
        "final_true_energy": lambda row: row["final"]["true_energy"],
        "direction_progress": lambda row: row["direction_progress"],
        "orthogonal_displacement_norm": (
            lambda row: row["orthogonal_displacement_norm"]
        ),
        "force_evaluations": lambda row: row["force_evaluations"],
        "wall_time_s": lambda row: row["wall_time_s"],
        "active_bound_fraction": (
            lambda row: row["active_bound_fraction"]
        ),
        "displacement_max": lambda row: row["displacement_max"],
        "gradient_norm": lambda row: row["gradient_norm"],
        "n_iter": lambda row: row["n_iter"],
    }
    result = {
        "certificate_rate": mean(
            float(group["quadratic"]["certificate_satisfied"])
            - float(group[BASELINE]["certificate_satisfied"])
            for group in task_groups
        )
    }
    for label, getter in metrics.items():
        differences = [
            float(getter(group["quadratic"]))
            - float(getter(group[BASELINE]))
            for group in task_groups
        ]
        result[f"mean_{label}"] = mean(differences)
        result[f"median_{label}"] = median(differences)
    return result


def _validate_row(row: Mapping[str, object]) -> None:
    if set(row) != ROW_KEYS:
        raise ValueError("row keys do not match the U3 schema")
    if row["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if row["arm_id"] not in ARM_IDS:
        raise ValueError("unknown arm_id")
    if row["observer_only_force_evaluations"] != 0:
        raise ValueError("observer-only force evaluations must be zero")
    if not isinstance(row["termination_reason"], str) or not row[
        "termination_reason"
    ]:
        raise ValueError("termination_reason must be a non-empty string")
    if row["optimizer_success"] not in {True, False, None}:
        raise ValueError("optimizer_success must be boolean or null")
    if (
        isinstance(row["n_iter"], bool)
        or not isinstance(row["n_iter"], int)
        or row["n_iter"] < 0
    ):
        raise ValueError("n_iter must be non-negative")
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
        "active_bound_fraction",
        "displacement_rms",
        "displacement_max",
        "gradient_norm",
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
