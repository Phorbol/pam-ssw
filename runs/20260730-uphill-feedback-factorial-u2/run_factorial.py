"""C60 U2 factorial decomposition of the existing uphill feedback law."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite
from typing import Callable

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.proposal_replay import (
    ProposalPointObservation,
    replay_proposal_task_observed,
    retarget_last_gaussian,
)
from pamssw.walker import ProposalRelaxationTask


ARM_IDS = (
    "feedback_on_on",
    "feedback_on_off",
    "feedback_off_on",
    "feedback_off_off",
)


def build_factorial_task(
    source: ProposalRelaxationTask,
    *,
    arm_id: str,
    base_sigma: float,
    inner_curvature: float,
    target_negative_curvature: float,
    bias_weight_min: float,
    bias_weight_max: float,
) -> ProposalRelaxationTask:
    """Construct one 2x2 arm in (sigma, weight/sigma**2) coordinates."""
    if arm_id not in ARM_IDS:
        raise ValueError(f"unsupported arm: {arm_id}")
    if arm_id == "feedback_on_on":
        return source

    current = source.biases[-1]
    current_sigma = _positive(current.sigma, "current sigma")
    feedback_free_sigma = _positive(base_sigma, "base sigma")
    current_effective_curvature = (
        _nonnegative(current.weight, "current weight")
        / (current_sigma * current_sigma)
    )
    feedback_free_effective_curvature = max(
        _finite(inner_curvature, "inner curvature")
        + _positive(
            target_negative_curvature,
            "target negative curvature",
        ),
        0.0,
    )
    lower = _nonnegative(bias_weight_min, "bias_weight_min")
    upper = _nonnegative(bias_weight_max, "bias_weight_max")
    if lower > upper:
        raise ValueError("bias-weight bounds are inverted")

    sigma_feedback_on = arm_id in {
        "feedback_on_off",
    }
    curvature_feedback_on = arm_id in {
        "feedback_off_on",
    }
    sigma = (
        current_sigma if sigma_feedback_on else feedback_free_sigma
    )
    effective_curvature = (
        current_effective_curvature
        if curvature_feedback_on
        else feedback_free_effective_curvature
    )
    weight = float(
        np.clip(
            sigma * sigma * effective_curvature,
            lower,
            upper,
        )
    )
    return retarget_last_gaussian(
        source,
        sigma=sigma,
        weight=weight,
    )


def run_factorial_arms(
    source: ProposalRelaxationTask,
    *,
    system: str,
    task_id: str,
    calculator_factory: Callable[[], object],
    optimizer: str,
    base_sigma: float,
    inner_curvature: float,
    target_negative_curvature: float,
    bias_weight_min: float,
    bias_weight_max: float,
) -> list[dict[str, object]]:
    source_hash = _task_sha256(source)
    rows = []
    for arm_id in ARM_IDS:
        task = build_factorial_task(
            source,
            arm_id=arm_id,
            base_sigma=base_sigma,
            inner_curvature=inner_curvature,
            target_negative_curvature=target_negative_curvature,
            bias_weight_min=bias_weight_min,
            bias_weight_max=bias_weight_max,
        )
        replay = replay_proposal_task_observed(
            task,
            calculator_factory(),
            optimizer=optimizer,
        )
        counts = replay.evaluation_counts
        rows.append(
            {
                "schema_version": 1,
                "system": system,
                "task_id": task_id,
                "arm_id": arm_id,
                "source_task_sha256": source_hash,
                "optimizer": optimizer,
                "proposal_fmax": float(task.fmax),
                "proposal_maxiter": int(task.maxiter),
                "last_bias_sigma": float(task.biases[-1].sigma),
                "last_bias_weight": float(task.biases[-1].weight),
                "effective_bias_curvature": float(
                    task.biases[-1].weight
                    / (task.biases[-1].sigma**2)
                ),
                "certificate_satisfied": replay.certificate_satisfied,
                "force_evaluations": counts.total,
                "purpose_counts": counts.as_dict(),
                "wall_time_s": replay.wall_time_s,
                "initial": _point_payload(replay.initial),
                "final": _point_payload(replay.final),
                "direction_progress": replay.direction_progress,
                "orthogonal_displacement_norm": (
                    replay.orthogonal_displacement_norm
                ),
                "endpoint_position_sha256": _position_sha256(
                    replay.result.state.positions
                ),
                "observer_only_force_evaluations": (
                    replay.observer_only_force_evaluations
                ),
            }
        )
    return rows


def _task_sha256(task: ProposalRelaxationTask) -> str:
    from pamssw.proposal_replay import proposal_task_to_payload
    import json

    canonical = json.dumps(
        proposal_task_to_payload(task),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _point_payload(point: ProposalPointObservation) -> dict[str, float]:
    return {
        "true_energy": point.true_energy,
        "bias_energy": point.bias_energy,
        "softening_energy": point.softening_energy,
        "total_energy": point.total_energy,
    }


def _position_sha256(positions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        np.asarray(positions, dtype="<f8")
    )
    return sha256(canonical.tobytes()).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result
