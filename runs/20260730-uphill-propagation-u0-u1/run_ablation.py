"""Fixed-prefix U0/U1 Gaussian uphill-propagation ablation."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import isfinite
from statistics import median
from typing import Callable, Iterable, Mapping

import numpy as np

from pamssw.accounting import (
    EvalCounter,
    EvaluationCounts,
    EvaluationPurpose,
)
from pamssw.proposal_replay import (
    ProposalPointObservation,
    proposal_task_to_payload,
    replay_proposal_task_observed,
    retarget_last_gaussian,
)
from pamssw.state import State
from pamssw.walker import ProposalPotential, ProposalRelaxationTask


ARM_IDS = (
    "current_full",
    "curvature_matched_no_feedback",
    "fixed_calibrated",
)


def build_arm_task(
    source: ProposalRelaxationTask,
    *,
    arm_id: str,
    base_sigma: float | None = None,
    inner_curvature: float | None = None,
    target_negative_curvature: float | None = None,
    fixed_sigma: float | None = None,
    fixed_weight: float | None = None,
    bias_weight_min: float | None = None,
    bias_weight_max: float | None = None,
) -> ProposalRelaxationTask:
    """Apply one arm to only the newest Gaussian of a frozen walk prefix."""
    if arm_id not in ARM_IDS:
        raise ValueError(f"unsupported arm: {arm_id}")
    if arm_id == "current_full":
        return source
    if arm_id == "fixed_calibrated":
        sigma = _finite_positive(fixed_sigma, "fixed sigma")
        weight = _finite_nonnegative(fixed_weight, "fixed weight")
        return retarget_last_gaussian(
            source,
            sigma=sigma,
            weight=weight,
        )

    sigma = _finite_positive(base_sigma, "base sigma")
    curvature = _finite(inner_curvature, "inner curvature")
    target = _finite_positive(
        target_negative_curvature,
        "target negative curvature",
    )
    weight = sigma * sigma * max(curvature + target, 0.0)
    if (bias_weight_min is None) != (bias_weight_max is None):
        raise ValueError("both bias-weight bounds must be provided")
    if bias_weight_min is not None and bias_weight_max is not None:
        lower = _finite_nonnegative(
            bias_weight_min,
            "bias_weight_min",
        )
        upper = _finite_nonnegative(
            bias_weight_max,
            "bias_weight_max",
        )
        if lower > upper:
            raise ValueError("bias-weight bounds are inverted")
        weight = float(np.clip(weight, lower, upper))
    return retarget_last_gaussian(
        source,
        sigma=sigma,
        weight=weight,
    )


def calibrate_fixed_parameters(
    records: Iterable[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Compute system-local fixed controls from calibration-only tasks."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        system = record.get("system")
        if not isinstance(system, str) or not system:
            raise ValueError("calibration system must be a non-empty string")
        _finite_positive(record.get("sigma"), "calibration sigma")
        _finite_nonnegative(record.get("weight"), "calibration weight")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("calibration task_id must be a non-empty string")
        source_hash = record.get("source_task_sha256")
        if not _is_sha256(source_hash):
            raise ValueError("calibration source_task_sha256 is invalid")
        grouped[system].append(record)

    result: dict[str, dict[str, object]] = {}
    for system, system_records in sorted(grouped.items()):
        if len(system_records) < 2:
            raise ValueError(
                f"{system} calibration requires at least two tasks"
            )
        ordered = sorted(system_records, key=lambda item: str(item["task_id"]))
        task_ids = [str(item["task_id"]) for item in ordered]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{system} calibration task IDs are duplicated")
        result[system] = {
            "sigma": float(
                median(float(item["sigma"]) for item in ordered)
            ),
            "weight": float(
                median(float(item["weight"]) for item in ordered)
            ),
            "task_ids": task_ids,
            "source_task_sha256": [
                str(item["source_task_sha256"]) for item in ordered
            ],
        }
    if not result:
        raise ValueError("calibration records are empty")
    return result


def task_sha256(task: ProposalRelaxationTask) -> str:
    payload = proposal_task_to_payload(task)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def run_task_arms(
    source: ProposalRelaxationTask,
    *,
    system: str,
    task_id: str,
    calculator_factory: Callable[[], object],
    optimizer: str,
    base_sigma: float,
    inner_curvature: float,
    target_negative_curvature: float,
    fixed_sigma: float,
    fixed_weight: float,
    bias_weight_min: float | None = None,
    bias_weight_max: float | None = None,
) -> list[dict[str, object]]:
    """Replay all arms with isolated ledgers on one frozen source task."""
    if not isinstance(system, str) or not system:
        raise ValueError("system must be a non-empty string")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    source_hash = task_sha256(source)
    rows: list[dict[str, object]] = []
    for arm_id in ARM_IDS:
        task = build_arm_task(
            source,
            arm_id=arm_id,
            base_sigma=base_sigma,
            inner_curvature=inner_curvature,
            target_negative_curvature=target_negative_curvature,
            fixed_sigma=fixed_sigma,
            fixed_weight=fixed_weight,
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
                "certificate_satisfied": replay.certificate_satisfied,
                "biased_proposal_relax_force_evaluations": (
                    counts.count(
                        EvaluationPurpose.BIASED_PROPOSAL_RELAX
                    )
                ),
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


def base_sigma_from_task(
    task: ProposalRelaxationTask,
    *,
    target_step_rms: float,
    max_step_rms: float,
    step_rms_scope: str,
    active_threshold: float,
) -> float:
    """Recover the feedback-free per-atom-RMS execution scale."""
    target = min(
        _finite_positive(target_step_rms, "target_step_rms"),
        _finite_positive(max_step_rms, "max_step_rms"),
    )
    threshold = _finite_nonnegative(
        active_threshold,
        "active_threshold",
    )
    if step_rms_scope not in {"all_atoms", "active_atoms"}:
        raise ValueError("step_rms_scope is invalid")
    direction = task.biases[-1].direction.reshape(
        task.initial_state.n_atoms,
        3,
    )
    movable_norms = np.linalg.norm(
        direction[task.initial_state.movable_mask],
        axis=1,
    )
    if movable_norms.size == 0:
        raise ValueError("task has no movable direction components")
    if step_rms_scope == "active_atoms":
        active = movable_norms > threshold * max(
            float(np.max(movable_norms)),
            1.0e-12,
        )
        movable_norms = movable_norms[active]
    direction_rms = float(
        np.sqrt(np.mean(movable_norms * movable_norms))
    )
    if direction_rms <= 0.0:
        raise ValueError("task direction has zero movable RMS")
    return target / direction_rms


def measure_inner_curvature(
    task: ProposalRelaxationTask,
    calculator,
    *,
    hvp_epsilon: float,
) -> tuple[float, EvaluationCounts]:
    """Measure curvature at the newest center before adding that Gaussian."""
    epsilon = _finite_positive(hvp_epsilon, "hvp_epsilon")
    last = task.biases[-1]
    center_state = State(
        numbers=task.initial_state.numbers.copy(),
        positions=last.center.reshape(task.initial_state.n_atoms, 3).copy(),
        cell=(
            None
            if task.initial_state.cell is None
            else task.initial_state.cell.copy()
        ),
        pbc=task.initial_state.pbc,
        fixed_mask=task.initial_state.fixed_mask.copy(),
        metadata=task.initial_state.metadata.copy(),
    )
    counter = EvalCounter(calculator)
    proposal = ProposalPotential(
        counter,
        biases=list(task.biases[:-1]),
        softening=task.softening,
    )
    direction = last.direction
    center = center_state.flatten_positions()
    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        plus_gradient = proposal.evaluate(
            center + epsilon * direction,
            center_state,
        )[1]
        minus_gradient = proposal.evaluate(
            center - epsilon * direction,
            center_state,
        )[1]
    hvp = (
        np.asarray(plus_gradient, dtype=float)
        - np.asarray(minus_gradient, dtype=float)
    ) / (2.0 * epsilon)
    return float(np.dot(direction, hvp)), counter.snapshot()


def _point_payload(
    point: ProposalPointObservation,
) -> dict[str, float]:
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


def _finite_positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
