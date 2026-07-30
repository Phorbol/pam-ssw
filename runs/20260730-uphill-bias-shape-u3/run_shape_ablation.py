"""Paired frozen-task test of Gaussian versus center-matched quadratic bias."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Callable

import numpy as np

from pamssw.bias import GaussianBiasTerm, QuadraticBiasTerm
from pamssw.proposal_replay import (
    ProposalPointObservation,
    proposal_task_to_payload,
    replay_proposal_task_observed,
)
from pamssw.walker import ProposalRelaxationTask


ARM_IDS = ("gaussian", "quadratic")


def biases_for_arm(
    task: ProposalRelaxationTask,
    arm_id: str,
) -> tuple[GaussianBiasTerm | QuadraticBiasTerm, ...]:
    """Replace only the newest Gaussian while preserving its local jet."""
    if arm_id not in ARM_IDS:
        raise ValueError(f"unsupported arm: {arm_id}")
    if arm_id == "gaussian":
        return tuple(task.biases)
    newest = task.biases[-1]
    return (
        *task.biases[:-1],
        QuadraticBiasTerm(
            center=newest.center,
            direction=newest.direction,
            sigma=newest.sigma,
            weight=newest.weight,
        ),
    )


def run_shape_arms(
    source: ProposalRelaxationTask,
    *,
    system: str,
    task_id: str,
    calculator_factory: Callable[[], object],
    optimizer: str,
) -> list[dict[str, object]]:
    source_hash = _task_sha256(source)
    rows = []
    for arm_id in ARM_IDS:
        biases = biases_for_arm(source, arm_id)
        replay = replay_proposal_task_observed(
            source,
            calculator_factory(),
            optimizer=optimizer,
            biases_override=biases,
        )
        newest = biases[-1]
        counts = replay.evaluation_counts
        rows.append(
            {
                "schema_version": 1,
                "system": system,
                "task_id": task_id,
                "arm_id": arm_id,
                "source_task_sha256": source_hash,
                "optimizer": optimizer,
                "proposal_fmax": float(source.fmax),
                "proposal_maxiter": int(source.maxiter),
                "coordinate_trust_radius": (
                    source.coordinate_trust_radius
                ),
                "last_bias_sigma": float(newest.sigma),
                "last_bias_weight": float(newest.weight),
                "effective_bias_curvature": float(
                    newest.weight / newest.sigma**2
                ),
                "certificate_satisfied": replay.certificate_satisfied,
                "gradient_norm": float(replay.result.gradient_norm),
                "n_iter": int(replay.result.n_iter),
                "termination_reason": (
                    replay.result.telemetry.termination_reason
                ),
                "optimizer_success": (
                    replay.result.telemetry.optimizer_success
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
                "active_bound_fraction": float(
                    replay.result.active_bound_fraction
                ),
                "displacement_rms": float(
                    replay.result.displacement_rms
                ),
                "displacement_max": float(
                    replay.result.displacement_max
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
    canonical = json.dumps(
        proposal_task_to_payload(task),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


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
    canonical = np.ascontiguousarray(np.asarray(positions, dtype="<f8"))
    return sha256(canonical.tobytes()).hexdigest()
