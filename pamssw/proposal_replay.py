"""Capture and replay fixed biased-PES proposal-relaxation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from .bias import GaussianBiasTerm
from .config import SSWConfig
from .coordinates import CartesianCoordinates, TangentVector
from .relax import Relaxer
from .result import RelaxResult
from .state import State
from .walker import ProposalPotential, ProposalRelaxationTask, SurfaceWalker


@dataclass(frozen=True)
class CapturedProposalTask:
    task: ProposalRelaxationTask
    evaluation_counts: EvaluationCounts


@dataclass(frozen=True)
class ProposalReplayResult:
    result: RelaxResult
    evaluation_counts: EvaluationCounts
    wall_time_s: float
    certificate_satisfied: bool


class _TaskCaptured(RuntimeError):
    def __init__(self, task: ProposalRelaxationTask) -> None:
        super().__init__("proposal-relaxation task captured")
        self.task = task


class _CapturingSurfaceWalker(SurfaceWalker):
    def __init__(self, *args, target_bias_count: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target_bias_count = target_bias_count

    def _relax_proposal_task(self, task, *, optimizer, trajectory_callback):
        if len(task.biases) == self._target_bias_count:
            raise _TaskCaptured(task)
        return super()._relax_proposal_task(
            task,
            optimizer=optimizer,
            trajectory_callback=trajectory_callback,
        )


def capture_proposal_task(
    seed_state: State,
    calculator,
    config: SSWConfig,
    *,
    target_bias_count: int,
) -> CapturedProposalTask:
    """Freeze one production task immediately before its relaxation starts."""
    if isinstance(target_bias_count, bool) or not isinstance(target_bias_count, int):
        raise TypeError("target_bias_count must be an integer")
    if target_bias_count <= 0:
        raise ValueError("target_bias_count must be positive")
    if target_bias_count > config.max_steps_per_walk:
        raise ValueError("target_bias_count exceeds max_steps_per_walk")
    walker = _CapturingSurfaceWalker(
        calculator=calculator,
        config=config,
        softening_enabled=False,
        target_bias_count=target_bias_count,
    )
    try:
        walker._walk_candidate_from_seed(seed_state)
    except _TaskCaptured as captured:
        return CapturedProposalTask(
            task=captured.task,
            evaluation_counts=walker.calculator.snapshot(),
        )
    raise RuntimeError("walk terminated before the requested proposal task")


def retarget_last_gaussian(
    task: ProposalRelaxationTask,
    *,
    sigma: float,
    weight: float,
) -> ProposalRelaxationTask:
    """Change only the newest Gaussian and its explicit starting displacement."""
    if not task.biases:
        raise ValueError("proposal task has no Gaussian bias")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")

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
    trial_state = CartesianCoordinates.from_state(center_state).displace(
        TangentVector(last.direction),
        sigma,
    )
    biases = (
        *task.biases[:-1],
        GaussianBiasTerm(
            center=last.center,
            direction=last.direction,
            sigma=sigma,
            weight=weight,
        ),
    )
    return ProposalRelaxationTask(
        initial_state=trial_state,
        biases=biases,
        softening=task.softening,
        fmax=task.fmax,
        maxiter=task.maxiter,
        coordinate_trust_radius=task.coordinate_trust_radius,
    )


def replay_proposal_task(
    task: ProposalRelaxationTask,
    calculator,
    *,
    optimizer: str,
) -> ProposalReplayResult:
    """Run one backend on one frozen task with an isolated exact ledger."""
    counter = EvalCounter(calculator)
    proposal = ProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    custom_lbfgs = optimizer in {
        "safe-lbfgs-total",
        "bias-separated-lbfgs",
    }
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer=optimizer,
        component_evaluator=proposal.evaluate_parts if custom_lbfgs else None,
    )
    started = perf_counter()
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=task.maxiter,
            coordinate_trust_radius=task.coordinate_trust_radius,
        )
    wall_time_s = perf_counter() - started
    certificate_satisfied = bool(
        np.isfinite(result.energy)
        and np.isfinite(result.gradient_norm)
        and np.all(np.isfinite(result.state.positions))
        and result.gradient_norm <= task.fmax
    )
    return ProposalReplayResult(
        result=result,
        evaluation_counts=counter.snapshot(),
        wall_time_s=wall_time_s,
        certificate_satisfied=certificate_satisfied,
    )


def proposal_task_to_payload(task: ProposalRelaxationTask) -> dict[str, Any]:
    if task.softening is not None:
        raise ValueError("local-softening task serialization is not supported")
    state = task.initial_state
    return {
        "schema_version": 1,
        "initial_state": {
            "numbers": state.numbers.tolist(),
            "positions": state.positions.tolist(),
            "cell": None if state.cell is None else state.cell.tolist(),
            "pbc": list(state.pbc),
            "fixed_mask": state.fixed_mask.tolist(),
        },
        "biases": [
            {
                "center": bias.center.tolist(),
                "direction": bias.direction.tolist(),
                "sigma": float(bias.sigma),
                "weight": float(bias.weight),
            }
            for bias in task.biases
        ],
        "fmax": float(task.fmax),
        "maxiter": int(task.maxiter),
        "coordinate_trust_radius": task.coordinate_trust_radius,
    }


def proposal_task_from_payload(payload: Mapping[str, Any]) -> ProposalRelaxationTask:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported proposal-task schema_version")
    state_payload = payload["initial_state"]
    state = State(
        numbers=np.asarray(state_payload["numbers"], dtype=int),
        positions=np.asarray(state_payload["positions"], dtype=float),
        cell=(
            None
            if state_payload["cell"] is None
            else np.asarray(state_payload["cell"], dtype=float)
        ),
        pbc=tuple(state_payload["pbc"]),
        fixed_mask=np.asarray(state_payload["fixed_mask"], dtype=bool),
    )
    biases = tuple(
        GaussianBiasTerm(
            center=np.asarray(item["center"], dtype=float),
            direction=np.asarray(item["direction"], dtype=float),
            sigma=float(item["sigma"]),
            weight=float(item["weight"]),
        )
        for item in payload["biases"]
    )
    return ProposalRelaxationTask(
        initial_state=state,
        biases=biases,
        softening=None,
        fmax=float(payload["fmax"]),
        maxiter=int(payload["maxiter"]),
        coordinate_trust_radius=(
            None
            if payload["coordinate_trust_radius"] is None
            else float(payload["coordinate_trust_radius"])
        ),
    )
