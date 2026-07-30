"""One-factor task construction for the serial-Gaussian uphill audit."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Mapping

import numpy as np

from pamssw.bias import GaussianBiasTerm
from pamssw.walker import ProposalRelaxationTask


def _array_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def bias_fingerprint(bias: GaussianBiasTerm) -> str:
    payload = {
        "center": _array_digest(bias.center),
        "direction": _array_digest(bias.direction),
        "sigma": float(bias.sigma),
        "weight": float(bias.weight),
    }
    return sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _state_fingerprint(task: ProposalRelaxationTask) -> str:
    state = task.initial_state
    payload = {
        "numbers": _array_digest(np.asarray(state.numbers, dtype=float)),
        "positions": _array_digest(state.positions),
        "cell": None if state.cell is None else _array_digest(state.cell),
        "pbc": list(state.pbc),
        "fixed_mask": [
            bool(value)
            for value in np.asarray(state.fixed_mask, dtype=bool)
        ],
    }
    return sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _softening_fingerprint(task: ProposalRelaxationTask) -> str | None:
    model = task.softening
    if model is None:
        return None
    payload = {
        "terms": [
            {
                "atom_i": int(term.atom_i),
                "atom_j": int(term.atom_j),
                "reference_distance": float(term.reference_distance),
                "width": float(term.width),
                "strength": float(term.strength),
            }
            for term in model.terms
        ],
        "cell": None if model.cell is None else _array_digest(model.cell),
        "pbc": list(model.pbc),
        "penalty": model.penalty,
        "xi": model.xi,
        "reference_scaled_xi": model.reference_scaled_xi,
        "cutoff": model.cutoff,
        "adaptive_strength": model.adaptive_strength,
        "max_strength_scale": model.max_strength_scale,
        "deviation_scale": model.deviation_scale,
    }
    return sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def physical_task_fingerprint(task: ProposalRelaxationTask) -> str:
    """Hash the modified PES and starting state, excluding iteration capacity."""

    payload = {
        "state": _state_fingerprint(task),
        "biases": [bias_fingerprint(bias) for bias in task.biases],
        "softening": _softening_fingerprint(task),
        "fmax": float(task.fmax),
        "coordinate_trust_radius": task.coordinate_trust_radius,
    }
    return sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def maxiter_arms(
    task: ProposalRelaxationTask,
    *,
    extended_maxiter: int,
) -> dict[str, ProposalRelaxationTask]:
    if extended_maxiter <= task.maxiter:
        raise ValueError("extended_maxiter must exceed the frozen task maxiter")
    return {
        f"maxiter{task.maxiter}": replace(task),
        f"maxiter{extended_maxiter}": replace(
            task,
            maxiter=extended_maxiter,
        ),
    }


def history_arms(
    task: ProposalRelaxationTask,
) -> dict[str, ProposalRelaxationTask]:
    if not task.biases:
        raise ValueError("history ablation requires at least one Gaussian bias")
    return {
        "cumulative": replace(task),
        "newest_only": replace(task, biases=(task.biases[-1],)),
    }


def softening_arms(
    task: ProposalRelaxationTask,
) -> dict[str, ProposalRelaxationTask]:
    if task.softening is None:
        raise ValueError("softening ablation requires proposal softening")
    return {
        "both": replace(task),
        "oracle_only": replace(task, softening=None),
    }


def task_component_diff(
    left: ProposalRelaxationTask,
    right: ProposalRelaxationTask,
) -> set[str]:
    differences: set[str] = set()
    if _state_fingerprint(left) != _state_fingerprint(right):
        differences.add("initial_state")
    left_biases = [bias_fingerprint(bias) for bias in left.biases]
    right_biases = [bias_fingerprint(bias) for bias in right.biases]
    if left_biases != right_biases:
        if (
            left_biases
            and right_biases
            and left_biases[-1] == right_biases[-1]
        ):
            differences.add("bias_history")
        else:
            differences.add("biases")
    if _softening_fingerprint(left) != _softening_fingerprint(right):
        differences.add("proposal_softening")
    if left.fmax != right.fmax:
        differences.add("fmax")
    if left.maxiter != right.maxiter:
        differences.add("maxiter")
    if left.coordinate_trust_radius != right.coordinate_trust_radius:
        differences.add("coordinate_trust_radius")
    return differences


def purpose_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    keys = set(before) | set(after)
    delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
    }
    if any(value < 0 for value in delta.values()):
        raise ValueError("evaluation counters must be monotonic")
    if "total" in delta:
        attributed = sum(
            value for key, value in delta.items() if key != "total"
        )
        if delta["total"] != attributed:
            raise ValueError("purpose accounting does not close")
    return dict(sorted(delta.items()))
