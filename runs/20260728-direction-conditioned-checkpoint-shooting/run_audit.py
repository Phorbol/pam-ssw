#!/usr/bin/env python3
"""Run the preregistered direction-conditioned checkpoint shooting audit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "balanced_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    },
    "deep_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
}
MEANINGFUL_ENERGY_DROP_EV = 0.001


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"state_id": state_id, "seed": seed, "arm": arm}
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


def classify_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
) -> str:
    if not checkpoints:
        raise ValueError("trajectory requires at least one checkpoint")
    observed = [int(row["step_index"]) for row in checkpoints]
    expected = list(range(1, len(checkpoints) + 1))
    if observed != expected:
        raise ValueError("checkpoint indices must be consecutive and ordered")
    productive = [bool(row["productive"]) for row in checkpoints]
    final_productive = productive[-1]
    earlier_productive = any(productive[:-1])
    if final_productive and earlier_productive:
        return "productive_earlier_and_final"
    if final_productive:
        return "productive_final"
    if earlier_productive:
        return "overshoot"
    return "no_productive_checkpoint"
