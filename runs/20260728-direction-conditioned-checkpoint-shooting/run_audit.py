#!/usr/bin/env python3
"""Run the preregistered direction-conditioned checkpoint shooting audit."""

from __future__ import annotations

from collections import Counter
import math
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


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoints = [
        checkpoint
        for row in rows
        for checkpoint in row["checkpoints"]
    ]
    return {
        "case_count": len(rows),
        "checkpoint_count": len(checkpoints),
        "certificate_count": sum(
            bool(checkpoint["certificate"]) for checkpoint in checkpoints
        ),
        "meaningful_checkpoint_count": sum(
            bool(checkpoint["productive"]) for checkpoint in checkpoints
        ),
        "generation_force_evaluations": sum(
            int(row["generation_force_evaluations"]) for row in rows
        ),
        "shooting_force_evaluations": sum(
            int(checkpoint["force_evaluations"])
            for checkpoint in checkpoints
        ),
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in rows).items())
        ),
    }


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (case["state_id"], case["seed"], case["arm"])
        for case in case_matrix()
    }
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("evidence requires the exact 12-case cohort")

    for row in rows:
        generation_purposes = row.get("generation_purpose_counts")
        if (
            row.get("status") != "completed"
            or not isinstance(generation_purposes, Mapping)
            or sum(int(value) for value in generation_purposes.values())
            != row.get("generation_force_evaluations")
            or generation_purposes.get("unattributed") != 0
        ):
            raise ValueError("generation purpose ledger does not close")
        if generation_purposes.get("direction_oracle") != (
            24 * int(row.get("direction_selection_count", -1))
        ):
            raise ValueError("direction ledger does not close")

        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("case requires at least one checkpoint")
        expected_classification = classify_trajectory(checkpoints)
        if row.get("classification") != expected_classification:
            raise ValueError("trajectory classification does not close")

        for checkpoint in checkpoints:
            purposes = checkpoint.get("purpose_counts")
            if (
                checkpoint.get("status") != "completed"
                or not isinstance(checkpoint.get("certificate"), bool)
                or not isinstance(checkpoint.get("productive"), bool)
                or not isinstance(purposes, Mapping)
                or sum(int(value) for value in purposes.values())
                != checkpoint.get("force_evaluations")
                or purposes.get("unattributed") != 0
                or purposes.get("direction_oracle") != 0
                or purposes.get("biased_proposal_relax") != 0
                or purposes.get("landing_true_quench", 0) <= 0
            ):
                raise ValueError("checkpoint purpose ledger does not close")
            expected_productive = bool(
                checkpoint["certificate"]
                and checkpoint.get("is_new_basin")
                and _finite(
                    checkpoint["landing_delta_eV"],
                    "checkpoint landing delta",
                )
                < -MEANINGFUL_ENERGY_DROP_EV
            )
            if checkpoint["productive"] != expected_productive:
                raise ValueError("checkpoint productive flag does not close")

    classification_counts = dict(
        sorted(Counter(row["classification"] for row in rows).items())
    )
    aggregate = _aggregate(rows)
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
            "checkpoint_count": aggregate["checkpoint_count"],
        },
        "classification_counts": classification_counts,
        "certificate_count": aggregate["certificate_count"],
        "meaningful_checkpoint_count": aggregate[
            "meaningful_checkpoint_count"
        ],
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "state_arm_results": {
            state_id: {
                arm: _aggregate(
                    [
                        row
                        for row in rows
                        if row["state_id"] == state_id
                        and row["arm"] == arm
                    ]
                )
                for arm in ARMS
            }
            for state_id in STATE_IDS
        },
        "totals": aggregate,
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive direction-conditioned checkpoint shooting audit; "
            "no stopping rule, direction arm, or selector is promoted"
        ),
        "cases": list(rows),
    }
