"""Pure protocol for current-action escape first-passage classification."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = ("D0_exact_anchor", "K4_discrete")
CHECKPOINT_HORIZONS = (1, 2, 4, 8)
LABELS = (
    "RETURN_STARTER",
    "ESCAPED_CERTIFIED",
    "AMBIGUOUS_MATCH",
    "INVALID_GEOMETRY",
    "FRAGMENTED",
    "QUENCH_UNCONVERGED",
    "BUDGET_EXHAUSTED",
)
UNLEARNABLE_LABELS = {
    "AMBIGUOUS_MATCH",
    "QUENCH_UNCONVERGED",
}


def case_matrix() -> list[dict[str, Any]]:
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "arm": arm,
        }
        for system in SYSTEMS
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


def classify_checkpoint(
    *,
    budget_exhausted: bool,
    fragmented: bool,
    geometry_valid: bool,
    certificate: bool,
    matcher_same: bool | None,
    descriptor_same: bool | None,
) -> str:
    """Give numerical validity priority over basin interpretation."""

    if budget_exhausted:
        return "BUDGET_EXHAUSTED"
    if fragmented:
        return "FRAGMENTED"
    if not geometry_valid:
        return "INVALID_GEOMETRY"
    if not certificate:
        return "QUENCH_UNCONVERGED"
    if matcher_same is None or descriptor_same is None:
        return "AMBIGUOUS_MATCH"
    if bool(matcher_same) != bool(descriptor_same):
        return "AMBIGUOUS_MATCH"
    return "RETURN_STARTER" if matcher_same else "ESCAPED_CERTIFIED"


def summarize_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    horizons = [int(row["horizon"]) for row in checkpoints]
    if horizons != sorted(horizons) or len(horizons) != len(set(horizons)):
        raise ValueError("checkpoint horizons must be unique and ordered")
    if any(horizon not in CHECKPOINT_HORIZONS for horizon in horizons):
        raise ValueError("unknown checkpoint horizon")
    labels = [str(row["label"]) for row in checkpoints]
    if any(label not in LABELS for label in labels):
        raise ValueError("unknown checkpoint label")
    by_horizon = dict(zip(horizons, labels, strict=True))
    complete = horizons == list(CHECKPOINT_HORIZONS)
    learnable = not any(label in UNLEARNABLE_LABELS for label in labels)
    return {
        "early_escape_then_h8_return": bool(
            learnable
            and by_horizon.get(8) == "RETURN_STARTER"
            and any(
                by_horizon.get(horizon) == "ESCAPED_CERTIFIED"
                for horizon in CHECKPOINT_HORIZONS
                if horizon < 8
            )
        ),
        "all_horizons_return_starter": bool(
            learnable
            and complete
            and all(label == "RETURN_STARTER" for label in labels)
        ),
        "learnable": learnable,
        "observed_horizons": horizons,
    }


def _validate_ledger(
    counts: Mapping[str, Any],
    expected_total: Any,
    *,
    name: str,
) -> int:
    if counts.get("unattributed") != 0:
        raise ValueError(f"{name} ledger contains unattributed evaluations")
    total = sum(int(value) for value in counts.values())
    if total != expected_total:
        raise ValueError(f"{name} ledger does not close")
    return total


def _case_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
    )


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_force_evaluations: int,
) -> dict[str, Any]:
    expected = {
        _case_key(case)
        for case in case_matrix()
    }
    observed = {_case_key(row) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("evidence requires the exact 24-case cohort")

    total = 0
    label_counts: Counter[str] = Counter()
    ambiguous_trajectories: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError("all generation paths must complete")
        total += _validate_ledger(
            row["generation_purpose_counts"],
            row["generation_force_evaluations"],
            name="generation",
        )
        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("each case requires at least one checkpoint")
        for checkpoint in checkpoints:
            total += _validate_ledger(
                checkpoint["purpose_counts"],
                checkpoint["force_evaluations"],
                name="checkpoint",
            )
            label = str(checkpoint["label"])
            if label not in LABELS:
                raise ValueError("unknown checkpoint label")
            label_counts[label] += 1
        summary = summarize_trajectory(checkpoints)
        if row.get("trajectory_summary") != summary:
            raise ValueError("trajectory summary does not close")
        if any(
            checkpoint["label"] in UNLEARNABLE_LABELS
            for checkpoint in checkpoints
        ):
            ambiguous_trajectories[str(row["system"])] += 1

    if total > max_force_evaluations:
        raise ValueError("force-evaluation budget exceeded")

    horizon_gate_contexts = []
    for system in SYSTEMS:
        for state_id in STATE_IDS:
            for arm in ARMS:
                context = [
                    row
                    for row in rows
                    if row["system"] == system
                    and row["state_id"] == state_id
                    and row["arm"] == arm
                ]
                overshoot_seeds = [
                    int(row["seed"])
                    for row in context
                    if row["trajectory_summary"][
                        "early_escape_then_h8_return"
                    ]
                ]
                if len(overshoot_seeds) >= 2:
                    horizon_gate_contexts.append(
                        {
                            "system": system,
                            "state_id": state_id,
                            "arm": arm,
                            "seed_count": len(overshoot_seeds),
                            "seeds": overshoot_seeds,
                        }
                    )

    action_support_gap_contexts = []
    for system in SYSTEMS:
        for state_id in STATE_IDS:
            context = [
                row
                for row in rows
                if row["system"] == system
                and row["state_id"] == state_id
            ]
            by_arm = {
                arm: [
                    row
                    for row in context
                    if row["arm"] == arm
                    and row["trajectory_summary"][
                        "all_horizons_return_starter"
                    ]
                ]
                for arm in ARMS
            }
            if all(len(by_arm[arm]) == len(SEEDS) for arm in ARMS):
                action_support_gap_contexts.append(
                    {
                        "system": system,
                        "state_id": state_id,
                    }
                )

    numerical_matcher_gate_systems = [
        system
        for system in SYSTEMS
        if ambiguous_trajectories[system] >= 2
    ]
    return {
        "schema_version": 1,
        "cohort": {
            "systems": list(SYSTEMS),
            "state_ids": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "checkpoint_horizons": list(CHECKPOINT_HORIZONS),
            "completed_generation_paths": len(rows),
            "checkpoint_count": sum(
                len(row["checkpoints"]) for row in rows
            ),
        },
        "label_counts": dict(sorted(label_counts.items())),
        "horizon_gate_contexts": horizon_gate_contexts,
        "action_support_gap_contexts": action_support_gap_contexts,
        "numerical_matcher_gate_systems": numerical_matcher_gate_systems,
        "unlearnable_trajectory_count_by_system": {
            system: ambiguous_trajectories[system] for system in SYSTEMS
        },
        "total_force_evaluations": total,
        "max_force_evaluations": max_force_evaluations,
        "unattributed_force_evaluations": 0,
        "production_default_changed": False,
        "claim_ceiling": (
            "fixed current-action first-passage mechanism classification; "
            "no adaptive horizon, new action family, or selector is promoted"
        ),
        "cases": list(rows),
    }
