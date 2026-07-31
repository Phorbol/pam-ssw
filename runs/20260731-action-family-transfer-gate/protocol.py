"""Pure analysis for the preregistered action-family transfer gate."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = (
    "D0_exact_anchor",
    "D1_anchor_krylov_d2",
    "K4_discrete",
)


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


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("arm rows cannot be empty")
    deltas = [float(row["landing_delta_eV"]) for row in rows]
    costs = [int(row["force_evaluations"]) for row in rows]
    return {
        "count": len(rows),
        "mean_landing_delta_eV": float(fmean(deltas)),
        "median_landing_delta_eV": float(median(deltas)),
        "mean_force_evaluations": float(fmean(costs)),
        "median_force_evaluations": float(median(costs)),
        "lower_landing_count": sum(delta < 0.0 for delta in deltas),
    }


def _is_dominated(
    selected_arm: str,
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str | None]:
    selected = summaries[selected_arm]
    selected_quality = float(selected["mean_landing_delta_eV"])
    selected_cost = float(selected["mean_force_evaluations"])
    for arm in ARMS:
        if arm == selected_arm:
            continue
        candidate = summaries[arm]
        candidate_quality = float(candidate["mean_landing_delta_eV"])
        candidate_cost = float(candidate["mean_force_evaluations"])
        if (
            candidate_quality <= selected_quality
            and candidate_cost <= selected_cost
            and (
                candidate_quality < selected_quality
                or candidate_cost < selected_cost
            )
        ):
            return True, arm
    return False, None


def summarize_campaign(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows cannot be empty")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    integrity_failures = []
    for row in rows:
        system = str(row["system"])
        arm = str(row["arm"])
        if system not in SYSTEMS or arm not in ARMS:
            raise ValueError("unknown system or arm")
        grouped[(system, arm)].append(row)
        if (
            not bool(row["certificate"])
            or not bool(row["landing_geometry_valid"])
            or bool(row["fragmented"])
        ):
            integrity_failures.append(
                {
                    "system": system,
                    "state_id": str(row["state_id"]),
                    "seed": int(row["seed"]),
                    "arm": arm,
                }
            )

    present_systems = tuple(
        system
        for system in SYSTEMS
        if any(group_system == system for group_system, _ in grouped)
    )
    missing = [
        (system, arm)
        for system in present_systems
        for arm in ARMS
        if (system, arm) not in grouped
    ]
    if missing:
        raise ValueError(f"missing system-arm groups: {missing}")

    by_system = {
        system: {
            arm: _arm_summary(grouped[(system, arm)])
            for arm in ARMS
        }
        for system in present_systems
    }
    held_out = {}
    transfer_pairs = (
        (("c60", "pdo"), ("pdo", "c60"))
        if set(present_systems) == set(SYSTEMS)
        else ()
    )
    for system, training_system in transfer_pairs:
        selected_arm = min(
            ARMS,
            key=lambda arm: (
                float(
                    by_system[training_system][arm][
                        "mean_landing_delta_eV"
                    ]
                ),
                ARMS.index(arm),
            ),
        )
        held_out_summary = by_system[system]
        selected_quality = float(
            held_out_summary[selected_arm]["mean_landing_delta_eV"]
        )
        uniform_mean = float(
            fmean(
                float(summary["mean_landing_delta_eV"])
                for summary in held_out_summary.values()
            )
        )
        dominated, dominating_arm = _is_dominated(
            selected_arm,
            held_out_summary,
        )
        held_out[system] = {
            "training_system": training_system,
            "selected_arm": selected_arm,
            "selected_mean_landing_delta_eV": selected_quality,
            "uniform_arm_mean_landing_delta_eV": uniform_mean,
            "improvement_over_uniform_eV": uniform_mean - selected_quality,
            "beats_uniform_mean": selected_quality < uniform_mean,
            "pareto_dominated": dominated,
            "dominating_arm": dominating_arm,
            "passes": selected_quality < uniform_mean and not dominated,
        }

    return {
        "by_system": by_system,
        "held_out": held_out,
        "integrity_failures": integrity_failures,
        "promotion_allowed": (
            not integrity_failures
            and set(present_systems) == set(SYSTEMS)
            and all(summary["passes"] for summary in held_out.values())
        ),
    }


__all__ = [
    "ARMS",
    "SEEDS",
    "STATE_IDS",
    "SYSTEMS",
    "case_matrix",
    "summarize_campaign",
]
