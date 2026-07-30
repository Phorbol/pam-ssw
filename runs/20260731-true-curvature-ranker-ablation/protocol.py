"""Pure paired analysis for the prospective true-curvature ranker ablation."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = ("static_score", "true_curvature")


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


def summarize_repeats(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
        return (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            str(row["arm"]),
        )

    first = {key(row): row for row in first_rows}
    second = {key(row): row for row in second_rows}
    if first.keys() != second.keys():
        raise ValueError("ranker repeats do not contain the same cases")
    expected = {
        (
            case["system"],
            case["state_id"],
            case["seed"],
            case["arm"],
        )
        for case in case_matrix()
        if any(
            row["state_id"] == case["state_id"]
            for row in first_rows
        )
    }
    if not first.keys() <= expected:
        raise ValueError("ranker repeat contains an unknown case")
    if not all(
        bool(row["certificate"]) and bool(row["landing_geometry_valid"])
        for row in list(first.values()) + list(second.values())
    ):
        raise ValueError("ranker repeat contains an uncertified terminal state")

    averaged: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for case_key in first:
        averaged[case_key] = {
            "landing_delta_eV": 0.5
            * (
                float(first[case_key]["landing_delta_eV"])
                + float(second[case_key]["landing_delta_eV"])
            ),
            "force_evaluations": 0.5
            * (
                int(first[case_key]["force_evaluations"])
                + int(second[case_key]["force_evaluations"])
            ),
        }
    paired: dict[tuple[str, str, int], dict[str, dict[str, float]]] = (
        defaultdict(dict)
    )
    for (system, state_id, seed, arm), values in averaged.items():
        paired[(system, state_id, seed)][arm] = values
    if any(set(arms) != set(ARMS) for arms in paired.values()):
        raise ValueError("each ranker group must contain both arms")

    by_system: dict[str, dict[str, Any]] = {}
    for system in sorted({key[0] for key in paired}):
        system_pairs = [
            arms
            for key, arms in paired.items()
            if key[0] == system
        ]
        energy_differences = [
            arms["true_curvature"]["landing_delta_eV"]
            - arms["static_score"]["landing_delta_eV"]
            for arms in system_pairs
        ]
        static_force_evaluations = sum(
            arms["static_score"]["force_evaluations"]
            for arms in system_pairs
        )
        true_force_evaluations = sum(
            arms["true_curvature"]["force_evaluations"]
            for arms in system_pairs
        )
        mean_energy_difference = sum(energy_differences) / len(
            energy_differences
        )
        median_energy_difference = float(median(energy_differences))
        force_difference = int(
            true_force_evaluations - static_force_evaluations
        )
        pareto_improved = bool(
            mean_energy_difference <= 0.0
            and median_energy_difference <= 0.0
            and force_difference <= 0
            and (
                mean_energy_difference < 0.0
                or median_energy_difference < 0.0
                or force_difference < 0
            )
        )
        by_system[system] = {
            "group_count": len(system_pairs),
            "true_curvature_energy_wins": sum(
                difference < 0.0
                for difference in energy_differences
            ),
            "ties": sum(
                difference == 0.0
                for difference in energy_differences
            ),
            "mean_energy_difference_eV": mean_energy_difference,
            "median_energy_difference_eV": median_energy_difference,
            "static_force_evaluations": int(
                static_force_evaluations
            ),
            "true_curvature_force_evaluations": int(
                true_force_evaluations
            ),
            "force_evaluation_difference": force_difference,
            "pareto_improved": pareto_improved,
        }
    return {
        "by_system": by_system,
        "promotion_allowed": bool(
            set(by_system) == set(SYSTEMS)
            and all(
                result["pareto_improved"]
                for result in by_system.values()
            )
        ),
    }
