"""Preregistered analysis for the two-arm short-uphill-rollout gate."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = ("static_score", "true_curvature")
HORIZONS = (1, 2)
PRIMARY_HORIZON = 2


def case_matrix() -> list[dict[str, Any]]:
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "arm": arm,
            "horizon": horizon,
        }
        for system in SYSTEMS
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
        for horizon in HORIZONS
    ]


def _probe_key(
    row: Mapping[str, Any],
) -> tuple[int, str, str, int, str, int]:
    return (
        int(row["repeat_id"]),
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
        int(row["horizon"]),
    )


def _terminal_key(
    row: Mapping[str, Any],
) -> tuple[int, str, str, int, str]:
    return (
        int(row["repeat_id"]),
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def summarize_campaign(
    probe_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assess one fixed physical rule without fitting or post-hoc weighting.

    The primary rule selects the arm with the larger true-PES energy rise after
    two serial Gaussian bias-relax steps. H=1 is diagnostic only.
    """

    if not probe_rows or not terminal_rows:
        raise ValueError("probe and terminal rows are required")
    if any(
        not bool(row["geometry_valid"])
        or bool(row["fragmented"])
        for row in probe_rows
    ):
        raise ValueError("probe contains invalid geometry")
    if any(
        not bool(row["first_direction_matches_terminal"])
        for row in probe_rows
    ):
        raise ValueError("probe first direction does not match terminal label")
    if any(
        not bool(row["certificate"])
        or not bool(row["landing_geometry_valid"])
        for row in terminal_rows
    ):
        raise ValueError("terminal label lacks a force certificate")

    probes = {_probe_key(row): row for row in probe_rows}
    terminals = {_terminal_key(row): row for row in terminal_rows}
    if len(probes) != len(probe_rows) or len(terminals) != len(
        terminal_rows
    ):
        raise ValueError("duplicate probe or terminal case")
    repeats = sorted({key[0] for key in probes})
    if repeats != sorted({key[0] for key in terminals}):
        raise ValueError("probe and terminal repeats differ")

    averaged_probes: dict[
        tuple[str, str, int, str, int], dict[str, float]
    ] = {}
    for system in SYSTEMS:
        for state_id in STATE_IDS:
            for seed in SEEDS:
                for arm in ARMS:
                    for horizon in HORIZONS:
                        rows = [
                            probes[
                                (
                                    repeat,
                                    system,
                                    state_id,
                                    seed,
                                    arm,
                                    horizon,
                                )
                            ]
                            for repeat in repeats
                        ]
                        averaged_probes[
                            (system, state_id, seed, arm, horizon)
                        ] = {
                            "escape_delta_eV": _mean(
                                [
                                    float(row["escape_delta_eV"])
                                    for row in rows
                                ]
                            ),
                            "force_evaluations": _mean(
                                [
                                    float(row["force_evaluations"])
                                    for row in rows
                                ]
                            ),
                            "wall_time_s": _mean(
                                [float(row["wall_time_s"]) for row in rows]
                            ),
                            "direction_step0_force_evaluations": _mean(
                                [
                                    float(
                                        row[
                                            "direction_step0_force_evaluations"
                                        ]
                                    )
                                    for row in rows
                                ]
                            ),
                        }

    averaged_terminals: dict[
        tuple[str, str, int, str], dict[str, float]
    ] = {}
    for system in SYSTEMS:
        for state_id in STATE_IDS:
            for seed in SEEDS:
                for arm in ARMS:
                    rows = [
                        terminals[
                            (repeat, system, state_id, seed, arm)
                        ]
                        for repeat in repeats
                    ]
                    averaged_terminals[
                        (system, state_id, seed, arm)
                    ] = {
                        "landing_delta_eV": _mean(
                            [
                                float(row["landing_delta_eV"])
                                for row in rows
                            ]
                        ),
                        "force_evaluations": _mean(
                            [
                                float(row["force_evaluations"])
                                for row in rows
                            ]
                        ),
                    }

    group_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for state_id in STATE_IDS:
            for seed in SEEDS:
                terminal = {
                    arm: averaged_terminals[
                        (system, state_id, seed, arm)
                    ]
                    for arm in ARMS
                }
                terminal_winner = min(
                    ARMS,
                    key=lambda arm: (
                        terminal[arm]["landing_delta_eV"],
                        ARMS.index(arm),
                    ),
                )
                for horizon in HORIZONS:
                    probe = {
                        arm: averaged_probes[
                            (system, state_id, seed, arm, horizon)
                        ]
                        for arm in ARMS
                    }
                    selected = max(
                        ARMS,
                        key=lambda arm: (
                            probe[arm]["escape_delta_eV"],
                            -ARMS.index(arm),
                        ),
                    )
                    lower_rise_arm = min(
                        ARMS,
                        key=lambda arm: (
                            probe[arm]["escape_delta_eV"],
                            ARMS.index(arm),
                        ),
                    )
                    repeat_selections = []
                    for repeat in repeats:
                        repeat_selections.append(
                            max(
                                ARMS,
                                key=lambda arm: (
                                    float(
                                        probes[
                                            (
                                                repeat,
                                                system,
                                                state_id,
                                                seed,
                                                arm,
                                                horizon,
                                            )
                                        ]["escape_delta_eV"]
                                    ),
                                    -ARMS.index(arm),
                                ),
                            )
                        )
                    rejected = next(
                        arm for arm in ARMS if arm != selected
                    )
                    estimated_online_fe = (
                        terminal[selected]["force_evaluations"]
                        + probe[rejected]["force_evaluations"]
                        - probe[rejected][
                            "direction_step0_force_evaluations"
                        ]
                    )
                    group_rows.append(
                        {
                            "system": system,
                            "state_id": state_id,
                            "seed": seed,
                            "horizon": horizon,
                            "selected_arm": selected,
                            "terminal_winner_arm": terminal_winner,
                            "prediction_correct": (
                                selected == terminal_winner
                            ),
                            "lower_rise_prediction_correct": (
                                lower_rise_arm == terminal_winner
                            ),
                            "prediction_stable": (
                                len(set(repeat_selections)) == 1
                            ),
                            "regret_eV": (
                                terminal[selected]["landing_delta_eV"]
                                - terminal[terminal_winner][
                                    "landing_delta_eV"
                                ]
                            ),
                            "difference_vs_static_eV": (
                                terminal[selected]["landing_delta_eV"]
                                - terminal["static_score"][
                                    "landing_delta_eV"
                                ]
                            ),
                            "probe_force_evaluations": sum(
                                probe[arm]["force_evaluations"]
                                for arm in ARMS
                            ),
                            "probe_serial_wall_time_s": sum(
                                probe[arm]["wall_time_s"]
                                for arm in ARMS
                            ),
                            "probe_ideal_parallel_wall_time_s": max(
                                probe[arm]["wall_time_s"]
                                for arm in ARMS
                            ),
                            "estimated_online_force_evaluations": (
                                estimated_online_fe
                            ),
                            "estimated_online_overhead_vs_static": (
                                estimated_online_fe
                                - terminal["static_score"][
                                    "force_evaluations"
                                ]
                            ),
                        }
                    )

    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        system_results: dict[str, Any] = {}
        for system in SYSTEMS:
            rows = [
                row
                for row in group_rows
                if row["system"] == system
                and row["horizon"] == horizon
            ]
            accuracy = _mean(
                [float(row["prediction_correct"]) for row in rows]
            )
            lower_rise_accuracy = _mean(
                [
                    float(row["lower_rise_prediction_correct"])
                    for row in rows
                ]
            )
            stability = _mean(
                [float(row["prediction_stable"]) for row in rows]
            )
            regrets = [float(row["regret_eV"]) for row in rows]
            differences = [
                float(row["difference_vs_static_eV"])
                for row in rows
            ]
            gate_passed = bool(
                accuracy >= (2.0 / 3.0)
                and stability >= (5.0 / 6.0)
                and median(regrets) <= 1.0e-12
                and _mean(differences) <= 0.0
                and median(differences) <= 0.0
            )
            system_results[system] = {
                "group_count": len(rows),
                "prediction_accuracy": accuracy,
                "lower_rise_prediction_accuracy": (
                    lower_rise_accuracy
                ),
                "prediction_stability": stability,
                "median_regret_eV": float(median(regrets)),
                "mean_regret_eV": _mean(regrets),
                "mean_difference_vs_static_eV": _mean(differences),
                "median_difference_vs_static_eV": float(
                    median(differences)
                ),
                "mean_probe_force_evaluations": _mean(
                    [
                        float(row["probe_force_evaluations"])
                        for row in rows
                    ]
                ),
                "mean_estimated_online_overhead_vs_static": _mean(
                    [
                        float(
                            row[
                                "estimated_online_overhead_vs_static"
                            ]
                        )
                        for row in rows
                    ]
                ),
                "gate_passed": gate_passed,
            }
        by_horizon[str(horizon)] = system_results

    by_system = by_horizon[str(PRIMARY_HORIZON)]
    return {
        "primary_rule": "larger_true_pes_energy_rise",
        "primary_horizon": PRIMARY_HORIZON,
        "by_system": by_system,
        "by_horizon": by_horizon,
        "group_rows": group_rows,
        "online_racing_stage_allowed": bool(
            all(
                by_system[system]["gate_passed"]
                for system in SYSTEMS
            )
        ),
    }
