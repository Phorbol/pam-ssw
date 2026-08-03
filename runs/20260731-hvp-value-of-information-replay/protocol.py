"""Pure analysis for the zero-new-FE initial K4 HVP value gate.

The counterfactual candidate campaign charged the four central finite-
difference HVPs once per shared pool and charged each forced continuation
separately.  This module therefore isolates the value of the *initial* K4
ranking.  It does not estimate the effect of removing HVPs from later walk
steps.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
CANDIDATE_INDICES = (0, 1, 2, 3)
INITIAL_POOL_HVP_COST = 8
NO_HVP_STRATEGIES = ("uniform_no_hvp", "family_rotation_no_hvp")


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
    )


def consolidate_repeats(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average the two repeated terminal observations without imputation."""

    consolidated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["repeat_certified"] = bool(
            row["certificate_first"]
            and row["certificate_second"]
            and row["landing_geometry_valid_first"]
            and row["landing_geometry_valid_second"]
        )
        row["mean_landing_delta_eV"] = 0.5 * (
            float(row["landing_delta_eV_first"])
            + float(row["landing_delta_eV_second"])
        )
        row["mean_force_evaluations"] = 0.5 * (
            float(row["force_evaluations_first"])
            + float(row["force_evaluations_second"])
        )
        consolidated.append(row)
    return sorted(
        consolidated,
        key=lambda row: (*_group_key(row), int(row["candidate_index"])),
    )


def family_rotation_schedule() -> dict[tuple[str, str, int], str]:
    """Return a result-independent 3:3 bond/random schedule per system.

    The alternating parity is fixed by the protocol axes, not by an observed
    energy, curvature, or candidate score.  Within the selected family the
    counterfactual is the equal-probability expectation over its two members.
    """

    schedule: dict[tuple[str, str, int], str] = {}
    for system in SYSTEMS:
        for state_index, state_id in enumerate(STATE_IDS):
            for seed_index, seed in enumerate(SEEDS):
                schedule[(system, state_id, seed)] = (
                    "bond"
                    if (state_index + seed_index) % 2 == 0
                    else "random"
                )
    return schedule


def _expected_strategy(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("an expectation requires at least one candidate")
    return {
        "landing_delta_eV": sum(
            float(row["mean_landing_delta_eV"]) for row in rows
        )
        / len(rows),
        "projected_force_evaluations": sum(
            float(row["mean_force_evaluations"]) for row in rows
        )
        / len(rows),
    }


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    rotated_family: str,
    initial_pool_hvp_cost: int = INITIAL_POOL_HVP_COST,
) -> dict[str, Any]:
    """Compare initial-selection rules on one exact four-candidate pool."""

    ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
    if [int(row["candidate_index"]) for row in ordered] != list(
        CANDIDATE_INDICES
    ):
        raise ValueError("each group must contain candidate indices 0..3")
    if not all(bool(row["repeat_certified"]) for row in ordered):
        raise ValueError("all candidates require paired strict certificates")
    static_winners = [
        row for row in ordered if int(row["static_rank"]) == 1
    ]
    if len(static_winners) != 1:
        raise ValueError("each group must contain one static K4 winner")
    family_rows = [
        row for row in ordered if str(row["kind"]) == rotated_family
    ]
    if len(family_rows) != 2:
        raise ValueError("the rotated family must contain exactly two candidates")

    best_delta = min(float(row["mean_landing_delta_eV"]) for row in ordered)
    static = static_winners[0]
    static_delta = float(static["mean_landing_delta_eV"])
    static_cost = (
        float(static["mean_force_evaluations"]) + initial_pool_hvp_cost
    )
    uniform = _expected_strategy(ordered)
    family = _expected_strategy(family_rows)
    return {
        "best_pool_landing_delta_eV": best_delta,
        "rotated_family": rotated_family,
        "static_k4": {
            "landing_delta_eV": static_delta,
            "terminal_regret_eV": static_delta - best_delta,
            "projected_force_evaluations": static_cost,
        },
        "uniform_no_hvp": {
            **uniform,
            "terminal_regret_eV": uniform["landing_delta_eV"] - best_delta,
        },
        "family_rotation_no_hvp": {
            **family,
            "terminal_regret_eV": family["landing_delta_eV"] - best_delta,
        },
    }


def _summarize_system(
    group_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for strategy in ("static_k4", *NO_HVP_STRATEGIES):
        strategy_rows = [row[strategy] for row in group_rows]
        result[strategy] = {
            "group_count": len(strategy_rows),
            "median_terminal_regret_eV": float(
                median(
                    float(row["terminal_regret_eV"])
                    for row in strategy_rows
                )
            ),
            "mean_terminal_regret_eV": sum(
                float(row["terminal_regret_eV"]) for row in strategy_rows
            )
            / len(strategy_rows),
            "median_projected_force_evaluations": float(
                median(
                    float(row["projected_force_evaluations"])
                    for row in strategy_rows
                )
            ),
            "mean_projected_force_evaluations": sum(
                float(row["projected_force_evaluations"])
                for row in strategy_rows
            )
            / len(strategy_rows),
        }
    return result


def evaluate_gate(
    by_system: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Apply the preregistered cross-system quality-and-cost decision."""

    decisions: dict[str, Any] = {}
    for strategy in NO_HVP_STRATEGIES:
        systems: dict[str, Any] = {}
        quality_deltas: list[float] = []
        for system in SYSTEMS:
            baseline = by_system[system]["static_k4"]
            candidate = by_system[system][strategy]
            quality_delta = float(
                candidate["median_terminal_regret_eV"]
                - baseline["median_terminal_regret_eV"]
            )
            cost_delta = float(
                candidate["median_projected_force_evaluations"]
                - baseline["median_projected_force_evaluations"]
            )
            quality_deltas.append(quality_delta)
            systems[system] = {
                "quality_delta_eV": quality_delta,
                "projected_force_evaluation_delta": cost_delta,
                "quality_not_worse": quality_delta <= 0.0,
                "projected_cost_lower": cost_delta < 0.0,
                "pass": quality_delta <= 0.0 and cost_delta < 0.0,
            }
        sign_flip = min(quality_deltas) < 0.0 < max(quality_deltas)
        decisions[strategy] = {
            "by_system": systems,
            "quality_sign_flip": sign_flip,
            "cross_system_pass": all(
                systems[system]["pass"] for system in SYSTEMS
            ),
        }
    decisions["live_selected_only_hvp_gate_allowed"] = any(
        decisions[strategy]["cross_system_pass"]
        for strategy in NO_HVP_STRATEGIES
    )
    decisions["initial_all_candidate_hvp_deletion_supported"] = bool(
        decisions["live_selected_only_hvp_gate_allowed"]
        and not any(
            decisions[strategy]["quality_sign_flip"]
            for strategy in NO_HVP_STRATEGIES
            if decisions[strategy]["cross_system_pass"]
        )
    )
    return decisions


def analyze_candidate_repeats(
    consolidated_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in consolidated_rows:
        grouped[_group_key(row)].append(dict(row))

    schedule = family_rotation_schedule()
    summaries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key in sorted(schedule):
        rows = grouped.get(key, [])
        reason = None
        if len(rows) != len(CANDIDATE_INDICES):
            reason = "incomplete_candidate_pool"
        elif not all(bool(row.get("repeat_certified")) for row in rows):
            reason = "uncertified_repeat_pair"
        if reason is not None:
            excluded.append(
                {
                    "system": key[0],
                    "state_id": key[1],
                    "seed": key[2],
                    "reason": reason,
                }
            )
            continue
        summaries.append(
            {
                "system": key[0],
                "state_id": key[1],
                "seed": key[2],
                **summarize_group(
                    rows,
                    rotated_family=schedule[key],
                ),
            }
        )

    by_system = {
        system: _summarize_system(
            [row for row in summaries if row["system"] == system]
        )
        for system in SYSTEMS
        if any(row["system"] == system for row in summaries)
    }
    gate = (
        evaluate_gate(by_system)
        if set(by_system) == set(SYSTEMS)
        else {
            "live_selected_only_hvp_gate_allowed": False,
            "initial_all_candidate_hvp_deletion_supported": False,
            "reason": "both systems require eligible groups",
        }
    )
    return {
        "quality": {
            "complete_expected_group_count": len(schedule),
            "eligible_group_count": len(summaries),
            "excluded_group_count": len(excluded),
            "excluded_groups": excluded,
        },
        "group_summaries": summaries,
        "by_system": by_system,
        "gate": gate,
        "scope": (
            "initial K4 candidate ranking only; continuation HVP policy is "
            "unchanged in every forced trajectory"
        ),
    }


def summarize_d0_k4_live_cases(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the independent live D0/K4 anchor without mixing pools."""

    result: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [
            row
            for row in cases
            if row["system"] == system
            and row["arm"] in {"D0_exact_anchor", "K4_discrete"}
            and bool(row["certificate"])
            and bool(row["landing_geometry_valid"])
            and not bool(row["fragmented"])
        ]
        by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(
            dict
        )
        for row in system_rows:
            by_key[(str(row["state_id"]), int(row["seed"]))][
                str(row["arm"])
            ] = row
        pairs = [
            arms
            for arms in by_key.values()
            if set(arms) == {"D0_exact_anchor", "K4_discrete"}
        ]
        result[system] = {
            "pair_count": len(pairs),
            "median_d0_minus_k4_landing_delta_eV": float(
                median(
                    float(pair["D0_exact_anchor"]["landing_delta_eV"])
                    - float(pair["K4_discrete"]["landing_delta_eV"])
                    for pair in pairs
                )
            ),
            "median_d0_minus_k4_force_evaluations": float(
                median(
                    float(pair["D0_exact_anchor"]["force_evaluations"])
                    - float(pair["K4_discrete"]["force_evaluations"])
                    for pair in pairs
                )
            ),
        }
    return result
